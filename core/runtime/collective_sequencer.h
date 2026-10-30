/* collective_sequencer.h -- process-wide collective sequencing across engines.
 *
 * Multiple engines (model runtimes) on the same GPU group must execute their collectives
 * (tensor-parallel all-reduces) in an identical global order across all ranks to prevent
 * crosswise deadlock between spinning peer flags (spec #1, ADR 0003).
 *
 * Mechanism:
 * - Rank 0 assigns monotonic tickets; other ranks follow that exact sequence.
 * - Device-side event chain orders stream execution without host stalling.
 * - Single-engine fast path: zero overhead (relaxed atomic check, no mutex/events).
 */
#pragma once
#include "../rad_core.h"
#include "../device/device.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_set>

namespace rad {

class CollectiveSequencer {
public:
    CollectiveSequencer() = default;
    ~CollectiveSequencer();

    CollectiveSequencer(const CollectiveSequencer&) = delete;
    CollectiveSequencer& operator=(const CollectiveSequencer&) = delete;

    /* Register / unregister an engine (tracked by owner pointer). */
    void register_engine(const void* owner, int world_size);
    void unregister_engine(const void* owner);

    /* Fast-path check: true when at most one engine is registered. */
    bool is_single_engine() const {
        return active_engines_.load(std::memory_order_relaxed) <= 1;
    }

    size_t active_engines() const {
        return (size_t)std::max(0, active_engines_.load(std::memory_order_relaxed));
    }

    /* Before issuing a collective on device (rank 0 assigns ticket order, other ranks follow): */
    uint64_t before_collective(const void* owner, int rank, int world_size, RadStream stream);

    /* After launching a collective on device (records completion event, advances ticket): */
    void     after_collective(const void* owner, int rank, int world_size, RadStream stream, uint64_t ticket);

    /* Abort waiting ranks on shutdown or device fault. */
    void abort();
    void reset();

private:
    static constexpr int kMaxRanks  = 8;
    static constexpr int kEventRing = 8;

    struct TicketEntry {
        uint64_t    id;
        const void* owner;
    };

    mutable std::mutex              m_;
    std::condition_variable         cv_;
    std::atomic<int>                active_engines_{ 0 };
    std::unordered_set<const void*> registered_owners_;
    int                             max_world_size_ = 1;
    bool                            aborted_        = false;

    uint64_t                        next_ticket_ = 0;
    std::deque<TicketEntry>         tickets_;
    uint64_t                        current_ticket_rank_[kMaxRanks] = { 0 };
    const void*                     last_owner_rank_[kMaxRanks]     = { nullptr };
    RadEvent                        last_event_rank_[kMaxRanks]     = { nullptr };

    RadEvent                        events_[kMaxRanks][kEventRing]  = {};
    bool                            events_init_[kMaxRanks]         = { false };

    void        ensure_events_locked(int rank);
    void        cleanup_tickets_locked();
    const void* ticket_owner_locked(uint64_t t) const;
};

/* shortcut: one sequencer per process, i.e. one GPU group; key it by device set when several groups are served. */
CollectiveSequencer& rad_collective_sequencer();

}  /* namespace rad */
