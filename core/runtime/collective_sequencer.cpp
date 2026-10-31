/* collective_sequencer.cpp -- group-wide collective order implementation. */
#include "collective_sequencer.h"

#include <algorithm>
#include <chrono>
#include <immintrin.h>

namespace rad {

CollectiveSequencer& rad_collective_sequencer() {
    static CollectiveSequencer s;
    return s;
}

CollectiveSequencer::~CollectiveSequencer() {
    reset();
}

void CollectiveSequencer::ensure_events_locked(int rank) {
    if (rank < 0 || rank >= kMaxRanks || events_init_[rank]) return;
    for (int i = 0; i < kEventRing; ++i) {
        if (!events_[rank][i]) {
            (void)rad_event_create_local(&events_[rank][i]);
        }
    }
    events_init_[rank] = true;
}

void CollectiveSequencer::reset() {
    std::lock_guard<std::mutex> lk(m_);
    aborted_ = true;
    cv_.notify_all();
    for (int r = 0; r < kMaxRanks; ++r) {
        for (int i = 0; i < kEventRing; ++i) {
            if (events_[r][i]) {
                rad_event_destroy(events_[r][i]);
                events_[r][i] = nullptr;
            }
        }
        events_init_[r] = false;
        current_ticket_rank_[r] = 0;
        last_owner_rank_[r] = nullptr;
        last_event_rank_[r] = nullptr;
    }
    tickets_.clear();
    next_ticket_ = 0;
    registered_owners_.clear();
    max_world_size_ = 1;
    active_engines_.store(0, std::memory_order_release);
    aborted_ = false;
}

void CollectiveSequencer::abort() {
    std::lock_guard<std::mutex> lk(m_);
    aborted_ = true;
    cv_.notify_all();
}

void CollectiveSequencer::register_engine(const void* owner, int world_size) {
    if (!owner) return;
    std::lock_guard<std::mutex> lk(m_);
    if (registered_owners_.insert(owner).second) {
        if (world_size > max_world_size_ && world_size <= kMaxRanks) {
            max_world_size_ = world_size;
        }
        active_engines_.store((int)registered_owners_.size(), std::memory_order_release);
    }
}

void CollectiveSequencer::unregister_engine(const void* owner) {
    if (!owner) return;
    std::lock_guard<std::mutex> lk(m_);
    if (registered_owners_.erase(owner) > 0) {
        active_engines_.store((int)registered_owners_.size(), std::memory_order_release);
        if (registered_owners_.empty()) {
            tickets_.clear();
            next_ticket_ = 0;
            max_world_size_ = 1;
            for (int r = 0; r < kMaxRanks; ++r) {
                current_ticket_rank_[r] = 0;
                last_owner_rank_[r] = nullptr;
                last_event_rank_[r] = nullptr;
            }
        }
        cv_.notify_all();
    }
}

const void* CollectiveSequencer::ticket_owner_locked(uint64_t t) const {
    if (tickets_.empty() || t < tickets_.front().id || t > tickets_.back().id) return nullptr;
    return tickets_[t - tickets_.front().id].owner;
}

void CollectiveSequencer::cleanup_tickets_locked() {
    uint64_t min_completed = current_ticket_rank_[0];
    const int bound = std::min(kMaxRanks, std::max(1, max_world_size_));
    for (int r = 1; r < bound; ++r) {
        min_completed = std::min(min_completed, current_ticket_rank_[r]);
    }
    while (!tickets_.empty() && tickets_.front().id < min_completed) {
        tickets_.pop_front();
    }
}

/* Spin before sleeping: the hand-off between the ranks of two engines is a few microseconds of
 * work, and a condition-variable sleep and wake costs ~0.5 ms on this host, once per collective. */
template <class Pred>
static void wait_turn(std::condition_variable& cv, std::unique_lock<std::mutex>& lk, Pred pred) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(300);
    while (!pred() && std::chrono::steady_clock::now() < until) {
        lk.unlock();
        for (int i = 0; i < 16; ++i) _mm_pause();
        lk.lock();
    }
    cv.wait(lk, pred);
}

uint64_t CollectiveSequencer::before_collective(const void* owner, int rank, int world_size, RadStream stream) {
    if (is_single_engine() || rank < 0 || rank >= kMaxRanks) return 0;

    uint64_t ticket = 0;
    RadEvent wait_ev = nullptr;

    {
        std::unique_lock<std::mutex> lk(m_);
        ensure_events_locked(rank);
        if (world_size > max_world_size_ && world_size <= kMaxRanks) {
            max_world_size_ = world_size;
        }

        if (rank == 0) {
            ticket = next_ticket_++;
            tickets_.push_back(TicketEntry{ ticket, owner });
            cv_.notify_all();

            wait_turn(cv_, lk, [&] {
                return aborted_ || current_ticket_rank_[0] == ticket;
            });
            if (aborted_) return ticket;
        } else {
            wait_turn(cv_, lk, [&] {
                if (aborted_) return true;
                if (current_ticket_rank_[rank] >= next_ticket_) return false;
                return ticket_owner_locked(current_ticket_rank_[rank]) == owner;
            });
            if (aborted_) return 0;
            ticket = current_ticket_rank_[rank];
        }

        if (last_owner_rank_[rank] != nullptr && last_owner_rank_[rank] != owner && last_event_rank_[rank] != nullptr) {
            wait_ev = last_event_rank_[rank];
        }
    }

    if (wait_ev && stream) {
        (void)rad_event_wait(stream, wait_ev);
    }

    return ticket;
}

void CollectiveSequencer::after_collective(const void* owner, int rank, int world_size, RadStream stream, uint64_t ticket) {
    if (is_single_engine() || rank < 0 || rank >= kMaxRanks) return;

    RadEvent ev = nullptr;
    {
        std::lock_guard<std::mutex> lk(m_);
        ensure_events_locked(rank);
        ev = events_[rank][ticket % kEventRing];
    }

    if (ev && stream) {
        (void)rad_event_record(ev, stream);
    }

    {
        std::lock_guard<std::mutex> lk(m_);
        last_event_rank_[rank] = ev;
        last_owner_rank_[rank] = owner;
        current_ticket_rank_[rank]++;
        cleanup_tickets_locked();
        cv_.notify_all();
    }
}

}  /* namespace rad */
