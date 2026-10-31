/* core/model/weight_store.h -- the host-RAM backing of a deactivated model (spec #1, ticket #12).
 *
 * A model's weights are put in GPU-ready form once: the loader reads the checkpoint or container,
 * cuts this rank's share and, where the resolved kernel wants another layout, rearranges it
 * (Engine::load_weights). The PreparedWeightStore keeps those finished bytes, a copy taken while
 * the first load already holds them in host memory (never read back from the card). The model
 * is then a vorbereitetes Model (GLOSSARY.md) once its Engine is destroyed, and the next activation
 * uploads the stored bytes through the load pool's bounded pinned staging ring: no checkpoint
 * parsing, no relayout, no D2H copy, and neither dequantisation nor requantisation, since the
 * bytes are the ones the kernels read.
 *
 * Backing. Every entry here is RAM backing: an anonymous allocation of this process, so it is
 * what the host budget of ticket #13 will count and evict. Weights the loader does not copy at all
 * (the mapped tier, e.g. an n-gram table read in place from the container's mapping, and the file
 * tier) have no entry: they are page-cache or disk backed, belong to the Engine that maps the
 * container, and are served from the page cache on every activation. A container weight the
 * loader reads unchanged could also be referenced in the container's mapping instead of copied;
 * that would save the RAM and give up the guarantee of "no disk read", so it is not done: a
 * prepared model reads no source bytes.
 *
 * Keys. A model's entries are valid under one `model key` (artifact identity, build identity,
 * the config that shapes the bytes); a load under another key drops them (the model is cold
 * again). Inside it each weight is found by rank, name and a fingerprint of its stored form
 * (encoding, layout tag, dtype, size, share), so a changed layout or a different TP share is a
 * miss, never a wrong buffer.
 *
 * The store belongs to the process and outlives every Engine; its memory is released when it is
 * destroyed.
 *
 * Host budget (ticket #13, docs/swap/host-budget.md). The store is also the process's ledger of
 * host RAM: its entries, the host pools of the active models (`claim` Pools) and the loads' staging
 * (`claim` Staging) are `used()`. With a budget (`cache.host_budget_mib`) the three together never
 * pass it by the store's doing: a copy that would not fit evicts the vorbereitete Model idle the
 * longest (LRU), which is then kalt, and if all of those are gone the copy is not made. A model
 * the Engine marks `activate`d is never evicted as a model; only its copy can be dropped, and only
 * when a mandatory claim (pools, or a host check of the Engine) cannot be met otherwise, after which
 * the model is kalt once deactivated. Without a budget the Engine's host checks (lockable_room)
 * decide, and evict through `evict_one` the same way when they refuse. */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rad_format.h"

namespace rad {

/* What one Engine::load_weights did, for status and metrics. */
struct WeightLoadReport {
    bool    warm = false;              /* every weight was placed from the store, none from disk */
    bool    captured = false;          /* this load put its weights in the store */
    int64_t weights = 0;               /* weights placed into a pool (mapped and file tier not counted) */
    int64_t weights_from_store = 0;
    int64_t bytes = 0;                 /* their stored bytes */
    int64_t bytes_from_store = 0;
    int64_t mapped_bytes = 0;          /* read in place from the container's mapping (page cache) */
    int64_t source_read_bytes = 0;     /* checkpoint or container bytes the loader read */
    int64_t disk_read_bytes = -1;      /* /proc/self/io read_bytes over the load; -1: unreadable */
    /* RAM-Streaming (ticket #18), from the placement plan summed over ranks: the static
     * (non-expert) weight bytes, and those of them read from the pinned host pool at run time
     * because the card had no room. Their ratio is the model's streamed share. */
    int64_t static_bytes = 0;
    int64_t static_streamed_bytes = 0;
    double  seconds = 0;
};

/* The artifact a path names: a file's path, size and mtime, or for a directory those of every file
 * in it. Empty if it cannot be read. */
std::string weight_artifact_identity(const std::string& path);

/* The process's read_bytes from /proc/self/io (what the block layer was asked for, page-cache
 * misses and mmap faults included); -1 if unreadable. */
int64_t proc_io_read_bytes();

/* Host RAM in use, by kind, and what the budget did (PreparedWeightStore::stats). */
struct HostMemoryStats {
    int64_t budget_bytes = 0;          /* cache.host_budget_mib; 0: derived from the host's limits */
    int64_t store_bytes = 0;           /* entries and loads in progress, prepared and active models */
    int64_t pool_bytes = 0;            /* host pools, prefix-cache tier, pinned tables of active models */
    int64_t staging_bytes = 0;         /* the loads' pinned staging ring and relayout window */
    int64_t evicted_prepared = 0;      /* vorbereitete Models made kalt */
    int64_t evicted_active_copies = 0; /* copies of active models dropped (kalt once deactivated) */
    int64_t evicted_bytes = 0;
    int64_t over_budget_claims = 0;    /* mandatory claims that stayed over the budget after eviction */
    int64_t used() const { return store_bytes + pool_bytes + staging_bytes; }
};

enum class HostUse { Pools, Staging };

/* A store buffer is an anonymous mmap: freeing it returns the memory to the host at once, which an
 * evicted model's room depends on (a malloc arena may keep it). */
struct WeightBufferUnmap {
    size_t bytes = 0;
    void operator()(uint8_t* p) const;
};

class PreparedWeightStore {
public:
    ~PreparedWeightStore();
    /* Starts a load of `model` under `key`: entries held under another key are dropped. */
    void begin(const std::string& model, const std::string& key);
    /* The prepared bytes of the weight `entry` (rank, name, fingerprint) of exactly `bytes`, or null. */
    const uint8_t* find(const std::string& model, const std::string& entry, int64_t bytes) const;
    /* A buffer of `bytes` the load fills with the weight's stored form; null if memory is short.
     * Not visible to find() until commit(). */
    uint8_t* reserve(const std::string& model, const std::string& entry, int64_t bytes);
    /* The load succeeded: its reserved buffers become entries. */
    void commit(const std::string& model);
    /* The load did not: its reserved buffers are freed. */
    void abort(const std::string& model);
    /* Whether the model holds entries, i.e. is a vorbereitetes Model once deactivated. */
    bool prepared(const std::string& model) const;
    int64_t ram_bytes(const std::string& model) const;
    int64_t ram_bytes() const;
    /* Frees the model's entries (it becomes cold). */
    void release(const std::string& model);

    /* ---- host budget (ticket #13) ---- */
    /* The explicit budget in bytes; 0: none, the Engine's host checks decide. */
    void set_budget(int64_t bytes);
    /* Called, under the store's lock (it must not call back), when a vorbereitetes Model is made
     * kalt by eviction, and by publish_state(). */
    using StateListener = std::function<void(const std::string& model, bool prepared, int64_t bytes)>;
    void set_state_listener(StateListener l);
    /* Reports the model's state to the listener (after its Engine is gone). */
    void publish_state(const std::string& model);
    /* The model has an Engine: it is not vorbereitet, its copy is evicted last. deactivate()
     * ends that, drops its claims, and stamps the model as used now (LRU). */
    void activate(const std::string& model);
    void deactivate(const std::string& model);
    /* The model's Engine takes `bytes` more host RAM of `kind`. Mandatory: it is recorded whatever
     * the budget says, after evicting copies (prepared models first, then active ones, the
     * model's own last) while a budget is exceeded. False if it still exceeds it. */
    bool claim(const std::string& model, HostUse kind, int64_t bytes);
    void unclaim(const std::string& model, HostUse kind, int64_t bytes);
    /* Room for `bytes` more of the model's copy under the budget, evicting prepared models LRU;
     * false if there is none (no budget: always true). */
    bool make_room(const std::string& model, int64_t bytes);
    /* Frees one copy that is not `requester`'s, for a host check that refused: the prepared model
     * idle longest; with `include_active`, then the active model's copy that was activated
     * longest ago, then the requester's own. False if nothing could be freed. */
    bool evict_one(const std::string& requester, bool include_active);
    HostMemoryStats stats() const;

    /* ---- expert heat (ticket #20, docs/swap/heat-warmup.md) ----
     * The model's expert heat at its deactivation, as a profile: layer, expert and share, no unit,
     * slot or address. Kept apart from the weight entries, so evicting the weights (kalt) or a
     * changed key leaves it; it is a few bytes per expert and not counted by the host budget. */
    void save_heat(const std::string& model, std::vector<RadFileProfile> heat);
    /* Empty if none was saved. */
    std::vector<RadFileProfile> heat(const std::string& model) const;

private:
    struct Buffer {
        std::unique_ptr<uint8_t, WeightBufferUnmap> data;
        int64_t bytes = 0;
    };
    struct Model {
        std::string key;
        std::unordered_map<std::string, Buffer> entries, pending;
        int64_t bytes = 0, pending_bytes = 0;
        int64_t pools = 0, staging = 0;
        bool active = false, loading = false;
        uint64_t used_at = 0;
    };
    void drop_entries_locked(Model& m);
    int64_t used_locked() const;
    bool evict_locked(const std::string& requester, bool include_active);
    mutable std::mutex m_;
    std::unordered_map<std::string, Model> models_;
    std::unordered_map<std::string, std::vector<RadFileProfile>> heat_;
    int64_t budget_ = 0;
    uint64_t clock_ = 0;
    HostMemoryStats stats_;
    StateListener listener_;
};

}  /* namespace rad */
