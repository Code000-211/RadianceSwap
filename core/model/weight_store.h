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
 * destroyed. There is no budget or eviction yet (ticket #13). */
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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
    double  seconds = 0;
};

/* The artifact a path names: a file's path, size and mtime, or for a directory those of every file
 * in it. Empty if it cannot be read. */
std::string weight_artifact_identity(const std::string& path);

/* The process's read_bytes from /proc/self/io (what the block layer was asked for, page-cache
 * misses and mmap faults included); -1 if unreadable. */
int64_t proc_io_read_bytes();

class PreparedWeightStore {
public:
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

private:
    struct Buffer {
        std::unique_ptr<uint8_t[]> data;
        int64_t bytes = 0;
    };
    struct Model {
        std::string key;
        std::unordered_map<std::string, Buffer> entries, pending;
        int64_t bytes = 0;
    };
    mutable std::mutex m_;
    std::unordered_map<std::string, Model> models_;
};

}  /* namespace rad */
