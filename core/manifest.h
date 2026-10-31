/* manifest.h -- the models file and the registry built from it (spec #1, ticket #7).
 *
 * `--manifest FILE` replaces `--model` with a JSON file:
 *
 *   { "gpu_group": { "devices": [0, 1], "tp": 2 },
 *     "cache":     { "host_budget_mib": 0, "staging_budget_mib": 0, "idle_timeout_ms": 0,
 *                    "max_queued_requests": 0, "warmup_budget_mib": 0,
 *                    "activation_backoff_ms": 0, "activation_backoff_max_ms": 0 },
 *     "models": [ { "id": "a", "path": "/m/a.rad", "default": true,
 *                   "engine": ["--max-model-len", "8192"] } ] }
 *
 * JSON because nlohmann/json is already a dependency and the project parses no YAML. A model's
 * `engine` is an argv-style list handed to config_parse over the command line's own options, so
 * there is one option parser and one validation. Every model is its own Engine on the same GPU
 * group (main.cpp, server/model_router.h): the default model is started at once, the others on
 * their first request. Each has a fixed VRAM share, and the group planner (place/group_planner.h)
 * divides the experts between them inside those shares. `cache.max_queued_requests` bounds the
 * requests waiting for one cold model (0: 64), and `cache.idle_timeout_ms` is how long an active
 * model may go without a request before it is deactivated (0: never). A model whose activation
 * failed is not tried again for `cache.activation_backoff_ms` (0: 1000), doubling with each failure
 * in a row up to `cache.activation_backoff_max_ms` (0: 60000). The gpu_group section and
 * the other cache values are only parsed and checked here.
 *
 * The registry belongs to the process (ADR 0001), not to an Engine. */
#pragma once

#include "rad_core.h"

#include <mutex>
#include <string>
#include <vector>

namespace rad {

/* KALT, VORBEREITEND, VORBEREITET, AKTIVIEREND, AKTIV, DEAKTIVIEREND, FEHLGESCHLAGEN (GLOSSARY.md). */
enum class ModelState { Cold, Preparing, Prepared, Activating, Active, Deactivating, Failed };
const char* model_state_name(ModelState s);

struct ManifestGroup {
    std::vector<int> devices;   /* empty: the visible cards, as without a manifest */
};

struct ManifestCache {
    int64_t host_budget_mib = 0;
    int64_t staging_budget_mib = 0;
    int64_t idle_timeout_ms = 0;
    int64_t max_queued_requests = 0;
    int64_t warmup_budget_mib = 0;
    int64_t activation_backoff_ms = 0;
    int64_t activation_backoff_max_ms = 0;
};

struct ModelRecord {
    std::string id;
    std::string path;
    bool        is_default = false;
    Config      config;        /* the command line's options with the model's own `engine` over them */
    ModelState  state = ModelState::Cold;
};

class ModelRegistry {
public:
    ManifestGroup group;
    ManifestCache cache;

    const std::vector<ModelRecord>& models() const { return models_; }
    const ModelRecord* find(const std::string& id) const;
    /* Valid only for a registry a manifest_* call filled. */
    const ModelRecord& default_model() const;
    /* The state is written by the activation threads (server/model_router.h) and read from any
     * thread: go through these two, not through ModelRecord::state, once the router runs. */
    void       set_state(const std::string& id, ModelState s);
    ModelState state_of(const std::string& id) const;
    void reset();
    void add(ModelRecord r) { models_.push_back(std::move(r)); }

private:
    std::vector<ModelRecord> models_;
    mutable std::mutex       state_m_;
};

/* The fixed VRAM share of a model started while its siblings are not (ticket #8). The budget
 * resolver (mem/vram_budget.h) claims everything free above --gpu-headroom-mib, so a model's share
 * is set by the headroom it is started with: the card's free memory now less the share, never
 * below `base_mib` (the --gpu-headroom-mib the model would have had). The last model starts with
 * its own headroom and takes what the others left. */
int64_t manifest_headroom_mib(int64_t free_mib, int64_t share_mib, int64_t base_mib);

/* Fills `out` from the manifest, each model's Config layered on `base` (the parsed command line).
 * RAD_E_INVAL, after logging what is wrong, for a malformed or inconsistent manifest. */
int manifest_parse(const std::string& text, const Config& base, ModelRegistry* out);
/* Reads `file`, then manifest_parse. */
int manifest_load(const std::string& file, const Config& base, ModelRegistry* out);

}  /* namespace rad */
