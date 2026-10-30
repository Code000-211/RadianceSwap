/* manifest.h -- the models file and the registry built from it (spec #1, ticket #7).
 *
 * `--manifest FILE` replaces `--model` with a JSON file:
 *
 *   { "gpu_group": { "devices": [0, 1], "tp": 2 },
 *     "cache":     { "host_budget_mib": 0, "staging_budget_mib": 0, "idle_timeout_ms": 0,
 *                    "max_queued_requests": 0, "warmup_budget_mib": 0 },
 *     "models": [ { "id": "a", "path": "/m/a.rad", "default": true,
 *                   "engine": ["--max-model-len", "8192"] } ] }
 *
 * JSON because nlohmann/json is already a dependency and the project parses no YAML. A model's
 * `engine` is an argv-style list handed to config_parse over the command line's own options, so
 * there is one option parser and one validation. The gpu_group and cache sections are only parsed
 * and checked here; nothing reads the cache values yet.
 *
 * The registry belongs to the process (ADR 0001), not to an Engine. */
#pragma once

#include "rad_core.h"

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
    void set_state(const std::string& id, ModelState s);
    void add(ModelRecord r) { models_.push_back(std::move(r)); }

private:
    std::vector<ModelRecord> models_;
};

/* Fills `out` from the manifest, each model's Config layered on `base` (the parsed command line).
 * RAD_E_INVAL, after logging what is wrong, for a malformed or inconsistent manifest. */
int manifest_parse(const std::string& text, const Config& base, ModelRegistry* out);
/* Reads `file`, then manifest_parse. */
int manifest_load(const std::string& file, const Config& base, ModelRegistry* out);

}  /* namespace rad */
