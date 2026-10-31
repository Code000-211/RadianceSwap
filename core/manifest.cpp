/* manifest.cpp -- see manifest.h. */
#include "manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <sys/stat.h>

namespace rad {

using json = nlohmann::json;

const char* model_state_name(ModelState s) {
    switch (s) {
    case ModelState::Cold:         return "cold";
    case ModelState::Preparing:    return "preparing";
    case ModelState::Prepared:     return "prepared";
    case ModelState::Activating:   return "activating";
    case ModelState::Active:       return "active";
    case ModelState::Deactivating: return "deactivating";
    case ModelState::Failed:       return "failed";
    }
    return "?";
}

const ModelRecord* ModelRegistry::find(const std::string& id) const {
    for (const ModelRecord& r : models_) if (r.id == id) return &r;
    return nullptr;
}

const ModelRecord& ModelRegistry::default_model() const {
    for (const ModelRecord& r : models_) if (r.is_default) return r;
    return models_.front();
}

void ModelRegistry::set_state(const std::string& id, ModelState s) {
    std::lock_guard<std::mutex> lk(state_m_);
    for (ModelRecord& r : models_) if (r.id == id) r.state = s;
}

void ModelRegistry::reset() {
    group = ManifestGroup();
    cache = ManifestCache();
    models_.clear();
}

ModelState ModelRegistry::state_of(const std::string& id) const {
    std::lock_guard<std::mutex> lk(state_m_);
    const ModelRecord* r = find(id);
    return r ? r->state : ModelState::Cold;
}

namespace {

/* Keys outside `allowed` are refused: a mistyped key would otherwise be a silent default. */
bool only_keys(const json& o, std::initializer_list<const char*> allowed, const std::string& where) {
    for (auto it = o.begin(); it != o.end(); ++it) {
        bool ok = false;
        for (const char* a : allowed) ok = ok || it.key() == a;
        if (!ok) { RAD_ERR("manifest: unknown key '%s' in %s", it.key().c_str(), where.c_str()); return false; }
    }
    return true;
}

bool read_group(const json& j, const Config& base, ModelRegistry* out, int* tp) {
    if (!j.is_object() || !only_keys(j, { "devices", "tp" }, "gpu_group")) {
        if (!j.is_object()) RAD_ERR("manifest: gpu_group must be an object");
        return false;
    }
    if (j.contains("devices")) {
        const json& d = j["devices"];
        if (!d.is_array() || d.empty()) {
            RAD_ERR("manifest: gpu_group.devices must be a non-empty array of device numbers");
            return false;
        }
        std::set<int> seen;
        for (const json& e : d) {
            if (!e.is_number_integer() || e.get<int64_t>() < 0 || e.get<int64_t>() > INT32_MAX) {
                RAD_ERR("manifest: gpu_group.devices holds non-negative whole numbers");
                return false;
            }
            if (!seen.insert(e.get<int>()).second) {
                RAD_ERR("manifest: gpu_group.devices names device %d twice", e.get<int>());
                return false;
            }
            out->group.devices.push_back(e.get<int>());
        }
    }
    int want = (int)out->group.devices.size();
    if (j.contains("tp")) {
        const json& t = j["tp"];
        if (!t.is_number_integer() || t.get<int64_t>() < 1 || t.get<int64_t>() > INT32_MAX) {
            RAD_ERR("manifest: gpu_group.tp must be a whole number of at least 1");
            return false;
        }
        if (want && t.get<int>() != want) {
            RAD_ERR("manifest: gpu_group.tp %d does not match its %d devices", t.get<int>(), want);
            return false;
        }
        want = t.get<int>();
    }
    if (want) {
        if (base.given.count("--tp") && base.tp != want) {
            RAD_ERR("manifest: gpu_group asks for tp %d, and --tp says %d; give one", want, base.tp);
            return false;
        }
        *tp = want;
    }
    return true;
}

bool read_cache(const json& j, ManifestCache* c) {
    if (!j.is_object()) { RAD_ERR("manifest: cache must be an object"); return false; }
    if (!only_keys(j, { "host_budget_mib", "staging_budget_mib", "idle_timeout_ms",
                        "max_queued_requests", "warmup_budget_mib" }, "cache"))
        return false;
    struct { const char* key; int64_t* field; } const fields[] = {
        { "host_budget_mib", &c->host_budget_mib }, { "staging_budget_mib", &c->staging_budget_mib },
        { "idle_timeout_ms", &c->idle_timeout_ms }, { "max_queued_requests", &c->max_queued_requests },
        { "warmup_budget_mib", &c->warmup_budget_mib },
    };
    for (const auto& f : fields) {
        if (!j.contains(f.key)) continue;
        const json& v = j[f.key];
        if (!v.is_number_integer() || v.get<int64_t>() < 0) {
            RAD_ERR("manifest: cache.%s must be a whole number, 0 or more", f.key);
            return false;
        }
        *f.field = v.get<int64_t>();
    }
    return true;
}

/* Options that belong to the manifest, not to one model: the model is its `path`, the group's tp
 * is shared, a served name is the id, and LoRA names an adapter of a single base. */
bool model_owned_flag(const std::string& a) {
    static const char* const owned[] = { "--model", "--manifest", "--tp", "--served-model-name",
                                         "--help", "-h" };
    for (const char* o : owned) if (a == o) return true;
    return a.rfind("--lora", 0) == 0 || a.rfind("--max-lora", 0) == 0;
}

bool read_model(const json& m, size_t n, const Config& base, int group_tp, ModelRecord* r) {
    const std::string where = "models[" + std::to_string(n) + "]";
    if (!m.is_object() || !only_keys(m, { "id", "path", "default", "engine" }, where)) {
        if (!m.is_object()) RAD_ERR("manifest: %s must be an object", where.c_str());
        return false;
    }
    if (!m.contains("id") || !m["id"].is_string() || m["id"].get<std::string>().empty()) {
        RAD_ERR("manifest: %s needs a non-empty string id", where.c_str());
        return false;
    }
    r->id = m["id"].get<std::string>();
    if (!m.contains("path") || !m["path"].is_string() || m["path"].get<std::string>().empty()) {
        RAD_ERR("manifest: model '%s' needs a non-empty string path", r->id.c_str());
        return false;
    }
    r->path = m["path"].get<std::string>();
    struct stat st;
    if (stat(r->path.c_str(), &st) != 0) {
        RAD_ERR("manifest: model '%s': path '%s' does not exist", r->id.c_str(), r->path.c_str());
        return false;
    }
    if (m.contains("default")) {
        if (!m["default"].is_boolean()) {
            RAD_ERR("manifest: model '%s': default must be true or false", r->id.c_str());
            return false;
        }
        r->is_default = m["default"].get<bool>();
    }

    std::vector<std::string> args{ "radiance" };
    if (m.contains("engine")) {
        if (!m["engine"].is_array()) {
            RAD_ERR("manifest: model '%s': engine must be an array of command-line words", r->id.c_str());
            return false;
        }
        for (const json& w : m["engine"]) {
            if (!w.is_string()) {
                RAD_ERR("manifest: model '%s': engine holds strings only", r->id.c_str());
                return false;
            }
            const std::string s = w.get<std::string>();
            if (model_owned_flag(s)) {
                RAD_ERR("manifest: model '%s': %s cannot be set per model", r->id.c_str(), s.c_str());
                return false;
            }
            args.push_back(s);
        }
    }
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());
    r->config = base;
    r->config.manifest.clear();
    r->config.model = r->path;
    r->config.served_model_name = r->id;
    r->config.given["--served-model-name"] = "manifest (the model id)";
    if (group_tp) {
        r->config.tp = group_tp;
        r->config.given["--tp"] = "manifest (gpu_group)";
    }
    if (config_parse((int)argv.size(), argv.data(), &r->config) != RAD_OK) {
        RAD_ERR("manifest: model '%s': the engine options are refused (see above)", r->id.c_str());
        return false;
    }
    return true;
}

}  /* namespace */

int64_t manifest_headroom_mib(int64_t free_mib, int64_t share_mib, int64_t base_mib) {
    return std::max(base_mib, free_mib - share_mib);
}

int manifest_parse(const std::string& text, const Config& base, ModelRegistry* out) {
    out->reset();
    const json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        RAD_ERR("manifest: not a JSON object");
        return RAD_E_INVAL;
    }
    if (!only_keys(j, { "gpu_group", "cache", "models" }, "the manifest")) return RAD_E_INVAL;
    if (base.given.count("--served-model-name")) {
        RAD_ERR("--served-model-name does not apply with --manifest: each model is served under its id");
        return RAD_E_INVAL;
    }
    int group_tp = 0;
    if (j.contains("gpu_group") && !read_group(j["gpu_group"], base, out, &group_tp)) return RAD_E_INVAL;
    if (j.contains("cache") && !read_cache(j["cache"], &out->cache)) return RAD_E_INVAL;
    if (!j.contains("models") || !j["models"].is_array() || j["models"].empty()) {
        RAD_ERR("manifest: models must be a non-empty array");
        return RAD_E_INVAL;
    }

    std::set<std::string> ids;
    std::vector<std::string> defaults;
    for (size_t i = 0; i < j["models"].size(); ++i) {
        ModelRecord r;
        if (!read_model(j["models"][i], i, base, group_tp, &r)) return RAD_E_INVAL;
        if (!ids.insert(r.id).second) {
            RAD_ERR("manifest: model id '%s' appears twice", r.id.c_str());
            return RAD_E_INVAL;
        }
        if (r.is_default) defaults.push_back(r.id);
        out->add(std::move(r));
    }
    if (defaults.size() != 1) {
        if (defaults.empty())
            RAD_ERR("manifest: no model is the default; set \"default\": true on exactly one");
        else {
            std::string all;
            for (const std::string& d : defaults) all += (all.empty() ? "" : ", ") + d;
            RAD_ERR("manifest: %zu models are marked default (%s); exactly one may be", defaults.size(),
                    all.c_str());
        }
        return RAD_E_INVAL;
    }
    return RAD_OK;
}

int manifest_load(const std::string& file, const Config& base, ModelRegistry* out) {
    std::ifstream f(file, std::ios::binary);
    if (!f) {
        RAD_ERR("--manifest: cannot read %s", file.c_str());
        return RAD_E_INVAL;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return manifest_parse(ss.str(), base, out);
}

}  /* namespace rad */
