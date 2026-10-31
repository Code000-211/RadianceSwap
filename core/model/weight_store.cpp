#include "model/weight_store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

#include <sys/mman.h>

namespace rad {

namespace fs = std::filesystem;

namespace {

std::string file_identity(const fs::path& p, const fs::path& shown, std::error_code& ec) {
    const auto size = fs::file_size(p, ec);
    if (ec) return {};
    const auto mtime = fs::last_write_time(p, ec);
    if (ec) return {};
    return shown.string() + ":" + std::to_string(size) + ":" +
           std::to_string(mtime.time_since_epoch().count());
}

}  /* namespace */

std::string weight_artifact_identity(const std::string& path) {
    std::error_code ec;
    const fs::path abs = fs::weakly_canonical(path, ec);
    if (ec) return {};
    if (!fs::is_directory(abs, ec)) return file_identity(abs, abs, ec);
    std::vector<std::string> files;
    for (fs::recursive_directory_iterator it(abs, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code e2;
        if (!it->is_regular_file(e2)) continue;
        const std::string one = file_identity(it->path(), fs::relative(it->path(), abs), e2);
        if (e2) return {};
        files.push_back(one);
    }
    if (ec) return {};
    std::sort(files.begin(), files.end());
    std::string id = abs.string() + "{";
    for (const std::string& f : files) id += f + ";";
    return id + "}";
}

int64_t proc_io_read_bytes() {
    FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return -1;
    char line[128];
    long long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "read_bytes: %lld", &v) == 1) break;
    std::fclose(f);
    return v;
}

void WeightBufferUnmap::operator()(uint8_t* p) const {
    if (p) ::munmap(p, bytes);
}

PreparedWeightStore::~PreparedWeightStore() = default;

void PreparedWeightStore::drop_entries_locked(Model& m) {
    m.entries.clear();
    m.pending.clear();
    m.bytes = m.pending_bytes = 0;
}

void PreparedWeightStore::begin(const std::string& model, const std::string& key) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    m.pending.clear();
    m.pending_bytes = 0;
    m.loading = true;   /* find() hands out pointers until commit/abort: not evictable meanwhile */
    m.used_at = ++clock_;
    if (m.key == key) return;
    m.entries.clear();
    m.bytes = 0;
    m.key = key;
}

const uint8_t* PreparedWeightStore::find(const std::string& model, const std::string& entry,
                                         int64_t bytes) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    if (m == models_.end()) return nullptr;
    const auto e = m->second.entries.find(entry);
    return e != m->second.entries.end() && e->second.bytes == bytes ? e->second.data.get() : nullptr;
}

uint8_t* PreparedWeightStore::reserve(const std::string& model, const std::string& entry,
                                      int64_t bytes) {
    if (bytes <= 0) return nullptr;
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    if (const auto old = m.pending.find(entry); old != m.pending.end()) {
        m.pending_bytes -= old->second.bytes;
        m.pending.erase(old);
    }
    /* Under the budget the copy evicts prepared models, never active ones. */
    while (budget_ > 0 && used_locked() + bytes > budget_)
        if (!evict_locked(model, false)) return nullptr;
    void* p = ::mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    Buffer& b = m.pending[entry];
    b.data = std::unique_ptr<uint8_t, WeightBufferUnmap>((uint8_t*)p, WeightBufferUnmap{ (size_t)bytes });
    b.bytes = bytes;
    m.pending_bytes += bytes;
    return b.data.get();
}

void PreparedWeightStore::commit(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    for (auto& [name, b] : m.pending) {
        if (const auto old = m.entries.find(name); old != m.entries.end()) m.bytes -= old->second.bytes;
        m.bytes += b.bytes;
        m.entries[name] = std::move(b);
    }
    m.pending.clear();
    m.pending_bytes = 0;
    m.loading = false;
}

void PreparedWeightStore::abort(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    if (m == models_.end()) return;
    m->second.pending.clear();
    m->second.pending_bytes = 0;
    m->second.loading = false;
}

bool PreparedWeightStore::prepared(const std::string& model) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    return m != models_.end() && !m->second.entries.empty();
}

int64_t PreparedWeightStore::ram_bytes(const std::string& model) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    return m == models_.end() ? 0 : m->second.bytes;
}

int64_t PreparedWeightStore::ram_bytes() const {
    std::lock_guard<std::mutex> lk(m_);
    int64_t n = 0;
    for (const auto& [id, m] : models_) n += m.bytes;
    return n;
}

void PreparedWeightStore::release(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    if (m != models_.end()) drop_entries_locked(m->second);
}

/* ---- host budget ---- */

int64_t PreparedWeightStore::used_locked() const {
    int64_t n = 0;
    for (const auto& [id, m] : models_) n += m.bytes + m.pending_bytes + m.pools + m.staging;
    return n;
}

bool PreparedWeightStore::evict_locked(const std::string& requester, bool include_active) {
    /* Prepared models first (idle longest), then active models' copies (activated longest ago),
     * the requester's own last. A model that is loading has handed out pointers: never. */
    Model* victim = nullptr;
    const std::string* victim_id = nullptr;
    int rank = 3;
    for (auto& [id, m] : models_) {
        if (m.entries.empty() || m.loading) continue;
        const int r = id == requester ? 2 : m.active ? 1 : 0;
        if ((r > 0 && !include_active) || r > rank) continue;
        if (r == rank && victim && m.used_at >= victim->used_at) continue;
        victim = &m;
        victim_id = &id;
        rank = r;
    }
    if (!victim) return false;
    const bool was_prepared = !victim->active;
    ++(was_prepared ? stats_.evicted_prepared : stats_.evicted_active_copies);
    stats_.evicted_bytes += victim->bytes;
    drop_entries_locked(*victim);
    if (was_prepared && listener_) listener_(*victim_id, false, 0);
    return true;
}

void PreparedWeightStore::set_budget(int64_t bytes) {
    std::lock_guard<std::mutex> lk(m_);
    budget_ = bytes > 0 ? bytes : 0;
}

void PreparedWeightStore::set_state_listener(StateListener l) {
    std::lock_guard<std::mutex> lk(m_);
    listener_ = std::move(l);
}

void PreparedWeightStore::publish_state(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    if (!listener_) return;
    const auto m = models_.find(model);
    const bool prepared = m != models_.end() && !m->second.entries.empty();
    listener_(model, prepared, prepared ? m->second.bytes : 0);
}

void PreparedWeightStore::save_heat(const std::string& model, std::vector<RadFileProfile> heat) {
    std::lock_guard<std::mutex> lk(m_);
    /* An engine that never routed leaves the earlier heat standing. */
    if (!heat.empty()) heat_[model] = std::move(heat);
}

std::vector<RadFileProfile> PreparedWeightStore::heat(const std::string& model) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto h = heat_.find(model);
    return h == heat_.end() ? std::vector<RadFileProfile>{} : h->second;
}

void PreparedWeightStore::activate(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    models_[model].active = true;
}

void PreparedWeightStore::deactivate(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    m.active = false;
    m.pools = m.staging = 0;
    m.used_at = ++clock_;
}

bool PreparedWeightStore::claim(const std::string& model, HostUse kind, int64_t bytes) {
    std::lock_guard<std::mutex> lk(m_);
    bool within = true;
    while (budget_ > 0 && used_locked() + bytes > budget_ && within) within = evict_locked(model, true);
    if (budget_ > 0 && used_locked() + bytes > budget_) {
        within = false;
        ++stats_.over_budget_claims;
    }
    (kind == HostUse::Pools ? models_[model].pools : models_[model].staging) += bytes;
    return within;
}

void PreparedWeightStore::unclaim(const std::string& model, HostUse kind, int64_t bytes) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    int64_t& n = kind == HostUse::Pools ? m.pools : m.staging;
    n = std::max<int64_t>(n - bytes, 0);
}

bool PreparedWeightStore::make_room(const std::string& model, int64_t bytes) {
    std::lock_guard<std::mutex> lk(m_);
    if (budget_ <= 0 || used_locked() + bytes <= budget_) return true;
    /* Evicting is of no use if every prepared model together would not make the room. */
    int64_t evictable = 0;
    for (const auto& [id, m] : models_)
        if (id != model && !m.active && !m.loading) evictable += m.bytes;
    if (used_locked() - evictable + bytes > budget_) return false;
    while (used_locked() + bytes > budget_) (void)evict_locked(model, false);
    return true;
}

bool PreparedWeightStore::evict_one(const std::string& requester, bool include_active) {
    std::lock_guard<std::mutex> lk(m_);
    return evict_locked(requester, include_active);
}

HostMemoryStats PreparedWeightStore::stats() const {
    std::lock_guard<std::mutex> lk(m_);
    HostMemoryStats s = stats_;
    s.budget_bytes = budget_;
    for (const auto& [id, m] : models_) {
        s.store_bytes += m.bytes + m.pending_bytes;
        s.pool_bytes += m.pools;
        s.staging_bytes += m.staging;
    }
    return s;
}

}  /* namespace rad */
