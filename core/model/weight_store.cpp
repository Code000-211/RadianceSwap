#include "model/weight_store.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <vector>

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

void PreparedWeightStore::begin(const std::string& model, const std::string& key) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    m.pending.clear();
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
    std::unique_ptr<uint8_t[]> data(new (std::nothrow) uint8_t[(size_t)bytes]);
    if (!data) return nullptr;
    std::lock_guard<std::mutex> lk(m_);
    Buffer& b = models_[model].pending[entry];
    b.data = std::move(data);
    b.bytes = bytes;
    return b.data.get();
}

void PreparedWeightStore::commit(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    Model& m = models_[model];
    for (auto& [name, b] : m.pending) {
        m.bytes += b.bytes;
        m.entries[name] = std::move(b);
    }
    m.pending.clear();
}

void PreparedWeightStore::abort(const std::string& model) {
    std::lock_guard<std::mutex> lk(m_);
    const auto m = models_.find(model);
    if (m != models_.end()) m->second.pending.clear();
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
    models_.erase(model);
}

}  /* namespace rad */
