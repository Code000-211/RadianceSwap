#include "model_router.h"

#include "manifest.h"
#include "oai.h"
#include "rad_internal.h"
#include "server.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace rad {
namespace server {

namespace {

HttpResponse refuse(const HttpRequest& q, const ApiError& e, int retry_after_s = 0) {
    HttpResponse r = HttpResponse::json_body(
        e.status, anthropic_path(q.path) ? anthropic_error_body(e) : e.dump());
    if (retry_after_s > 0) r.set_header("Retry-After", std::to_string(retry_after_s));
    return r;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

/* The generation endpoints that are not text: they belong to one model's Server alone. */
bool media_endpoint(const std::string& path) {
    for (const char* p : { "/v1/images/", "/images/", "/v1/audio/", "/audio/", "/v1/videos",
                           "/v1/realtime" })
        if (starts_with(path, p)) return true;
    return false;
}

/* A content part that carries a picture, a clip or a video, in any of the API shapes (OpenAI's
 * chat and Responses parts, Anthropic's blocks). */
bool has_media_part(const nlohmann::json& j) {
    if (j.is_object()) {
        const auto t = j.find("type");
        if (t != j.end() && t->is_string()) {
            const std::string& v = t->get_ref<const std::string&>();
            if (v == "image_url" || v == "input_image" || v == "image" || v == "video_url" ||
                v == "input_video" || v == "video" || v == "input_audio" || v == "audio_url" ||
                v == "audio")
                return true;
        }
        for (const auto& kv : j.items())
            if (has_media_part(kv.value())) return true;
    } else if (j.is_array()) {
        for (const auto& e : j)
            if (has_media_part(e)) return true;
    }
    return false;
}

}  /* namespace */

ModelRouter::~ModelRouter() {
    stop();
    join();
}

void ModelRouter::add_cold(const std::string& id, ModelLauncher launch) {
    std::lock_guard<std::mutex> lk(m_);
    cold_[id].launch = std::move(launch);
}

void ModelRouter::activate(const std::string& id) {
    std::lock_guard<std::mutex> lk(m_);
    const auto c = cold_.find(id);
    if (c != cold_.end() && !c->second.running && !stopped_) start_locked(id, c->second);
}

void ModelRouter::join() {
    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (auto& [id, c] : cold_)
            if (c.thread.joinable()) threads.push_back(std::move(c.thread));
        if (idle_thread_.joinable()) threads.push_back(std::move(idle_thread_));
    }
    for (std::thread& t : threads) t.join();
}

bool ModelRouter::deactivating(const std::string& id) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto c = cold_.find(id);
    return c != cold_.end() && c->second.deactivating;
}

void ModelRouter::note_vram_released(const std::string& id, int64_t mib) {
    std::lock_guard<std::mutex> lk(m_);
    if (const auto c = cold_.find(id); c != cold_.end()) c->second.vram_released_mib = mib;
}

int64_t ModelRouter::queued(const std::string& id) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto c = cold_.find(id);
    return c == cold_.end() ? 0 : c->second.waiters;
}

/* Every poll: stamps a model in use as busy, and deactivates one that has not been for the idle
 * timeout. The poll is a fifth of the timeout, within [1, 100] ms: a stream that ends is seen
 * idle that much late. */
void ModelRouter::monitor_idle() {
    const auto idle = std::chrono::milliseconds(act_.idle_timeout_ms);
    const auto poll = std::chrono::milliseconds(std::clamp<int64_t>(act_.idle_timeout_ms / 5, 1, 100));
    std::unique_lock<std::mutex> lk(m_);
    while (!stopped_) {
        const auto now = Clock::now();
        for (auto& [id, c] : cold_) {
            const auto it = servers_.find(id);
            if (it == servers_.end() || !c.running || c.deactivating) continue;
            if (c.users > 0 || c.waiters > 0 || it->second->in_flight() > 0) {
                c.busy_at = now;
                continue;
            }
            if (now - c.busy_at < idle) continue;
            RAD_INFO("model '%s' has been idle for %lld ms: deactivating it", id.c_str(),
                     (long long)act_.idle_timeout_ms);
            servers_.erase(it);
            c.deactivating = true;
            c.began = now;
            if (registry_) registry_->set_state(id, ModelState::Deactivating);
        }
        cv_.notify_all();
        cv_.wait_for(lk, poll, [&] { return stopped_; });
    }
}

void ModelRouter::attach(const std::string& id, Server* s) {
    std::lock_guard<std::mutex> lk(m_);
    servers_[id] = s;
    if (const auto c = cold_.find(id); c != cold_.end()) {
        Cold& m = c->second;
        const auto now = Clock::now();
        m.attached = true;
        m.busy_at = now;
        ++m.activations;
        m.activation_s += std::chrono::duration<double>(now - m.began).count();
    }
    if (registry_) registry_->set_state(id, ModelState::Active);
    /* Once: the table is read by the listener, and a model attached again after a deactivation
     * finds it there. */
    if (!routes_added_) {
        routes_added_ = true;
        for (const std::string& sig : s->router().routes()) {
            const size_t sp = sig.find(' ');
            router_.add(sig.substr(0, sp), sig.substr(sp + 1),
                        [this](const HttpRequest& q) { return route(q); });
        }
    }
    cv_.notify_all();
}

void ModelRouter::detach(const std::string& id) {
    std::lock_guard<std::mutex> lk(m_);
    servers_.erase(id);
}

int ModelRouter::serve_model(const std::string& id, Server* s) {
    attach(id, s);
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] {
            const auto c = cold_.find(id);
            return (stop_done_ && !serving_) || (c != cold_.end() && c->second.deactivating);
        });
    }
    detach(id);
    return RAD_OK;
}

int ModelRouter::run() {
    TransportOptions t;
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return stopped_ || servers_.size() >= n_models_; });
        if (stopped_) return RAD_OK;
        t = servers_.at(default_id_)->transport_options();
        for (const auto& [id, s] : servers_)
            if (id != default_id_) t.max_threads += s->queue_cap();
        /* A request waiting for its cold model holds a transport thread. */
        t.max_threads += (int)(act_.max_queued * (int64_t)cold_.size());
        serving_ = true;
        /* Only now: the options above were read from the default model's Server, which a
         * deactivation would take away. */
        if (act_.idle_timeout_ms > 0) idle_thread_ = std::thread([this] { monitor_idle(); });
    }
    const int rc = transport_.serve(router_, t);
    {
        std::lock_guard<std::mutex> lk(m_);
        serving_ = false;
        cv_.notify_all();
    }
    return rc;
}

void ModelRouter::stop() {
    std::vector<Server*> servers;
    {
        std::lock_guard<std::mutex> lk(m_);
        if (stopped_) return;
        stopped_ = true;
        for (const auto& [id, s] : servers_) servers.push_back(s);
        cv_.notify_all();
    }
    transport_.stop();
    for (Server* s : servers) s->stop();
    std::lock_guard<std::mutex> lk(m_);
    stop_done_ = true;
    cv_.notify_all();
}

HttpResponse ModelRouter::route(const HttpRequest& q) {
    if (media_endpoint(q.path)) {
        ApiError e = err_not_implemented(
            q.path + " is not served while several models are active: the image, audio and video "
                     "endpoints are available only with a single model");
        return refuse(q, e);
    }
    std::string id = default_id_;
    bool named = false;
    if (!q.body.empty()) {
        const nlohmann::json j = nlohmann::json::parse(q.body, nullptr, false);
        if (j.is_object()) {
            if (has_media_part(j)) {
                ApiError e = err_not_implemented(
                    "image, audio and video input is not served while several models are "
                    "active; it is available only with a single model");
                return refuse(q, e);
            }
            const auto m = j.find("model");
            if (m != j.end() && m->is_string() && !m->get_ref<const std::string&>().empty()) {
                id = m->get<std::string>();
                named = true;
            }
        }
    }

    Server* s = nullptr;
    Cold* used = nullptr;   /* raised under the lock that found `s`; lowered when dispatch ends */
    std::string served;
    bool cold = false, unattached = false;
    {
        std::lock_guard<std::mutex> lk(m_);
        const auto it = servers_.find(id);
        const auto c = cold_.find(id);
        if (it != servers_.end()) {
            s = it->second;
            if (c != cold_.end()) {
                used = &c->second;
                ++used->users;
            }
        } else if (c != cold_.end()) {
            /* A GET names no model: a probe or a scrape is not a reason to activate the default. */
            if (q.method == "GET" && !named) unattached = true;
            else cold = true;
        } else {
            for (const auto& kv : servers_) served += (served.empty() ? "" : ", ") + kv.first;
            for (const auto& kv : cold_) served += (served.empty() ? "" : ", ") + kv.first;
        }
    }
    if (unattached) return answer_unattached(q);
    if (cold) {
        HttpResponse refusal;
        if (!(s = await_model(id, q, &refusal))) return refusal;
        std::lock_guard<std::mutex> lk(m_);
        used = &cold_.at(id);
    }
    if (!s) {
        ApiError e;
        e.status = 404;
        e.type = "not_found_error";
        e.param = "model";
        e.code = "model_not_found";
        e.message = "no such model: " + id + " (served: " + served + ")";
        return refuse(q, e);
    }
    struct Done {
        ModelRouter& r;
        Cold* c;
        ~Done() {
            if (!c) return;
            std::lock_guard<std::mutex> lk(r.m_);
            --c->users;
            c->busy_at = Clock::now();
        }
    } done{ *this, used };
    HttpResponse r = s->router().dispatch(q);
    if (q.method == "GET" && q.path == "/metrics" && !r.streaming() && r.status == 200)
        r.body += metrics_text();
    return r;
}

HttpResponse ModelRouter::answer_unattached(const HttpRequest& q) const {
    if (q.path == "/metrics") {
        HttpResponse r;
        r.content_type = "text/plain; version=0.0.4; charset=utf-8";
        r.body = metrics_text();
        return r;
    }
    if (q.path == "/health" || q.path == "/ping") return HttpResponse::text(200, "");
    if (q.path == "/v1/models" || q.path == "/models") {
        nlohmann::json data = nlohmann::json::array();
        {
            std::lock_guard<std::mutex> lk(m_);
            for (const auto& [id, c] : cold_)
                data.push_back({ { "id", id }, { "object", "model" }, { "owned_by", "radiance" } });
        }
        return HttpResponse::json_body(
            200, nlohmann::json{ { "object", "list" }, { "data", std::move(data) } }.dump());
    }
    ApiError e = err_unavailable("the default model " + default_id_ + " is not active; a request "
                                 "that names a model, or a generation request, activates it");
    e.code = "model_not_active";
    return refuse(q, e, act_.retry_after_s);
}

Server* ModelRouter::await_model(const std::string& id, const HttpRequest& q,
                                 HttpResponse* refusal) {
    const auto deadline = Clock::now() + std::chrono::milliseconds(act_.wait_ms);
    auto fail = [&](ApiError e, const char* code) {
        e.code = code;
        *refusal = refuse(q, e, act_.retry_after_s);
        return nullptr;
    };

    std::unique_lock<std::mutex> lk(m_);
    Cold& c = cold_.at(id);
    /* The Server found is handed to route(), which lowers `users` when its dispatch ends. */
    auto take = [&](Server* s) {
        ++c.users;
        return s;
    };
    if (const auto it = servers_.find(id); it != servers_.end()) return take(it->second);
    if (stopped_) return fail(err_unavailable("the server is shutting down"), "server_stopping");
    if (c.waiters >= act_.max_queued)
        return fail(err_overloaded("model " + id + " is not active yet and its queue of " +
                                   std::to_string(act_.max_queued) + " waiting requests is full"),
                    "model_queue_full");
    ++c.waiters;
    struct Leave {
        Cold& c;
        ~Leave() { --c.waiters; }
    } leave{ c };

    uint64_t attempt = 0;   /* the activation this request waits on; 0: none yet */
    for (;;) {
        if (const auto it = servers_.find(id); it != servers_.end()) return take(it->second);
        if (stopped_) return fail(err_unavailable("the server is shutting down"), "server_stopping");
        if (attempt && c.failed_attempt >= attempt)
            return fail(err_unavailable(c.error), "model_activation_failed");
        /* The attempt ended without failing: the model was active and is deactivated again. */
        if (attempt && !c.running) attempt = 0;
        if (!attempt) {
            if (!c.running) {
                RAD_INFO("model '%s' is cold: activating it for a request", id.c_str());
                attempt = start_locked(id, c);
            } else if (!c.deactivating) {
                attempt = c.attempt;
            }   /* else: wait for the deactivation to finish, then activate */
        }
        if (Clock::now() >= deadline)
            return fail(err_unavailable("model " + id + " was not active within " +
                                        std::to_string(act_.wait_ms) + " ms"),
                        "model_activation_timeout");
        lk.unlock();
        const bool gone = q.peer_gone();
        lk.lock();
        if (gone) {
            *refusal = HttpResponse::text(499, "client closed request\n");
            return nullptr;
        }
        cv_.wait_for(lk, std::chrono::milliseconds(20));
    }
}

uint64_t ModelRouter::start_locked(const std::string& id, Cold& c) {
    if (c.thread.joinable()) c.thread.join();   /* the attempt before: it has only to return */
    c.running = true;
    c.attached = false;
    c.error.clear();
    c.began = Clock::now();
    ++c.attempt;
    if (registry_) registry_->set_state(id, ModelState::Activating);
    c.thread = std::thread([this, id, &c] {
        const int rc = c.launch(id);
        std::lock_guard<std::mutex> g(m_);
        c.running = false;
        if (c.deactivating) {
            /* The Launcher has destroyed the Engine: the model's VRAM is free. */
            const auto now = Clock::now();
            c.deactivating = false;
            c.attached = false;
            ++c.deactivations;
            c.deactivation_s += std::chrono::duration<double>(now - c.began).count();
            if (registry_) registry_->set_state(id, ModelState::Cold);
            RAD_INFO("model '%s' is deactivated (cold) after %.3f s%s", id.c_str(),
                     std::chrono::duration<double>(now - c.began).count(),
                     rc < 0 ? "; its runtime reported an error on the way out" : "");
        } else if (!c.attached && !stopped_) {
            c.failed_attempt = c.attempt;
            c.error = "activation of model " + id + " failed: " + rad_strerror(rc < 0 ? rc : RAD_E_STATE);
            if (registry_) registry_->set_state(id, ModelState::Failed);
            RAD_ERR("%s", c.error.c_str());
        }
        cv_.notify_all();
    });
    return c.attempt;
}

std::string ModelRouter::status_json() const {
    std::lock_guard<std::mutex> lk(m_);
    return status_json_locked();
}

std::string ModelRouter::status_json_locked() const {
    nlohmann::json models = nlohmann::json::array();
    for (const auto& [id, c] : cold_) {
        const char* state = servers_.count(id)  ? "active"
                          : c.deactivating      ? "deactivating"
                          : c.running           ? "activating"
                          : c.attempt && c.failed_attempt == c.attempt ? "failed" : "cold";
        nlohmann::json m = { { "id", id }, { "state", state },
                             { "default", id == default_id_ },
                             { "activations", c.activations },
                             { "activation_seconds", c.activation_s },
                             { "deactivations", c.deactivations },
                             { "deactivation_seconds", c.deactivation_s },
                             { "queued_requests", c.waiters } };
        if (c.vram_released_mib >= 0) m["vram_released_mib"] = c.vram_released_mib;
        models.push_back(std::move(m));
    }
    return nlohmann::json{ { "object", "list" }, { "idle_timeout_ms", act_.idle_timeout_ms },
                           { "data", std::move(models) } }.dump();
}

std::string ModelRouter::metrics_text() const {
    std::lock_guard<std::mutex> lk(m_);
    std::string o;
    auto line = [&](const char* name, const std::string& id, const std::string& extra, double v) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.9g", v);
        o += std::string(name) + "{model=\"" + id + "\"" + extra + "} " + buf + "\n";
    };
    o += "# HELP radiance:model_state 1 for the state each configured model is in\n"
         "# TYPE radiance:model_state gauge\n";
    for (const auto& [id, c] : cold_) {
        const char* state = servers_.count(id)  ? "active"
                          : c.deactivating      ? "deactivating"
                          : c.running           ? "activating"
                          : c.attempt && c.failed_attempt == c.attempt ? "failed" : "cold";
        line("radiance:model_state", id, std::string(",state=\"") + state + "\"", 1);
    }
    o += "# HELP radiance:model_activations_total Activations that completed\n"
         "# TYPE radiance:model_activations_total counter\n";
    for (const auto& [id, c] : cold_) line("radiance:model_activations_total", id, "", (double)c.activations);
    o += "# HELP radiance:model_activation_seconds_total Time spent activating\n"
         "# TYPE radiance:model_activation_seconds_total counter\n";
    for (const auto& [id, c] : cold_) line("radiance:model_activation_seconds_total", id, "", c.activation_s);
    o += "# HELP radiance:model_deactivations_total Deactivations that completed\n"
         "# TYPE radiance:model_deactivations_total counter\n";
    for (const auto& [id, c] : cold_) line("radiance:model_deactivations_total", id, "", (double)c.deactivations);
    o += "# HELP radiance:model_deactivation_seconds_total Time spent deactivating\n"
         "# TYPE radiance:model_deactivation_seconds_total counter\n";
    for (const auto& [id, c] : cold_) line("radiance:model_deactivation_seconds_total", id, "", c.deactivation_s);
    o += "# HELP radiance:model_vram_released_mib VRAM the last deactivation freed (MiB)\n"
         "# TYPE radiance:model_vram_released_mib gauge\n";
    for (const auto& [id, c] : cold_)
        if (c.vram_released_mib >= 0) line("radiance:model_vram_released_mib", id, "", (double)c.vram_released_mib);
    return o;
}

}  /* namespace server */
}  /* namespace rad */
