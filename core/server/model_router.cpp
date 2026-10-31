#include "model_router.h"

#include "manifest.h"
#include "oai.h"
#include "rad_internal.h"
#include "server.h"

#include <nlohmann/json.hpp>

#include <chrono>
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

void ModelRouter::join() {
    std::vector<std::thread> threads;
    {
        std::lock_guard<std::mutex> lk(m_);
        for (auto& [id, c] : cold_)
            if (c.thread.joinable()) threads.push_back(std::move(c.thread));
    }
    for (std::thread& t : threads) t.join();
}

int64_t ModelRouter::queued(const std::string& id) const {
    std::lock_guard<std::mutex> lk(m_);
    const auto c = cold_.find(id);
    return c == cold_.end() ? 0 : c->second.waiters;
}

void ModelRouter::attach(const std::string& id, Server* s) {
    std::lock_guard<std::mutex> lk(m_);
    servers_[id] = s;
    if (const auto c = cold_.find(id); c != cold_.end()) c->second.attached = true;
    if (registry_) registry_->set_state(id, ModelState::Active);
    if (servers_.size() == 1)
        for (const std::string& sig : s->router().routes()) {
            const size_t sp = sig.find(' ');
            router_.add(sig.substr(0, sp), sig.substr(sp + 1),
                        [this](const HttpRequest& q) { return route(q); });
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
        cv_.wait(lk, [&] { return stop_done_ && !serving_; });
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
            if (m != j.end() && m->is_string() && !m->get_ref<const std::string&>().empty())
                id = m->get<std::string>();
        }
    }

    Server* s = nullptr;
    std::string served;
    bool cold = false;
    {
        std::lock_guard<std::mutex> lk(m_);
        const auto it = servers_.find(id);
        if (it != servers_.end()) s = it->second;
        else if (cold_.count(id)) cold = true;
        else {
            for (const auto& kv : servers_) served += (served.empty() ? "" : ", ") + kv.first;
            for (const auto& kv : cold_) served += (served.empty() ? "" : ", ") + kv.first;
        }
    }
    if (cold) {
        HttpResponse refusal;
        if (!(s = await_model(id, q, &refusal))) return refusal;
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
    return s->router().dispatch(q);
}

Server* ModelRouter::await_model(const std::string& id, const HttpRequest& q,
                                 HttpResponse* refusal) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(act_.wait_ms);
    auto fail = [&](ApiError e, const char* code) {
        e.code = code;
        *refusal = refuse(q, e, act_.retry_after_s);
        return nullptr;
    };

    std::unique_lock<std::mutex> lk(m_);
    Cold& c = cold_.at(id);
    if (const auto it = servers_.find(id); it != servers_.end()) return it->second;
    if (stopped_) return fail(err_unavailable("the server is shutting down"), "server_stopping");
    if (c.waiters >= act_.max_queued)
        return fail(err_overloaded("model " + id + " is not active yet and its queue of " +
                                   std::to_string(act_.max_queued) + " waiting requests is full"),
                    "model_queue_full");
    if (!c.running) {
        if (c.thread.joinable()) c.thread.join();   /* a failed attempt: it has only to return */
        c.running = true;
        c.attached = false;
        c.error.clear();
        ++c.attempt;
        if (registry_) registry_->set_state(id, ModelState::Activating);
        RAD_INFO("model '%s' is cold: activating it for a request", id.c_str());
        c.thread = std::thread([this, id, &c] {
            const int rc = c.launch(id);
            std::lock_guard<std::mutex> g(m_);
            c.running = false;
            if (!c.attached && !stopped_) {
                c.failed_attempt = c.attempt;
                c.error = "activation of model " + id + " failed: " + rad_strerror(rc < 0 ? rc : RAD_E_STATE);
                if (registry_) registry_->set_state(id, ModelState::Failed);
                RAD_ERR("%s", c.error.c_str());
            }
            cv_.notify_all();
        });
    }
    const uint64_t attempt = c.attempt;
    ++c.waiters;
    struct Leave {
        Cold& c;
        ~Leave() { --c.waiters; }
    } leave{ c };

    for (;;) {
        if (const auto it = servers_.find(id); it != servers_.end()) return it->second;
        if (c.failed_attempt >= attempt)
            return fail(err_unavailable(c.error), "model_activation_failed");
        if (stopped_) return fail(err_unavailable("the server is shutting down"), "server_stopping");
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

}  /* namespace server */
}  /* namespace rad */
