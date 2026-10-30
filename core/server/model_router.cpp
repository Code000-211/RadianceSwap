#include "model_router.h"

#include "oai.h"
#include "server.h"

#include <nlohmann/json.hpp>

#include <vector>

namespace rad {
namespace server {

namespace {

HttpResponse refuse(const HttpRequest& q, const ApiError& e) {
    return HttpResponse::json_body(
        e.status, anthropic_path(q.path) ? anthropic_error_body(e) : e.dump());
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

void ModelRouter::attach(const std::string& id, Server* s) {
    std::lock_guard<std::mutex> lk(m_);
    servers_[id] = s;
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
        cv_.wait(lk, [&] { return stopped_ || servers_.size() == n_models_; });
        if (stopped_) return RAD_OK;
        t = servers_.at(default_id_)->transport_options();
        for (const auto& [id, s] : servers_)
            if (id != default_id_) t.max_threads += s->queue_cap();
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

HttpResponse ModelRouter::route(const HttpRequest& q) const {
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
    {
        std::lock_guard<std::mutex> lk(m_);
        const auto it = servers_.find(id);
        if (it != servers_.end()) s = it->second;
        else for (const auto& kv : servers_) served += (served.empty() ? "" : ", ") + kv.first;
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

}  /* namespace server */
}  /* namespace rad */
