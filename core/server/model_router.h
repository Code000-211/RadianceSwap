/* core/server/model_router.h -- the process's HTTP listener over several models (spec #1, #8).
 *
 * One ModelRouter owns the listener and the route table; each active model keeps its own Server
 * (frontend, scheduler bridge, limits, admission), built by that model's Engine. A request's
 * `model` is read here, before any Server tokenises or renders a template, and the request is
 * handed to that model's Server through its own Router. A request naming no model goes to the
 * default one, an unknown id is a 404, and the image, audio and video endpoints (and media parts
 * in a text request) are refused: the manifest mode serves text models only.
 *
 * Lifetime: a Server is attached from its Engine's thread and must stay alive until serve_model
 * returns, which it does only after the listener has stopped and every attached Server was told to
 * stop, so no handler runs against a Server that is being destroyed. */
#pragma once

#include "http.h"

#include <condition_variable>
#include <map>
#include <mutex>
#include <string>

namespace rad {
namespace server {

class Server;

class ModelRouter {
public:
    /* `n_models`: how many Servers run() waits for before it listens. */
    ModelRouter(std::string default_id, size_t n_models)
        : default_id_(std::move(default_id)), n_models_(n_models) {}

    /* The process's routes: those of the first attached Server (every Server registers the same
     * table), each resolving the request's model first. Complete once n_models are attached. */
    const Router& router() const { return router_; }

    void attach(const std::string& id, Server* s);
    void detach(const std::string& id);

    /* attach, block until stop() has finished and the listener is down, detach. Returns RAD_OK. */
    int serve_model(const std::string& id, Server* s);

    /* Waits for every model, then listens with the default model's transport options (the pool
     * grown by the other models' admission ceilings). Blocks until stop(); a bind failure is
     * returned. Returns RAD_OK at once if stop() came first. */
    int run();

    int port() const { return transport_.port(); }   /* 0 until run() is accepting */

    /* Idempotent, any thread: stops the listener, then every attached Server. */
    void stop();

private:
    HttpResponse route(const HttpRequest& q) const;

    const std::string default_id_;
    const size_t      n_models_;
    Router            router_;
    HttpTransport     transport_;

    mutable std::mutex      m_;
    std::condition_variable cv_;
    std::map<std::string, Server*> servers_;
    bool stopped_ = false, stop_done_ = false, serving_ = false;
};

}  /* namespace server */
}  /* namespace rad */
