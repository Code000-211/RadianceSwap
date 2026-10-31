/* core/server/model_router.h -- the process's HTTP listener over several models (spec #1, #8).
 *
 * One ModelRouter owns the listener and the route table; each active model keeps its own Server
 * (frontend, scheduler bridge, limits, admission), built by that model's Engine. A request's
 * `model` is read here, before any Server tokenises or renders a template, and the request is
 * handed to that model's Server through its own Router. A request naming no model goes to the
 * default one, an unknown id is a 404, and the image, audio and video endpoints (and media parts
 * in a text request) are refused: the manifest mode serves text models only.
 *
 * Activation (ticket #9): a configured model that is not attached is cold. The first request that
 * names it starts its Launcher on a thread of its own (the model's activation thread, owned here),
 * so no HTTP thread builds an Engine; at most one runs per model. The request, and any that follow
 * before the model is attached, wait on the HTTP thread that carries them, bounded three ways: at
 * most `max_queued` waiters per model (429 beyond it), a deadline (503), and the client leaving
 * (`closed`, which frees the place and leaves the activation running for the next request). The
 * model is active once its Server is attached, which the Launcher does by serving the model once
 * its Engine is ready on every rank. A Launcher that returns before attaching has failed: its
 * waiters get a 503 naming the status, and the next request starts a new attempt.
 *
 * Lifetime: a Server is attached from its Engine's thread and must stay alive until serve_model
 * returns, which it does only after the listener has stopped and every attached Server was told to
 * stop, so no handler runs against a Server that is being destroyed. */
#pragma once

#include "http.h"

#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace rad {

class ModelRegistry;

namespace server {

class Server;

/* Builds the cold model `id` and serves it: Engine::start, start_execution, then serve_model with
 * its Server. Runs on the model's activation thread and returns when the model stops serving (the
 * router stopped) or, with a negative status, when the activation failed. */
using ModelLauncher = std::function<int(const std::string& id)>;

struct ActivationOptions {
    static constexpr int64_t kDefaultMaxQueued = 64;
    int64_t max_queued = 0;        /* waiting requests per cold model; 0: kDefaultMaxQueued */
    int64_t wait_ms = 600000;      /* how long a request waits for its model */
    int     retry_after_s = 1;
};

class ModelRouter {
public:
    /* `n_models`: how many Servers run() waits for before it listens (the models active from the
     * start). `registry`, if given, follows each model's state. */
    ModelRouter(std::string default_id, size_t n_models, ModelRegistry* registry = nullptr,
                ActivationOptions act = {})
        : default_id_(std::move(default_id)), n_models_(n_models), registry_(registry),
          act_(act) {
        if (act_.max_queued <= 0) act_.max_queued = ActivationOptions::kDefaultMaxQueued;
    }
    /* Stops, then joins the activation threads. */
    ~ModelRouter();

    /* A configured model that is not active yet; before run(). */
    void add_cold(const std::string& id, ModelLauncher launch);
    /* Joins the activation threads; they end once stop() has run. */
    void join();

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

    /* Requests now waiting for the cold model `id` to attach; 0 for any other id. */
    int64_t queued(const std::string& id) const;

    int port() const { return transport_.port(); }   /* 0 until run() is accepting */

    /* Idempotent, any thread: stops the listener, then every attached Server. */
    void stop();

private:
    HttpResponse route(const HttpRequest& q);
    /* Waits for the cold model `id` to attach; the Server to dispatch to, or the refusal. */
    Server* await_model(const std::string& id, const HttpRequest& q, HttpResponse* refusal);

    struct Cold {
        ModelLauncher launch;
        std::thread   thread;
        bool          running = false, attached = false;
        int64_t       waiters = 0;
        uint64_t      attempt = 0, failed_attempt = 0;
        std::string   error;
    };

    const std::string default_id_;
    const size_t      n_models_;
    ModelRegistry* const registry_;
    ActivationOptions act_;
    Router            router_;
    HttpTransport     transport_;

    mutable std::mutex      m_;
    std::condition_variable cv_;
    std::map<std::string, Server*> servers_;
    std::map<std::string, Cold>    cold_;
    bool stopped_ = false, stop_done_ = false, serving_ = false;
};

}  /* namespace server */
}  /* namespace rad */
