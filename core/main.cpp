/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "build/startup.h"
#include "hf_model.h"
#include "manifest.h"
#include "place/group_planner.h"
#include "server/model_router.h"
#include "util/stallwatch.h"

#include "device/device.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <memory>
#include <mutex>
#include <thread>

namespace {
std::atomic<bool> g_signalled{false};   /* set by the SIGINT/SIGTERM handler, read by serve_manifest */
}

/* Manifest mode: every model is its own Engine (a model runtime) on the same GPU group, each with a
 * fixed share of the free VRAM, behind one listener that routes by the request's `model`
 * (server/model_router.h). Only the default model is started here; the others are cold and the
 * router starts them on their first request, each on its own activation thread. A model idle for
 * `cache.idle_timeout_ms` is deactivated (0: never), the default one too, and its Engine destroyed.
 * An activation that fails returns from its Launcher with the Engine destroyed (every partial
 * allocation released); the router backs the model off and, for a device error, marks the whole
 * group unhealthy.
 * The registry and the router belong to the process; an Engine belongs to the thread that serves
 * it and is destroyed when that thread returns.
 *
 * The fixed share caps a model's KV cache and static weights until KV is lent between models
 * (#17). The group planner (place/group_planner.h) divides only the experts, inside those shares:
 * every engine, the default one and a cold one at its activation, joins it when it plans and
 * leaves it when it is destroyed. */
static int serve_manifest(rad::ModelRegistry& registry) {
    const std::vector<rad::ModelRecord>& models = registry.models();
    const size_t n = models.size();
    /* Shares are fixed, so no model reserves pool for one not started (to_start 0). Declared before
     * the engines and the router: both end before it. */
    rad::GroupPlanner group(0);

    /* The free VRAM of the group's tightest card, in MiB; -1 where no card reports a capacity. */
    auto free_mib = [](int tp) {
        int64_t least = -1;
        for (int d = 0; d < std::max(tp, 1); ++d) {
            RadDeviceProps p{};
            if (rad_dev_props(d, &p) < 0 || p.vram_bytes <= 0) return int64_t(-1);
            const int64_t f = p.vram_free >> 20;
            least = least < 0 ? f : std::min(least, f);
        }
        return least;
    };

    /* One model is built at a time: the budget resolver claims what the card has free, and a
     * model's step staging comes out of the room it kept (start_execution), before the next
     * model's budget reads the card. Models already serving are not held up by it. */
    std::mutex build_m;
    int64_t share_mib = -1;
    /* Starts the engine of model `i`: its VRAM share, start(), start_execution(). A negative status
     * has been reported. 1 is --debug-graph, done with `*dump_rc`. */
    auto build = [&](size_t i, bool at_startup, std::unique_ptr<rad::Engine>* out, int* dump_rc) {
        std::lock_guard<std::mutex> lk(build_m);
        rad::Config cfg = models[i].config;
        /* An explicit --gpu-headroom-mib, on the command line or in the model's `engine`, wins. */
        if (n > 1 && !cfg.given.count("--gpu-headroom-mib")) {
            const int64_t free = free_mib(cfg.tp);
            if (free > 0) {
                if (share_mib < 0) share_mib = (free - cfg.gpu_headroom_mib) / (int64_t)n;
                cfg.gpu_headroom_mib = rad::manifest_headroom_mib(free, share_mib, cfg.gpu_headroom_mib);
                RAD_INFO("manifest: model '%s' takes about %lld MiB of the group's VRAM "
                         "(--gpu-headroom-mib %lld)", models[i].id.c_str(), (long long)share_mib,
                         (long long)cfg.gpu_headroom_mib);
            }
        }
        if (!rad::resolve_hf_model(cfg)) return (int)RAD_E_NOTFOUND;
        auto engine = std::make_unique<rad::Engine>(std::move(cfg));
        engine->set_registry(&registry);
        engine->set_group_planner(&group);

        /* A startup that stops logging reports where it waits (util/stallwatch.h). */
        rad::StallWatch watch(("startup " + models[i].id).c_str(), 120000);
        int s = engine->start();
        if (s < 0) {
            engine->print_startup_report(s);
            return s;
        }
        if (at_startup && engine->config().debug_graph) {
            *dump_rc = engine->dump();
            return 1;
        }
        s = engine->start_execution();
        if (s < 0) return s;
        *out = std::move(engine);
        return 0;
    };

    std::atomic<bool> serve_failed{false};
    rad::server::ModelRouter* router_ptr = nullptr;
    /* Serves `e` until the router stops or deactivates the model. A failure of a model that had
     * attached (its step loop) stops the process, and so does a failure of the default model's
     * first start (`startup`): nothing listens before it. A cold model whose serve() fails before
     * it attached is an activation failure, which the router reports to the requests waiting. */
    auto serve_engine = [&](rad::Engine& e, const std::string& id, bool startup) {
        const int st = e.serve(nullptr, router_ptr);
        if (st < 0) {
            if (!e.startup_report_printed()) e.print_startup_report(st);
            if (startup || registry.state_of(id) == rad::ModelState::Active) {
                serve_failed = true;
                router_ptr->stop();
            }
        }
        return st;
    };

    const std::string default_id = registry.default_model().id;
    rad::server::ActivationOptions act;
    act.max_queued = registry.cache.max_queued_requests;
    act.idle_timeout_ms = registry.cache.idle_timeout_ms;
    act.backoff_ms = registry.cache.activation_backoff_ms;
    act.backoff_max_ms = registry.cache.activation_backoff_max_ms;
    rad::server::ModelRouter router(default_id, 1, &registry, act);
    router_ptr = &router;

    /* --debug-graph: the default model's first start dumps its graph instead of serving. */
    std::atomic<bool> dumped{false};
    std::atomic<int> dump_exit{0};
    std::atomic<bool> default_started{false};

    /* Every model, the default one too, is built and served by its Launcher, and the Engine is
     * destroyed when the Launcher returns: at the router stopping, or at a deactivation, which
     * frees the model's VRAM (the free memory before and after is reported). */
    for (size_t i = 0; i < n; ++i) {
        const bool is_default = models[i].id == default_id;
        router.add_cold(models[i].id, [&, i, is_default](const std::string& id) {
            const bool startup = is_default && !default_started.exchange(true);
            std::unique_ptr<rad::Engine> e;
            int dump_rc = 0;
            const int64_t free_before = free_mib(models[i].config.tp);
            const int s = build(i, startup, &e, &dump_rc);
            if (s < 0) {
                /* build() has destroyed the Engine, and with it every partial allocation. */
                const int64_t free_after = free_mib(models[i].config.tp);
                if (free_before >= 0 && free_after >= 0)
                    RAD_INFO("model '%s': the failed activation is rolled back; the group's tightest "
                             "card has %lld MiB free (%lld MiB before it began)", id.c_str(),
                             (long long)free_after, (long long)free_before);
            }
            if (s != 0) {
                if (startup) {
                    if (s < 0) serve_failed = true;
                    else { dumped = true; dump_exit = dump_rc; }
                    router_ptr->stop();
                }
                return s < 0 ? s : 0;
            }
            const int64_t before = free_mib(models[i].config.tp);
            const int st = serve_engine(*e, id, startup);
            const bool deactivated = router_ptr->deactivating(id);
            e.reset();
            if (deactivated) {
                const int64_t after = free_mib(models[i].config.tp);
                if (before >= 0 && after >= 0) {
                    router_ptr->note_vram_released(id, after - before);
                    RAD_INFO("model '%s' deactivated: %lld MiB of VRAM freed on the group's "
                             "tightest card (%lld MiB free now; the rest of its use is other "
                             "models and the process's persistent overhead)", id.c_str(),
                             (long long)(after - before), (long long)after);
                }
            }
            return st;
        });
    }

    /* The signals belong to the process, not to a model's watcher: the default model can be
     * deactivated. They are taken once the default model is up, as before: a signal during startup
     * still ends the process at once. */
    std::atomic<bool> finished{false};
    std::thread signals([&] {
        while (!finished && registry.state_of(default_id) != rad::ModelState::Active)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        if (finished) return;
        struct sigaction sa {};
        sa.sa_handler = [](int) { g_signalled.store(true, std::memory_order_relaxed); };
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESETHAND;      /* a second one takes the default disposition and kills */
        sigaction(SIGINT,  &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);
        while (!finished) {
            if (g_signalled.load(std::memory_order_relaxed)) {
                RAD_INFO("stopping: draining in-flight requests");
                router_ptr->stop();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    router.activate(default_id);
    const int s = router.run();
    router.stop();
    router.join();
    finished = true;
    signals.join();
    if (dumped) return dump_exit;
    return s < 0 || serve_failed ? 1 : 0;
}

int main(int argc, char** argv) {
    /* Process-lifetime owner of the tools and plugin registry (ADR 0001). */
    struct ProcessOwner {
        ~ProcessOwner() {
            rad::rad_tools_registry().close();
        }
    } process_owner;

    rad::Config cfg;
    int s = rad::config_parse(argc, argv, &cfg);
    if (s == 1) { rad::config_usage(argv[0]); return 0; }
    if (s < 0)  { rad::config_usage(argv[0]); return 2; }

    /* Process-lifetime owner of the model records (ADR 0001). */
    rad::ModelRegistry registry;
    if (!cfg.manifest.empty()) {
        if (rad::manifest_load(cfg.manifest, cfg, &registry) < 0) return 2;
        return serve_manifest(registry);
    }

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    rad::Engine engine(std::move(cfg));

    /* A startup that stops logging reports where it waits (util/stallwatch.h). Two minutes
     * comfortably covers the slowest normal step, such as a container read from a slow disk. The
     * watch goes on into serve() and ends when the server listens. */
    auto watch = std::make_unique<rad::StallWatch>("startup", 120000);
    s = engine.start();
    if (s < 0 || engine.config().debug_graph) watch.reset();
    if (s < 0) {
        /* start() logged the error; the report adds the machine and the decisions made so far. */
        engine.print_startup_report(s);
        return 1;
    }

    if (engine.config().debug_graph) return engine.dump();
    s = engine.serve(std::move(watch));
    if (s < 0) {
        /* serve() can also refuse before listening (e.g. a template default the template
         * rejects, an unholdable thinking budget). */
        if (!engine.startup_report_printed()) engine.print_startup_report(s);
        return 1;
    }
    return 0;
}
