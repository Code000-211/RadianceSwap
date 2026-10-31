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
#include <memory>
#include <mutex>
#include <thread>

/* Manifest mode: every model is its own Engine (a model runtime) on the same GPU group, each with a
 * fixed share of the free VRAM, behind one listener that routes by the request's `model`
 * (server/model_router.h). Only the default model is started here; the others are cold and the
 * router starts them on their first request, each on its own activation thread. The registry and
 * the router belong to the process; an Engine belongs to the thread that serves it and is destroyed
 * when that thread returns.
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
    /* Serves `e` until the router stops. A failure of the default model, or of a model that had
     * attached (its step loop), stops the process; a cold model whose serve() fails before it
     * attached is an activation failure, which the router reports to the requests waiting for it. */
    auto serve_engine = [&](rad::Engine& e, const std::string& id) {
        const int st = e.serve(nullptr, router_ptr);
        if (st < 0) {
            if (!e.startup_report_printed()) e.print_startup_report(st);
            if (id == registry.default_model().id || registry.state_of(id) == rad::ModelState::Active) {
                serve_failed = true;
                router_ptr->stop();
            }
        }
        return st;
    };

    const std::string default_id = registry.default_model().id;
    rad::server::ActivationOptions act;
    act.max_queued = registry.cache.max_queued_requests;
    rad::server::ModelRouter router(default_id, 1, &registry, act);
    router_ptr = &router;
    size_t default_idx = 0;
    for (size_t i = 0; i < n; ++i) {
        if (models[i].id == default_id) { default_idx = i; continue; }
        router.add_cold(models[i].id, [&, i](const std::string& id) {
            std::unique_ptr<rad::Engine> e;
            int dump_rc = 0;
            const int s = build(i, false, &e, &dump_rc);
            return s < 0 ? s : serve_engine(*e, id);
        });
    }

    std::unique_ptr<rad::Engine> engine;
    int dump_rc = 0;
    registry.set_state(default_id, rad::ModelState::Activating);
    const int bs = build(default_idx, true, &engine, &dump_rc);
    if (bs < 0) return 1;
    if (bs > 0) return dump_rc;

    std::thread serving([&] { serve_engine(*engine, default_id); });
    const int s = router.run();
    router.stop();
    serving.join();
    router.join();
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
