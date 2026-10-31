/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "build/startup.h"
#include "hf_model.h"
#include "manifest.h"
#include "server/model_router.h"
#include "util/stallwatch.h"

#include "device/device.h"

#include <algorithm>
#include <thread>

/* Manifest mode: every model is its own Engine (a model runtime), started one after the other on
 * the same GPU group, each with a fixed share of the free VRAM, then served behind one listener
 * that routes by the request's `model` (server/model_router.h). The registry and the router belong
 * to the process; the engines are stopped in the reverse order of their start. */
static int serve_manifest(rad::ModelRegistry& registry) {
    const std::vector<rad::ModelRecord>& models = registry.models();
    const size_t n = models.size();

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

    std::vector<std::unique_ptr<rad::Engine>> engines;
    auto fail = [&](int code) {
        while (!engines.empty()) engines.pop_back();
        return code;
    };
    int64_t share_mib = -1;
    for (size_t i = 0; i < n; ++i) {
        rad::Config cfg = models[i].config;
        /* An explicit --gpu-headroom-mib, on the command line or in the model's `engine`, wins. */
        if (n > 1 && i + 1 < n && !cfg.given.count("--gpu-headroom-mib")) {
            const int64_t free = free_mib(cfg.tp);
            if (free > 0) {
                if (share_mib < 0) share_mib = (free - cfg.gpu_headroom_mib) / (int64_t)n;
                cfg.gpu_headroom_mib = rad::manifest_headroom_mib(free, share_mib, cfg.gpu_headroom_mib);
                RAD_INFO("manifest: model '%s' takes about %lld MiB of the group's VRAM "
                         "(--gpu-headroom-mib %lld)", models[i].id.c_str(), (long long)share_mib,
                         (long long)cfg.gpu_headroom_mib);
            }
        }
        if (!rad::resolve_hf_model(cfg)) return fail(1);
        engines.push_back(std::make_unique<rad::Engine>(std::move(cfg)));
        rad::Engine& e = *engines.back();
        e.set_registry(&registry);

        /* A startup that stops logging reports where it waits (util/stallwatch.h). */
        rad::StallWatch watch(("startup " + models[i].id).c_str(), 120000);
        int s = e.start();
        if (s < 0) {
            e.print_startup_report(s);
            return fail(1);
        }
        if (e.config().debug_graph) {
            const int rc = e.dump();
            return fail(rc);
        }
        /* Its step staging comes out of the room the budget kept, before the next model's budget
         * reads the card. */
        s = e.start_execution();
        if (s < 0) return fail(1);
        registry.set_state(models[i].id, rad::ModelState::Active);
    }

    rad::server::ModelRouter router(registry.default_model().id, n);
    std::vector<int> status(n, 0);
    std::vector<std::thread> threads;
    for (size_t i = 0; i < n; ++i)
        threads.emplace_back([&, i] {
            status[i] = engines[i]->serve(nullptr, &router);
            if (status[i] < 0) {
                if (!engines[i]->startup_report_printed()) engines[i]->print_startup_report(status[i]);
                router.stop();
            }
        });
    int s = router.run();
    router.stop();
    for (std::thread& t : threads) t.join();
    int rc = s < 0 ? 1 : 0;
    for (int st : status) if (st < 0) rc = 1;
    return fail(rc);
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
