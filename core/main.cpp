/* main.cpp -- the entry point: turns a command line into engine.h's startup sequence and a status
 * into an exit code.
 */
#include "engine.h"
#include "build/startup.h"
#include "hf_model.h"
#include "manifest.h"
#include "util/stallwatch.h"

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

    /* Process-lifetime owner of the model records (ADR 0001); only the default model is started
     * for now, and every model is listed by /v1/models. */
    rad::ModelRegistry registry;
    if (!cfg.manifest.empty()) {
        if (rad::manifest_load(cfg.manifest, cfg, &registry) < 0) return 2;
        cfg = registry.default_model().config;
    }

    if (!rad::resolve_hf_model(cfg)) return 1;   /* a repo id: fetched, and --model points at it */

    const bool manifest_mode = !registry.models().empty();
    rad::Engine engine(std::move(cfg));
    if (manifest_mode) engine.set_registry(&registry);

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

    if (manifest_mode) registry.set_state(registry.default_model().id, rad::ModelState::Active);

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
