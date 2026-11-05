/* manifest_test.cpp -- the models file (core/manifest.h) and the options that select it.
 *
 *   Taken     a valid manifest fills one ModelRecord per model, with the command line's options
 *             under each model's own `engine` options
 *   Refused   each broken manifest stops the start, naming what is wrong
 *   Unchanged without --manifest, --model parses as before
 *
 * No device, no model: the "models" are empty files.
 */
#include "rad_test.h"

#include "manifest.h"
#include "rad_core.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace rad;

namespace {

std::string touch(const std::string& name) {
    const std::string p = radtest::tmpdir() + "/" + name;
    std::ofstream(p).put('x');
    return p;
}

/* One model object of a manifest, as JSON text. */
std::string model(const std::string& id, bool is_default, const std::string& engine = "[]",
                  const std::string& path = "") {
    return "{\"id\":\"" + id + "\",\"path\":\"" + (path.empty() ? touch(id + ".rad") : path) +
           "\",\"default\":" + (is_default ? "true" : "false") + ",\"engine\":" + engine + "}";
}

int parse_text(const std::string& text, ModelRegistry* r, const Config& base = Config()) {
    return manifest_parse(text, base, r);
}

/* The manifest must be refused, and the message must say `why`. */
void refused(const std::string& text, const std::string& why, const Config& base = Config()) {
    ModelRegistry r;
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = parse_text(text, &r, base); });
    CHECK_EQ(st, (int)RAD_E_INVAL);
    if (log.find(why) == std::string::npos)
        std::fprintf(stderr, "    expected '%s' in: %s", why.c_str(), log.c_str());
    CHECK(log.find(why) != std::string::npos);
}

}  /* namespace */

TEST(a_manifest_fills_one_record_per_model_with_the_engine_options_applied) {
    Config base;
    Argv cli({ "--manifest", "m.json", "--port", "8123" });
    REQUIRE_EQ(config_parse(cli.argc(), cli.argv.data(), &base), (int)RAD_OK);
    const std::string text =
        "{\"gpu_group\":{\"devices\":[0,1],\"tp\":2},"
        "\"cache\":{\"host_budget_mib\":1024,\"staging_budget_mib\":64,\"idle_timeout_ms\":30000,"
        "\"max_queued_requests\":16,\"warmup_budget_mib\":256,"
        "\"activation_backoff_ms\":500,\"activation_backoff_max_ms\":20000,\"max_active_models\":2},\"models\":[" +
        model("alpha", true, "[\"--max-model-len\",\"8192\"]") + "," +
        model("beta", false, "[\"--kv-cache-dtype\",\"fp8\"]") + "]}";
    ModelRegistry r;
    REQUIRE_EQ(parse_text(text, &r, base), (int)RAD_OK);

    CHECK_EQ(r.group.devices.size(), (size_t)2);
    CHECK_EQ(r.cache.host_budget_mib, (int64_t)1024);
    CHECK_EQ(r.cache.staging_budget_mib, (int64_t)64);
    CHECK_EQ(r.cache.idle_timeout_ms, (int64_t)30000);
    CHECK_EQ(r.cache.max_queued_requests, (int64_t)16);
    CHECK_EQ(r.cache.warmup_budget_mib, (int64_t)256);
    CHECK_EQ(r.cache.activation_backoff_ms, (int64_t)500);
    CHECK_EQ(r.cache.activation_backoff_max_ms, (int64_t)20000);
    CHECK_EQ(r.cache.max_active_models, (int64_t)2);

    REQUIRE_EQ(r.models().size(), (size_t)2);
    CHECK_EQ(r.default_model().id, std::string("alpha"));
    const ModelRecord& a = *r.find("alpha");
    const ModelRecord& b = *r.find("beta");
    CHECK_EQ(a.path, touch("alpha.rad"));
    CHECK_EQ(a.config.model, a.path);
    CHECK_EQ(a.config.max_ctx, (int64_t)8192);
    CHECK_EQ(b.config.max_ctx, (int64_t)0);                /* alpha's option is alpha's alone */
    CHECK_EQ(a.config.port, 8123);                         /* the command line is the base */
    CHECK_EQ(b.config.port, 8123);
    CHECK_EQ(a.config.tp, 2);
    CHECK_EQ(a.config.served_model_name, std::string("alpha"));
    CHECK_EQ(a.state, ModelState::Cold);
    CHECK(r.find("gamma") == nullptr);

    r.set_state("alpha", ModelState::Active);
    CHECK(r.find("alpha")->state == ModelState::Active);
    CHECK_EQ(std::string(model_state_name(ModelState::Deactivating)), std::string("deactivating"));
}

TEST(a_manifest_with_only_models_is_enough) {
    ModelRegistry r;
    REQUIRE_EQ(parse_text("{\"models\":[" + model("solo", true) + "]}", &r), (int)RAD_OK);
    CHECK_EQ(r.default_model().id, std::string("solo"));
    CHECK_EQ(r.default_model().config.tp, 1);
}

TEST(a_broken_manifest_is_refused_by_name) {
    refused("not json", "not a JSON object");
    refused("{\"models\":[]}", "models must be a non-empty array");
    refused("{\"models\":[" + model("a", true) + "," + model("a", false) + "]}", "'a' appears twice");
    refused("{\"models\":[" + model("a", false) + "," + model("b", false) + "]}", "no model is the default");
    refused("{\"models\":[" + model("a", true) + "," + model("b", true) + "]}", "2 models are marked default");
    refused("{\"models\":[" + model("a", true, "[]", "/nonexistent/a.rad") + "]}", "does not exist");
    refused("{\"models\":[{\"id\":\"\",\"path\":\"x\"}]}", "non-empty string id");
    refused("{\"models\":[" + model("a", true) + "],\"extra\":1}", "unknown key 'extra'");
    refused("{\"gpu_group\":{\"devices\":[0,1],\"tp\":1},\"models\":[" + model("a", true) + "]}",
            "does not match its 2 devices");
    refused("{\"gpu_group\":{\"devices\":[0,0]},\"models\":[" + model("a", true) + "]}", "device 0 twice");
    refused("{\"cache\":{\"host_budget_mib\":-1},\"models\":[" + model("a", true) + "]}",
            "cache.host_budget_mib");
    refused("{\"cache\":{\"idle_timeout\":5},\"models\":[" + model("a", true) + "]}", "unknown key 'idle_timeout'");
}

TEST(a_models_engine_options_run_through_the_command_line_validation) {
    /* The same refusals the command line gives: an unknown option, a value out of range. */
    refused("{\"models\":[" + model("a", true, "[\"--no-such-option\"]") + "]}", "unknown option");
    refused("{\"models\":[" + model("a", true, "[\"--kv-cache-dtype\",\"bogus\"]") + "]}", "the engine options are refused");
    refused("{\"models\":[" + model("a", true, "[\"--max-model-len\",\"8x\"]") + "]}", "--max-model-len");
    /* Options that belong to the manifest, not to a model. */
    refused("{\"models\":[" + model("a", true, "[\"--model\",\"/x.rad\"]") + "]}", "--model cannot be set per model");
    refused("{\"models\":[" + model("a", true, "[\"--tp\",\"2\"]") + "]}", "--tp cannot be set per model");
    refused("{\"models\":[" + model("a", true, "[\"--served-model-name\",\"z\"]") + "]}", "cannot be set per model");
}

TEST(lora_is_refused_with_a_manifest) {
    refused("{\"models\":[" + model("a", true, "[\"--lora\",\"x=/p\"]") + "]}", "--lora cannot be set per model");
    refused("{\"models\":[" + model("a", true, "[\"--max-loras\",\"2\"]") + "]}", "--max-loras cannot be set per model");

    Config c;
    Argv cli({ "--manifest", "m.json", "--lora", "x=/p" });
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = config_parse(cli.argc(), cli.argv.data(), &c); });
    CHECK_EQ(st, (int)RAD_E_INVAL);
    CHECK(log.find("--lora is not served with --manifest") != std::string::npos);
}

TEST(the_command_line_takes_a_model_or_a_manifest_not_both) {
    int st = RAD_OK;
    Config c;
    Argv both({ "--model", "/m/x.rad", "--manifest", "m.json" });
    std::string log = stderr_of([&] { st = config_parse(both.argc(), both.argv.data(), &c); });
    CHECK_EQ(st, (int)RAD_E_INVAL);
    CHECK(log.find("--manifest lists the models") != std::string::npos);

    Config n;
    Argv none({ "--port", "9" });
    log = stderr_of([&] { st = config_parse(none.argc(), none.argv.data(), &n); });
    CHECK_EQ(st, (int)RAD_E_INVAL);
    CHECK(log.find("--model is required") != std::string::npos);

    Config only;
    Argv m({ "--manifest", "m.json" });
    CHECK_EQ(config_parse(m.argc(), m.argv.data(), &only), (int)RAD_OK);
    CHECK_EQ(only.manifest, std::string("m.json"));
    CHECK(only.model.empty());
}

TEST(a_served_name_on_the_command_line_conflicts_with_manifest_ids) {
    Config base;
    Argv cli({ "--manifest", "m.json", "--served-model-name", "x" });
    REQUIRE_EQ(config_parse(cli.argc(), cli.argv.data(), &base), (int)RAD_OK);
    refused("{\"models\":[" + model("a", true) + "]}", "--served-model-name does not apply", base);
}

TEST(without_a_manifest_model_parses_as_before) {
    Config c;
    Argv a({ "--model", "/m/x.rad", "--max-model-len", "4096" });
    REQUIRE_EQ(config_parse(a.argc(), a.argv.data(), &c), (int)RAD_OK);
    CHECK_EQ(c.model, std::string("/m/x.rad"));
    CHECK(c.manifest.empty());
    CHECK(c.served_model_name.empty());
    CHECK_EQ(c.max_ctx, (int64_t)4096);
}

TEST(manifest_load_reads_a_file_and_names_a_missing_one) {
    ModelRegistry r;
    const std::string f = radtest::tmpdir() + "/models.json";
    std::ofstream(f) << "{\"models\":[" << model("filed", true) << "]}";
    CHECK_EQ(manifest_load(f, Config(), &r), (int)RAD_OK);
    CHECK_EQ(r.default_model().id, std::string("filed"));
    int st = RAD_OK;
    const std::string log = stderr_of([&] { st = manifest_load(radtest::tmpdir() + "/none.json", Config(), &r); });
    CHECK_EQ(st, (int)RAD_E_INVAL);
    CHECK(log.find("cannot read") != std::string::npos);
}
TEST(a_models_headroom_leaves_its_share_and_never_less_than_the_default) {
    /* 20000 MiB free, 8000 for this model: leave 12000 unclaimed. */
    CHECK_EQ(manifest_headroom_mib(20000, 8000, 96), (int64_t)12000);
    /* A share the card cannot spare keeps the model's own headroom. */
    CHECK_EQ(manifest_headroom_mib(5000, 8000, 96), (int64_t)96);
}

/* The driver's free figure as read once per poll; the last value repeats. */
struct Card {
    std::vector<int64_t> reads;
    size_t at = 0;
    int64_t slept = 0;
    int64_t settle(int64_t poll = 20, int64_t quiet = 300, int64_t max = 5000) {
        return manifest_settled_free_mib(
            [&] { return reads[std::min(at++, reads.size() - 1)]; },
            [&](int64_t ms) { slept += ms; }, poll, quiet, max);
    }
};

TEST(the_free_vram_after_a_deactivation_is_read_once_the_driver_has_released_it) {
    /* #62 on an RX 9060 XT: the figure stayed at about 8 GiB for some 60 ms after ~Engine, then
     * the 6 GiB the model held came back at once. */
    Card c{ { 8247, 8247, 8247, 8247, 14208, 14214, 14199 } };
    CHECK_EQ(c.settle(), (int64_t)14199);
    CHECK_EQ(c.slept, (int64_t)(4 * 20 + 300));   /* the release, then a quiet stretch */

    /* A release that comes in steps counts each step; the display's few MiB of drift do not. */
    Card s{ { 1000, 1000, 3000, 3000, 6000, 6010, 5995, 6004 } };
    CHECK_EQ(s.settle(), (int64_t)6004);
    CHECK_EQ(s.slept, (int64_t)(4 * 20 + 300));
}

TEST(a_settled_read_never_waits_past_its_bound_and_stops_at_an_unknown_figure) {
    Card rising{ {} };
    for (int64_t v = 0; v < 1000; ++v) rising.reads.push_back(v * 100);
    rising.settle(20, 300, 1000);
    CHECK_EQ(rising.slept, (int64_t)1000);

    Card none{ { -1 } };
    CHECK_EQ(none.settle(), (int64_t)-1);
    CHECK_EQ(none.slept, (int64_t)0);
    Card lost{ { 5000, -1 } };
    CHECK_EQ(lost.settle(), (int64_t)-1);
    CHECK_EQ(lost.slept, (int64_t)20);
}

TEST(only_another_active_activating_or_deactivating_model_holds_vram) {
    ModelRegistry r;
    for (const char* id : { "a", "b", "c" }) {
        ModelRecord m;
        m.id = id;
        r.add(std::move(m));
    }
    CHECK(!manifest_others_hold_vram(r, "a"));
    r.set_state("a", ModelState::Active);
    CHECK(!manifest_others_hold_vram(r, "a"));   /* itself does not count */
    CHECK(manifest_others_hold_vram(r, "b"));
    r.set_state("a", ModelState::Prepared);
    r.set_state("c", ModelState::Failed);
    CHECK(!manifest_others_hold_vram(r, "b"));
    for (ModelState s : { ModelState::Activating, ModelState::Deactivating }) {
        r.set_state("c", s);
        CHECK(manifest_others_hold_vram(r, "b"));
    }
}

RAD_TEST_MAIN()
