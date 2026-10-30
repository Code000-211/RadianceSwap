/* tests/collective_order_gpu_test.cpp -- GPU acceptance test for collective order (Issue #6).
 *
 * Acceptance criteria:
 * - Zwei Engines derselben Model-Datei laufen unter Dauerlast parallel auf TP=2 ohne Hänger.
 * - Ausgaben jeder Runtime sind deterministisch gleich wie im Einzelbetrieb.
 * - Overhead der Serialisierung ist gemessen.
 */
#include "rad_test.h"
#include "engine.h"
#include "server/sink.h"
#include "runtime/collective_sequencer.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <vector>

#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
#include <hip/hip_runtime.h>
#endif

namespace {

static bool has_two_gpus() {
#if defined(RAD_HAVE_HIP) && RAD_HAVE_HIP
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 2) {
        return false;
    }
    return true;
#else
    return false;
#endif
}

static std::string find_test_model() {
    /* Candidate paths for small .rad model */
    const std::vector<std::string> candidates = {
        "/home/orca/projects/AdaptiveMTP/runs/gh-60/20261026T225122Z-cpu-08b-convert-r01/qwen35-08b-fp8.rad",
        "/home/orca/projects/AdaptiveMTP/runs/gh-60/20261026T225122Z-cpu-08b-convert-r01/qwen35-08b.rad",
        "/home/orca/Projects/Quant/night-ws-A/build-night/convert-test/out/text.rad",
    };
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path;
        }
    }
    return "";
}

}  /* namespace */

TEST(gpu_two_engines_tp2_parallel_collective_order) {
    if (!has_two_gpus()) {
        printf("    [SKIP] Requires at least 2 HIP GPUs for TP=2 test\n");
        return;
    }

    const std::string model_path = find_test_model();
    if (model_path.empty()) {
        printf("    [SKIP] No small .rad model found on disk\n");
        return;
    }

    const std::string home = find_radiance_home();
    REQUIRE(!home.empty());

    printf("    Testing with model: %s\n", model_path.c_str());

    constexpr int kSteps = 200;
    const std::vector<int32_t> prompt = { 151644, 872, 198, 279, 1457 };

    auto make_cfg = [&](int port) {
        rad::Config c;
        c.model = model_path;
        c.radiance_home = home;
        c.tp = 2;
        c.port = port;
        c.max_seqs = 4;
        c.max_ctx = 2048;
        return c;
    };

    /* ------------------------------------------------------------- 1. Solo baseline */
    std::vector<int32_t> solo_tokens;
    double solo_ms = 0.0;
    {
        printf("    --- 1. Starting single engine baseline (TP=2) ---\n");
        rad::Engine engine_solo(make_cfg(0));
        REQUIRE_EQ(engine_solo.start(), RAD_OK);
        REQUIRE_EQ(engine_solo.start_execution(), RAD_OK);

        CHECK(rad::rad_collective_sequencer().is_single_engine());

        rad::server::Sink sink;
        auto req = std::make_unique<rad::Request>();
        req->id = 1;
        req->prompt = prompt;
        req->max_tokens = kSteps;
        req->ignore_eos = true;
        req->sp.temp = 0.0f;  /* greedy for deterministic comparison */
        req->sink = &sink;

        const auto t0 = std::chrono::steady_clock::now();
        REQUIRE_EQ(engine_solo.scheduler().add(std::move(req)), RAD_OK);

        rad::server::SinkToken stbuf[256];
        while (true) {
            size_t n = sink.read(stbuf, 256);
            for (size_t i = 0; i < n; ++i) solo_tokens.push_back(stbuf[i].token);
            if (sink.done() && n == 0) break;
            if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto t1 = std::chrono::steady_clock::now();
        solo_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        engine_solo.scheduler().reap(1);

        printf("    Solo finished: %zu tokens in %.2f ms (%.2f ms/tok, %.1f tok/s)\n",
               solo_tokens.size(), solo_ms, solo_ms / kSteps, 1000.0 * kSteps / solo_ms);

        CHECK_EQ(engine_solo.stop_execution(), RAD_OK);
    }

    CHECK(rad::rad_collective_sequencer().is_single_engine());

    /* ------------------------------------------------------------- 2. Parallel run */
    std::vector<int32_t> tokens1, tokens2;
    double parallel_ms = 0.0;
    {
        printf("    --- 2. Starting two parallel engines on same GPU group (TP=2) ---\n");
        rad::Engine engine1(make_cfg(0));
        rad::Engine engine2(make_cfg(0));

        REQUIRE_EQ(engine1.start(), RAD_OK);
        REQUIRE_EQ(engine1.start_execution(), RAD_OK);

        REQUIRE_EQ(engine2.start(), RAD_OK);
        REQUIRE_EQ(engine2.start_execution(), RAD_OK);

        /* Sequencer must detect multi-engine mode */
        CHECK(!rad::rad_collective_sequencer().is_single_engine());
        CHECK_EQ(rad::rad_collective_sequencer().active_engines(), 2u);

        rad::server::Sink sink1, sink2;
        auto req1 = std::make_unique<rad::Request>();
        req1->id = 101;
        req1->prompt = prompt;
        req1->max_tokens = kSteps;
        req1->ignore_eos = true;
        req1->sp.temp = 0.0f;
        req1->sink = &sink1;

        auto req2 = std::make_unique<rad::Request>();
        req2->id = 102;
        req2->prompt = prompt;
        req2->max_tokens = kSteps;
        req2->ignore_eos = true;
        req2->sp.temp = 0.0f;
        req2->sink = &sink2;

        const auto t0 = std::chrono::steady_clock::now();

        std::thread client1([&] {
            REQUIRE_EQ(engine1.scheduler().add(std::move(req1)), RAD_OK);
            rad::server::SinkToken stbuf[256];
            while (true) {
                size_t n = sink1.read(stbuf, 256);
                for (size_t i = 0; i < n; ++i) tokens1.push_back(stbuf[i].token);
                if (sink1.done() && n == 0) break;
                if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            engine1.scheduler().reap(101);
        });

        std::thread client2([&] {
            REQUIRE_EQ(engine2.scheduler().add(std::move(req2)), RAD_OK);
            rad::server::SinkToken stbuf[256];
            while (true) {
                size_t n = sink2.read(stbuf, 256);
                for (size_t i = 0; i < n; ++i) tokens2.push_back(stbuf[i].token);
                if (sink2.done() && n == 0) break;
                if (n == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            engine2.scheduler().reap(102);
        });

        client1.join();
        client2.join();

        const auto t1 = std::chrono::steady_clock::now();
        parallel_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("    Parallel finished: 2 x %d tokens in %.2f ms (aggregate %.1f tok/s)\n",
               kSteps, parallel_ms, 1000.0 * (2 * kSteps) / parallel_ms);

        CHECK_EQ(engine1.stop_execution(), RAD_OK);
        CHECK_EQ(engine2.stop_execution(), RAD_OK);
    }

    CHECK(rad::rad_collective_sequencer().is_single_engine());

    /* ------------------------------------------------------------- 3. Assertions & Measurement */
    REQUIRE_EQ(tokens1.size(), (size_t)kSteps);
    REQUIRE_EQ(tokens2.size(), (size_t)kSteps);
    REQUIRE_EQ(solo_tokens.size(), (size_t)kSteps);

    /* Criterion 2: Deterministic bit-identical output parity */
    for (size_t i = 0; i < (size_t)kSteps; ++i) {
        CHECK_EQ(tokens1[i], solo_tokens[i]);
        CHECK_EQ(tokens2[i], solo_tokens[i]);
    }
    printf("    Output check: All %d tokens bit-for-bit identical between solo and parallel runs!\n", kSteps);

    /* Criterion 3: Measure serialization overhead */
    const double solo_step_ms = solo_ms / kSteps;
    const double par_step_ms  = parallel_ms / kSteps;
    const double overhead_pct = ((par_step_ms - solo_step_ms) / solo_step_ms) * 100.0;
    const double agg_throughput_solo = (1000.0 * kSteps) / solo_ms;
    const double agg_throughput_par  = (1000.0 * 2 * kSteps) / parallel_ms;

    printf("\n    ================ Collective Sequencing Benchmark ================\n");
    printf("    Solo run (1 engine, fast-path)       : %6.2f ms (%6.2f ms/step, %6.1f tok/s)\n",
           solo_ms, solo_step_ms, agg_throughput_solo);
    printf("    Parallel run (2 engines, sequenced)  : %6.2f ms (%6.2f ms/step, %6.1f tok/s agg)\n",
           parallel_ms, par_step_ms, agg_throughput_par);
    printf("    Per-step wall latency change         : %+.2f ms/step (%+.1f%%)\n",
           par_step_ms - solo_step_ms, overhead_pct);
    printf("    Aggregate throughput efficiency      : %.1f%%\n",
           (agg_throughput_par / (2.0 * agg_throughput_solo)) * 100.0);
    printf("    =================================================================\n\n");
}

RAD_TEST_MAIN()
