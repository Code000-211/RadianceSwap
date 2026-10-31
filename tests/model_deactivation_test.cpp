/* tests/model_deactivation_test.cpp -- an active model that stays idle is deactivated
 * (core/server/model_router.h, ticket #10).
 *
 * The launcher is the seam the real process fills with "build the Engine, serve it, destroy it":
 * here a FakeEngine (vocabulary, scheduler, frontend, Server) is built on the model's thread,
 * served through ModelRouter::serve_model and destroyed when that returns, with counters on the
 * way. Every model, the default one too, has a launcher, so every model can be deactivated.
 */
#include "server_fakes.h"

#include "manifest.h"
#include "server/model_router.h"
#include "text/token_view.h"

namespace {

/* What an Engine's vocabulary is to the frontend: shared, and counted. Reading it after its
 * destruction is a use-after-free; the magic number makes the fakes notice it (and a sanitizer
 * build sees it directly). */
struct Counted {
    static inline std::atomic<int> alive{ 0 };
    uint32_t magic = 0xC0FFEE;
    Counted() { ++alive; }
    ~Counted() {
        magic = 0;
        --alive;
    }
};
std::atomic<bool> g_vocab_used_after_free{ false };

struct VocabTokenizer final : FakeTokenizer {
    std::shared_ptr<Counted> vocab;
    std::vector<int32_t> encode(const std::string& s, bool add_special,
                                bool parse_special) const override {
        if (vocab->magic != 0xC0FFEE) g_vocab_used_after_free = true;
        return FakeTokenizer::encode(s, add_special, parse_special);
    }
};

struct Gate {
    std::mutex m;
    std::condition_variable cv;
    bool open = false;
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return open; });
    }
    void release() {
        std::lock_guard<std::mutex> lk(m);
        open = true;
        cv.notify_all();
    }
};

/* One model's Engine: its own vocabulary and scheduler, the frontend over them, and the Server.
 * The Server goes first, as run_server's locals do. */
struct FakeEngine {
    std::atomic<int>& destroyed;
    std::shared_ptr<Counted> vocab = std::make_shared<Counted>();
    FakeScheduler sched;
    std::shared_ptr<ModelFrontend> frontend = std::make_shared<ModelFrontend>();
    std::unique_ptr<Server> srv;

    FakeEngine(const std::string& id, const std::string& reply, std::atomic<int>& d) : destroyed(d) {
        sched.reply = reply;
        auto tok = std::make_shared<VocabTokenizer>();
        tok->vocab = vocab;
        frontend->tok = tok;
        frontend->chat = std::make_shared<FakeChatTemplate>();
        Deps deps;
        deps.sched = &sched;
        deps.frontend = frontend;
        ServerOptions o;
        o.model_id = id;
        o.max_ctx = 200;
        o.host = "127.0.0.1";
        o.port = 0;
        srv = std::make_unique<Server>(deps, o);
    }
    ~FakeEngine() {
        srv.reset();
        ++destroyed;
    }
};

std::string chat(const std::string& model, bool stream = false) {
    json b = { { "model", model },
               { "messages", json::array({ { { "role", "user" }, { "content", "hi" } } }) },
               { "max_tokens", 8 } };
    if (stream) b["stream"] = true;
    return b.dump();
}

std::string reply_of(const HttpResponse& r) {
    return json::parse(r.body)["choices"][0]["message"]["content"].get<std::string>();
}

/* Models "a" (the default), "b" and "c", each with a launcher; "a" is activated at once, as the
 * process does. `teardown` holds a launcher after its Server is gone, as a slow drain would;
 * `manual` makes the schedulers wait for the test to write the tokens. Both are set before the
 * first activation: a model that is idle for a hundred milliseconds is gone. */
struct Fleet {
    ModelRegistry reg;
    std::atomic<int> launches{ 0 }, destroyed{ 0 };
    Gate* const teardown;
    const bool manual;
    std::mutex m;
    std::map<std::string, FakeEngine*> live;
    std::unique_ptr<ModelRouter> mr;
    std::thread listener;   /* run(): the idle monitor starts with the listener */

    explicit Fleet(int64_t idle_ms, bool manual_sched = false, Gate* slow_teardown = nullptr)
        : teardown(slow_teardown), manual(manual_sched) {
        REQUIRE_EQ(Counted::alive.load(), 0);
        for (const char* id : { "a", "b", "c" }) {
            ModelRecord r;
            r.id = id;
            r.is_default = std::string(id) == "a";
            reg.add(std::move(r));
        }
        ActivationOptions act;
        act.idle_timeout_ms = idle_ms;
        mr = std::make_unique<ModelRouter>("a", 1, &reg, act);
        for (const char* id : { "a", "b", "c" })
            mr->add_cold(id, [this](const std::string& model) {
                ++launches;
                FakeEngine e(model, model == "a" ? "alpha" : model == "b" ? "beta" : "gamma",
                             destroyed);
                e.sched.manual = manual;   /* before the Server can be reached */
                {
                    std::lock_guard<std::mutex> lk(m);
                    live[model] = &e;
                }
                const int rc = mr->serve_model(model, e.srv.get());
                /* Engine::stop_execution(Drain): what the scheduler still holds finishes. */
                while (e.sched.in_flight() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (teardown) teardown->wait();
                {
                    std::lock_guard<std::mutex> lk(m);
                    live.erase(model);
                }
                return rc;
            });
        mr->activate("a");
        listener = std::thread([this] { mr->run(); });
        REQUIRE(eventually([&] { return mr->port() > 0; }));
    }
    /* The router must go before the models it serves. */
    ~Fleet() {
        mr->stop();
        if (listener.joinable()) listener.join();
        mr.reset();
    }

    HttpResponse post(const std::string& model, bool stream = false) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = chat(model, stream);
        return mr->router().dispatch(std::move(q));
    }
    HttpResponse get(const std::string& path) {
        HttpRequest q;
        q.method = "GET";
        q.path = path;
        return mr->router().dispatch(std::move(q));
    }
    ModelState state(const char* id) { return reg.state_of(id); }
    FakeEngine* engine(const char* id) {
        std::lock_guard<std::mutex> lk(m);
        const auto it = live.find(id);
        return it == live.end() ? nullptr : it->second;
    }
    json status(const char* id) {
        const json all = json::parse(mr->status_json());
        for (const json& m : all["data"])
            if (m["id"] == id) return m;
        return json();
    }
};

}  // namespace

TEST(an_idle_model_is_deactivated_and_its_engine_destroyed) {
    Fleet f(100);
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    CHECK_EQ(f.destroyed.load(), 1);
    CHECK(f.engine("a") == nullptr);
    CHECK_EQ(Counted::alive.load(), 0);
    CHECK_EQ(f.launches.load(), 1);
    CHECK_EQ(f.status("a")["state"].get<std::string>(), std::string("cold"));
    CHECK_EQ(f.status("a")["deactivations"].get<int>(), 1);
}

TEST(a_model_that_keeps_getting_requests_stays_active) {
    Fleet f(200);
    for (int i = 0; i < 15; ++i) {
        CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.launches.load(), 1);
    CHECK_EQ(f.destroyed.load(), 0);
}

TEST(no_idle_timeout_never_deactivates) {
    Fleet f(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.destroyed.load(), 0);
}

TEST(a_streaming_response_holds_the_model_until_its_last_chunk) {
    Fleet f(100, true);
    HttpResponse r = f.post("a", true);
    REQUIRE(r.streaming());
    /* dispatch has returned and the generation has not started: far more than the timeout. */
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.destroyed.load(), 0);

    FakeScheduler& sched = f.engine("a")->sched;
    sched.push(0, "alpha");
    sched.finish(0, Finish::Stop);
    std::string all, chunk;
    while (r.next(chunk)) {
        all += chunk;
        chunk.clear();
    }
    all += chunk;
    CHECK(all.find("alpha") != std::string::npos);
    if (r.on_close) r.on_close(true);
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    CHECK_EQ(f.destroyed.load(), 1);
}

TEST(a_running_request_is_never_cut_and_delays_the_deactivation) {
    Fleet f(100, true);
    HttpResponse r;
    std::thread t([&] { r = f.post("a"); });
    REQUIRE(eventually([&] { return f.engine("a") && f.engine("a")->sched.in_flight() == 1; }));
    FakeScheduler& sched = f.engine("a")->sched;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.destroyed.load(), 0);
    CHECK(sched.cancelled.empty());

    sched.push(0, "alpha");
    sched.finish(0, Finish::Stop);
    t.join();
    CHECK_EQ(r.status, 200);
    CHECK_EQ(reply_of(r), std::string("alpha"));
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    CHECK_EQ(f.destroyed.load(), 1);
}

TEST(a_request_during_deactivation_waits_and_then_reactivates_the_model) {
    Gate slow;
    Fleet f(100, false, &slow);
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Deactivating; }));
    CHECK_EQ(f.status("a")["state"].get<std::string>(), std::string("deactivating"));

    HttpResponse r;
    std::thread t([&] { r = f.post("a"); });
    REQUIRE(eventually([&] { return f.mr->queued("a") == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    CHECK_EQ(f.launches.load(), 1);   /* the old runtime is still going away */
    CHECK_EQ(f.mr->queued("a"), (int64_t)1);

    slow.release();
    t.join();
    CHECK_EQ(r.status, 200);
    CHECK_EQ(reply_of(r), std::string("alpha"));
    CHECK_EQ(f.launches.load(), 2);
    CHECK_EQ(f.status("a")["deactivations"].get<int>(), 1);
}

TEST(activate_deactivate_activate_cycles_work) {
    Fleet f(150);
    for (int cycle = 1; cycle <= 3; ++cycle) {
        CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
        REQUIRE(eventually([&] { return f.state("b") == ModelState::Cold; }));
        CHECK_EQ(f.status("b")["activations"].get<int>(), cycle);
        CHECK_EQ(f.status("b")["deactivations"].get<int>(), cycle);
    }
    /* The default model went the same way meanwhile, and wakes on a request naming no model. */
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    json body = { { "messages", json::array({ { { "role", "user" }, { "content", "hi" } } }) },
                  { "max_tokens", 8 } };
    HttpRequest q;
    q.method = "POST";
    q.path = "/v1/chat/completions";
    q.body = body.dump();
    CHECK_EQ(reply_of(f.mr->router().dispatch(std::move(q))), std::string("alpha"));
    CHECK_EQ(f.launches.load(), 5);   /* a, b x3, a again */
    CHECK_EQ(f.status("a")["activations"].get<int>(), 2);
}

TEST(a_lease_outlives_the_engine_that_made_it) {
    Fleet f(500);
    FrontendLease lease = f.engine("a")->srv->frontend();
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    CHECK_EQ(f.destroyed.load(), 1);
    /* The engine's vocabulary is gone with it; the lease's tokenizer still holds its own share. */
    CHECK_EQ(Counted::alive.load(), 1);
    CHECK_EQ(lease->tok->encode("hey", false, false).size(), (size_t)3);
    CHECK(!g_vocab_used_after_free.load());
    lease.reset();
    CHECK_EQ(Counted::alive.load(), 0);
}

/* The bridges Engine::run_server builds, over a vocabulary the "engine" drops at once: the
 * frontend alone keeps tokenizer and grammar compiler working. */
TEST(the_real_frontend_bridges_own_the_vocabulary_they_read) {
    ModelFrontend fe;
    {
        VocabBuild vb;
        vb.tokens = { "a", "b", "<|im_end|>" };
        vb.types = { RAD_TT_NORMAL, RAD_TT_NORMAL, RAD_TT_CONTROL };
        vb.eos = 2;
        auto vocab = std::make_shared<Vocab>();
        REQUIRE(vocab->load(std::move(vb)) >= 0);
        auto view = std::make_shared<TokenView>();
        REQUIRE(view->init(vocab) >= 0);
        fe.tok = std::make_shared<TokenizerBridge>(vocab);
        fe.grammar = std::make_shared<GrammarBridge>(std::shared_ptr<const VocabView>(view));
    }   /* the engine's own references to the vocabulary end here */
    const std::vector<int32_t> ids = fe.tok->encode("ab", false, false);
    CHECK_EQ(ids.size(), (size_t)2);
    CHECK_EQ(fe.tok->decode(ids), std::string("ab"));
    CHECK_EQ(fe.tok->vocab_info().n_tokens, 3);
    std::shared_ptr<const GbnfProgram> program;
    std::string err;
    CHECK_EQ(fe.grammar->compile("root ::= \"a\"", &program, err), RAD_OK);
    CHECK(program != nullptr);
}

TEST(the_default_models_probes_and_scrapes_do_not_activate_it) {
    Fleet f(100);
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    const int launches = f.launches.load();

    const HttpResponse health = f.get("/health");
    CHECK_EQ(health.status, 200);
    const HttpResponse models = f.get("/v1/models");
    CHECK_EQ(models.status, 200);
    CHECK_EQ(json::parse(models.body)["data"].size(), (size_t)3);
    const HttpResponse metrics = f.get("/metrics");
    CHECK_EQ(metrics.status, 200);
    CHECK(metrics.body.find("radiance:model_state{model=\"a\",state=\"cold\"} 1") != std::string::npos);
    CHECK(f.get("/props").status == 503);
    CHECK_EQ(f.launches.load(), launches);
}

TEST(status_and_metrics_report_state_counts_and_durations_per_model) {
    Fleet f(400);
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    REQUIRE(eventually([&] { return f.state("b") == ModelState::Cold; }));
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    f.mr->note_vram_released("b", 1234);

    const json st = json::parse(f.get("/radiance/models").body);
    CHECK_EQ(st["idle_timeout_ms"].get<int>(), 400);
    json b, c;
    for (const json& m : st["data"]) {
        if (m["id"] == "b") b = m;
        if (m["id"] == "c") c = m;
    }
    CHECK_EQ(b["state"].get<std::string>(), std::string("active"));
    CHECK_EQ(b["activations"].get<int>(), 2);
    CHECK_EQ(b["deactivations"].get<int>(), 1);
    CHECK(b["activation_seconds"].get<double>() >= 0);
    CHECK(b["deactivation_seconds"].get<double>() >= 0);
    CHECK_EQ(b["vram_released_mib"].get<int>(), 1234);
    CHECK_EQ(c["state"].get<std::string>(), std::string("cold"));
    CHECK_EQ(c["activations"].get<int>(), 0);

    const std::string m = f.mr->metrics_text();
    CHECK(m.find("radiance:model_state{model=\"b\",state=\"active\"} 1") != std::string::npos);
    CHECK(m.find("radiance:model_state{model=\"c\",state=\"cold\"} 1") != std::string::npos);
    CHECK(m.find("radiance:model_activations_total{model=\"b\"} 2") != std::string::npos);
    CHECK(m.find("radiance:model_deactivations_total{model=\"b\"} 1") != std::string::npos);
    CHECK(m.find("radiance:model_vram_released_mib{model=\"b\"} 1234") != std::string::npos);
    CHECK(m.find("radiance:model_deactivation_seconds_total{model=\"b\"}") != std::string::npos);

    /* An active default model's own /metrics carries them too. */
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
    const HttpResponse scrape = f.get("/metrics");
    CHECK(scrape.body.find("vllm:") != std::string::npos);
    CHECK(scrape.body.find("radiance:model_activations_total{model=\"a\"} 2") != std::string::npos);
}

TEST(stopping_during_a_deactivation_does_not_hang) {
    Gate slow;
    Fleet f(100, false, &slow);
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Deactivating; }));
    f.mr->stop();
    slow.release();
    f.mr->join();
    CHECK_EQ(f.destroyed.load(), 1);
}

RAD_TEST_MAIN()
