/* tests/model_pressure_test.cpp -- a model that finds the VRAM taken makes the router deactivate the
 * model idle the longest (core/server/model_router.h, ticket #16, docs/swap/pressure-deactivation.md).
 *
 * The launcher plays the Engine against a VRAM ledger: it takes the model's cost from the ledger
 * (RAD_E_NOMEM, before attaching, if the ledger cannot give it), serves a fake model, and gives the
 * cost back when the Server is gone, as ~Engine does. Nothing touches a GPU.
 */
#include "server_fakes.h"

#include "manifest.h"
#include "server/model_router.h"

namespace {

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

std::string code_of(const HttpResponse& r) {
    const json j = json::parse(r.body);
    return j["error"]["code"].is_string() ? j["error"]["code"].get<std::string>() : "";
}

struct Served {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    std::unique_ptr<Server> srv;
    Served(const std::string& id, const std::string& reply, bool manual) {
        sched.reply = reply;
        sched.manual = manual;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        ServerOptions o;
        o.model_id = id;
        o.max_ctx = 200;
        o.port = 0;
        srv = std::make_unique<Server>(d, o);
    }
};

/* Models "a" (the default), "b" and "c"; none is cold-started but "a". `capacity` MiB of VRAM
 * and a `cost` per model (default 100). A model that does not fit gets RAD_E_NOMEM. */
struct Fleet {
    ModelRegistry reg;
    std::unique_ptr<ModelRouter> mr;
    std::mutex m;
    int64_t capacity, used = 0;
    std::map<std::string, int64_t> cost{ { "a", 100 }, { "b", 100 }, { "c", 100 } };
    std::map<std::string, Served*> live;
    std::map<std::string, int> launches;
    std::atomic<int> fail_rc{ 0 };   /* a non-VRAM failure for every launch of "b" */
    const bool manual;
    std::thread listener;

    Fleet(int64_t cap, bool manual_sched, ActivationOptions act = {})
        : capacity(cap), manual(manual_sched) {
        for (const char* id : { "a", "b", "c" }) {
            ModelRecord r;
            r.id = id;
            r.is_default = std::string(id) == "a";
            reg.add(std::move(r));
        }
        mr = std::make_unique<ModelRouter>("a", 1, &reg, act);
        for (const char* id : { "a", "b", "c" })
            mr->add_cold(id, [this](const std::string& model) {
                {
                    std::lock_guard<std::mutex> lk(m);
                    ++launches[model];
                    if (model == "b" && fail_rc.load() < 0) return fail_rc.load();
                    if (used + cost[model] > capacity) return (int)RAD_E_NOMEM;
                    used += cost[model];
                }
                struct Give {
                    Fleet& f;
                    const std::string& id;
                    ~Give() {
                        std::lock_guard<std::mutex> lk(f.m);
                        f.used -= f.cost[id];
                    }
                } give{ *this, model };
                Served s(model, model == "a" ? "alpha" : model == "b" ? "beta" : "gamma", manual);
                {
                    std::lock_guard<std::mutex> lk(m);
                    live[model] = &s;
                }
                const int rc = mr->serve_model(model, s.srv.get());
                while (s.sched.in_flight() > 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                std::lock_guard<std::mutex> lk(m);
                live.erase(model);
                return rc;
            });
        mr->activate("a");
        listener = std::thread([this] { mr->run(); });
        REQUIRE(eventually([&] { return mr->port() > 0; }));
    }
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
    ModelState state(const char* id) { return reg.state_of(id); }
    int launched(const char* id) {
        std::lock_guard<std::mutex> lk(m);
        return launches[id];
    }
    int64_t used_mib() {
        std::lock_guard<std::mutex> lk(m);
        return used;
    }
    Served* engine(const char* id) {
        std::lock_guard<std::mutex> lk(m);
        const auto it = live.find(id);
        return it == live.end() ? nullptr : it->second;
    }
    int pressure_deactivations(const char* id) {
        const json all = json::parse(mr->status_json());
        for (const json& j : all["data"])
            if (j["id"] == id) return j["pressure_deactivations"].get<int>();
        return -1;
    }
};

void nap() { std::this_thread::sleep_for(std::chrono::milliseconds(40)); }

}  // namespace

TEST(the_model_idle_the_longest_is_deactivated_first_and_the_activation_then_succeeds) {
    Fleet f(200, false);
    REQUIRE(f.state("a") == ModelState::Active);   /* attached at the start: the oldest */
    nap();
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    CHECK_EQ(f.used_mib(), (int64_t)200);
    nap();

    /* No room for "c": "a" has been idle the longest. */
    CHECK_EQ(reply_of(f.post("c")), std::string("gamma"));
    CHECK(f.state("a") == ModelState::Cold);
    CHECK(f.state("b") == ModelState::Active);
    CHECK(f.state("c") == ModelState::Active);
    CHECK_EQ(f.used_mib(), (int64_t)200);
    CHECK_EQ(f.launched("c"), 2);   /* the first attempt found no room, the second did */
    CHECK_EQ(f.pressure_deactivations("a"), 1);
    CHECK_EQ(f.pressure_deactivations("b"), 0);

    /* "b" is used again, so "c" is now the one idle the longest. */
    nap();
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    nap();
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
    CHECK(f.state("c") == ModelState::Cold);
    CHECK(f.state("b") == ModelState::Active);
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.pressure_deactivations("c"), 1);
    CHECK(f.mr->metrics_text().find("radiance:model_pressure_deactivations_total{model=\"c\"} 1") !=
          std::string::npos);
}

TEST(a_model_with_a_running_request_is_never_deactivated_even_when_it_is_the_oldest) {
    Fleet f(200, true);
    /* "a" is the oldest model and holds a running generation; "b" is newer and idle. */
    HttpResponse stream = f.post("a", true);
    REQUIRE(stream.streaming());
    nap();
    HttpResponse rb;
    std::thread tb([&] { rb = f.post("b"); });
    REQUIRE(eventually([&] { return f.engine("b") && f.engine("b")->sched.in_flight() == 1; }));
    f.engine("b")->sched.push(0, "beta");
    f.engine("b")->sched.finish(0, Finish::Stop);
    tb.join();
    CHECK_EQ(reply_of(rb), std::string("beta"));
    nap();

    HttpResponse rc;
    std::thread tc([&] { rc = f.post("c"); });
    REQUIRE(eventually([&] { return f.state("b") == ModelState::Cold; }));
    REQUIRE(eventually([&] { return f.engine("c") && f.engine("c")->sched.in_flight() == 1; }));
    f.engine("c")->sched.push(0, "gamma");
    f.engine("c")->sched.finish(0, Finish::Stop);
    tc.join();
    CHECK_EQ(reply_of(rc), std::string("gamma"));
    CHECK(f.state("a") == ModelState::Active);
    CHECK(f.engine("a")->sched.cancelled.empty());
    CHECK_EQ(f.pressure_deactivations("a"), 0);
    CHECK_EQ(f.pressure_deactivations("b"), 1);

    f.engine("a")->sched.push(0, "alpha");
    f.engine("a")->sched.finish(0, Finish::Stop);
    std::string chunk;
    while (stream.next(chunk)) chunk.clear();
    if (stream.on_close) stream.on_close(true);
}

TEST(with_every_other_model_in_use_the_activation_waits_and_goes_on_once_one_is_idle) {
    Fleet f(100, true);
    HttpResponse stream = f.post("a", true);
    REQUIRE(stream.streaming());

    HttpResponse rb;
    std::thread tb([&] { rb = f.post("b"); });
    REQUIRE(eventually([&] { return f.launched("b") == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    CHECK(f.state("a") == ModelState::Active);   /* never cut for "b" */
    CHECK(f.engine("a")->sched.cancelled.empty());
    CHECK_EQ(f.launched("b"), 1);                /* and no activation spins meanwhile */
    CHECK_EQ(f.mr->queued("b"), (int64_t)1);

    f.engine("a")->sched.push(0, "alpha");
    f.engine("a")->sched.finish(0, Finish::Stop);
    std::string chunk;
    while (stream.next(chunk)) chunk.clear();
    if (stream.on_close) stream.on_close(true);
    REQUIRE(eventually([&] { return f.engine("b") && f.engine("b")->sched.in_flight() == 1; }));
    f.engine("b")->sched.push(0, "beta");
    f.engine("b")->sched.finish(0, Finish::Stop);
    tb.join();
    CHECK_EQ(reply_of(rb), std::string("beta"));
    CHECK(f.state("a") == ModelState::Cold);
    CHECK_EQ(f.launched("b"), 2);
}

TEST(with_every_other_model_in_use_the_wait_ends_at_the_deadline) {
    ActivationOptions act;
    act.wait_ms = 300;
    Fleet f(100, true, act);
    HttpResponse stream = f.post("a", true);
    REQUIRE(stream.streaming());
    const HttpResponse r = f.post("b");
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("model_activation_timeout"));
    CHECK(r.body.find("VRAM") != std::string::npos);
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.launched("b"), 1);

    f.engine("a")->sched.push(0, "alpha");
    f.engine("a")->sched.finish(0, Finish::Stop);
    std::string chunk;
    while (stream.next(chunk)) chunk.clear();
    if (stream.on_close) stream.on_close(true);
}

TEST(without_another_model_to_deactivate_the_activation_fails_at_once_naming_the_vram) {
    ActivationOptions act;
    act.idle_timeout_ms = 100;
    Fleet f(100, false, act);
    REQUIRE(eventually([&] { return f.state("a") == ModelState::Cold; }));   /* nothing is active */
    {
        std::lock_guard<std::mutex> lk(f.m);
        f.capacity = 50;   /* a foreign process took the rest */
    }
    const HttpResponse r = f.post("b");
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("model_activation_failed"));
    CHECK(r.body.find("not enough VRAM") != std::string::npos);
    CHECK(r.body.find("no other model is active") != std::string::npos);
    CHECK_EQ(f.launched("b"), 1);
    CHECK_EQ(f.pressure_deactivations("a"), 0);
}

TEST(a_model_that_cannot_fit_even_alone_ends_after_one_round_per_model_not_in_a_loop) {
    Fleet f(200, false);
    CHECK_EQ(reply_of(f.post("c")), std::string("gamma"));   /* "a" and "c" are active */
    {
        std::lock_guard<std::mutex> lk(f.m);
        f.cost["b"] = 300;
    }
    const HttpResponse r = f.post("b");
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("model_activation_failed"));
    CHECK(f.state("a") == ModelState::Cold);
    CHECK(f.state("c") == ModelState::Cold);
    CHECK_EQ(f.launched("b"), 3);   /* the first attempt and one after each of the two deactivations */
    CHECK_EQ(f.used_mib(), (int64_t)0);
}

TEST(a_failure_that_is_not_a_lack_of_vram_deactivates_nobody) {
    Fleet f(200, false);
    f.fail_rc = RAD_E_STATE;
    const HttpResponse r = f.post("b");
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("model_activation_failed"));
    CHECK(f.state("a") == ModelState::Active);
    CHECK_EQ(f.launched("b"), 1);
    CHECK_EQ(f.pressure_deactivations("a"), 0);
}

TEST(the_share_follows_the_number_of_models_meant_to_be_active) {
    /* 20000 MiB free above 96 MiB of headroom. */
    CHECK_EQ(manifest_share_mib(20000, 96, 3, 0), (int64_t)6634);   /* all three at once */
    CHECK_EQ(manifest_share_mib(20000, 96, 3, 2), (int64_t)9952);   /* two of the three */
    CHECK_EQ(manifest_share_mib(20000, 96, 3, 1), (int64_t)19904);
    CHECK_EQ(manifest_share_mib(20000, 96, 3, 9), (int64_t)6634);   /* never fewer than all */
    CHECK_EQ(manifest_share_mib(50, 96, 3, 2), (int64_t)0);
    CHECK(manifest_has_room(10100, 9952, 96));
    CHECK(manifest_has_room(9952 + 96 - 600, 9952, 96));   /* a sixteenth may be missing */
    CHECK(!manifest_has_room(9952 + 96 - 700, 9952, 96));
}

RAD_TEST_MAIN()
