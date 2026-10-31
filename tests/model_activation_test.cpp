/* tests/model_activation_test.cpp -- a cold model is activated by its first request
 * (core/server/model_router.h, ticket #9).
 *
 * The launcher is the seam the real process fills with "build the Engine, serve it": here it builds
 * a fake model (scheduler, tokenizer, template, Server) after a delay the test controls, and
 * attaches it. Requests go through Router::dispatch, the call the transport makes, with the
 * client's `closed` callback where a test needs one.
 */
#include "server_fakes.h"

#include "manifest.h"
#include "server/model_router.h"

namespace {

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

/* A model's scheduler, frontend and Server. */
struct Served {
    FakeScheduler sched;
    FakeTokenizer tok;
    FakeChatTemplate tmpl;
    std::unique_ptr<Server> srv;
    explicit Served(const std::string& id, const std::string& reply) {
        sched.reply = reply;
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
    size_t submitted() {
        std::lock_guard<std::mutex> lk(sched.mu);
        return sched.prompts.size();
    }
};

std::string chat(const std::string& model) {
    return json({ { "model", model },
                  { "messages", json::array({ { { "role", "user" }, { "content", "hi" } } }) },
                  { "max_tokens", 8 } }).dump();
}

std::string reply_of(const HttpResponse& r) {
    return json::parse(r.body)["choices"][0]["message"]["content"].get<std::string>();
}

std::string code_of(const HttpResponse& r) {
    const json j = json::parse(r.body);
    return j["error"]["code"].is_string() ? j["error"]["code"].get<std::string>() : "";
}

std::string header_of(const HttpResponse& r, const std::string& k) {
    for (const auto& h : r.headers) if (h.first == k) return h.second;
    return "";
}

/* Model "a" is active from the start; "b" and "c" are cold. Their launcher counts its calls, waits
 * for `gate` if one is set, returns `fail_rc` if that is negative, and otherwise serves a fake. */
struct Fleet {
    ModelRegistry reg;
    Served a{ "a", "alpha" };
    std::atomic<int> launches{ 0 };
    std::atomic<int> fail_rc{ 0 };
    Gate* gate = nullptr;
    int delay_ms = 0;
    std::unique_ptr<ModelRouter> mr;

    explicit Fleet(ActivationOptions act = {}) {
        for (const char* id : { "a", "b", "c" }) {
            ModelRecord r;
            r.id = id;
            r.is_default = std::string(id) == "a";
            reg.add(std::move(r));
        }
        mr = std::make_unique<ModelRouter>("a", 1, &reg, act);
        for (const char* id : { "b", "c" })
            mr->add_cold(id, [this](const std::string& model) {
                ++launches;
                if (delay_ms) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                if (gate) gate->wait();
                if (fail_rc.load() < 0) return fail_rc.load();
                Served s(model, model == "b" ? "beta" : "gamma");
                return mr->serve_model(model, s.srv.get());
            });
        mr->attach("a", a.srv.get());
    }
    /* The router must go before the models it serves. */
    ~Fleet() { mr.reset(); }

    HttpResponse post(const std::string& model, std::function<bool()> closed = nullptr) {
        HttpRequest q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = chat(model);
        q.closed = std::move(closed);
        return mr->router().dispatch(std::move(q));
    }
    ModelState state(const char* id) { return reg.state_of(id); }
};

}  // namespace

TEST(models_are_cold_until_a_request_names_them_and_active_after) {
    Fleet f;
    CHECK(f.state("b") == ModelState::Cold);
    CHECK_EQ(f.launches.load(), 0);
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
    CHECK_EQ(f.launches.load(), 0);   /* a request for an active model starts nothing */
    CHECK(f.state("a") == ModelState::Active);

    Gate gate;
    f.gate = &gate;
    HttpResponse r;
    std::thread t([&] { r = f.post("b"); });
    CHECK(eventually([&] { return f.state("b") == ModelState::Activating; }));
    CHECK_EQ(f.mr->queued("b"), (int64_t)1);
    gate.release();
    t.join();
    CHECK_EQ(r.status, 200);
    CHECK_EQ(reply_of(r), std::string("beta"));
    CHECK(f.state("b") == ModelState::Active);
    CHECK_EQ(f.mr->queued("b"), (int64_t)0);
    CHECK(f.state("c") == ModelState::Cold);
    /* Active models are not activated again. */
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    CHECK_EQ(f.launches.load(), 1);
}

TEST(concurrent_first_requests_trigger_exactly_one_activation) {
    Fleet f;
    f.delay_ms = 150;
    const int n = 12;
    std::vector<std::thread> ts;
    std::vector<HttpResponse> got((size_t)n);
    for (int i = 0; i < n; ++i) ts.emplace_back([&, i] { got[(size_t)i] = f.post("b"); });
    for (auto& t : ts) t.join();
    for (const HttpResponse& r : got) {
        CHECK_EQ(r.status, 200);
        CHECK_EQ(reply_of(r), std::string("beta"));
    }
    CHECK_EQ(f.launches.load(), 1);
}

TEST(requests_beyond_the_queue_limit_are_refused_with_429) {
    ActivationOptions act;
    act.max_queued = 2;
    act.retry_after_s = 3;
    Fleet f(act);
    Gate gate;
    f.gate = &gate;
    HttpResponse r1, r2;
    std::thread t1([&] { r1 = f.post("b"); });
    std::thread t2([&] { r2 = f.post("b"); });
    REQUIRE(eventually([&] { return f.mr->queued("b") == 2; }));

    const HttpResponse full = f.post("b");
    CHECK_EQ(full.status, 429);
    CHECK_EQ(code_of(full), std::string("model_queue_full"));
    CHECK_EQ(json::parse(full.body)["error"]["type"].get<std::string>(), std::string("rate_limit_error"));
    CHECK_EQ(header_of(full, "Retry-After"), std::string("3"));
    /* The queue of one model does not limit another's. */
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));

    gate.release();
    t1.join();
    t2.join();
    CHECK_EQ(r1.status, 200);
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(f.launches.load(), 1);
}

TEST(a_client_that_leaves_while_waiting_frees_its_place_and_the_activation_goes_on) {
    ActivationOptions act;
    act.max_queued = 1;
    Fleet f(act);
    Gate gate;
    f.gate = &gate;
    std::atomic<bool> gone{ false };
    HttpResponse r;
    std::thread t([&] { r = f.post("b", [&] { return gone.load(); }); });
    REQUIRE(eventually([&] { return f.mr->queued("b") == 1; }));
    gone = true;
    t.join();
    CHECK_EQ(r.status, 499);
    CHECK_EQ(f.mr->queued("b"), (int64_t)0);
    CHECK(f.state("b") == ModelState::Activating);

    /* The place is free again, and the next request waits on the same activation. */
    HttpResponse r2;
    std::thread t2([&] { r2 = f.post("b"); });
    REQUIRE(eventually([&] { return f.mr->queued("b") == 1; }));
    gate.release();
    t2.join();
    CHECK_EQ(r2.status, 200);
    CHECK_EQ(f.launches.load(), 1);
}

TEST(a_request_waits_no_longer_than_its_deadline) {
    ActivationOptions act;
    act.wait_ms = 100;
    Fleet f(act);
    Gate gate;
    f.gate = &gate;
    const HttpResponse r = f.post("b");
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("model_activation_timeout"));
    CHECK_EQ(f.mr->queued("b"), (int64_t)0);
    /* The activation was not cancelled: it finishes, and the next request is served. */
    gate.release();
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    CHECK_EQ(f.launches.load(), 1);
}

TEST(a_running_model_is_served_while_another_activates) {
    Fleet f;
    Gate gate;
    f.gate = &gate;
    HttpResponse rb;
    std::thread t([&] { rb = f.post("b"); });
    REQUIRE(eventually([&] { return f.state("b") == ModelState::Activating; }));
    for (int i = 0; i < 5; ++i) CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));
    CHECK_EQ(f.mr->queued("b"), (int64_t)1);
    gate.release();
    t.join();
    CHECK_EQ(rb.status, 200);
    CHECK_EQ(f.a.submitted(), (size_t)5);
}

TEST(a_failed_activation_is_reported_to_every_waiter_and_the_next_request_tries_again) {
    Fleet f;
    Gate gate;
    f.gate = &gate;
    f.fail_rc = RAD_E_NOMEM;
    const int n = 4;
    std::vector<std::thread> ts;
    std::vector<HttpResponse> got((size_t)n);
    for (int i = 0; i < n; ++i) ts.emplace_back([&, i] { got[(size_t)i] = f.post("b"); });
    REQUIRE(eventually([&] { return f.mr->queued("b") == n; }));
    gate.release();
    for (auto& t : ts) t.join();
    for (const HttpResponse& r : got) {
        CHECK_EQ(r.status, 503);
        CHECK_EQ(code_of(r), std::string("model_activation_failed"));
        CHECK(r.body.find("activation of model b failed") != std::string::npos);
    }
    CHECK_EQ(f.launches.load(), 1);
    CHECK(f.state("b") == ModelState::Failed);
    CHECK_EQ(f.mr->queued("b"), (int64_t)0);
    CHECK_EQ(reply_of(f.post("a")), std::string("alpha"));

    f.fail_rc = 0;
    CHECK_EQ(reply_of(f.post("b")), std::string("beta"));
    CHECK_EQ(f.launches.load(), 2);
    CHECK(f.state("b") == ModelState::Active);
}

TEST(two_cold_models_activate_independently) {
    Fleet f;
    f.delay_ms = 100;
    HttpResponse rb, rc;
    std::thread tb([&] { rb = f.post("b"); });
    std::thread tc([&] { rc = f.post("c"); });
    tb.join();
    tc.join();
    CHECK_EQ(reply_of(rb), std::string("beta"));
    CHECK_EQ(reply_of(rc), std::string("gamma"));
    CHECK_EQ(f.launches.load(), 2);
}

TEST(an_unknown_model_is_still_a_404_and_lists_the_cold_ones) {
    Fleet f;
    const HttpResponse r = f.post("z");
    CHECK_EQ(r.status, 404);
    CHECK(r.body.find("a, b, c") != std::string::npos);
    CHECK_EQ(f.launches.load(), 0);
}

TEST(stopping_releases_the_waiters_and_joins_the_activation) {
    Fleet f;
    Gate gate;
    f.gate = &gate;
    HttpResponse r;
    std::thread t([&] { r = f.post("b"); });
    REQUIRE(eventually([&] { return f.mr->queued("b") == 1; }));
    f.mr->stop();
    t.join();
    CHECK_EQ(r.status, 503);
    CHECK_EQ(code_of(r), std::string("server_stopping"));
    gate.release();
    f.mr->join();
    CHECK_EQ(code_of(f.post("c")), std::string("server_stopping"));
}

RAD_TEST_MAIN()
