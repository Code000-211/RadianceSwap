/* tests/model_router_test.cpp -- one listener over several models (core/server/model_router.h).
 *
 * Two fake models, each with its own tokenizer, chat template, context limit and scheduler, behind
 * one ModelRouter. Router::dispatch is the call the transport makes, so a request is routed exactly
 * as over a socket; nothing is bound.
 */
#include "server_fakes.h"

#include "server/model_router.h"

#include "cpp-httplib/httplib.h"

namespace {

/* A template that marks every prompt with the model it belongs to. */
struct TaggedTemplate final : FakeChatTemplate {
    std::string tag;
    explicit TaggedTemplate(std::string t) : tag(std::move(t)) {}
    int apply(const json& messages, const json& tools, const ChatRenderOptions& opt,
              ChatRender* out, std::string* why) const override {
        const int rc = FakeChatTemplate::apply(messages, tools, opt, out, why);
        out->prompt = tag + out->prompt;
        return rc;
    }
};

/* A model's frontend and scheduler, and the Server over them. */
struct FakeModel {
    FakeScheduler sched;
    FakeTokenizer tok;
    TaggedTemplate tmpl;
    std::unique_ptr<Server> srv;

    FakeModel(const std::string& id, const std::string& tag, int64_t max_ctx, const std::string& reply,
              bool add_bos)
        : tmpl(tag) {
        sched.reply = reply;
        tok.add_bos = add_bos;
        Deps d;
        d.sched = &sched;
        d.tok = &tok;
        d.chat = &tmpl;
        ServerOptions o;
        o.model_id = id;
        o.max_ctx = max_ctx;
        o.host = "127.0.0.1";
        o.port = 0;
        srv = std::make_unique<Server>(d, o);
    }
    std::string prompt(size_t i) {
        std::lock_guard<std::mutex> lk(sched.mu);
        std::string s;
        for (int32_t t : sched.prompts.at(i)) s += (char)t;
        return s;
    }
    size_t submitted() {
        std::lock_guard<std::mutex> lk(sched.mu);
        return sched.prompts.size();
    }
};

std::string chat(const std::string& model, const std::string& content = "hi",
                 const std::string& extra = "") {
    json b = { { "messages", json::array({ { { "role", "user" }, { "content", content } } }) },
               { "max_tokens", 8 } };
    if (!model.empty()) b["model"] = model;
    std::string s = b.dump();
    if (!extra.empty()) s.insert(s.size() - 1, "," + extra);
    return s;
}

std::string reply_of(const HttpResponse& r) {
    return json::parse(r.body)["choices"][0]["message"]["content"].get<std::string>();
}

std::string code_of(const HttpResponse& r) {
    const json j = json::parse(r.body);
    return j["error"]["code"].is_string() ? j["error"]["code"].get<std::string>() : "";
}

/* Model "a": a long context and a tokenizer that adds BOS. Model "b": a short context and none. */
struct Pair {
    FakeModel a{ "a", "[A]", 200, "alpha", true };
    FakeModel b{ "b", "[B]", 40, "beta", false };
    ModelRouter mr;
    explicit Pair(const std::string& default_id = "a") : mr(default_id, 2) {
        mr.attach("a", a.srv.get());
        mr.attach("b", b.srv.get());
    }
    HttpResponse post(const std::string& body, const std::string& path = "/v1/chat/completions") {
        return mr.router().call("POST", path, body);
    }
};

}  // namespace

TEST(a_request_lands_at_the_frontend_and_scheduler_of_the_model_it_names) {
    Pair p;
    HttpResponse ra = p.post(chat("a"));
    HttpResponse rb = p.post(chat("b"));
    CHECK_EQ(ra.status, 200);
    CHECK_EQ(rb.status, 200);
    CHECK_EQ(reply_of(ra), std::string("alpha"));
    CHECK_EQ(reply_of(rb), std::string("beta"));
    /* Each prompt went through that model's template and tokenizer only. */
    REQUIRE_EQ(p.a.submitted(), (size_t)1);
    REQUIRE_EQ(p.b.submitted(), (size_t)1);
    CHECK_EQ(p.a.prompt(0), std::string("\x01[A]<user>hi<assistant>"));
    CHECK_EQ(p.b.prompt(0), std::string("[B]<user>hi<assistant>"));
    CHECK_EQ(json::parse(ra.body)["model"].get<std::string>(), std::string("a"));
    CHECK_EQ(json::parse(rb.body)["model"].get<std::string>(), std::string("b"));
}

TEST(each_model_keeps_its_own_context_limit) {
    Pair p;
    const std::string long_prompt(60, 'x');   /* one token a byte: fits 200, not 40 */
    CHECK_EQ(p.post(chat("a", long_prompt)).status, 200);
    const HttpResponse rb = p.post(chat("b", long_prompt));
    CHECK_EQ(rb.status, 400);
    CHECK_EQ(p.b.submitted(), (size_t)0);
}

TEST(requests_for_both_models_run_at_the_same_time) {
    Pair p;
    const int n = 12;
    std::vector<std::thread> ts;
    std::vector<std::string> got((size_t)n);
    for (int i = 0; i < n; ++i)
        ts.emplace_back([&, i] { got[(size_t)i] = reply_of(p.post(chat(i % 2 ? "b" : "a"))); });
    for (auto& t : ts) t.join();
    for (int i = 0; i < n; ++i) CHECK_EQ(got[(size_t)i], std::string(i % 2 ? "beta" : "alpha"));
    CHECK_EQ(p.a.submitted(), (size_t)(n / 2));
    CHECK_EQ(p.b.submitted(), (size_t)(n / 2));
}

TEST(a_streamed_request_is_routed_too) {
    Pair p;
    HttpResponse r = p.post(chat("b", "hi", "\"stream\":true"));
    CHECK_EQ(r.status, 200);
    CHECK(r.streaming());
    const std::string wire = drain(r);
    CHECK(wire.find("\"model\":\"b\"") != std::string::npos);
    CHECK_EQ(p.a.submitted(), (size_t)0);
    CHECK_EQ(p.b.submitted(), (size_t)1);
}

TEST(an_unknown_model_is_a_404_that_names_the_served_ones) {
    Pair p;
    const HttpResponse r = p.post(chat("c"));
    CHECK_EQ(r.status, 404);
    CHECK_EQ(code_of(r), std::string("model_not_found"));
    const std::string msg = json::parse(r.body)["error"]["message"].get<std::string>();
    CHECK(msg.find("no such model: c") != std::string::npos);
    CHECK(msg.find("a, b") != std::string::npos);
    CHECK_EQ(p.a.submitted() + p.b.submitted(), (size_t)0);
    /* In Anthropic's envelope on its paths. */
    const HttpResponse am = p.post(R"({"model":"c","max_tokens":4,"messages":[{"role":"user","content":"hi"}]})",
                                   "/v1/messages");
    CHECK_EQ(am.status, 404);
    CHECK_EQ(json::parse(am.body)["type"].get<std::string>(), std::string("error"));
}

TEST(a_request_without_a_model_goes_to_the_default) {
    Pair a_default("a");
    CHECK_EQ(reply_of(a_default.post(chat(""))), std::string("alpha"));
    CHECK_EQ(a_default.a.submitted(), (size_t)1);
    CHECK_EQ(a_default.b.submitted(), (size_t)0);
    Pair b_default("b");
    CHECK_EQ(reply_of(b_default.post(chat(""))), std::string("beta"));
    CHECK_EQ(b_default.b.submitted(), (size_t)1);
    /* ...and so does a body that is not JSON: the default model answers its own 400. */
    CHECK_EQ(b_default.post("{not json").status, 400);
    CHECK_EQ(b_default.a.submitted(), (size_t)0);
}

TEST(media_input_and_the_media_endpoints_are_refused) {
    Pair p;
    const std::string image = R"({"model":"a","messages":[{"role":"user","content":[
        {"type":"text","text":"what is this"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]})";
    const std::string audio = R"({"model":"b","messages":[{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]})";
    const std::string video = R"({"model":"b","messages":[{"role":"user","content":[
        {"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}]}]})";
    for (const std::string& body : { image, audio, video }) {
        const HttpResponse r = p.post(body);
        CHECK_EQ(r.status, 501);
        CHECK(r.body.find("several models") != std::string::npos);
    }
    for (const char* path : { "/v1/images/generations", "/v1/images/edits", "/v1/audio/speech",
                              "/v1/audio/transcriptions", "/v1/videos", "/audio/speech" })
        CHECK_EQ(p.post(R"({"model":"a","prompt":"x"})", path).status, 501);
    CHECK_EQ(p.mr.router().call("GET", "/v1/realtime").status, 501);
    CHECK_EQ(p.mr.router().call("GET", "/v1/videos").status, 501);
    CHECK_EQ(p.a.submitted() + p.b.submitted(), (size_t)0);
    /* A plain text request is served as before. */
    CHECK_EQ(p.post(chat("a")).status, 200);
}

TEST(a_text_request_naming_no_media_is_not_mistaken_for_one) {
    Pair p;
    /* "image" as text and as a JSON value is not an image part. */
    CHECK_EQ(p.post(chat("a", "draw an image_url please", "\"user\":\"image\"")).status, 200);
}

TEST(routes_not_about_a_model_are_answered_by_the_default) {
    Pair p;
    const HttpResponse h = p.mr.router().call("GET", "/health");
    CHECK_EQ(h.status, 200);
    CHECK_EQ(p.mr.router().call("GET", "/nope").status, 404);
    CHECK_EQ(p.mr.router().call("GET", "/v1/chat/completions").status, 405);
}

TEST(a_single_model_server_is_unchanged_by_the_router_existing) {
    /* Without a router the Server answers a `model` it does not know with its own model, and
     * media parts reach its own handling (here: no vision encoder, 501 by its own message). */
    FakeModel solo("solo", "[S]", 200, "single", false);
    CHECK_EQ(reply_of(solo.srv->router().call("POST", "/v1/chat/completions", chat("anything"))),
             std::string("single"));
    CHECK_EQ(reply_of(solo.srv->router().call("POST", "/v1/chat/completions", chat(""))),
             std::string("single"));
    const HttpResponse r = solo.srv->router().call(
        "POST", "/v1/chat/completions",
        R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]})");
    CHECK_EQ(r.status, 501);
    CHECK(r.body.find("vision encoder") != std::string::npos);
}

TEST(stopping_releases_every_attached_model_after_the_listener_is_down) {
    Pair p;
    int rc_a = -1, rc_b = -1;
    std::atomic<bool> a_back{false};
    /* serve_model blocks until stop(); run() was never started, so the listener is down. */
    ModelRouter mr("a", 2);
    std::thread ta([&] { rc_a = mr.serve_model("a", p.a.srv.get()); a_back = true; });
    std::thread tb([&] { rc_b = mr.serve_model("b", p.b.srv.get()); });
    /* Not mr.router().routes(): attach() is adding to that table. */
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!a_back.load());
    mr.stop();
    ta.join();
    tb.join();
    CHECK_EQ(rc_a, (int)RAD_OK);
    CHECK_EQ(rc_b, (int)RAD_OK);
    /* Stopping twice is harmless, and run() after a stop returns at once. */
    mr.stop();
    CHECK_EQ(mr.run(), (int)RAD_OK);
}

TEST(the_listener_routes_over_a_real_socket_and_stops_in_order) {
    Pair p;
    ModelRouter mr("a", 2);
    std::thread ta([&] { mr.serve_model("a", p.a.srv.get()); });
    std::thread tb([&] { mr.serve_model("b", p.b.srv.get()); });
    int rc = -1;
    std::thread tl([&] { rc = mr.run(); });
    REQUIRE(eventually([&] { return mr.port() > 0; }));
    httplib::Client c("127.0.0.1", mr.port());
    const auto ra = c.Post("/v1/chat/completions", chat("a"), "application/json");
    const auto rb = c.Post("/v1/chat/completions", chat("b"), "application/json");
    const auto rn = c.Post("/v1/chat/completions", chat("c"), "application/json");
    const auto rd = c.Post("/v1/chat/completions", chat(""), "application/json");
    REQUIRE(ra && rb && rn && rd);
    CHECK_EQ(ra->status, 200);
    CHECK_EQ(rb->status, 200);
    CHECK_EQ(rn->status, 404);
    CHECK_EQ(json::parse(ra->body)["choices"][0]["message"]["content"].get<std::string>(), std::string("alpha"));
    CHECK_EQ(json::parse(rb->body)["choices"][0]["message"]["content"].get<std::string>(), std::string("beta"));
    CHECK_EQ(json::parse(rd->body)["model"].get<std::string>(), std::string("a"));
    mr.stop();
    tl.join();
    ta.join();
    tb.join();
    CHECK_EQ(rc, (int)RAD_OK);
}

RAD_TEST_MAIN()
