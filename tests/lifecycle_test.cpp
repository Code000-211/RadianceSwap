/* tests/lifecycle_test.cpp -- what every request leaves behind, however it ends.
 *
 * A request ends in one of a dozen ways (answered, cut by a stop string, failed by the engine,
 * dropped by its client, ended by shutdown, refused at the door), and each way must leave the same
 * bookkeeping: the scheduler told to reap it exactly once and only after its sink finished (a
 * missed reap leaks a slot, an early one hands a freed slot's tokens to another request), its
 * admission places returned, and /metrics counting it once under the right reason. A sweep sends
 * every route malformed input, since the next route added inherits none of the per-route checks.
 */
#include "server_fakes.h"
#include "engine.h"
#include "tiny_model.h"
#include "build/startup.h"

#include <cstring>
#include <filesystem>

namespace {

/* Tears the server down the way the engine does (scheduler first) and checks the reap contract
 * over everything the case submitted. */
void finish_and_check_reaped(Fixture& f) {
    f.sched.join_all();
    f.srv.reset();
    f.sched.check_reaped();
}

HttpRequest request(const char* path, const std::string& body,
                    std::function<bool()> closed = nullptr) {
    HttpRequest q;
    q.method = "POST";
    q.path = path;
    q.body = body;
    q.closed = std::move(closed);
    return q;
}

const char* const kChat = R"({"messages":[{"role":"user","content":"hi"}]})";

}  /* namespace */

/* ================================================================== reaped exactly once */

/* Answered requests, buffered and streamed, in every API's shape and with n choices: each id is
 * reaped once, after its sink finished. */
TEST(every_answered_request_is_reaped_once) {
    Fixture f;
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}],"n":3})").status, 200);
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":["a","b"],"n":2})").status, 200);
    CHECK_EQ(f.post("/v1/messages", R"({"model":"m","max_tokens":8,
                                       "messages":[{"role":"user","content":"hi"}]})").status, 200);
    CHECK_EQ(f.post("/v1/responses", R"({"input":"hi"})").status, 200);
    CHECK_EQ(f.post("/completion", R"({"prompt":"hi"})").status, 200);
    for (const char* path : { "/v1/chat/completions", "/v1/messages", "/v1/responses" }) {
        const std::string body = std::string(path) == "/v1/chat/completions"
            ? R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"n":2})"
            : std::string(path) == "/v1/messages"
            ? R"({"model":"m","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]})"
            : R"({"input":"hi","stream":true})";
        HttpResponse s = f.post(path, body);
        CHECK_EQ(s.status, 200);
        drain(s);
    }
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);
    CHECK_EQ(f.sched.submitted.size(), (size_t)14);
    finish_and_check_reaped(f);
}

/* A stop string cuts the reply and cancels the request while its sink is still open; the server
 * keeps the sink until the scheduler finishes it and only then reaps. */
TEST(a_request_cut_by_a_stop_string_is_reaped_once_its_sink_finishes) {
    Fixture f;
    f.sched.reply = std::string("abEND") + std::string(200, 'x');
    for (int i = 0; i < 4; ++i)
        CHECK_EQ(f.post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],
                                                    "stop":["END"],"max_tokens":500})").status, 200);
    HttpResponse s = f.post("/v1/completions", R"({"prompt":"x","stop":["END"],"max_tokens":500,
                                                   "stream":true})");
    drain(s);
    CHECK(f.sched.saw_cancel());
    finish_and_check_reaped(f);
}

/* An engine failure, buffered and streamed: the request is reaped after its failed sink. */
TEST(a_request_the_engine_failed_is_reaped_once) {
    Fixture f;
    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_IO;
    CHECK_EQ(f.post("/v1/chat/completions", kChat).status, 500);
    HttpResponse s = f.post("/v1/completions", R"({"prompt":"x","stream":true,"n":2})");
    CHECK_EQ(s.status, 200);
    drain(s);
    finish_and_check_reaped(f);
}

/* A client that leaves, before a buffered answer (499) or in the middle of a stream: every choice
 * is cancelled and reaped once its sink finishes, and the admission places come back. */
TEST(a_dropped_request_is_reaped_once_whether_buffered_or_streamed) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;
    std::atomic<bool> gone{false};
    std::thread hangup([&] {
        if (eventually([&] { return f.srv->in_flight() == 2; })) gone.store(true);
    });
    HttpResponse r = f.srv->router().dispatch(
        request("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],"n":2})",
                [&gone] { return gone.load(); }));
    hangup.join();
    CHECK_EQ(r.status, 499);
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);

    std::atomic<bool> never{false};
    HttpResponse s = f.srv->router().dispatch(
        request("/v1/chat/completions",
                R"({"messages":[{"role":"user","content":"hi"}],"stream":true,"n":3})",
                [&never] { return never.load(); }));
    CHECK_EQ(s.status, 200);
    for (int i = 0; i < 3; ++i) { std::string c; s.next(c); }
    s.on_close(false);
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);
    finish_and_check_reaped(f);
}

/* A server stopping under a buffered request answers it 429 "shutting down" and still reaps
 * what it had submitted. */
TEST(a_request_waiting_when_the_server_stops_is_reaped_once) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;
    HttpResponse r;
    std::thread client([&] {
        r = f.srv->router().dispatch(request("/v1/completions", R"({"prompt":"x","n":2})"));
    });
    CHECK(eventually([&] { return f.srv->in_flight() == 2; }));
    f.srv->stop();
    client.join();
    CHECK_EQ(r.status, 429);
    CHECK(has(r.body, "shutting down"));
    finish_and_check_reaped(f);
}

/* A request the scheduler refused got id 0, which it never issued: nothing is reaped for it,
 * and the client hears the failure. */
TEST(a_request_the_scheduler_refused_is_never_reaped) {
    Fixture f;
    f.sched.refuse = true;
    CHECK_EQ(f.post("/v1/chat/completions", kChat).status, 500);
    HttpResponse s = f.post("/v1/completions", R"({"prompt":"x","stream":true,"n":2})");
    CHECK_EQ(s.status, 200);
    CHECK(has(drain(s), "\"finish_reason\":\"error\""));
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);
    finish_and_check_reaped(f);
    CHECK(f.sched.reaped.empty());
}

/* A prompt the KV cache can never hold is refused by the scheduler at submission (RAD_E_FULL, the
 * sink failed with the reason naming the cache's size) and the client hears it at once: a 400
 * context overflow carrying that reason, before any stream starts. Otherwise the request would
 * wait at the head of the queue with no reply. */
TEST(a_prompt_the_kv_cache_can_never_hold_is_a_400_at_once) {
    Fixture f;
    f.sched.refuse = true;
    f.sched.refuse_status = RAD_E_FULL;
    f.sched.refuse_why = "This model's maximum context length is 240 tokens, all the KV cache "
                         "holds for one sequence: the prompt exceeds the available context size";
    for (const auto& [path, body] : std::vector<std::pair<const char*, const char*>>{
             { "/v1/chat/completions", kChat },
             { "/v1/completions", R"({"prompt":"x","stream":true,"n":2})" },
             { "/completion", R"({"prompt":"x"})" } }) {
        const HttpResponse r = f.post(path, body);
        CHECK_EQ(r.status, 400);
        CHECK(has(r.body, "context_length_exceeded"));
        CHECK(has(r.body, "maximum context length is 240 tokens"));
    }
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);
    finish_and_check_reaped(f);
    CHECK(f.sched.reaped.empty());
}

/* Embeddings run through the scheduler like completions; each input is reaped once, after a
 * success and after a failure. */
TEST(every_embedding_input_is_reaped_once) {
    FakeEmbedder emb;
    emb.width = 4;
    Fixture f(Fixture::Tweak{ [&](ServerOptions&, Deps& d) { d.embed = &emb; } });
    f.sched.embedding = { 1.0f, 0.0f, 0.0f, 0.0f };
    CHECK_EQ(f.post("/v1/embeddings", R"({"input":["a","b","c"]})").status, 200);
    f.sched.join_all();
    f.sched.fail_status = RAD_E_IO;
    CHECK_EQ(f.post("/v1/embeddings", R"({"input":["a","b"]})").status, 500);
    finish_and_check_reaped(f);
}

/* ================================================================== admission */

/* A request needing more admission places than the queue has can never be admitted, so it is a
 * 400 naming the bound, not a 429 a client would retry forever. At the bound it is admitted. */
TEST(a_request_that_could_never_be_admitted_is_a_400_naming_the_bound) {
    FakeEmbedder emb;
    Fixture f(Fixture::Tweak{ [&](ServerOptions& o, Deps& d) {
        o.queue_depth = 4;
        d.embed = &emb;
    } });
    struct Case { const char* path; const char* body; const char* param; };
    for (const Case& c : { Case{ "/v1/completions", R"({"prompt":["a","b","c"],"n":2})", "prompt" },
                           Case{ "/v1/chat/completions",
                                 R"({"messages":[{"role":"user","content":"hi"}],"n":5})", "n" },
                           Case{ "/v1/embeddings", R"({"input":["a","b","c","d","e"]})", "input" },
                           Case{ "/completion", R"({"prompt":["a","b","c","d","e"]})", "prompt" } }) {
        HttpResponse r = f.post(c.path, c.body);
        CHECK_EQ(r.status, 400);
        const json e = json::parse(r.body, nullptr, false);
        CHECK_EQ(str_at(e["error"], "param"), std::string(c.param));
        CHECK(has(e["error"].value("message", ""), "at most 4"));
    }
    CHECK(f.sched.submitted.empty());
    CHECK_EQ(f.srv->in_flight(), (int64_t)0);
    /* At the bound it is admitted. */
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":["a","b"],"n":2})").status, 200);
    finish_and_check_reaped(f);
}

/* ================================================================== the route sweep */

namespace {

/* What every response must be, whatever the input: a status a client can act on, and for JSON a
 * body that parses with an error object when the status is an error (Anthropic's envelope on
 * /v1/messages). */
void check_answer(const std::string& what, const HttpResponse& r) {
    const int s = r.status;
    /* 426: a plain GET of a WebSocket route (/v1/realtime) is told to upgrade. */
    const bool known = s == 200 || s == 400 || s == 403 || s == 404 || s == 405 || s == 413 ||
                       s == 422 || s == 426 || s == 429 || s == 501;
    CHECK(known);
    if (!known) fprintf(stderr, "    %s -> %d %s\n", what.c_str(), s, r.body.substr(0, 200).c_str());
    if (r.streaming() || r.content_type.rfind("application/json", 0) != 0) return;
    const json j = json::parse(r.body, nullptr, false);
    CHECK(!j.is_discarded());
    if (j.is_discarded()) {
        fprintf(stderr, "    %s -> %d, body not JSON: %s\n", what.c_str(), s,
                r.body.substr(0, 200).c_str());
        return;
    }
    if (s < 400) return;
    const bool anthropic = what.find("/v1/messages") != std::string::npos;
    const bool shaped = anthropic ? j.value("type", "") == "error" && j["error"].is_object()
                                  : j.contains("error") && j["error"].is_object() &&
                                        j["error"].contains("message");
    CHECK(shaped);
    if (!shaped) fprintf(stderr, "    %s -> %d %s\n", what.c_str(), s, r.body.substr(0, 200).c_str());
}

/* Sends every registered route the bodies or queries no handler should choke on. With `deep`,
 * also a body nested 60000 levels, far past the bound every body is parsed under and deep enough
 * to overflow a worker's stack in any code that walks it recursively; it costs a few
 * milliseconds a route to scan, so only one sweep sends it. */
void sweep(Server& srv, bool deep) {
    const std::string nested = "{\"x\":" + std::string(60000, '[') + std::string(60000, ']') + "}";
    const char* const kBodies[] = { "", "{", "[]", "null", "{\"x\":", "\"text\"", "17",
                                    "{\"messages\":5,\"prompt\":{},\"input\":true}",
                                    "{\"n\":-1,\"stream\":\"yes\",\"max_tokens\":\"x\"}",
                                    "\xff\xfe{}" };
    for (const std::string& sig : srv.router().routes()) {
        const size_t sp = sig.find(' ');
        const std::string method = sig.substr(0, sp);
        std::string path = sig.substr(sp + 1);
        for (size_t at; (at = path.find(':')) != std::string::npos || (at = path.find('*')) != std::string::npos;) {
            const size_t end = path.find('/', at);
            path.replace(at, end == std::string::npos ? std::string::npos : end - at, "x");
        }
        auto run = [&](const std::string& what, HttpRequest q) {
            try {
                check_answer(what, srv.router().dispatch(std::move(q)));
            } catch (const std::exception& e) {
                CHECK(false);
                fprintf(stderr, "    %s threw %s\n", what.c_str(), e.what());
            }
        };
        if (method == "GET") {
            HttpRequest q;
            q.method = "GET";
            parse_target(path + "?x=%ZZ&&=&id=%00&stream=true&model=" + std::string(300, 'm'),
                         &q.path, &q.query);
            run(sig + " junk query", std::move(q));
            continue;
        }
        std::vector<std::string> bodies(std::begin(kBodies), std::end(kBodies));
        if (deep) bodies.push_back(nested);
        for (const std::string& b : bodies) {
            HttpRequest q;
            q.method = method;
            q.path = path;
            q.body = b;
            q.headers["Content-Type"] = "application/json";
            run(sig + " body " + std::to_string(b.size()) + "B", std::move(q));
        }
        /* The multipart form of an upload, with junk fields. */
        HttpRequest m;
        m.method = method;
        m.path = path;
        m.form = { FormPart{ "n", "", "", "x" }, FormPart{ "image", "a.png", "image/png", "notpng" } };
        run(sig + " multipart", std::move(m));
    }
}

}  /* namespace */

/* Every route, fully wired (chat, grammar, media, embeddings, images): malformed input is a 4xx
 * with an error object, never a 500, an exception or a hang. */
TEST(every_route_answers_malformed_input_with_an_error_object) {
    FakeMultimodal mm;
    FakeEmbedder emb;
    FakeImageGen img;
    Fixture f(Fixture::Tweak{ [&](ServerOptions& o, Deps& d) {
        d.mm = &mm;
        d.embed = &emb;
        d.images = &img;
        o.allow_image = o.allow_audio = o.allow_video = true;
        o.decision_type = "d1";
    } });
    sweep(*f.srv, true);
    CHECK(f.sched.submitted.empty());
    finish_and_check_reaped(f);
}

/* The same with only a scheduler and a tokenizer, so the routes whose component is missing answer
 * their 501s. */
TEST(every_route_answers_malformed_input_with_nothing_optional_attached) {
    FakeScheduler sched;
    FakeTokenizer tok;
    Deps d;
    d.sched = &sched;
    d.tok = &tok;
    ServerOptions o;
    o.model_id = "bare";
    o.default_max_tokens = 8;
    {
        Server srv(d, o);
        sweep(srv, false);
    }
    CHECK(sched.submitted.empty());
}

/* ================================================================== /metrics, every outcome */

namespace {

double sample(const std::string& body, const std::string& name) {
    const size_t at = body.find("\n" + name);
    if (at == std::string::npos) return -1.0;
    const size_t sp = body.find(' ', at + 1 + name.size());
    return std::stod(body.substr(sp + 1, body.find('\n', sp) - sp - 1));
}

}  /* namespace */

/* A request cut by max_tokens counts under finished_reason "length", buffered or streamed, on
 * chat and completions alike; one ended by a stop string under "stop". Dashboards read the share
 * of length-cut replies to size max_tokens. */
TEST(a_request_cut_by_max_tokens_counts_as_length) {
    Fixture f;
    f.sched.reply = "abcdefgh";
    const std::string chat = R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":2)";
    CHECK_EQ(f.post("/v1/chat/completions", chat + "}").status, 200);
    HttpResponse s = f.post("/v1/chat/completions", chat + R"(,"stream":true})");
    drain(s);
    HttpResponse t = f.post("/v1/completions", R"({"prompt":"x","max_tokens":2,"stream":true})");
    drain(t);
    CHECK_EQ(f.post("/completion", R"({"prompt":"x","n_predict":2})").status, 200);
    CHECK_EQ(f.post("/v1/completions", R"({"prompt":"x","stop":["cd"]})").status, 200);
    f.sched.join_all();
    const std::string m = f.get("/metrics").body;
    const std::string L = "vllm:request_success_total{model_name=\"test-model\",engine=\"0\"";
    CHECK_EQ(sample(m, L + ",finished_reason=\"length\"}"), 4.0);
    CHECK_EQ(sample(m, L + ",finished_reason=\"stop\"}"), 1.0);
    finish_and_check_reaped(f);
}

/* One request of each outcome, and /metrics counts each once under its reason: success by
 * finish reason, aborts from both a buffered and a streamed client leaving, the engine's failure,
 * the 429, the status classes, the disconnects, and the prompt and generated tokens of every
 * request that ran. The queue is empty after. */
TEST(metrics_count_every_request_once_under_its_outcome) {
    Fixture f(Fixture::Tweak{ [](ServerOptions& o, Deps&) { o.queue_depth = 2; } });
    /* The tokens request number `i` has produced; a hanging request stops at three. */
    auto produced = [&](size_t i) {
        std::lock_guard<std::mutex> lk(f.sched.mu);
        return i < f.sched.submitted.size() ? f.sched.sinks[f.sched.submitted[i]]->n_produced()
                                            : (int64_t)-1;
    };
    const std::string L = "{model_name=\"test-model\",engine=\"0\"";
    const std::string tools = R"(,"tools":[{"type":"function","function":{"name":"f"}}])";
    auto prompt_len = [&](const std::string& extra) {
        HttpResponse r = f.post("/v1/chat/completions/input_tokens",
                                R"({"messages":[{"role":"user","content":"hi"}])" + extra + "}");
        return json::parse(r.body).value("input_tokens", (int64_t)-1);
    };
    const int64_t p_plain = prompt_len(""), p_tools = prompt_len(tools);
    REQUIRE(p_plain > 0 && p_tools > p_plain);

    int64_t prompt = 0, generated = 0;
    f.sched.reply = "hello";
    CHECK_EQ(f.post("/v1/chat/completions", kChat).status, 200);                 /* stop */
    prompt += p_plain; generated += 5;
    CHECK_EQ(f.post("/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],
                                               "max_tokens":2})").status, 200);  /* length */
    prompt += p_plain; generated += 2;
    f.sched.reply = R"(TOOL:f:{"a":1})";
    CHECK_EQ(f.post("/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}])" + tools + "}").status, 200);
    prompt += p_tools; generated += (int64_t)f.sched.reply.size();
    CHECK_EQ(f.post("/v1/chat/completions", "{").status, 400);
    f.sched.join_all();

    /* Two streams that never finish hold the queue; a third request is the 429. A buffered one
     * then leaves. */
    f.sched.reply = "abcdefgh";
    f.sched.hang = true;
    std::atomic<bool> never{false};
    HttpResponse s1 = f.srv->router().dispatch(request(
        "/v1/chat/completions", R"({"messages":[{"role":"user","content":"hi"}],"stream":true})",
        [&never] { return never.load(); }));
    CHECK_EQ(s1.status, 200);
    std::atomic<bool> gone{false};
    HttpResponse b;
    std::thread client([&] {
        b = f.srv->router().dispatch(request("/v1/chat/completions", kChat,
                                             [&gone] { return gone.load(); }));
    });
    CHECK(eventually([&] { return f.srv->in_flight() == 2; }));
    CHECK_EQ(f.post("/v1/chat/completions", kChat).status, 429);
    /* Requests 3 and 4 are the stream and the buffered one; each leaves after its three tokens. */
    CHECK(eventually([&] { return produced(3) == 3 && produced(4) == 3; }));
    gone.store(true);
    client.join();
    CHECK_EQ(b.status, 499);
    prompt += p_plain; generated += 3;
    s1.on_close(false);
    prompt += p_plain; generated += 3;
    f.sched.join_all();
    f.sched.hang = false;

    f.sched.reply = "he";
    f.sched.fail_status = RAD_E_IO;
    CHECK_EQ(f.post("/v1/chat/completions", kChat).status, 500);                 /* error */
    prompt += p_plain; generated += 2;
    f.sched.join_all();

    const std::string m = f.get("/metrics").body;
    auto reason = [&](const char* r) {
        return sample(m, "vllm:request_success_total" + L + ",finished_reason=\"" + r + "\"}");
    };
    CHECK_EQ(reason("stop"), 1.0);
    CHECK_EQ(reason("length"), 1.0);
    CHECK_EQ(reason("tool_calls"), 1.0);
    CHECK_EQ(reason("abort"), 2.0);
    CHECK_EQ(reason("error"), 1.0);
    /* The two token counts, then stop, length and tool_calls; a dropped stream's 200 is not a
     * response that completed. */
    CHECK_EQ(sample(m, "radiance:http_responses_total" + L + ",class=\"2xx\"}"), 5.0);
    CHECK_EQ(sample(m, "radiance:http_responses_total" + L + ",class=\"4xx\"}"), 3.0);
    CHECK_EQ(sample(m, "radiance:http_responses_total" + L + ",class=\"5xx\"}"), 1.0);
    CHECK_EQ(sample(m, "radiance:client_disconnect_total" + L + "}"), 2.0);
    CHECK_EQ(sample(m, "radiance:http_admission_rejected_total" + L + "}"), 1.0);
    CHECK_EQ(sample(m, "radiance:http_queue_depth" + L + "}"), 0.0);
    CHECK_EQ(sample(m, "vllm:prompt_tokens_total" + L + "}"), (double)prompt);
    CHECK_EQ(sample(m, "vllm:generation_tokens_total" + L + "}"), (double)generated);

    /* The occupancy and the ceiling, under the names /health, /load and /metrics agree on. */
    const json h = json::parse(f.get("/health").body);
    CHECK_EQ(h.value("in_flight", (int64_t)-1), (int64_t)0);
    CHECK_EQ(h.value("queue_cap", (int64_t)-1), (int64_t)2);
    const json l = json::parse(f.get("/load").body);
    CHECK_EQ(l.value("server_load", (int64_t)-1), (int64_t)0);
    CHECK_EQ(l.value("queue_cap", (int64_t)-1), (int64_t)2);
    finish_and_check_reaped(f);
}

/* ================================================================== engine and registry lifecycle */

namespace {
int s_fake_init_count = 0;
int s_fake_fini_count = 0;

int fake_kernel_init(const RadParam*, int, int, int, void** out) {
    ++s_fake_init_count;
    static int dummy = 1;
    *out = &dummy;
    return RAD_OK;
}

void fake_kernel_fini(void*) {
    ++s_fake_fini_count;
}
}  /* namespace */

TEST(kernel_instances_tracked_and_finalized_per_owner) {
    s_fake_init_count = 0;
    s_fake_fini_count = 0;

    RadKernelInfo fake_info{};
    fake_info.name = "fake_kernel";
    fake_info.init = fake_kernel_init;
    fake_info.fini = fake_kernel_fini;

    rad::KernelRow fake_row{};
    fake_row.info = &fake_info;
    fake_row.plugin = "fake_plugin";

    rad::Resolved r1{};
    r1.row = &fake_row;
    rad::Resolved r2{};
    r2.row = &fake_row;
    rad::Resolved r3{};
    r3.row = &fake_row;

    rad::Registry reg;
    const void* owner1 = reinterpret_cast<const void*>(0x1111);
    const void* owner2 = reinterpret_cast<const void*>(0x2222);

    CHECK_OK(reg.init_instance(owner1, r1, 0, 1));
    CHECK_OK(reg.init_instance(owner2, r2, 0, 1));
    CHECK_OK(reg.init_instance(owner2, r3, 0, 1));

    CHECK_EQ(s_fake_init_count, 3);
    CHECK_EQ(s_fake_fini_count, 0);
    CHECK_EQ(reg.n_instances(), (size_t)3);
    CHECK_EQ(reg.n_instances(owner1), (size_t)1);
    CHECK_EQ(reg.n_instances(owner2), (size_t)2);

    /* fini_instances(owner1) finalises only owner1 instances; owner2 instances live on. */
    reg.fini_instances(owner1);
    CHECK_EQ(s_fake_fini_count, 1);
    CHECK_EQ(reg.n_instances(), (size_t)2);
    CHECK_EQ(reg.n_instances(owner1), (size_t)0);
    CHECK_EQ(reg.n_instances(owner2), (size_t)2);

    /* Repeated fini_instances(owner1) is a no-op. */
    reg.fini_instances(owner1);
    CHECK_EQ(s_fake_fini_count, 1);

    /* close() finalises remaining instances exactly once. */
    reg.close();
    CHECK_EQ(s_fake_fini_count, 3);
    CHECK_EQ(reg.n_instances(), (size_t)0);
}

/* Two Engine instances run sequentially in the same process: the first engine's destruction does
 * not close or tear down the process-lifetime rad_tools_registry (ADR 0001), so the second engine
 * starts cleanly without use-after-free or duplicate plugin registration errors. */
TEST(two_engine_lifecycles_run_sequentially_without_use_after_free) {
    std::string home = find_radiance_home();
    REQUIRE(!home.empty());

    tiny::TempDir dir("engine_seq");
    REQUIRE(dir.ok());
    REQUIRE(tiny::write_qwen35(dir.path, {}));

    rad::Config cfg1;
    cfg1.model = dir.path;
    cfg1.radiance_home = home;
    cfg1.kernel_hierarchy = { "libavx" };

    {
        rad::Engine engine1(cfg1);
        int s = engine1.start();
        CHECK_EQ(s, RAD_OK);
        CHECK(!rad::rad_tools_registry().plugins().empty());
    }

    /* Process-lifetime registry must still hold loaded plugins after engine1 teardown. */
    CHECK(!rad::rad_tools_registry().plugins().empty());

    rad::Config cfg2;
    cfg2.model = dir.path;
    cfg2.radiance_home = home;
    cfg2.kernel_hierarchy = { "libavx" };

    {
        rad::Engine engine2(cfg2);
        int s = engine2.start();
        CHECK_EQ(s, RAD_OK);
        CHECK(!rad::rad_tools_registry().plugins().empty());
    }

    CHECK(!rad::rad_tools_registry().plugins().empty());
}

TEST(registry_lifecycle_and_process_ownership) {
    std::string home = find_radiance_home();
    REQUIRE(!home.empty());

    rad::LoadedPlugins lp1, lp2;
    CHECK_OK(rad::rad_tools_load_plugins(home, { "libavx" }, "qwen35", "", &lp1));
    CHECK(!rad::rad_tools_registry().plugins().empty());

    /* A second call against the process registry reuses loaded plugins without duplicate errors. */
    CHECK_OK(rad::rad_tools_load_plugins(home, { "libavx" }, "qwen35", "", &lp2));
    CHECK_EQ(lp1.arch_plugin, lp2.arch_plugin);

    /* Registry can be closed explicitly by process runtime owner, resetting plugins. */
    rad::rad_tools_registry().close();
    CHECK(rad::rad_tools_registry().plugins().empty());

    /* Re-opening after close reloads cleanly without use-after-free or error. */
    rad::LoadedPlugins lp3;
    CHECK_OK(rad::rad_tools_load_plugins(home, { "libavx" }, "qwen35", "", &lp3));
    CHECK(!rad::rad_tools_registry().plugins().empty());
    CHECK_EQ(lp3.arch_plugin, lp1.arch_plugin);
}

RAD_TEST_MAIN()
