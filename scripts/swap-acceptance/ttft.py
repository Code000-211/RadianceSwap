#!/usr/bin/env python3
"""ttft.py CASE [N] : one TTFT-split case against a running manifest server (m_ttft.json, models a and b).

Every request reads its split from the response's `timings`: activation_ms (router wait), queue_ms
(scheduler queue), prompt_ms (prefill), ttft_ms (their sum plus parse/tokenisation). Rows go to
ttft_CASE.jsonl (appended); `ttft.py summary` prints p50/p95 per case.
  active : N sequential requests to an active b (b is activated first)
  cold   : one request to a cold b while a is active (the first b of a fresh process)
  warm   : N times: wait until nothing is active, then one request to a or b (both prepared)
  load   : N requests to a from 32 concurrent clients (max-num-seqs 8: the rest queue); decode 64 tokens
  loadact: the load on a, and meanwhile one request to a cold or prepared b (activation under load)
"""
import json, random, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
U = "http://localhost:18080"
words = [w for w in open("/usr/share/dict/words").read().split()] if False else [f"w{i}x" for i in range(3000)]
def get(p): return json.load(urllib.request.urlopen(U + p, timeout=60))
def post(model, n_words, max_tokens):
    prompt = f"Story {random.random()}: " + " ".join(random.choice(words) for _ in range(n_words))
    r = urllib.request.Request(U + "/v1/completions", json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}).encode(), {"content-type": "application/json"})
    t = time.time(); j = json.load(urllib.request.urlopen(r, timeout=900)); wall = (time.time() - t) * 1000
    tm = j["timings"]
    return dict(model=model, wall=wall, act=tm["activation_ms"], queue=tm["queue_ms"], prefill=tm["prompt_ms"], ttft=tm["ttft_ms"], prompt_n=tm["prompt_n"])
def idle():
    for _ in range(300):
        if all(d["state"] not in ("active", "deactivating", "activating") for d in get("/radiance/models")["data"]): return
        time.sleep(0.2)
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
def save(case, rows):
    with open(f"ttft_{case}.jsonl", "a") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    for k in ("act", "queue", "prefill", "ttft"):
        xs = [r[k] for r in rows]
        print(f"{case:8s} n={len(rows):3d} {k:8s} p50 {pct(xs,.5):9.1f} ms  p95 {pct(xs,.95):9.1f} ms", flush=True)
if sys.argv[1] == "summary":
    for case in ("active", "cold", "warm", "load", "loadact"):
        try: rows = [json.loads(l) for l in open(f"ttft_{case}.jsonl")]
        except FileNotFoundError: continue
        if case == "loadact": rows = [r for r in rows if r["act"] > 0] or rows
        save_n = len(rows)
        print(case, save_n, " | ".join(f"{k} p50 {pct([r[k] for r in rows], .5):.1f} p95 {pct([r[k] for r in rows], .95):.1f}" for k in ("act", "queue", "prefill", "ttft")))
    sys.exit()
case = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
PROMPT = 300   # words, ~1700 tokens (b has a 2048-token context); prompt_n is recorded
if case == "active":
    post("b", 20, 4)   # activates b if needed
    save(case, [post("b", PROMPT, 4) for _ in range(N)])
elif case == "cold":
    save(case, [post("b", PROMPT, 4)])
elif case == "warm":
    rows = []
    for i in range(N):
        idle(); rows.append(post("ab"[i % 2], PROMPT, 4))
    save(case, rows)
elif case in ("load", "loadact"):
    extra = []
    def side():
        time.sleep(3); extra.append(post("b", PROMPT, 4))
    th = threading.Thread(target=side) if case == "loadact" else None
    post("a", 20, 4)
    if th: th.start()
    with ThreadPoolExecutor(32) as ex: rows = list(ex.map(lambda _: post("a", PROMPT, 64), range(N)))
    if th: th.join(); rows += extra
    save(case, rows)
