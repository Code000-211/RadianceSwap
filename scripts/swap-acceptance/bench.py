#!/usr/bin/env python3
"""bench.py PORT MODEL[,MODEL2] [prefill_tokens] : C1 decode, C8 decode (aggregate), prefill. With two models the same loads run on both concurrently."""
import json, sys, urllib.request, time, random, threading
port = sys.argv[1]; models = sys.argv[2].split(","); PF = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
def post(body):
    r = urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon".split()
def rnd(n): return " ".join(random.choice(words) for _ in range(n))
def run(model, kind, out):
    if kind == "c1":
        j = post({"model": model, "prompt": "Write a long story about a robot.", "max_tokens": 256, "temperature": 0, "ignore_eos": True})
        out[(model, kind)] = j["timings"]["predicted_per_second"]
    elif kind == "c8":
        res = [None]*8
        def one(i): res[i] = post({"model": model, "prompt": f"Story {i}: " + rnd(5), "max_tokens": 256, "temperature": 0, "ignore_eos": True})
        t = time.time(); th = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        [x.start() for x in th]; [x.join() for x in th]; dt = time.time() - t
        out[(model, kind)] = sum(r["usage"]["completion_tokens"] for r in res) / dt
    elif kind == "prefill":
        j = post({"model": model, "prompt": rnd(PF), "max_tokens": 1, "temperature": 0})
        out[(model, kind)] = (j["usage"]["prompt_tokens"], j["timings"]["prompt_per_second"], j["timings"]["ttft_ms"])
for kind in ["c1", "c8", "prefill"]:
    for rep in range(3):
        out = {}
        th = [threading.Thread(target=run, args=(m, kind, out)) for m in models]
        [x.start() for x in th]; [x.join() for x in th]
        print(kind, rep, {f"{k[0]}": (round(v, 1) if not isinstance(v, tuple) else tuple(round(x, 1) for x in v)) for k, v in out.items()}, flush=True)
