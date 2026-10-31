#!/usr/bin/env python3
"""gen.py PORT MODEL OUTFILE : greedy completions over a fixed prompt set; writes json of texts (+token logprob ids)."""
import json, sys, urllib.request, time
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = [
 "The capital of France is",
 "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
 "Explain in two sentences why the sky is blue.",
 "1, 1, 2, 3, 5, 8, 13,",
 "Translate to German: The weather is nice today, let us go for a walk.\n",
 "Once upon a time in a small village by the sea, " * 40,   # longer prompt
]
def post(body, timeout=600):
    r = urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
res = []
for p in PROMPTS:
    t = time.time()
    j = post({"model": model, "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 1})
    c = j["choices"][0]
    res.append({"text": c["text"], "tokens": (c.get("logprobs") or {}).get("tokens"), "finish": c["finish_reason"], "ptok": j["usage"]["prompt_tokens"], "s": round(time.time()-t,3)})
json.dump(res, open(out, "w"), indent=1)
print(out, [r["finish"] for r in res], [r["ptok"] for r in res], [r["s"] for r in res])
