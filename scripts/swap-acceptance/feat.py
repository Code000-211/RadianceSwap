#!/usr/bin/env python3
"""feat.py PORT MODEL OUT : sampling(seed), streaming, long prompt, abort; writes json"""
import json, sys, urllib.request, time, http.client, random
port, model, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
def post(body, timeout=300):
    r = urllib.request.Request(f"http://localhost:{port}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
res = {}
# sampling with a seed: two identical requests, and the same across servers
s1 = post({"model": model, "prompt": "Tell me a short story about a lighthouse.", "max_tokens": 64, "temperature": 0.8, "top_p": 0.9, "seed": 42, "logprobs": 1})
s2 = post({"model": model, "prompt": "Tell me a short story about a lighthouse.", "max_tokens": 64, "temperature": 0.8, "top_p": 0.9, "seed": 42, "logprobs": 1})
res["sample"] = s1["choices"][0]["text"]; res["sample_repeat_equal"] = s1["choices"][0]["text"] == s2["choices"][0]["text"]
# streaming
conn = http.client.HTTPConnection("localhost", port, timeout=120)
conn.request("POST", "/v1/completions", json.dumps({"model": model, "prompt": "The capital of France is", "max_tokens": 24, "temperature": 0, "stream": True}), {"content-type": "application/json"})
r = conn.getresponse(); chunks = 0; text = ""; done = False
for line in r:
    line = line.decode().strip()
    if line.startswith("data:"):
        d = line[5:].strip()
        if d == "[DONE]": done = True; break
        chunks += 1; text += json.loads(d)["choices"][0].get("text", "")
conn.close(); res["stream_chunks"] = chunks; res["stream_done"] = done; res["stream_text"] = text
ns = post({"model": model, "prompt": "The capital of France is", "max_tokens": 24, "temperature": 0})
res["stream_equals_nonstream"] = ns["choices"][0]["text"] == text
# long prompt
random.seed(5); words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
lp = " ".join(random.choice(words) for _ in range(2200))
l = post({"model": model, "prompt": lp + "\nThe next word is", "max_tokens": 16, "temperature": 0})
res["long_prompt_tokens"] = l["usage"]["prompt_tokens"]; res["long_text"] = l["choices"][0]["text"]
# abort: open a long stream, read 3 chunks, drop the connection
conn = http.client.HTTPConnection("localhost", port, timeout=120)
conn.request("POST", "/v1/completions", json.dumps({"model": model, "prompt": "Write a very long story.", "max_tokens": 2000, "temperature": 0, "ignore_eos": True, "stream": True}), {"content-type": "application/json"})
r = conn.getresponse(); n = 0
for line in r:
    if line.startswith(b"data:"):
        n += 1
        if n >= 3: break
conn.close(); res["abort_after_chunks"] = n
time.sleep(1.5)
st = json.load(urllib.request.urlopen(f"http://localhost:{port}/stats?model={model}")); res["in_flight_after_abort"] = st["in_flight"]; res["running_after_abort"] = st["running"]
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, str) else v[:50]) for k, v in res.items()}))
