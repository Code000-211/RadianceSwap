#!/usr/bin/env python3
"""elastic.py LOG PHASES : the elastic expert slab (#35). Two models m1/m2 are active; PHASES is a
comma list of MODEL:SECONDS (load only that model, C8 diverse prompts) or idle:SECONDS. VRAM per card
(rocm-smi) is sampled every 3 s and printed with the server log's 'group:' lines, and greedy tokens of
both models are checked after every phase. e.g. elastic.py /tmp/t35/mm.log m1:60,idle:20,m2:90,idle:20"""
import hashlib, json, random, re, subprocess, sys, threading, time, urllib.request
log, phases = sys.argv[1], sys.argv[2].split(",")
PORT = 18080
TOP = ["Explain how a transformer language model works.", "Write a python function that merges two sorted lists.", "Summarize the causes of the French Revolution.",
 "Translate into German: The meeting was moved to Thursday afternoon.", "What is the derivative of x^3 sin(x)?", "Give a recipe for a vegetarian lasagna.",
 "Describe the plot of Romeo and Juliet.", "Explain the difference between TCP and UDP.", "Write a haiku about the sea.", "List ten prime numbers and explain what makes them prime."]
PROMPTS = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "Explain in two sentences why the sky is blue.",
 "1, 1, 2, 3, 5, 8, 13,", "Translate to German: The weather is nice today, let us go for a walk.\n", "Once upon a time in a small village by the sea, " * 40]
def post(body, timeout=900):
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def vram():
    o = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
    return [int(m) >> 20 for m in re.findall(r"Total Used Memory \(B\): (\d+)", o)]
def slab():
    """per model and rank: (committed, target, offer) MiB of the expert slab, from /radiance/models"""
    j = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/radiance/models", timeout=30))
    out = {}
    for m in j["data"]:
        if "expert_slab" in m:
            out[m["id"]] = [(v["committed_bytes"] >> 20, v["target_bytes"] >> 20, v["offer_bytes"] >> 20) for v in m["expert_slab"]]
    return out
def slab_line(tag):
    sl = slab()
    s = f"   slab {tag}: " + "; ".join(f"{k} committed/target/offer MiB {v}" for k, v in sorted(sl.items()))
    if len(sl) == 2:
        for r in range(len(next(iter(sl.values())))):
            tot = sum(v[r][0] for v in sl.values()); pool = sum(v[r][2] for v in sl.values())
            s += f" | rank {r}: sum committed {tot} <= pool {pool}: {tot <= pool}"
    return s
def digest(model):
    h = hashlib.sha256()
    for p in PROMPTS:
        c = post({"model": model, "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 1})["choices"][0]
        h.update(c["text"].encode()); h.update(json.dumps((c.get("logprobs") or {}).get("tokens")).encode())
    return h.hexdigest()[:12]
def loglines(frm):
    return [l.rstrip() for l in open(log, errors="replace").readlines()[frm:] if "group:" in l or "group planner" in l or "would not back" in l]
def nlines(): return len(open(log, errors="replace").readlines())
t0 = time.time(); series = []; stop = threading.Event()
def sampler():
    while not stop.is_set():
        series.append((round(time.time() - t0, 1), vram())); stop.wait(3)
threading.Thread(target=sampler, daemon=True).start()
ref = {m: digest(m) for m in ("m1", "m2")}
print("reference tokens", ref, "VRAM MiB", vram(), flush=True)
for ph in phases:
    who, secs = ph.split(":"); secs = float(secs); start = time.time(); frm = nlines(); v0 = vram()
    print(slab_line(f"{ph} start"), flush=True)
    done = [0]; bad = []
    def w(i):
        while time.time() < start + secs:
            try: done[0] += post({"model": who, "prompt": random.choice(TOP) + f" ({i})", "max_tokens": 128, "temperature": 0})["usage"]["completion_tokens"]
            except Exception as e: bad.append(str(e))
    th = [threading.Thread(target=w, args=(i,)) for i in range(8 if who != "idle" else 0)]
    [t.start() for t in th]
    if who == "idle": time.sleep(secs)
    [t.join() for t in th]
    print(slab_line(f"{ph} end"), flush=True)
    v1 = vram()
    try: d = {m: digest(m) for m in ("m1", "m2")}
    except Exception as e: d = {"error": str(e)}
    print(f"== {ph}: VRAM MiB {v0} -> {v1}; tokens {done[0]} errors {len(bad)} {bad[:1]}; greedy tokens ok: {d == ref} {d}", flush=True)
    for l in loglines(frm)[:40]: print("   ", l)
stop.set()
print("VRAM series (s, [card0, card1] MiB):")
for t, v in series[::2]: print("  ", t, v)
