#!/usr/bin/env python3
"""lpstress.py LOG CYCLES PHASE_SECONDS : reproduction for the elastic slab (#35). Two models m1/m2 are active.
Each cycle loads m1 alone, then m2 alone (the idle one gives its pages up, the busy one takes them), and every
load request asks for logprobs, so a non-finite logit in any step shows up as an HTTP 500. Prints per phase the
requests, the errors, the committed expert bytes per engine and the VRAM; at the end the error rate per cycle.
A thread asks /radiance/models every 100 ms: the server answers from one planner snapshot, so each answer must have
sum committed <= pool (the sum of the offers, + POOL_SLACK_MIB) per rank (#51); the total line counts samples and violations."""
import json, random, re, subprocess, sys, threading, time, urllib.request, urllib.error
log, cycles, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
PORT = 18080
TOP = ["Explain how a transformer language model works.", "Write a python function that merges two sorted lists.", "Summarize the causes of the French Revolution.",
 "Translate into German: The meeting was moved to Thursday afternoon.", "What is the derivative of x^3 sin(x)?", "Give a recipe for a vegetarian lasagna.",
 "Describe the plot of Romeo and Juliet.", "Explain the difference between TCP and UDP.", "Write a haiku about the sea.", "List ten prime numbers and explain what makes them prime.",
 "The capital of France is", "1, 1, 2, 3, 5, 8, 13,"]
PROBES = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n", "Explain in two sentences why the sky is blue.",
          "1, 1, 2, 3, 5, 8, 13,", "Translate to German: The weather is nice today, let us go for a walk.\n", "Once upon a time in a small village by the sea, " * 40]
def post(body, timeout=900):
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def vram():
    o = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
    return [int(m) >> 20 for m in re.findall(r"Total Used Memory \(B\): (\d+)", o)]
def slab():
    j = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/radiance/models", timeout=30))
    return {m["id"]: [v["committed_bytes"] >> 20 for v in m["expert_slab"]] for m in j["data"] if "expert_slab" in m}
POOL_SLACK_MIB = 2   # MiB rounding of the four numbers per rank (see elastic.py)
pool_checks = {"n": 0, "bad": 0, "worst": 0}
def pool_sampler():
    while True:
        try:
            ms = [m["expert_slab"] for m in json.load(urllib.request.urlopen(f"http://localhost:{PORT}/radiance/models", timeout=30))["data"] if "expert_slab" in m]
            if len(ms) == 2:
                pool_checks["n"] += 1
                over = max(sum(m[r]["committed_bytes"] for m in ms) - sum(m[r]["offer_bytes"] for m in ms) for r in range(len(ms[0]))) >> 20
                if over > POOL_SLACK_MIB: pool_checks["bad"] += 1; pool_checks["worst"] = max(pool_checks["worst"], over)
        except Exception: pass
        time.sleep(0.1)
threading.Thread(target=pool_sampler, daemon=True).start()
nlog = lambda: len(open(log, errors="replace").readlines())
tot = {"req": 0, "err": 0, "nonfinite_logs": 0}; per_cycle = []
for c in range(cycles):
    errs_c = 0
    for who in ("m1", "m2"):
        end = time.time() + secs; done = [0]; bad = []; frm = nlog()
        def w(i):
            while time.time() < end:
                try:
                    post({"model": who, "prompt": random.choice(TOP) + f" ({i})", "max_tokens": 48, "temperature": 0, "logprobs": 1}); done[0] += 1
                except urllib.error.HTTPError as e: bad.append(e.code)
                except Exception as e: bad.append(str(e)[:40])
        th = [threading.Thread(target=w, args=(i,)) for i in range(8)]; [t.start() for t in th]; [t.join() for t in th]
        # the other engine has been idle (cold, shrunk) for the whole phase: its first requests now
        other = "m2" if who == "m1" else "m1"; probe_bad = []; probes = [0]
        for k in range(12):
            try:
                post({"model": other, "prompt": PROBES[k % len(PROBES)], "max_tokens": 32, "temperature": 0, "logprobs": 1}); probes[0] += 1
            except urllib.error.HTTPError as e: probe_bad.append((other, k, e.code))
            except Exception as e: probe_bad.append((other, k, str(e)[:40]))
            time.sleep(0.3)
        bad += probe_bad
        nf = sum("no finite logit" in l for l in open(log, errors="replace").readlines()[frm:])
        tot["req"] += done[0] + len(bad); tot["err"] += len(bad); tot["nonfinite_logs"] += nf; errs_c += len(bad)
        print(f"cycle {c} {who}: ok {done[0]} probes {probes[0]} errors {len(bad)} {bad[:3]} nonfinite-logs {nf} committed MiB {slab()} VRAM {vram()}", flush=True)
    per_cycle.append(errs_c)
print("POOL sum committed <= pool:", pool_checks, flush=True)
print("TOTAL", tot, "cycles with errors", sum(1 for e in per_cycle if e), "of", cycles, flush=True)
