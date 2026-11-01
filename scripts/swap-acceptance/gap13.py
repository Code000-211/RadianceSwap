#!/usr/bin/env python3
"""gap13.py OUTDIR ROUNDS REPS VARIANT,... : the 13 % gap between a MoE alone and the same MoE in a two-model manifest (§15).
Variants (all the MoE of m_dm.json with its engine options; the second model of m_dm.json stays cold):
  model   `--model` with the MoE's options          m1     manifest with the MoE only (m_gap1.json)
  m2      manifest with MoE + cold dense (m_dm.json) m2off  m2 with RADIANCE_ELASTIC_SLAB=off (plain pool)
Each round starts one server per variant (rotating order), warms with 2 C1 requests, runs REPS C1 requests
(256 tokens, greedy, ignore_eos) and prints tok/s plus the expert residency of /stats; one server at a time."""
import json, os, subprocess, sys, time, urllib.request
out, rounds, reps, variants = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(out, exist_ok=True)
MOE = json.load(open(f"{here}/m_dm.json"))["models"][0]
ARGS = {"model": ["--model", MOE["path"], "--tp", "2"] + MOE["engine"],
        "m1": ["--manifest", f"{here}/m_gap1.json", "--tp", "2"],
        "m2": ["--manifest", f"{here}/m_dm.json", "--tp", "2"]}
ARGS["m2off"] = ARGS["m2"]
def get(path): return json.load(urllib.request.urlopen(f"http://localhost:18080{path}", timeout=60))
def c1(model):
    body = {"prompt": "Write a long story about a robot.", "max_tokens": 256, "temperature": 0, "ignore_eos": True}
    if model: body["model"] = model
    r = urllib.request.Request("http://localhost:18080/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["timings"]["predicted_per_second"]
res = {v: [] for v in variants}
for rd in range(rounds):
    for v in variants[rd % len(variants):] + variants[:rd % len(variants)]:
        env = dict(os.environ, T21=out, **({"RADIANCE_ELASTIC_SLAB": "off"} if v == "m2off" else {}))
        log = f"{out}/{v}.{rd}.log"
        p = subprocess.run([f"{here}/start.sh", log, "1500"] + ARGS[v], env=env, capture_output=True, text=True)
        print(v, rd, p.stdout.strip().splitlines()[-1:], flush=True)
        if "UP" not in p.stdout: sys.exit(1)
        pid = open(f"{out}/pid").read().strip()
        try:
            model = "moe" if v != "model" else None
            for _ in range(2): c1(model)
            t = [c1(model) for _ in range(reps)]
            res[v] += t
            st = get("/stats")["experts"]
            tiers = {x["name"]: x["units"] for x in st["tiers"]}
            print(f"  {v} round {rd}: C1 tok/s {[round(x, 1) for x in t]} median {sorted(t)[len(t)//2]:.1f}; "
                  f"routed_res/routed {st['routed_res']}/{st['routed']} = {100*st['routed_res']/max(st['routed'],1):.2f}%; tiers {tiers}; "
                  f"flex {st['flex_bytes']/2**20:.0f}/{st['flex_capacity']/2**20:.0f} MiB; promotions {st['promotions']} demotions {st['demotions']} heat {st['heat']}", flush=True)
            if v != "model":
                for m in get("/radiance/models")["data"]:
                    if "expert_slab" in m: print("   slab", m["id"], [(s["committed_bytes"] >> 20, s["target_bytes"] >> 20, s["offer_bytes"] >> 20) for s in m["expert_slab"]])
        finally:
            subprocess.run([f"{here}/stop.sh", pid], capture_output=True, env=env)
for v, t in res.items():
    t.sort(); print(f"{v}: n={len(t)} min {t[0]:.1f} median {t[len(t)//2]:.1f} max {t[-1]:.1f} mean {sum(t)/len(t):.1f}")
