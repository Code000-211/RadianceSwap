#!/usr/bin/env python3
"""host_budget_stream.py MODEL_DIR [--tp 2] [--a-headroom-mib N] [--b-headroom-mib 96] [--extra ARGS] [--phases natural,refuse]

GPU acceptance of ticket #46 (docs/swap/host-budget.md, "Host-Pool-Wachstum"): the scenario of PR #43 that ended
in exit 137 (kernel OOM kill): a MoE `a` holds the cards' VRAM and a large share of the host RAM, then MoE `b`
activates, streams its static weights and grows its pinned host pool. Every host pool now goes through the host
checks (MemAvailable, cgroups, --host-headroom-mib) and cache.host_budget_mib before it is allocated.

  natural  cache.host_budget_mib 0 (derived from MemAvailable / cgroups). The first request to `b` must end in one
           of the two clean ways: 200 (b active, streaming; the store copies, prepared models first, were dropped to
           make room) or 503 whose message says "not enough host memory". Either way the server process must be
           alive (no exit 137), `a` must still be active and answer what it answered before.
  refuse   a budget that holds `a`'s pools plus 256 MiB only: `b` cannot fit, so its activation must be refused with
           503 "not enough host memory", `a` stays active and unchanged, `b` is `failed` in /radiance/models and
           over_budget_claims counted. (Needs `b`'s pool growth above 256 MiB: any MoE streaming statics does.)

`a` and `b` are both MODEL_DIR (a MoE); give --a-headroom-mib so `a` leaves `b` too little VRAM and `b` streams
(static_streaming.py's phase `full` derives such a value for a model: its docs/swap/static-streaming.md
"GPU-Abnahme"). `--extra '--ngram-placement disk'` keeps a's host RAM low enough to load a MoE twice.
Needs the GPUs to itself (start.sh refuses when a radiance runs) and a build in ./build.
Exit status 0 when every check passed; logs and json stay in --out.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--out", default="/tmp/swap-acceptance/hb")
ap.add_argument("--phases", default="natural,refuse")
ap.add_argument("--a-headroom-mib", type=int, required=True, help="a's --gpu-headroom-mib: what it leaves b must be short of b's static weights")
ap.add_argument("--b-headroom-mib", type=int, default=96)
ap.add_argument("--extra", default="", help="more engine arguments for both models")
ap.add_argument("--maxlen", type=int, default=4096)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.environ["T21"] = args.out
PORT = 18080
FAILED = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def start(log, manifest):
    r = subprocess.run([os.path.join(HERE, "start.sh"), log, "7200", "--manifest", manifest], capture_output=True, text=True)
    print(r.stdout.strip(), flush=True)
    if "UP pid=" not in r.stdout:
        print(r.stderr)
        sys.exit("the server did not come up; see " + log)
    return int(open(os.path.join(args.out, "pid")).read().strip())


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop(pid):
    print(subprocess.run([os.path.join(HERE, "stop.sh"), str(pid)], capture_output=True, text=True).stdout.strip(), flush=True)


def get(path):
    return json.loads(urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=60).read().decode())


def post(model, prompt="The capital of France is"):
    """(status, text or error message)."""
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions",
                               json.dumps({"model": model, "prompt": prompt, "max_tokens": 8, "temperature": 0}).encode(),
                               {"content-type": "application/json"})
    try:
        return 200, json.load(urllib.request.urlopen(r, timeout=3600))["choices"][0]["text"]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def manifest(name, budget_mib):
    engine = ["--max-model-len", str(args.maxlen), "--max-num-seqs", "1"] + args.extra.split()
    p = os.path.join(args.out, name + ".json")
    json.dump({"gpu_group": {"devices": list(range(args.tp)), "tp": args.tp},
               "cache": {"idle_timeout_ms": 0, "host_budget_mib": budget_mib},
               "models": [{"id": "a", "path": args.model, "default": True,
                           "engine": ["--max-model-len", "262144", "--max-num-seqs", "64"] + engine[4:] + ["--gpu-headroom-mib", str(args.a_headroom_mib)]},
                          {"id": "b", "path": args.model,
                           "engine": engine + ["--gpu-headroom-mib", str(args.b_headroom_mib)]}]}, open(p, "w"), indent=1)
    return p


def states():
    j = get("/radiance/models")
    return {d["id"]: d for d in j["data"]}, j.get("host_memory", {})


def scenario(name, budget_mib, must_refuse):
    print(f"== {name}: cache.host_budget_mib {budget_mib}")
    log = os.path.join(args.out, name + ".log")
    pid = start(log, manifest(name, budget_mib))
    code_a, text_a = post("a")
    check(code_a == 200, "a answers")
    st, host = states()
    print("   host_memory with a active:", host)
    code_b, body_b = post("b")           # activates b: its host pool grows here
    print(f"   first request to b: HTTP {code_b}: {body_b[:200].strip()}")
    check(alive(pid), "the server is alive after b's activation (no exit 137)")
    st, host = states()
    print("   host_memory after:", host)
    print("   b:", {k: st["b"].get(k) for k in ("state", "static_streamed_fraction", "error")})
    check(code_b in (200, 503), "b ends cleanly: 200 or 503")
    if code_b == 503:
        check("not enough host memory" in body_b, "the 503 says not enough host memory")
        check(st["b"]["state"] == "failed", "b is failed, not active")
    else:
        check(not must_refuse, "b was admitted")
        check(st["b"]["state"] == "active", "b is active")
    if must_refuse:
        check(code_b == 503, "b is refused")
        check(host.get("over_budget_claims", 0) > 0, "over_budget_claims counted")
    check(st["a"]["state"] == "active", "a is still active")
    check(post("a") == (200, text_a), "a answers the same as before")
    check(alive(pid), "the server is alive at the end")
    stop(pid)
    log_text = open(log).read()
    check("Killed" not in log_text, "no kill trace in the log")
    return host


phases = args.phases.split(",")
if "natural" in phases:
    scenario("natural", 0, False)
if "refuse" in phases:
    # a's pools plus a little: measured from a run with a roomy budget (a alone, then the budget is fixed)
    print("== refuse: measuring a's pools")
    pid = start(os.path.join(args.out, "probe.log"), manifest("probe", 0))
    post("a")
    pools = states()[1].get("pool_bytes", 0)
    stop(pid)
    check(pools > 0, "a's host pools are measured")
    scenario("refuse", int(pools / 1048576) + 256, True)

print("\nFAILED: " + "; ".join(FAILED) if FAILED else "\nall checks passed")
sys.exit(1 if FAILED else 0)
