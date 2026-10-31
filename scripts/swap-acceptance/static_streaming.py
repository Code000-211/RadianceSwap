#!/usr/bin/env python3
"""static_streaming.py MODEL_DIR [--pct 50] [--tp 2] [--seqs 1] [--extra ARGS] [--hog PATH] [--out DIR] [--phases ref,stream,status,full]

GPU acceptance of ticket #18 (RAM-Streaming of static weights, docs/swap/static-streaming.md).
Needs the GPUs to itself (start.sh refuses when a radiance runs) and a build in ./build.

  ref     one model, everything on the card: greedy completions (gen.py) and the throughput
          numbers (bench.py) are the baseline; the budget report gives the numbers the next
          phases derive their VRAM from.
  stream  the same model with --gpu-headroom-mib raised until `pct` of its static weights stream
          from the pinned host pool (found from the report, corrected up to 4 times). Completions
          must equal the baseline's bit for bit (text and tokens); bench.py gives what streaming
          costs (decode C1/C8, prefill).
  status  the same as a one-model manifest: /radiance/models and /metrics must report the
          streamed share (static_streamed_fraction, radiance:model_static_streamed_*), and the
          completions equal the baseline again.
  full    the real case: model `a` (default, large KV) takes the cards' free VRAM leaving about
          what `b` needs for `pct` streaming; the first request to `b` activates it at full VRAM.
          b must answer (equal to the baseline), a must still answer, b's status must show a
          streamed share; the activation seconds and b's first-request wall time are printed.

Exit status 0 when every check passed. Every phase leaves its logs and json in --out.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--pct", type=float, default=50.0, help="target share of the static weights to stream")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--out", default="/tmp/swap-acceptance/ss")
ap.add_argument("--phases", default="ref,stream,status,full")
ap.add_argument("--maxlen", type=int, default=4096)
ap.add_argument("--seqs", type=int, default=1, help="--max-num-seqs of the measured model (8 makes bench.py's C8 a real batch)")
ap.add_argument("--a-headroom-mib", type=int, default=0, help="phase full: override a's headroom")
ap.add_argument("--extra", default="", help="more engine arguments for every measured model (a MoE: e.g. '--ngram-placement disk')")
ap.add_argument("--hog", default="", help="phase full: model path of `a`, the one that takes the VRAM (default: the model itself; "
                "a large MoE needs a small one here, two of it do not fit the host RAM)")
ap.add_argument("--hog-extra", default="", help="phase full: more engine arguments for `a`")
ap.add_argument("--prefill", type=int, default=3000, help="bench.py prefill tokens")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.environ["T21"] = args.out          # start.sh writes its pid file here
PORT = 18080
FAILED = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def size_mib(text):
    m = re.search(r"([\d.]+)\s*(B|KiB|MiB|GiB)\b", text)
    return float(m.group(1)) * {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}[m.group(2)]


def report(log):
    """The numbers of the startup budget report, in MiB."""
    t = open(log).read()
    def line(tag):
        m = re.search(r"^\s*" + re.escape(tag) + r"\s+(.*)$", t, re.M)
        return size_mib(m.group(1)) if m else None
    total, held = line("card total"), line("already held")
    share = re.search(r"of the ([\d.]+ \w+) of static weights \(([\d.]+)%\)", t)
    return dict(free=total - held if total is not None and held is not None else None,
                claimable=line("claimable"), statics=line("static weights"), kv=line("vram_kv"),
                share=float(share.group(2)) if share else 0.0, streamed="read from the pinned host pool" in t)


def start(log, extra, timeout=7200):
    r = subprocess.run([os.path.join(HERE, "start.sh"), log, str(timeout)] + extra,
                       capture_output=True, text=True)
    print(r.stdout.strip(), flush=True)
    m = re.search(r"UP pid=(\d+)", r.stdout)
    if not m:
        print(r.stderr)
        sys.exit("the server did not come up; see " + log)
    pid = open(os.path.join(args.out, "pid")).read().strip()
    return pid


def stop(pid):
    print(subprocess.run([os.path.join(HERE, "stop.sh"), pid], capture_output=True, text=True).stdout.strip(),
          flush=True)


def get(path):
    return urllib.request.urlopen(f"http://localhost:{PORT}{path}", timeout=60).read().decode()


def post(model, prompt, n=8):
    t = time.time()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions",
                               json.dumps({"model": model, "prompt": prompt, "max_tokens": n,
                                           "temperature": 0}).encode(), {"content-type": "application/json"})
    j = json.load(urllib.request.urlopen(r, timeout=1800))
    return j["choices"][0]["text"], time.time() - t


def run(script, *a, out=None):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + list(a), capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-400:], flush=True)
    if out:
        open(out, "w").write(r.stdout)
    return r.returncode == 0


def gen(model, out):
    return run("gen.py", str(PORT), model, out)


def same(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    return len(A) == len(B) and all(x["text"] == y["text"] and x["tokens"] == y["tokens"] for x, y in zip(A, B))


ENGINE = ["--max-model-len", str(args.maxlen), "--max-num-seqs", str(args.seqs)] + args.extra.split()
BASE = ["--model", args.model, "--tp", str(args.tp)] + ENGINE
ref_json = os.path.join(args.out, "ref.json")
state_f = os.path.join(args.out, "numbers.json")
phases = args.phases.split(",")
num = json.load(open(state_f)) if os.path.exists(state_f) else {}

# -------------------------------------------------------------------------------------------- ref
if "ref" in phases:
    print("== ref: everything on the card")
    log = os.path.join(args.out, "ref.log")
    pid = start(log, BASE)
    num = report(log)
    check(not num["streamed"], "the baseline streams nothing")
    check(gen(args.model, ref_json), "baseline completions")
    run("bench.py", str(PORT), args.model, str(args.prefill), out=os.path.join(args.out, "bench_ref.txt"))
    stop(pid)
    json.dump(num, open(state_f, "w"))
    print("baseline numbers (MiB):", num)

for k in ("free", "claimable", "statics", "kv"):
    if num.get(k) is None and set(phases) - {"ref"}:
        sys.exit(f"no '{k}' from the baseline report; run the ref phase first")

default_headroom = 96            # --gpu-headroom-mib's default; the report was made with it
target_card = num.get("kv", 0) + num.get("statics", 0) * (1 - args.pct / 100.0)   # claimable for pct streaming


def headroom_for(want_claimable, have_claimable, have_headroom):
    return int(round(have_headroom + have_claimable - want_claimable))


# ---------------------------------------------------------------------------------------- stream
H = headroom_for(target_card, num.get("claimable", 0), default_headroom)
if "stream" in phases:
    print(f"== stream: {args.pct:.0f}% of {num['statics']:.0f} MiB of static weights, first try --gpu-headroom-mib {H}")
    pid = None
    for attempt in range(5):
        log = os.path.join(args.out, f"stream{attempt}.log")
        pid = start(log, BASE + ["--gpu-headroom-mib", str(H)])
        r = report(log)
        print(f"   attempt {attempt}: headroom {H} MiB -> streams {r['share']:.1f}%")
        if abs(r["share"] - args.pct) <= 15.0 and r["streamed"]:
            break
        # more headroom, less claimable, a larger share; the share is a staircase, so step in proportion
        H += int((args.pct - r["share"]) / 100.0 * num["statics"])
        stop(pid)
        pid = None
    check(pid is not None, f"a headroom of {H} MiB streams about {args.pct:.0f}%")
    if pid is not None:
        s_json = os.path.join(args.out, "stream.json")
        check(gen(args.model, s_json), "streamed completions")
        check(same(ref_json, s_json), "streamed completions equal the baseline's, text and tokens")
        run("bench.py", str(PORT), args.model, str(args.prefill), out=os.path.join(args.out, "bench_stream.txt"))
        stop(pid)
        print("cost: compare bench_ref.txt (c1/c8 decode tok/s, prefill tok/s) with bench_stream.txt")
        num["stream_headroom_mib"] = H
        json.dump(num, open(state_f, "w"))

H = num.get("stream_headroom_mib", H)


def manifest(models):
    p = os.path.join(args.out, "manifest.json")
    json.dump({"gpu_group": {"devices": list(range(args.tp)), "tp": args.tp},
               "cache": {"idle_timeout_ms": 0}, "models": models}, open(p, "w"), indent=1)
    return p


# ---------------------------------------------------------------------------------------- status
if "status" in phases:
    print("== status: one-model manifest at the streaming headroom")
    m = manifest([{"id": "b", "path": args.model, "default": True,
                   "engine": ENGINE + ["--gpu-headroom-mib", str(H)]}])
    pid = start(os.path.join(args.out, "status.log"), ["--manifest", m])
    check(gen("b", os.path.join(args.out, "status.json")), "completions through the manifest")
    check(same(ref_json, os.path.join(args.out, "status.json")), "manifest completions equal the baseline's")
    st = {d["id"]: d for d in json.loads(get("/radiance/models"))["data"]}["b"]
    print("   /radiance/models:", {k: st.get(k) for k in ("state", "static_bytes", "static_streamed_bytes",
                                                          "static_streamed_fraction")})
    check(st.get("static_streamed_fraction", 0) > 0, "/radiance/models reports a streamed share")
    met = [l for l in get("/metrics").splitlines() if l.startswith("radiance:model_static_")]
    print("   /metrics:", *met, sep="\n     ")
    check(len(met) == 3 and any("fraction" in l and not l.endswith(" 0") for l in met), "/metrics reports it")
    stop(pid)

# ------------------------------------------------------------------------------------------ full
if "full" in phases:
    print("== full: b activates while a holds the cards' VRAM")
    # 64 MiB above the target: b's fixed costs (arena, staging, queues) are checked before any pool is
    # sized, and a's drift moves what it leaves; at 100 % the exact target fails by ~20 MiB and the
    # pressure path deactivates a instead (which is correct, but not what this phase checks).
    T = target_card + 64
    # a leaves F - 16 - H_a free (drift), b's claimable is then claimable0 - used_a (derivation in
    # docs/swap/static-streaming.md); we want it at T.
    Ha = args.a_headroom_mib or int(num["free"] - 16 - num["claimable"] + T)
    print(f"   a: --gpu-headroom-mib {Ha} (leaves b about {T:.0f} MiB claimable); b: default headroom")
    big = ["--max-model-len", "262144", "--max-num-seqs", "64"] + args.hog_extra.split()
    m = manifest([{"id": "a", "path": args.hog or args.model, "default": True,
                   "engine": big + ["--gpu-headroom-mib", str(Ha)]},
                  {"id": "b", "path": args.model, "engine": ENGINE + ["--gpu-headroom-mib", "96"]}])
    pid = start(os.path.join(args.out, "full.log"), ["--manifest", m])
    text_a, _ = post("a", "The capital of France is")
    free_before = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True,
                                 text=True).stdout
    print("   VRAM with a active:", re.findall(r"Used Memory \(B\): (\d+)", free_before))
    text_b, wall = post("b", "The capital of France is")        # activates b
    print(f"   first request to b (activation + answer): {wall:.2f} s")
    st = {d["id"]: d for d in json.loads(get("/radiance/models"))["data"]}
    print("   b:", {k: st["b"].get(k) for k in ("state", "activation_seconds", "static_streamed_fraction")})
    check(st["b"]["state"] == "active", "b is active")
    check(st["b"].get("static_streamed_fraction", 0) > 0, "b streams part of its static weights at full VRAM")
    check(st["a"]["state"] == "active", "a is still active")
    check(post("a", "The capital of France is")[0] == text_a, "a still answers the same")
    check(gen("b", os.path.join(args.out, "full_b.json")), "b completions at full VRAM")
    check(same(ref_json, os.path.join(args.out, "full_b.json")), "b's completions equal the baseline's")
    stop(pid)

print("\n" + ("ALL CHECKS PASSED" if not FAILED else "FAILED: " + "; ".join(FAILED)))
sys.exit(1 if FAILED else 0)
