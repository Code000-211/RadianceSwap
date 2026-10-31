#!/usr/bin/env python3
"""static_streaming.py MODEL_DIR [--pct 50] [--tp 2] [--seqs 1] [--extra ARGS] [--hog PATH] [--out DIR] [--phases ref,stream,status,full,promote]

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

  promote ticket #19 (docs/swap/static-streaming.md): as `full`, then `a` goes idle and is deactivated
          (cache.idle_timeout_ms = --idle-ms). While it does, `b` is kept busy with logprobs requests;
          b's streamed share must fall to 0 (promoted into the VRAM `a` left), every reply must be
          the baseline's text and tokens, no request may fail and the log must hold no non-finite
          logit. Decode (C1, C8) and prefill of b are measured streaming, and again after the
          promotion; the baseline's numbers (ref) are printed beside them.

Exit status 0 when every check passed. Every phase leaves its logs and json in --out.
"""
import argparse, json, os, re, subprocess, sys, threading, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--pct", type=float, default=50.0, help="target share of the static weights to stream")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--out", default="/tmp/swap-acceptance/ss")
ap.add_argument("--phases", default="ref,stream,status,full,promote")
ap.add_argument("--maxlen", type=int, default=4096)
ap.add_argument("--seqs", type=int, default=1, help="--max-num-seqs of the measured model (8 makes bench.py's C8 a real batch)")
ap.add_argument("--a-headroom-mib", type=int, default=0, help="phase full: override a's headroom")
ap.add_argument("--extra", default="", help="more engine arguments for every measured model (a MoE: e.g. '--ngram-placement disk')")
ap.add_argument("--hog", default="", help="phase full: model path of `a`, the one that takes the VRAM (default: the model itself; "
                "a large MoE needs a small one here, two of it do not fit the host RAM)")
ap.add_argument("--hog-extra", default="", help="phase full: more engine arguments for `a`")
ap.add_argument("--prefill", type=int, default=3000, help="bench.py prefill tokens")
ap.add_argument("--idle-ms", type=int, default=40000, help="phase promote: cache.idle_timeout_ms (a is deactivated after it)")
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


class Load:
    """Concurrent logprobs requests to `model` (greedy, 32 tokens, a few prompts): the keep-alive of a model and the
    stress of its weights. A non-finite logit is an HTTP 500 ("row 0 has no finite logit"); `texts` collects the
    distinct replies per prompt, since the same prompt must always give the same text."""
    PROMPTS = ["The capital of France is", "1, 1, 2, 3, 5, 8, 13,", "Explain in two sentences why the sky is blue."]

    def __init__(self, model, n=4):
        self.model, self.n = model, n
        self.req = self.err = self.nonfinite = 0
        self.texts = {}
        self.done = threading.Event()
        self.th = [threading.Thread(target=self.loop, args=(i,)) for i in range(n)]
        self.lock = threading.Lock()

    def loop(self, i):
        k = i
        while not self.done.is_set():
            pr = self.PROMPTS[k % len(self.PROMPTS)]
            k += 1
            try:
                r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions",
                                           json.dumps({"model": self.model, "prompt": pr, "max_tokens": 32,
                                                       "temperature": 0, "logprobs": 1}).encode(),
                                           {"content-type": "application/json"})
                j = json.load(urllib.request.urlopen(r, timeout=300))
                lp = (j["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
                with self.lock:
                    self.req += 1
                    if any(x is None or x != x or abs(x) == float("inf") for x in lp): self.nonfinite += 1
                    t = self.texts.setdefault(pr, {})
                    t[j["choices"][0]["text"]] = t.get(j["choices"][0]["text"], 0) + 1
            except urllib.error.HTTPError as e:
                with self.lock: self.err += 1
                print("   HTTP", e.code, e.read()[:120], flush=True)
            except Exception as e:
                with self.lock: self.err += 1
                print("   error", str(e)[:80], flush=True)

    def start(self):
        [t.start() for t in self.th]

    def stop(self):
        self.done.set()
        [t.join() for t in self.th]

    def distinct(self):
        """Replies beyond the first for a prompt: 0 when every prompt always gave one text."""
        return sum(len(t) - 1 for t in self.texts.values())

    def summary(self):
        return f"{self.req} requests, {self.err} failed, {self.nonfinite} non-finite, {self.distinct()} extra distinct replies"


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


def manifest(models, idle_ms=0):
    p = os.path.join(args.out, "manifest.json")
    json.dump({"gpu_group": {"devices": list(range(args.tp)), "tp": args.tp},
               "cache": {"idle_timeout_ms": idle_ms}, "models": models}, open(p, "w"), indent=1)
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

# ------------------------------------------------------------------------------------------- ctl
if "ctl" in phases:
    print("== ctl: the promote phase's logprobs load on one model with everything on the card")
    pid = start(os.path.join(args.out, "ctl.log"), BASE)
    load = Load(args.model, 4)
    load.start(); time.sleep(30); load.stop()
    print("   ", load.summary())
    for pr, t in load.texts.items():
        if len(t) > 1: print("   distinct replies to", repr(pr[:30]), {k[:60]: v for k, v in t.items()})
    stop(pid)

# --------------------------------------------------------------------------------------- promote
if "promote" in phases:
    import statistics
    print("== promote: b streams at full VRAM, a is deactivated, b's statics come back")
    T = target_card + 64
    Ha = args.a_headroom_mib or int(num["free"] - 16 - num["claimable"] + T)
    print(f"   a: --gpu-headroom-mib {Ha} (leaves b about {T:.0f} MiB claimable); b: default headroom; "
          f"a is deactivated after {args.idle_ms} ms idle")
    big = ["--max-model-len", "262144", "--max-num-seqs", "64"] + args.hog_extra.split()
    m = manifest([{"id": "a", "path": args.hog or args.model, "default": True,
                   "engine": big + ["--gpu-headroom-mib", str(Ha)]},
                  {"id": "b", "path": args.model, "engine": ENGINE + ["--gpu-headroom-mib", "96"]}],
                 idle_ms=args.idle_ms)
    log = os.path.join(args.out, "promote.log")
    pid = start(log, ["--manifest", m])

    def vram():
        o = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
        return [int(x) >> 20 for x in re.findall(r"Total Used Memory \(B\): (\d+)", o)]

    def state(i):
        return {d["id"]: d for d in json.loads(get("/radiance/models"))["data"]}[i]

    def summary(path):
        """Median of each bench.py line kind: c1/c8 tok/s, prefill tok/s and TTFT ms."""
        t = open(path).read()
        out = {}
        for kind in ("c1", "c8"):
            v = [float(x) for x in re.findall(r"^" + kind + r" \d \{'[^']*': ([\d.]+)\}", t, re.M)]
            out[kind] = statistics.median(v) if v else None
        pf = re.findall(r"^prefill \d \{'[^']*': \(([\d.]+), ([\d.]+), ([\d.]+)\)\}", t, re.M)
        out["prefill"] = statistics.median(float(x[1]) for x in pf) if pf else None
        out["ttft_ms"] = statistics.median(float(x[2]) for x in pf) if pf else None
        out["ptok"] = float(pf[0][0]) if pf else None
        return out

    text_a, _ = post("a", "The capital of France is")
    vram_a = vram()
    text_b, wall = post("b", "The capital of France is")        # activates b at full VRAM
    st0 = state("b")
    print(f"   b activated ({wall:.2f} s), streams {100 * st0.get('static_streamed_fraction', 0):.1f}% "
          f"({st0.get('static_streamed_bytes')} of {st0.get('static_bytes')} bytes); VRAM MiB with a and b: {vram()}")
    check(st0.get("static_streamed_fraction", 0) > 0, "b streams part of its static weights while a holds the cards")
    check(gen("b", os.path.join(args.out, "promote_before.json")), "b completions while streaming")
    check(same(ref_json, os.path.join(args.out, "promote_before.json")), "b's streamed completions equal the baseline's")

    # a stays active (a tiny request every few seconds) while b is measured streaming
    stop_keep = threading.Event()
    def keep_a():
        while not stop_keep.wait(max(args.idle_ms / 4000.0, 2)):
            try: post("a", "ok", 1)
            except Exception: pass
    kt = threading.Thread(target=keep_a); kt.start()
    stress0 = Load("b", 4)
    stress0.start(); time.sleep(15); stress0.stop()
    print("   streaming, 4 concurrent logprobs requests for 15 s:", stress0.summary())
    check(stress0.err == 0 and stress0.nonfinite == 0, "no failure and no non-finite logit while streaming under load")
    run("bench.py", str(PORT), "b", str(args.prefill), out=os.path.join(args.out, "bench_promote_before.txt"))
    stop_keep.set(); kt.join()
    st1 = state("b")
    check(st1.get("static_streamed_fraction", 0) > 0 and state("a")["state"] == "active",
          "b still streams and a is still active after the streaming measurement")

    # a idles out now. b stays busy with one logprobs request after the other: the keep-alive, and the
    # exactness check (alone in its batch, the same prompt must give the same text at every moment
    # of the promotion; a non-finite logit is an HTTP 500: "row 0 has no finite logit"). Several
    # requests at once batch differently and are not bit-identical even on the baseline
    # (phase ctl), so the concurrent stress comes before and after, not during.
    load = Load("b", 1)
    load.start()
    t0 = time.time(); a_gone = None; zero = None; frac = []
    while time.time() - t0 < args.idle_ms / 1000.0 + 180 and zero is None:
        time.sleep(1.0)
        sa, sb = state("a"), state("b")
        f = sb.get("static_streamed_fraction")
        frac.append((round(time.time() - t0, 1), sa["state"], round(f, 3) if f is not None else None))
        if a_gone is None and sa["state"] != "active": a_gone = time.time() - t0
        if f == 0: zero = time.time() - t0
    print("   (seconds since a's last request, a's state, b's streamed fraction):", frac[::max(len(frac) // 12, 1)], frac[-1:])
    time.sleep(10)                                              # more stress on the promoted weights
    load.stop()
    vram_after = vram()
    check(a_gone is not None, f"a was deactivated by its idle timeout ({a_gone and round(a_gone, 1)} s)")
    check(zero is not None, f"b's streamed share fell to 0 ({zero and round(zero, 1)} s after a's last request"
                            f"; {zero and a_gone and round(zero - a_gone, 1)} s after a was gone)")
    st2 = state("b")
    check(st2.get("static_streamed_fraction") == 0 and st2.get("static_streamed_bytes") == 0 and st2.get("state") == "active",
          "/radiance/models: b active, streamed bytes 0")
    met = [l for l in get("/metrics").splitlines() if l.startswith("radiance:model_static_streamed_")]
    print("   /metrics:", *met, sep="\n     ")
    check(any("fraction" in l and l.rstrip().endswith(" 0") for l in met), "/metrics: streamed fraction 0")
    print("   one request at a time while it happened:", load.summary())
    for pr, t in load.texts.items():
        if len(t) > 1: print("   distinct replies to", repr(pr[:30]), {k[:60]: v for k, v in t.items()})
    check(load.req > 0 and load.err == 0 and load.nonfinite == 0,
          "no request failed and no non-finite logit during the promotion")
    check(load.distinct() == 0, "every reply to a prompt was the same text during the promotion")
    stress1 = Load("b", 4)
    stress1.start(); time.sleep(15); stress1.stop()
    print("   promoted, 4 concurrent logprobs requests for 15 s:", stress1.summary())
    check(stress1.err == 0 and stress1.nonfinite == 0, "no failure and no non-finite logit when promoted, under load")
    nf = sum("no finite logit" in l for l in open(log, errors="replace"))
    check(nf == 0, f"the log holds no non-finite logit ({nf})")
    print("   VRAM MiB: a and b", vram_a, "->", vram_after, "(b alone, promoted)")
    promoted = [l.strip() for l in open(log, errors="replace") if "promoted static weights" in l]
    print("   log:", promoted[:2], "...", promoted[-1:])
    check(gen("b", os.path.join(args.out, "promote_after.json")), "b completions after the promotion")
    check(same(ref_json, os.path.join(args.out, "promote_after.json")), "b's completions after the promotion equal the baseline's")
    run("bench.py", str(PORT), "b", str(args.prefill), out=os.path.join(args.out, "bench_promote_after.txt"))
    check(post("b", "The capital of France is")[0] == text_b, "b still answers the same")
    stop(pid)
    rows = {k: summary(os.path.join(args.out, f"bench_{k}.txt")) for k in ("ref", "promote_before", "promote_after")
            if os.path.exists(os.path.join(args.out, f"bench_{k}.txt"))}
    print("   throughput (median of 3): C1 tok/s, C8 tok/s, prefill tok/s, TTFT ms, prompt tokens")
    for k, v in rows.items():
        print(f"     {k:16s}", v)
    json.dump({"streaming_before": st0.get("static_streamed_fraction"), "bench": rows, "load": {"req": load.req, "err": load.err, "nonfinite": load.nonfinite, "extra_distinct": load.distinct()},
               "stress_streaming": stress0.summary(), "stress_promoted": stress1.summary(),
               "a_gone_s": a_gone, "zero_s": zero, "vram_mib": {"a_active": vram_a, "after": vram_after}},
              open(os.path.join(args.out, "promote_numbers.json"), "w"), indent=1)

print("\n" + ("ALL CHECKS PASSED" if not FAILED else "FAILED: " + "; ".join(FAILED)))
sys.exit(1 if FAILED else 0)
