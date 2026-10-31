#!/usr/bin/env python3
"""kvgrow.py LOG SEQS PROMPT_TOKENS GEN_TOKENS [IDLE_SECONDS] : the cross-model KV partition (#17, part 2).

Server: scripts/swap-acceptance/start.sh $T21/kv.log 1500 --manifest scripts/swap-acceptance/m_kv.json --tp 2
  m1 = A: small cache ceiling, so its share goes to experts and its slab is full when it is warm.
  m2 = B: a cache ceiling (--max-model-len 32768 x --max-num-seqs 16) far above the carve its share leaves it.

What it does:
  1. reference: SEQS distinct long prompts, greedy with logprobs, one after the other on B (its carve holds one
     of them), and a few short prompts on A.
  2. A warm: load A for 20 s, so its slab holds its whole pool share. Snapshot S0.
  3. B load: all SEQS prompts at once on B (SEQS x PROMPT_TOKENS should exceed B's carve: the server log at start
     says "kv: ... the cache can grow to N tokens" and the planner's KV line names the carve). B has to grow its
     partition; A is idle, so its slab gives pages back. Sampled every 2 s from /radiance/models:
     A and B expert-slab committed bytes, B's `kv_growth` (grown, target), VRAM per card.
  4. Idle: IDLE_SECONDS (default 45), B's chunks go back (5 s after the claim ends, only when free) and A regrows.
  5. A single probe per model, with logprobs.

Checks (printed PASS/FAIL, exit 1 on any FAIL):
  b_got_kv           B's kv grown_bytes was > 0 at some sample
  a_slab_sank        A's committed slab fell below its warm level
  never_twice        at every sample, per rank: committed(A) + committed(B) + grown(A) + grown(B) <= offer(A) + offer(B)
  no_errors          every B request answered 200 and no 'no finite logit' in the log (the lpstress check)
  finite_logprobs    every returned token logprob is a finite number
  running_undisturbed  each prompt's greedy tokens under contention equal its reference tokens. Greedy tokens of a
                     batched run can differ from a single run in the last bits; a mismatch is printed with the
                     index of the first differing token. Early (< 8) differences or text that is not language
                     point at KV bytes that moved, which is the failure this ticket must not have.
  returned           after the idle phase B's grown_bytes is 0 again and A's slab is back within 256 MiB of S0
"""
import json, math, random, re, subprocess, sys, threading, time, urllib.request, urllib.error
log = sys.argv[1]; seqs = int(sys.argv[2]); ptok = int(sys.argv[3]); gen = int(sys.argv[4])
idle_s = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
A, B = "m1", "m2"
PORT = 18080
WORDS = ("river stone lamp winter garden engine paper window silver orchard thunder market bridge candle harbor "
         "meadow copper lantern violin compass anchor mirror forest island canyon desert signal harvest velvet "
         "quartz ember falcon glacier horizon ivory jungle kettle ladder marble nebula oyster pillar rocket").split()

def prompt(i):
    r = random.Random(1000 + i)
    return f"Document {i}. " + " ".join(r.choice(WORDS) for _ in range(ptok * 3 // 4)) + "\nContinue the story of this document:"
def post(model, body, timeout=1800):
    body = dict(body, model=model)
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))
def vram():
    o = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
    return [int(m) >> 20 for m in re.findall(r"Total Used Memory \(B\): (\d+)", o)]
def models():
    j = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/radiance/models", timeout=30))
    out = {}
    for m in j["data"]:
        if "expert_slab" in m:
            out[m["id"]] = {"slab": m["expert_slab"], "kv": m.get("kv_growth", [])}
    return out
def finite(c):
    lp = (c.get("logprobs") or {}).get("token_logprobs") or []
    return all(x is None or (isinstance(x, (int, float)) and math.isfinite(x)) for x in lp) and len(lp) > 0
def run(model, i, g):
    c = post(model, {"prompt": prompt(i), "max_tokens": g, "temperature": 0, "logprobs": 1})["choices"][0]
    return c["text"], (c.get("logprobs") or {}).get("tokens") or [], finite(c)
def nlines(): return len(open(log, errors="replace").readlines())
MiB = 1 << 20
samples = []; stop = threading.Event(); t0 = time.time()
def sampler():
    while not stop.is_set():
        try:
            m = models(); samples.append((round(time.time() - t0, 1), m, vram()))
        except Exception as e:
            samples.append((round(time.time() - t0, 1), None, str(e)[:60]))
        stop.wait(2)
def line(tag, m):
    if not m: return f"  {tag}: -"
    s = "; ".join(f"{k} slab committed {[v['committed_bytes'] // MiB for v in x['slab']]} MiB"
                  + (f", kv grown/target {[(v['grown_bytes'] // MiB, v['target_bytes'] // MiB) for v in x['kv']]} MiB" if x["kv"] else "")
                  for k, x in sorted(m.items()))
    return f"  {tag}: {s}"

print("== reference", flush=True)
ref = {}
for i in range(seqs):
    ref[i] = run(B, i, gen)
    print(f"  B prompt {i}: {len(ref[i][1])} tokens, finite {ref[i][2]}", flush=True)
aref = [run(A, 100 + k, 24) for k in range(3)]
print("== A warm (20 s)", flush=True)
end = time.time() + 20
def awarm(k):
    while time.time() < end:
        try: run(A, 200 + k, 32)
        except Exception as e: print("  A warm error", str(e)[:80])
th = [threading.Thread(target=awarm, args=(k,)) for k in range(4)]; [t.start() for t in th]; [t.join() for t in th]
time.sleep(1)
s0 = models(); print(line("S0 (A warm)", s0), "VRAM", vram(), flush=True)
a0 = [v["committed_bytes"] for v in s0[A]["slab"]]
threading.Thread(target=sampler, daemon=True).start()

print(f"== B load: {seqs} sequences of ~{ptok} tokens, {gen} generated", flush=True)
frm = nlines(); res = {}; bad = []
def bwork(i):
    try: res[i] = run(B, i, gen)
    except urllib.error.HTTPError as e: bad.append((i, e.code, e.read()[:120]))
    except Exception as e: bad.append((i, str(e)[:100]))
th = [threading.Thread(target=bwork, args=(i,)) for i in range(seqs)]; [t.start() for t in th]; [t.join() for t in th]
print(line("B load end", models()), flush=True)

print(f"== idle {idle_s:.0f} s", flush=True)
time.sleep(idle_s)
final = models(); print(line("idle end", final), "VRAM", vram(), flush=True)
probe = {}
for mdl in (A, B):
    try: probe[mdl] = run(mdl, 300, 32)
    except Exception as e: probe[mdl] = (str(e)[:80], [], False)
stop.set()
loglines = open(log, errors="replace").readlines()[frm:]
nonfin = sum("no finite logit" in l for l in loglines)
print("== server log (kv / group):")
for l in loglines:
    if "kv:" in l and ("backed chunk" in l or "gave a chunk" in l or "could not back" in l) or "would not back" in l:
        print("   ", l.rstrip())
print("== samples (s: A committed | B committed | B kv grown/target | VRAM)")
for t, m, v in samples[::3]:
    if not m: print("  ", t, m, v); continue
    print("  ", t, [x["committed_bytes"] // MiB for x in m[A]["slab"]], [x["committed_bytes"] // MiB for x in m[B]["slab"]],
          [(x["grown_bytes"] // MiB, x["target_bytes"] // MiB) for x in m[B]["kv"]], v)

ok = True
def check(name, cond, why=""):
    global ok
    ok = ok and cond
    print(("PASS " if cond else "FAIL ") + name + (f"  {why}" if why else ""), flush=True)
good = [s for s in samples if s[1]]
gmax = max((max([v["grown_bytes"] for v in s[1][B]["kv"]] or [0]) for s in good), default=0)
check("b_got_kv", gmax > 0, f"max grown {gmax // MiB} MiB")
amin = min((min(v["committed_bytes"] for v in s[1][A]["slab"]) for s in good), default=0)
check("a_slab_sank", amin < min(a0) - 64 * MiB, f"A warm {[x // MiB for x in a0]} MiB, lowest {amin // MiB} MiB")
worst = None
for t, m, _ in good:
    for r in range(len(m[A]["slab"])):
        used = sum(m[k]["slab"][r]["committed_bytes"] + (m[k]["kv"][r]["grown_bytes"] if r < len(m[k]["kv"]) else 0) for k in (A, B))
        pool = sum(m[k]["slab"][r]["offer_bytes"] for k in (A, B))
        if used > pool and (worst is None or used - pool > worst[0]): worst = (used - pool, t, r)
check("never_twice", worst is None, "" if worst is None else f"over the pool by {worst[0] // MiB} MiB at {worst[1]} s rank {worst[2]}")
check("no_errors", not bad and nonfin == 0 and len(res) == seqs, f"errors {bad[:2]}, 'no finite logit' x{nonfin}, answered {len(res)}/{seqs}")
check("finite_logprobs", all(x[2] for x in res.values()) and all(x[2] for x in ref.values()) and all(v[2] for v in probe.values()))
diff = []
for i, (txt, toks, _) in res.items():
    r = ref[i][1]
    n = next((k for k in range(min(len(r), len(toks))) if r[k] != toks[k]), None)
    if n is None and len(r) != len(toks): n = min(len(r), len(toks))
    if n is not None: diff.append((i, n))
check("running_undisturbed", not diff, f"differing prompts (index, first differing token) {diff}")
gend = max([v["grown_bytes"] for v in final[B]["kv"]] or [0])
aend = [v["committed_bytes"] for v in final[A]["slab"]]
check("returned", gend == 0 and all(e >= s - 256 * MiB for e, s in zip(aend, a0)),
      f"B grown {gend // MiB} MiB, A committed {[x // MiB for x in aend]} vs S0 {[x // MiB for x in a0]} MiB")
sys.exit(0 if ok else 1)
