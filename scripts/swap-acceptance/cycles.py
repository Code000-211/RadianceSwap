#!/usr/bin/env python3
"""cycles.py N : alternate a/b requests; wait until deactivated; sample VRAM (rocm-smi) + host RSS/threads/fds each cycle."""
import glob, json, subprocess, sys, time, urllib.request, os, re, statistics
N = int(sys.argv[1]); pid = int(sys.argv[2]); MODE = sys.argv[3] if len(sys.argv) > 3 else "ab"
def get(path): return json.load(urllib.request.urlopen("http://localhost:18080" + path, timeout=60))
def post(m):
    r = urllib.request.Request("http://localhost:18080/v1/completions", json.dumps({"model": m, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["text"]
def vram():
    o = subprocess.run(["/opt/rocm/bin/rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
    return [int(x) // 1048576 for x in re.findall(r"Used Memory \(B\): (\d+)", o)]
def host():
    s = open(f"/proc/{pid}/status").read()
    g = lambda k: int(re.search(k + r":\s+(\d+)", s).group(1))
    return dict(rss=g("VmRSS") // 1024, hwm=g("VmHWM") // 1024, thr=g("Threads"), fds=len(os.listdir(f"/proc/{pid}/fd")),
                pinned=int(re.search(r"Pss_Anon:\s+(\d+)", open(f"/proc/{pid}/smaps_rollup").read()).group(1)) // 1024)
rows = []; prev = {"a": 0.0, "b": 0.0}; txt0 = {}
for i in range(N):
    m = MODE[i % len(MODE)]
    t = time.time(); txt = post(m); wall = time.time() - t
    txt0.setdefault(m, txt); assert txt == txt0[m], (txt, txt0[m])
    st = {d["id"]: d for d in get("/radiance/models")["data"]}
    act = st[m]["activation_seconds"] - prev[m]; prev[m] = st[m]["activation_seconds"]
    # wait until everything is deactivated
    for _ in range(200):
        s2 = {d["id"]: d["state"] for d in get("/radiance/models")["data"]}
        if all(v != "active" and v != "deactivating" for v in s2.values()): break
        time.sleep(0.1)
    time.sleep(0.3)
    v = vram(); h = host()
    kp = "/sys/class/kfd/kfd/proc/" + os.listdir("/sys/class/kfd/kfd/proc")[0]
    h["kq"] = len(os.listdir(kp + "/queues")); h["kvram"] = int(open(glob.glob(kp + "/vram_*")[0]).read()) // 1048576
    rows.append(dict(i=i, m=m, kind=st[m]["last_weight_load"]["kind"], wall=round(wall, 3), act=round(act, 3), wl=st[m]["last_weight_load"]["seconds"], vram0=v[0], vram1=v[1], **h))
    if i % 10 == 0 or i == N - 1: print(rows[-1], flush=True)
json.dump(rows, open("cycles.json", "w"))
def pct(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
for kind in ("cold", "warm"):
    xs = [r["act"] for r in rows if r["kind"] == kind]; ws = [r["wall"] for r in rows if r["kind"] == kind]
    if xs: print(kind, "n", len(xs), "activation_s p50 %.3f p95 %.3f max %.3f | request wall p50 %.3f p95 %.3f" % (pct(xs, .5), pct(xs, .95), max(xs), pct(ws, .5), pct(ws, .95)))
# leak check: linear slope over the second half
def slope(ys):
    n = len(ys); xm = (n - 1) / 2; ym = sum(ys) / n
    return sum((x - xm) * (y - ym) for x, y in enumerate(ys)) / sum((x - xm) ** 2 for x in range(n))
for k in ("vram0", "vram1", "kvram", "kq", "rss", "pinned", "thr", "fds"):
    ys = [r[k] for r in rows]; h = ys[len(ys) // 4:]
    print(k, "first", ys[0], "min", min(ys), "max", max(ys), "last", ys[-1], "slope/cycle (after 25%%) %.4f" % slope(h))
