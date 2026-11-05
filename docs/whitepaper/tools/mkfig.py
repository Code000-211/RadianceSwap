#!/usr/bin/env python3
"""Generate the whitepaper's SVG figures. Data from docs/swap/gpu-acceptance.md and docs/GUIDE.md §4.12."""
import sys, os

OUT = sys.argv[1]

STYLE = """
<style>
  .bg{fill:#fcfcfb}
  .t1{fill:#0b0b0b} .t2{fill:#52514e} .t3{fill:#7a7975}
  .grid{stroke:#e6e5e1;stroke-width:1}
  .axis{stroke:#b9b8b2;stroke-width:1}
  .s1{fill:#2a78d6} .s2{fill:#eb6834} .s3{fill:#1baf7a}
  .l1{stroke:#2a78d6} .l2{stroke:#eb6834}
  .box{fill:#f3f2ee;stroke:#c9c8c2;stroke-width:1}
  .box1{fill:#e3eefb;stroke:#2a78d6;stroke-width:1.2}
  .box2{fill:#fde9e0;stroke:#eb6834;stroke-width:1.2}
  .box3{fill:#ddf3ea;stroke:#1baf7a;stroke-width:1.2}
  .boxd{fill:none;stroke:#9a9993;stroke-width:1.2;stroke-dasharray:5 4}
  .arrow{stroke:#52514e;stroke-width:1.6;fill:none}
  .ah{fill:#52514e}
  .gap{stroke:#fcfcfb;stroke-width:2}
  text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
  @media (prefers-color-scheme: dark){
    .bg{fill:#1a1a19}
    .t1{fill:#f0efec} .t2{fill:#c3c2b7} .t3{fill:#9a998f}
    .grid{stroke:#2e2e2c} .axis{stroke:#55544f}
    .s1{fill:#3987e5} .s2{fill:#d95926} .s3{fill:#199e70}
    .l1{stroke:#3987e5} .l2{stroke:#d95926}
    .box{fill:#252523;stroke:#4a4945}
    .box1{fill:#16263b;stroke:#3987e5}
    .box2{fill:#3a1f14;stroke:#d95926}
    .box3{fill:#12301f;stroke:#199e70}
    .boxd{stroke:#77766f}
    .arrow{stroke:#c3c2b7} .ah{fill:#c3c2b7}
    .gap{stroke:#1a1a19}
  }
</style>
<defs>
  <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" class="ah"/>
  </marker>
</defs>
"""

def svg(name, w, h, body, title):
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
         f'role="img" aria-label="{title}">\n<title>{title}</title>\n{STYLE}\n'
         f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>\n{body}\n</svg>\n')
    with open(os.path.join(OUT, name), "w") as f:
        f.write(s)

def T(x, y, s, cls="t1", size=13, anchor="start", weight="normal"):
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{s}</text>')

def rbar(x, y, w, h, cls, r=4, end="right"):
    """Horizontal bar, rounded only at the data end."""
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    if end == "right":
        d = (f"M{x},{y} H{x+w-r} Q{x+w},{y} {x+w},{y+r} V{y+h-r} Q{x+w},{y+h} {x+w-r},{y+h} H{x} Z")
    else:
        d = f"M{x},{y} H{x+w} V{y+h} H{x} Z"
    return f'<path d="{d}" class="{cls}"/>'

def vbar(x, y0, w, h, cls, r=4, round_top=True):
    """Vertical bar from baseline y0 upwards."""
    if h <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    y = y0 - h
    if round_top:
        d = (f"M{x},{y0} V{y+r} Q{x},{y} {x+r},{y} H{x+w-r} Q{x+w},{y} {x+w},{y+r} V{y0} Z")
    else:
        d = f"M{x},{y0} V{y} H{x+w} V{y0} Z"
    return f'<path d="{d}" class="{cls}"/>'

def legend(x, y, items):
    out, cx = [], x
    for cls, label in items:
        out.append(f'<rect x="{cx}" y="{y-10}" width="12" height="12" rx="3" class="{cls}"/>')
        out.append(T(cx + 18, y, label, "t2", 12))
        cx += 30 + len(label) * 6.6
    return "\n".join(out)

# ---------------------------------------------------------------- 1. activation times
def fig_activation():
    W, H = 760, 350
    b = [T(24, 34, "Activation time: cold (from disk) vs. warm (from the RAM store)", "t1", 16, weight="600"),
         T(24, 54, "2× Radeon AI PRO R9700, TP=2 · seconds, shorter is better", "t2", 12),
         legend(24, 82, [("s2", "cold"), ("s1", "warm")])]
    panels = [
        ("Dense · Qwen3.5-0.8B", 3.0, [0, 1, 2, 3], [("cold", 2.61, "s2", "2.6 s"), ("warm", 2.125, "s1", "2.1 s (p50, 120 cycles)")]),
        ("MoE · Qwen3.8-Flash-Next (113.6 GiB)", 80, [0, 20, 40, 60, 80], [("cold", 70.2, "s2", "70 s"), ("warm", 60.0, "s1", "60 s")]),
    ]
    x0, x1 = 120, 600
    y = 112
    for title, vmax, ticks, rows in panels:
        b.append(T(24, y + 4, title, "t1", 13, weight="600"))
        top = y + 16
        bh, gap = 22, 10
        bottom = top + len(rows) * (bh + gap) - gap
        for t in ticks:
            gx = x0 + (x1 - x0) * t / vmax
            b.append(f'<line x1="{gx}" y1="{top-4}" x2="{gx}" y2="{bottom+4}" class="grid"/>')
            b.append(T(gx, bottom + 18, f"{t} s", "t3", 11, "middle"))
        b.append(f'<line x1="{x0}" y1="{top-4}" x2="{x0}" y2="{bottom+4}" class="axis"/>')
        for i, (lab, v, cls, txt) in enumerate(rows):
            by = top + i * (bh + gap)
            b.append(T(x0 - 10, by + 15, lab, "t2", 12, "end"))
            w = (x1 - x0) * v / vmax
            b.append(rbar(x0, by, w, bh, cls))
            b.append(T(x0 + w + 8, by + 15, txt, "t1", 12))
        y = bottom + 46
    b.append(T(24, H - 18, "MoE warm with --ngram-placement disk: weights in 6.3–7.6 s (the n-gram table is then not copied).", "t2", 11))
    svg("activation.svg", W, H, "\n".join(b), "Activation time, cold vs. warm")

# ---------------------------------------------------------------- 2. static streaming cost
def fig_streaming():
    W, H = 760, 410
    b = [T(24, 34, "What streaming static weights over PCIe costs", "t1", 16, weight="600"),
         T(24, 54, "Decode C1 as % of throughput without streaming · x: share of static weights read from host RAM", "t2", 12),
         legend(24, 82, [("s1", "Dense 0.8B (base 326 tok/s)"), ("s2", "MoE Flash-Next (base ≈160 tok/s)")])]
    x0, x1, y0, y1 = 70, 720, 334, 120
    def px(v): return x0 + (x1 - x0) * v / 100
    def py(v): return y0 - (y0 - y1) * v / 100
    for t in [0, 25, 50, 75, 100]:
        b.append(f'<line x1="{x0}" y1="{py(t)}" x2="{x1}" y2="{py(t)}" class="grid"/>')
        b.append(T(x0 - 8, py(t) + 4, f"{t} %", "t3", 11, "end"))
        b.append(T(px(t), y0 + 18, f"{t} %", "t3", 11, "middle"))
    b.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')
    dense = [(0, 100, ""), (25.6, 60.9, "198"), (50.8, 19.2, "63"), (85.5, 9.9, "")]
    moe = [(0, 100, ""), (15.4, 23.1, "37"), (42.7, 15.1, "24"), (96.5, 7.1, "")]
    for pts, cls, lcls, dy in [(dense, "s1", "l1", -12), (moe, "s2", "l2", 18)]:
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(a):.1f},{py(v):.1f}" for i, (a, v, _) in enumerate(pts))
        b.append(f'<path d="{d}" class="{lcls}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
        for a, v, lab in pts:
            b.append(f'<circle cx="{px(a):.1f}" cy="{py(v):.1f}" r="5" class="{cls} gap"/>')
            if lab:
                b.append(T(px(a) + 8, py(v) + dy, f"{lab} tok/s", "t1", 11))
    b.append(T(px(85.5) - 8, py(9.9) - 10, "32 tok/s", "t1", 11, "end"))
    b.append(T(px(96.5) + 2, py(7.1) - 12, "11 tok/s", "t1", 11, "end"))
    b.append(T(24, H - 32, "PCIe ≈ 28 GB/s vs. 600–870 GB/s from VRAM. Output stays bit-identical; streamed weights move back", "t2", 11))
    b.append(T(24, H - 16, "onto the card as soon as VRAM frees up (measured: full throughput again after 1–2 s).", "t2", 11))
    svg("streaming.svg", W, H, "\n".join(b), "Cost of streaming static weights")

# ---------------------------------------------------------------- 3. two models at once
def fig_concurrency():
    W, H = 760, 340
    b = [T(24, 34, "Two models busy at once: who gets how much?", "t1", 16, weight="600"),
         T(24, 54, "Dense 0.8B, one request per model (C1), tok/s · bar = sum of both models", "t2", 12),
         legend(24, 82, [("s1", "Model A"), ("s2", "Model B")])]
    x0, x1 = 230, 640
    vmax = 350
    rows = [("A alone", [329], "329 tok/s"),
            ("A + B, before fix 2", [31, 31], "31 + 31 = 62 tok/s"),
            ("A + B, today", [154, 154], "154 + 154 = 308 tok/s")]
    top, bh, gap = 110, 30, 22
    bottom = top + len(rows) * (bh + gap) - gap
    for t in [0, 100, 200, 300]:
        gx = x0 + (x1 - x0) * t / vmax
        b.append(f'<line x1="{gx}" y1="{top-6}" x2="{gx}" y2="{bottom+6}" class="grid"/>')
        b.append(T(gx, bottom + 22, str(t), "t3", 11, "middle"))
    b.append(f'<line x1="{x0}" y1="{top-6}" x2="{x0}" y2="{bottom+6}" class="axis"/>')
    for i, (lab, vals, txt) in enumerate(rows):
        by = top + i * (bh + gap)
        b.append(T(x0 - 12, by + 20, lab, "t2", 12, "end"))
        cx = x0
        for j, v in enumerate(vals):
            w = (x1 - x0) * v / vmax
            last = j == len(vals) - 1
            cls = "s1" if j == 0 else "s2"
            b.append(rbar(cx, by, w - (0 if last else 2), bh, cls, end="right" if last else "none"))
            cx += w
        b.append(T(cx + 8, by + 20, txt, "t1", 12))
    b.append(T(24, H - 34, "All models' collectives run in one group-wide order (ADR 0003): no deadlock, but no parallel communication", "t2", 11))
    b.append(T(24, H - 18, "either. Fix 2: the hand-off spins 300 µs before the thread sleeps (before: ~0.5 ms wake-up cost × 48 per decode step).", "t2", 11))
    svg("concurrency.svg", W, H, "\n".join(b), "Throughput of two concurrently active models")

# ---------------------------------------------------------------- 4. elastic slab
def fig_elastic():
    W, H = 760, 350
    b = [T(24, 34, "Elastic expert slab: VRAM follows the load", "t1", 16, weight="600"),
         T(24, 54, "Two Flash-Next entries, rank 0, committed MiB per model · shared pool 20,680 MiB", "t2", 12),
         legend(24, 82, [("s1", "m1"), ("s2", "m2")])]
    x0, x1 = 200, 690
    vmax = 21000
    rows = [("load on m1 only", 13448, 7231),
            ("switching to m2", 8613, 8894),
            ("load on m2 only", 7220, 13460)]
    top, bh, gap = 110, 30, 24
    bottom = top + len(rows) * (bh + gap) - gap
    def gx(v): return x0 + (x1 - x0) * v / vmax
    for t in [0, 5000, 10000, 15000, 20000]:
        b.append(f'<line x1="{gx(t)}" y1="{top-6}" x2="{gx(t)}" y2="{bottom+6}" class="grid"/>')
        b.append(T(gx(t), bottom + 22, f"{t//1000} GiB" if t else "0", "t3", 11, "middle"))
    b.append(f'<line x1="{x0}" y1="{top-6}" x2="{x0}" y2="{bottom+6}" class="axis"/>')
    for i, (lab, a, c) in enumerate(rows):
        by = top + i * (bh + gap)
        b.append(T(x0 - 12, by + 20, lab, "t2", 12, "end"))
        wa, wc = gx(a) - x0, gx(c) - x0
        b.append(rbar(x0, by, wa - 2, bh, "s1", end="none"))
        b.append(rbar(x0 + wa, by, wc, bh, "s2"))
        if wa > 60:
            b.append(f'<text x="{x0 + wa/2}" y="{by+20}" font-size="11" text-anchor="middle" fill="#ffffff">{a:,}' + "</text>")
        if wc > 60:
            b.append(f'<text x="{x0 + wa + wc/2}" y="{by+20}" font-size="11" text-anchor="middle" fill="#ffffff">{c:,}' + "</text>")
    pool = gx(20680)
    b.append(f'<line x1="{pool}" y1="{top-14}" x2="{pool}" y2="{bottom+6}" class="arrow" stroke-dasharray="4 3"/>')
    b.append(T(pool, top - 18, "pool", "t2", 11, "middle"))
    off = gx(10314)
    b.append(f'<line x1="{off}" y1="{top-14}" x2="{off}" y2="{bottom+6}" class="axis" stroke-dasharray="2 3"/>')
    b.append(T(off, top - 18, "offer per model (10.3 GiB)", "t3", 11, "middle"))
    b.append(T(24, H - 34, "The busy model grows ~3 GiB past its offer, the idle one falls to its minimum. Σ committed ≤ pool in every", "t2", 11))
    b.append(T(24, H - 18, "sample. Moves in 128 MiB chunks, target reached in 20–30 s; greedy output of both models unchanged.", "t2", 11))
    svg("elastic-slab.svg", W, H, "\n".join(b), "Elastic expert slab under changing load")

# ---------------------------------------------------------------- 5. memory tiers (diagram)
def fig_tiers():
    W, H = 760, 420
    b = [T(24, 34, "Where a model's weights can live", "t1", 16, weight="600"),
         T(24, 54, "Faster at the top, larger at the bottom · bandwidths measured on 2× R9700 / PCIe", "t2", 12)]
    tiers = [
        ("box1", "VRAM (31.9 GiB per card)", "600–870 GB/s", "active models: static weights, KV cache, hot experts"),
        ("box3", "Pinned host pool (RAM)", "≈ 28 GB/s over PCIe", "RAM streaming: cold experts, static weights that do not fit in VRAM"),
        ("box2", "Prepared store (RAM)", "upload ≈ 0.25 s / 2.4 GB", "deactivated models, packed GPU-ready: no parsing, no relayout"),
        ("box", "Disk / page cache", "2.5–3.1 GB/s measured", "cold models: .rad container or HF checkpoint, full load path"),
    ]
    y = 78
    for i, (cls, name, bw, what) in enumerate(tiers):
        inset = i * 26
        x, w, h = 24 + inset, W - 48 - 2 * inset, 64
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" class="{cls}"/>')
        b.append(T(x + 16, y + 26, name, "t1", 14, weight="600"))
        b.append(T(x + w - 16, y + 26, bw, "t2", 12, "end"))
        b.append(T(x + 16, y + 48, what, "t2", 12))
        y += h + 14
    b.append(T(24, H - 16, "States: active = VRAM (+ host pool) · prepared = store · cold = disk only. Page cache does not count as RAM backing.", "t2", 11))
    svg("memory-tiers.svg", W, H, "\n".join(b), "RadianceSwap memory tiers")

# ---------------------------------------------------------------- 6. one card, VRAM split (diagram)
def fig_card():
    W, H = 760, 350
    b = [T(24, 34, "One card, two active models: how VRAM is shared", "t1", 16, weight="600"),
         T(24, 54, "Schematic, not to scale", "t2", 12)]
    x0, x1, y, h = 24, W - 24, 104, 64
    b.append(f'<rect x="{x0}" y="{y}" width="{x1-x0}" height="{h}" rx="8" class="box"/>')
    segs = [("box1", 90, "A: static"), ("box1", 100, "A: KV"), ("box1", 150, "A: expert slab"),
            ("box2", 150, "B: expert slab"), ("box2", 100, "B: KV"), ("box2", 90, "B: static")]
    cx = x0 + 10
    mids = []
    for cls, w, lab in segs:
        b.append(f'<rect x="{cx}" y="{y+10}" width="{w-4}" height="{h-20}" rx="5" class="{cls}"/>')
        b.append(T(cx + (w - 4) / 2, y + 37, lab, "t1", 12, "middle"))
        mids.append(cx + (w - 4) / 2)
        cx += w
    b.append(T(cx + 12, y + 37, "free", "t3", 11))
    # elastic boundary arrow between slabs
    bx = x0 + 10 + 90 + 100 + 150 - 2
    b.append(f'<path d="M{bx-60},{y+h+20} H{bx+60}" class="arrow" marker-start="url(#ah)" marker-end="url(#ah)"/>')
    b.append(T(bx, y + h + 40, "elastic boundary: 128 MiB chunks move to the model with more expert heat", "t2", 12, "middle"))
    kva, kvb = mids[1], mids[4]
    b.append(f'<path d="M{kva},{y+10} C{kva},{y-14} {kvb},{y-14} {kvb},{y+10}" class="arrow" stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    b.append(T((kva + kvb) / 2, y - 24, "KV lending: KV grows into slab bytes given back", "t2", 11, "middle"))
    y2 = y + h + 74
    b.append(T(24, y2, "Order in which freed bytes are handed out", "t1", 13, weight="600"))
    order = [("box1", "1  Streamed statics", "back onto the card"),
             ("box3", "2  KV cache", "of a model with waiting prompts"),
             ("box2", "3  Expert slab", "growth by group heat")]
    ox = 24
    for i, (cls, a, c) in enumerate(order):
        b.append(f'<rect x="{ox}" y="{y2+14}" width="220" height="54" rx="8" class="{cls}"/>')
        b.append(T(ox + 14, y2 + 37, a, "t1", 13, weight="600"))
        b.append(T(ox + 14, y2 + 56, c, "t2", 11))
        if i < 2:
            b.append(f'<path d="M{ox+224},{y2+41} H{ox+244}" class="arrow" marker-end="url(#ah)"/>')
        ox += 248
    b.append(T(24, H - 14, "Running sequences never lose their KV cache; every model keeps a minimum share of experts.", "t2", 11))
    svg("vram-card.svg", W, H, "\n".join(b), "VRAM split between two active models")

# ---------------------------------------------------------------- 7. llama-swap vs RadianceSwap (diagram)
def fig_vs():
    W, H = 760, 420
    b = [T(24, 34, "Two ways to “one endpoint, many models”", "t1", 16, weight="600")]
    # left: llama-swap
    lx, rx, cw = 24, 396, 340
    for x, title, sub in [(lx, "llama-swap", "Proxy in front of separate server processes"),
                          (rx, "RadianceSwap", "One process, one engine, shared cards")]:
        b.append(T(x, 66, title, "t1", 14, weight="600"))
        b.append(T(x, 84, sub, "t2", 12))
    # llama-swap boxes
    b.append(f'<rect x="{lx}" y="100" width="{cw}" height="40" rx="8" class="box"/>')
    b.append(T(lx + cw / 2, 125, "llama-swap (Go): reads model, starts/stops cmd", "t1", 12, "middle"))
    for i, (cls, lab, st) in enumerate([("box1", "Process A", "llama-server · running"), ("boxd", "Process B", "stopped · started on demand")]):
        y = 170 + i * 70
        b.append(f'<rect x="{lx+20}" y="{y}" width="{cw-40}" height="52" rx="8" class="{cls}"/>')
        b.append(T(lx + 36, y + 22, lab, "t1", 13, weight="600"))
        b.append(T(lx + 36, y + 40, st, "t2", 11))
        b.append(f'<path d="M{lx+cw/2},{140 if i == 0 else 140} V{y}" class="arrow" marker-end="url(#ah)" {"stroke-dasharray=\"4 3\"" if i else ""}/>' if i == 0 else
                 f'<path d="M{lx+cw-6},{140} V{y+26} H{lx+cw-18}" class="arrow" stroke-dasharray="4 3" marker-end="url(#ah)"/>')
    b.append(f'<rect x="{lx}" y="312" width="{cw}" height="40" rx="8" class="box"/>')
    b.append(T(lx + cw / 2, 330, "GPU: VRAM fixed per process", "t1", 12, "middle"))
    b.append(T(lx + cw / 2, 345, "co-residency via groups / matrix / capacity (operator)", "t2", 11, "middle"))
    # RadianceSwap boxes
    b.append(f'<rect x="{rx}" y="100" width="{cw}" height="252" rx="10" class="box"/>')
    b.append(T(rx + 16, 122, "radiance --manifest (C++/HIP)", "t1", 12, weight="600"))
    b.append(f'<rect x="{rx+16}" y="134" width="{cw-32}" height="30" rx="6" class="box3"/>')
    b.append(T(rx + cw / 2, 154, "ModelRouter → ModelManager (queue, activation)", "t1", 12, "middle"))
    for i, (cls, lab) in enumerate([("box1", "Runtime A"), ("box2", "Runtime B")]):
        x = rx + 16 + i * ((cw - 32) / 2 + 4)
        w = (cw - 32) / 2 - 4
        b.append(f'<rect x="{x}" y="176" width="{w}" height="52" rx="6" class="{cls}"/>')
        b.append(T(x + w / 2, 198, lab, "t1", 13, "middle", "600"))
        b.append(T(x + w / 2, 216, "runs in parallel", "t2", 11, "middle"))
    b.append(f'<rect x="{rx+16}" y="240" width="{cw-32}" height="30" rx="6" class="box"/>')
    b.append(T(rx + cw / 2, 260, "CollectiveSequencer · GroupPlanner · Store", "t1", 12, "middle"))
    b.append(f'<rect x="{rx+16}" y="282" width="{cw-32}" height="56" rx="6" class="box"/>')
    b.append(T(rx + cw / 2, 304, "GPU group (TP): VRAM shared dynamically", "t1", 12, "middle"))
    b.append(T(rx + cw / 2, 322, "expert slab, KV lending, static streaming", "t2", 11, "middle"))
    b.append(T(24, 384, "llama-swap: any OpenAI-compatible backend, any platform, mature. Switch = stop one process, start another.", "t2", 11))
    b.append(T(24, 402, "RadianceSwap: Radiance engine only (ROCm), experimental. Switch = runtime in the same process, weights warm in RAM.", "t2", 11))
    svg("vs-llama-swap.svg", W, H, "\n".join(b), "llama-swap and RadianceSwap compared")

for f in [fig_activation, fig_streaming, fig_concurrency, fig_elastic, fig_tiers, fig_card, fig_vs]:
    f()
print("ok")
