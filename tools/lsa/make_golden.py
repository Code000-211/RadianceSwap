"""Write the golden data `ref_test`'s LSA cases compare libref against (tests/data/golden/lsa/).

The oracle is latent-sparse-attention itself - its Ward merge (`lsa.core.ward`) on the index keys its served
projection makes - with the selection taken at page granularity as docs/LSA.md specifies. The map is a real one,
Qwen3.8-27B's distilled D-LSA map, so the keys go through the trained projection rather than a random one.
Run it from a checkout of latent-sparse-attention, whose environment has torch:

    uv run --project ../latent-sparse-attention python tools/lsa/make_golden.py \\
        --map qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt --layer 3

Every file is raw little-endian with no header; the shapes are the constants below, which ref_test repeats.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from lsa.core.ward import pool_keys, ward_merge

# Qwen3.8-27B's global layer
HEADS, KV_HEADS, HEAD_DIM, ROT = 24, 4, 256, 64
FREE = HEAD_DIM - ROT
INDEX_DIM = 64
# lsa_index on a few rows - q alone is 12 KiB a row
INDEX_ROWS = 8
# Three cluster blocks of the served design, and a selection scaled down so it is not every page
RATIO, BLOCK, BUCKETS, CAP = 4, 512, 64, 16
TOKENS = 3 * BLOCK
WINDOW, SINK, TOP_P, TOP_K = 128, 4, 8, 64
QUERIES = (520, 1023, 1100, 1535)
TABLE_WIDTH = 64

OUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "golden" / "lsa"


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def index(q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """latent-sparse-attention's `KeyAndQueryProjection` on rotated, normed q [T, 24, 256] and k [T, 4, 256]."""
    key = k[..., ROT:].reshape(len(k), KV_HEADS * FREE).to(torch.bfloat16)
    summed = q[..., ROT:].reshape(len(q), KV_HEADS, HEADS // KV_HEADS, FREE).sum(2, dtype=torch.float32)
    summed = summed.reshape(len(q), -1).to(torch.bfloat16)
    ik = F.normalize((key @ wk.to(torch.bfloat16)).float(), dim=-1).to(torch.bfloat16)
    iq = F.normalize((summed @ wq.to(torch.bfloat16)).float(), dim=-1)
    return iq, ik


def cluster(ik: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each block's centroids [C, B, n] (bf16 values) and pools [C, B, CAP / RATIO] (-1 padded)."""
    blocks = len(ik) // BLOCK
    pools = pool_keys(ik.float().view(blocks, BLOCK, -1), RATIO)
    sizes = torch.full(pools.shape[:2], float(RATIO))
    labels = ward_merge(pools, sizes, BUCKETS, CAP)
    members = torch.full((blocks, BUCKETS, CAP // RATIO), -1, dtype=torch.int64)
    centroids = torch.zeros(blocks, BUCKETS, INDEX_DIM)
    for c in range(blocks):
        for b in range(BUCKETS):
            held = (labels[c] == b).nonzero().flatten()
            members[c, b, : len(held)] = held
            centroids[c, b] = F.normalize(pools[c, held].sum(0), dim=-1)
    return bf16(centroids), members


def select(iq: torch.Tensor, position: int, centroids: torch.Tensor, members: torch.Tensor) -> list[int]:
    """docs/LSA.md's table for one query, as logical pages ascending."""
    closed = position // BLOCK
    scores = (bf16(iq) @ centroids[:closed].reshape(-1, INDEX_DIM).T) if closed else torch.empty(0)
    picked = torch.argsort(scores, descending=True, stable=True)[:TOP_P].tolist()
    window_page = max(0, position + 1 - WINDOW) // RATIO
    pages = list(range(min(SINK // RATIO, window_page)))
    taken = 0
    for bucket in picked:
        c, b = divmod(bucket, BUCKETS)
        for pool in members[c, b].tolist():
            if pool < 0 or taken >= TOP_K // RATIO:
                break
            page = c * (BLOCK // RATIO) + pool
            if SINK // RATIO <= page < window_page:
                pages.append(page)
                taken += 1
    pages += range(window_page, position // RATIO + 1)
    return sorted(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    maps = torch.load(args.map, map_location="cpu", weights_only=True)
    wq, wk = maps[f"{args.layer}.W_Q_idx"], maps[f"{args.layer}.W_K_idx"]
    generator = torch.Generator().manual_seed(args.seed)
    # Unit-RMS heads, as q_norm and k_norm leave them, in bf16 as the attention holds them
    q = torch.randn(INDEX_ROWS, HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
    k = torch.randn(TOKENS, KV_HEADS, HEAD_DIM, generator=generator).to(torch.bfloat16)
    iq_rows, ik_rows = index(q.float(), k[:INDEX_ROWS].float(), wq, wk)
    _, ik = index(q[:1].float().expand(TOKENS, -1, -1), k.float(), wq, wk)
    # Index queries for the selection: some keys' own directions, perturbed, so the scores separate
    iq_sel = F.normalize(ik[list(QUERIES)].float() + 0.5 * torch.randn(len(QUERIES), INDEX_DIM, generator=generator), dim=-1)
    centroids, members = cluster(ik)
    tables = [select(iq_sel[i], p, centroids, members) for i, p in enumerate(QUERIES)]

    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "wq.bf16": wq.T.contiguous().to(torch.bfloat16),       # [n, K], a projection's [N][K]
        "wk.bf16": wk.T.contiguous().to(torch.bfloat16),
        "q.bf16": q.reshape(INDEX_ROWS, -1),
        "k.bf16": k[:INDEX_ROWS].reshape(INDEX_ROWS, -1),
        "iq_rows.f32": iq_rows.float(),
        "ik_rows.bf16": ik_rows,
        "ik.bf16": ik,
        "iq_sel.f32": iq_sel.float(),
        "centroids.f32": centroids,
        "members.i32": members.to(torch.int32),
        # Each query's logical pages ascending, -1 past its count, TABLE_WIDTH wide
        "tables.i32": torch.tensor([t + [-1] * (TABLE_WIDTH - len(t)) for t in tables], dtype=torch.int32),
    }
    for name, tensor in files.items():
        (OUT / name).write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    meta = {
        "map": args.map.name, "layer": args.layer, "seed": args.seed,
        "queries": list(QUERIES), "tables": tables,
        "window": WINDOW, "sink": SINK, "top_p": TOP_P, "top_k": TOP_K,
    }
    (OUT / "golden.json").write_text(json.dumps(meta, indent=1) + "\n")
    for position, table in zip(QUERIES, tables):
        print(f"position {position}: {len(table)} pages")


if __name__ == "__main__":
    main()
