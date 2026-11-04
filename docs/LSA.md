# LSA on Qwen4-Exp: a design for serving Latent Sparse Attention in place of QSA

Status: **proposal, nothing implemented.** This document plans how D-LSA (Flat), the distilled indexer of
Latent Sparse Attention ([arXiv 2610.42147](https://arxiv.org/abs/2610.42147),
[latent-sparse-attention](https://github.com/Code000-211/latent-sparse-attention)), replaces QSA's selection on
the twelve global layers of Qwen3.8-Flash-Next. Read [QSA.md](QSA.md) first: LSA keeps QSA's
page-table contract and its attention, and changes only how the table is chosen.

Why: QSA's indexer scores every closed 4-token block, so its cost grows linearly with context at decode
and quadratically at prefill. LSA clusters the history once per 512-token block and scores the cluster
centroids instead. In the paper it gives 3.5x prefill per layer at 1M and 22.7x at 10M on GLM-5.3-Flash over
the model's own sparse attention, and the gain grows with context. The target here is **2M tokens**
(YaRN x9 over the native 262,144).

## 1. What LSA computes (the reference)

The reference is the SGLang plugin of latent-sparse-attention as served on Qwen3.8-27B
(`configs/evaluation/accuracy/qwen38_27b/lsa_flat_tp1.yaml`). File:line references below point into that
repository. Flash-Next's global layer has the same head layout as Qwen3.8-27B's, except for 2 KV heads
where 27B has 4.

### 1.1 The map

A map is two matrices per global layer, trained offline and frozen:

```
W_K, W_Q   [384, 64]   one pair per layer 3, 7, ..., 47      (torch .pt: "<layer>.W_K_idx", "<layer>.W_Q_idx")
```

**384** is the position-free part of the logit: dims 64..255 of each of the 2 KV heads, side by side. RoPE
rotates dims 0..63 only (`partial_rotary_factor` 0.25), so dims 64..255 of the stored key equal the pre-rope
key after `k_norm`. The training-free variant fits `W_K` = `W_Q` to the top 64 principal directions of the
layer's keys. The distilled variant trains them against the model's attention. Both have the same format.

### 1.2 Index keys and queries, per token

```
k_free[t]   = concat_h  k[t, h, 64:256]                                   [384]  bf16  (h = 0, 1)
q_free[t]   = concat_h  sum_{j<12} q[t, 12h + j, 64:256]   (f32 sum)      [384]  -> bf16
idx_k[t]    = bf16( L2norm( f32( k_free @ W_K ) ) )                       [64]   unit, bf16
idx_q[t]    = L2norm( f32( q_free @ W_Q ) )                               [64]   unit, f32
```

The GEMMs take bf16 operands with f32 accumulation and round their output to bf16 before the norm
(`sglang/index/layer_index.py:295-355`). `q` and `k` are the attention's own, after `q_norm`/`k_norm`.

### 1.3 Clustering, once per closed 512-token block

The cluster block is C = `window_size` = 512 tokens. A block is clustered once all its tokens exist, and
never again:

```
pool[i]     = sum_{t in 4i..4i+3} idx_k[t]          f32, not renormalised        128 pools per block
Ward        bucket b: key sum S_b (f32), size n_b (in tokens)
            cost(a, b) = |S_a| + |S_b| - |S_a + S_b|      (spherical Ward)
            allowed:   a != b, both alive, n_a + n_b <= 16       (bucket_cap: at most 4 pools)
            round:     each alive bucket finds its cheapest allowed partner, ties to the lower index;
                       mutual pairs merge into the lower index; if more pairs than alive - 64, only the
                       cheapest (alive - 64) merge
            until      alive == 64                               (bucket_size 8: 64 buckets of 4..16 keys)
centroid[b] = bf16( L2norm( S_b ) )                  [64]
members[b]  = the bucket's pools, ascending
```

`core/ward.py:29-90` is the reference, and `core/kernels/triton/ward.py` the kernel, which computes the
gram matrix in tf32x3 and sums squares in a fixed order. `bucket_cap >= 2 * bucket_size` guarantees the loop
reaches exactly 64. Survivors are numbered in order of their lowest pool.

### 1.4 Selection, per query at position p

```
visible buckets   vis(p) = floor(p / 512) * 64                 (only blocks closed before p's own)
score[i]          = bf16(idx_q) . centroid[i], f32 accumulate; ties to the lower bucket id
top_p             = the best min(1024, vis) buckets
expansion         best bucket first, take each member key m with 4 <= m < ws = max(0, p + 1 - 512),
                  until 4096 keys are taken; the last bucket is cut, lowest positions first
attended set      sinks [0, 4) + expanded keys + window [ws, p]          width 4 + 4096 + 512 = 4612
```

The selection is exact: the reference's threshold estimate and scan only make it faster
(`core/kernels/scan_select.py`), and the result is the exact top-1024. The paper's numbers are measured at
these constants (`settings/design.py`: window 512, sink 4, top_k 4096, top_p 1024, index_dim 64, pool 4,
bucket_size 8, bucket_cap 16).

**Exactness bound.** For p <= 4611, every long-range key in [4, p - 511) fits the 4096 budget, and top_p
only binds from p >= 8704. So below 4612 tokens of reach the selection is every key, and LSA equals dense
causal attention.

## 2. How it maps onto Radiance

### 2.1 Pages, not keys

QSA pages the attention KV group at 4, and `attn_paged` reads a table of page ids
(`r4d_attn_paged_h256_gqa12.hip`, `C_EQ("block_size", 4)`). LSA's pools are 4 consecutive tokens, the same
4 tokens as a page, and a bucket is a set of whole pools. **A bucket's members are attention pages**, so
LSA's selection is a page table and the attention, `kv_resolve`, streaming and lending are QSA's, unchanged.

Three of the reference's boundaries are per key, and Radiance's are per page:

| boundary | reference (keys) | here (pages) | difference |
|---|---|---|---|
| sinks | keys 0..3 | page 0 | none |
| window | keys [p - 511, p] | pages floor((p - 511) / 4) .. floor(p / 4) | up to 3 more keys, before the window |
| budget | 4096 keys, the last bucket cut | 1024 pages, the last bucket cut by pages | none, since 4096 = 1024 x 4 |
| page straddling `ws` | its keys < ws are bucket members, the rest window | the whole page is window | the same keys, counted toward the window rather than the budget |

The attended set is therefore the reference's set plus at most 3 keys a query. **This is a deviation from
the trained recipe and must be measured** (§5). A per-key table would need a page-1 attention kernel, which
the QSA attention units do not build.

### 2.2 The table contract

Width `W` = 1 (sink) + 1024 (budget) + 129 (window) = 1154 page entries, plus QSA's spare column, so
`qsa_topk = 1154`. The attention sums its softmax over pages in table order (QSA.md, "Cost"), so:

- **ascending page index**, deduplicated: sink page, bucket pages and window pages are disjoint by
  construction, because a bucket page at or past the window's first page is skipped;
- **the partial page last**, present iff `(p + 1) % 4 != 0`, and `seqused = (nsel - 1) * 4 + p % 4 + 1`, as
  QSA's;
- below the exactness bound the table is every visible page, ascending, so the dense path
  (`qsa_exact_to`) and the sparse one must produce the same bytes. **That is the test** (§5).

At page granularity the bound moves slightly. The window's first page is `wp = floor((p - 511) / 4)`, and the
long-range candidates are pages `1 .. wp - 1`. The budget of 1024 pages binds from `wp >= 1026`, that is from
p >= 4615, and top_p binds only from p >= 8704. So every visible page is selected for a reach of up to 4615
tokens, and `qsa_exact_to = 4615`.

Expansion walks buckets best-first, but the output must be ascending. `lsa_select` therefore collects at most
1154 logical page indices per row and sorts them in LDS (a 2048-wide bitonic sort) before mapping them
through the block table.

### 2.3 Where LSA reads q and k

QSA projects the normed hidden `aw.h` with its own `index_qk_proj`, so it runs before `attn.step()`. LSA reads
the **attention's own q and k after `q_norm`/`k_norm` and rope**, which exist only inside
`AttnGatedFP8::step` (`arch/common/rad_block_attn_gated_fp8.h`: prologue, `op_fused*` or
`op_q_norm`/`op_rope_*`, then `store_cache`, then the sparse issue). `step()` is split into
`step_prologue()` (projections through the KV store) and `step_read()` (the dense or sparse issue). LSA's
ops run between the two. QSA keeps calling the two in a row, so its graph is unchanged.

The keys are taken from `w.k` before the store, not out of the KV cache: with `--kv-dtype fp8_v4` or
`k6_v4`, the cache holds quantised keys the map was not trained on. The trunk projections themselves are int8
or fp8 under the served recipe (`qwen4exp-w4nl64-i8-hc8m.recipe`), so q and k already differ from the bf16
checkpoint by the recipe's error. That is the same error QSA serves under, and it is part of what §5 measures.

### 2.4 KV groups

| group | kind | geometry | holds | bytes / token / layer |
|---|---|---|---|---|
| `kv_attn` | FULL, block 4 | unchanged | K, V | unchanged |
| `kv_lsa_key` | FULL, block 4, dense-read | `n_head_kv` 1, `head_dim` 32 | `idx_k`, one 64-wide bf16 row a token | 128 |
| `kv_lsa_desc` | FULL, block 512, dense-read | `n_head_kv` 1, `head_dim` 5 (10 KiB page) | one block's descriptor (below) | 20 |

`kv_lsa_key` is written a token at a time, like K and V. A pool's 4 keys are in one page, and the sum happens
at clustering. So, unlike QSA's block keys, **LSA needs no tail group**: a decode step writes its token's
row, and a page completes when its fourth token arrives.

A block's descriptor is written once, by the step that completes the block, into the block's own 512-token
page of `kv_lsa_desc`:

```
centroids   [64, 64]  bf16     8192 B
members     [64, 4]   i16      512 B     pool offset within the block, -1 padded, ascending
```

It lives in the page of the token that completes the block, so a prefix-cache hit shares it only with
sequences whose 512 tokens are identical (`core/mem/prefix.h`). Partially written descriptor pages are never
shared. The paged group's 512 enters the chunk quantum (`core/sched/geometry.cpp`, the lcm of block sizes), so
`--max-num-batched-tokens` must be at least 512, and `--checkpoint-interval` becomes a multiple of 512. The
default of 2048 already is. Hits on this hybrid model are clamped to linear-state checkpoints anyway.

At 2M tokens and 12 layers the two groups add (128 + 20) x 12 x 2,097,152 = **3.5 GiB** a sequence, against
48 GiB of attention KV at bf16 and 1.5 GiB of QSA block keys. With LSA serving every global layer, QSA's
block-key and tail groups are needed only for the MTP head (§2.7).

Both groups are FULL and not streamed, so they lend, tier and checkpoint like any other
(`KVManager::lendable`, `kvtier.h`). Deactivation in manifest mode drops them with the engine, as it drops
the KV.

### 2.5 Ops

All five ops are new, with a libref row each (CPU, unconstrained), a schema in `docs/OPS.md`, an entry in
`tests/ref_test.cpp`'s vocabulary, and later libr4d and libavx rows.

| op | rows | reads | writes | notes |
|---|---|---|---|---|
| `lsa_index` | T | `w.q [T, 24*256]`, `w.k [T, 2*256]`, `W_Q`, `W_K` | `idx_q [T, 64]` f32, `idx_k [T, 64]` bf16 | §1.2. Slices dims 64..255, sums query heads per group in f32, bf16 GEMM with f32 accumulation, L2 norm. Parameters `heads`, `kv_heads`, `head_dim`, `rot_dim`, `n` |
| `lsa_key_store` | T | `idx_k`, positions, `kv_lsa_key` table | `kv_lsa_key` | the KV store's addressing at row width 64. Could fold into `lsa_index` |
| `lsa_work` | ceil(T / 512) + n_seq | `cu_seqlens`, positions, `kv_lsa_desc` and `kv_lsa_key` tables | work list (seq, block, desc page) | the blocks this step completes, fixed width, `page = -1` past the count; recorded decode passes replay it unchanged. Capacity parameter `work` is `P_CAP` |
| `lsa_cluster` | work items | work list, `kv_lsa_key` | `kv_lsa_desc` | §1.3. One workgroup a block: 128 pools in LDS, a 128 x 128 f32 gram, at most 128 rounds, deterministic ties. Stateful, so listed in `rad_kernel_concurrent`'s `kStateful` |
| `lsa_select` | T | `idx_q`, `kv_lsa_desc` table and pages, attention table, positions, `cu` | `sel [T, W+1]`, `nsel`, `seqused` | §1.4 and §2.2: score `vis(p)` centroids, exact top-1024 with QSA's tie rule, expansion by pages to 1024, sink and window pages, ascending sort, partial page last |

`lsa_select` scores in one pass where QSA has two ops (`qsa_score` plus a plane). At 2M a row's scores are
262,144 f32 (1 MiB), and the reference's threshold-and-scan selection avoids writing them out. libr4d's first
form may keep a plane, `[rows, max_ctx / 8]` f32. That is half of QSA's at the same context and is bounded
the same way (QSA.md, "The score plane"). The fused scan is the second form, and the two must produce the
same bytes, like `qsa_select`'s split and unsplit forms.

Issue order in a global layer:
`attn.step_prologue` → `lsa_index` → `lsa_key_store` → `lsa_work` → `lsa_cluster` → `lsa_select` → `attn.step_read`.
`lsa_cluster` runs before `lsa_select`. A prefill chunk's queries never score the blocks the chunk itself
closes past their own position, because `vis(p)` bounds each row. They do score blocks closed earlier in the
same chunk, as the reference does.

### 2.6 Loading the map

The maps travel inside the container. Radiance serves no stray side files out of a checkpoint directory
(`core/format/checkpoint.cpp`, `shard_list`), and a merged weight gets placement, TP replication, the
warm store and the fingerprint for free:

```
lsa export-radiance maps.pt lsa_maps.safetensors      # latent-sparse-attention, to be written
rad-convert <hf_ckpt> --reuse flash.rad --in-place --add lsa_maps.safetensors \
    --set qwen4exp.lsa_map=<name of the map>
```

Tensor names are `model.language_model.layers.<L>.self_attn.lsa.{k,q}_map.weight`, each `[64, 384]` (the
transpose of `W`, as `gemm_nt` reads `[N, K]`), bf16, declared `optional`, `RAD_SHARD_NONE`, `grp_layer(L)`.
The exporter checks the layer set against the checkpoint's global layers and refuses a map whose
`input_dim` is not 384.

### 2.7 Switching it on

An engine flag, `--sparse-indexer qsa|lsa`, defaulting to `qsa`. It sets a metadata key
`qwen4exp.sparse_indexer` the way `--rope-scaling` rewrites `rope_parameters` (`Engine::apply_rope_scaling`,
`core/engine_bringup.cpp`). That needs no ABI change, and manifest models get it through their `engine`
argv list. `lsa` with no map in the container is refused at declare with the missing tensor named. The flag
does not fall back to QSA silently.

The **MTP head keeps QSA**: no LSA map is trained for it, since distillation covers layers 3..47. Its
`QsaIndexer` and `kv_qsa_bkey`/`kv_qsa_tail` stay, and the shared `sel` buffer is declared at
max(1155, 513) columns.

YaRN: the map is fitted under the rotation it is served with. Serve with `--rope-scaling yarn:9`, the factor
the map was fitted under (`qwen38_flash_next` model card). [YARN.md](YARN.md) validates Flash-Next at factor 2
only. Factor 9 is part of §5.

## 3. What does not change

- `attn_paged` and its rows, `kv_resolve`, KV streaming, lending, tiering, prefix caching. The table is a
  QSA-shaped table.
- QSA itself: `--sparse-indexer qsa` is today's graph, byte for byte. `arch_test` pins its order.
- The trunk, the MoE, PLE and the gated residual.

## 4. Plan

| phase | contents | where it is checked |
|---|---|---|
| 0 | `lsa export-radiance` in latent-sparse-attention: `.pt` to safetensors, layer and shape checks. A Python oracle that writes golden `idx_k`, descriptors and selections for random keys from `core/reference.py`, at page granularity (§2.1) | Python tests |
| 1 | libref: the five ops, schemas, shapes, `OPS.md` rows, `ref_test` literals and goldens from phase 0; `AttnGatedFP8` split into prologue and read; `rad_lsa.h`; qwen4exp wiring behind `--sparse-indexer`; `rad-convert --add` path | CPU build: `ref_test`, `arch_test` (graph order, dense-below-bound), `ops_doc`. Runs here without ROCm |
| 2 | libr4d: `lsa_index` (a GEMV-shaped GEMM plus norm), `lsa_key_store`, `lsa_work`, `lsa_cluster` (one workgroup a block), `lsa_select` (plane form first) | `r4d_emu` on CPU against libref, then `r4d_selftest` oracles on an R9700, including split against unsplit byte identity on mostly tied scores |
| 3 | libavx rows, so a host build serves without `--debug-accept-reference-kernels` | `avx_check` |
| 4 | the fused threshold-and-scan `lsa_select`; `--perflsa` in `r4d_selftest` | timings against `--perfqsa` at 128k, 512k, 1M, 2M |

## 5. Acceptance

1. **Below the bound** (reach <= 4615, §2.2): `--sparse-indexer lsa` produces the same text as dense attention,
   byte for byte. This is QSA's own test (QSA.md, "Exactness bound").
2. **Selection against the reference:** for random keys and for keys dumped from a served layer, the page set
   equals the reference's key set rounded to pages (§2.1), on every row.
3. **Accuracy at 128k, 512k, 1M and 2M** (RULER, PaperTrail), against QSA on the same container and rotation:
   the training-free map first, the distilled one if the training-free map falls short.
4. **Speed:** a layer's indexer time at the same contexts, against `qsa_score` + `qsa_select`. Decode tokens per
   second and time to first token at 512k, 1M and 2M.
5. **YaRN x9:** QSA's own retrieval at 1M and 2M under `--rope-scaling yarn:9`, before LSA is judged against it.

## 6. Open questions

- **Cluster block 512 against 8192.** The reference clusters per 512-token block. A larger block gives better
  clusters and fewer descriptor pages, but it puts 8192 into the chunk quantum. A LINEAR-state descriptor
  ring would avoid that (`{ceil(max_ctx / C), desc}` per sequence, snapshotted with the linear state). The map
  is trained at 512, so 512 is the plan.
- **Prefix hits.** A hit resumes at a linear-state checkpoint, so descriptors of blocks before it come from
  the cache, and those after it are rebuilt by the prefill. Nothing in LSA changes this. It needs a test that
  resumes inside a block.
- **Speculative decoding.** A rejected token that completed a block wrote a descriptor that the re-accepted
  token rewrites. Queries are bounded by their own position, so a stale descriptor is never read past a
  rejection. `arch_test` should pin this with a spec step that straddles a 512 boundary.
- **The recipe's int8 q/k.** If served accuracy falls short, re-fit the PCA map on q/k dumped from Radiance
  under the served recipe, not from bf16 transformers.
