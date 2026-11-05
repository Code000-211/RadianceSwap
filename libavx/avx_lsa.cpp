/* avx_lsa.cpp -- LSA's indexer, D-LSA (Flat): lsa_index, lsa_key_store, lsa_cluster and lsa_select
 * (docs/OPS.md, "Sparse attention (LSA)"; docs/LSA.md).
 *
 * libref/ref_lsa.cpp is the contract, and this is it line for line in libavx's helpers: the same
 * refusals in the same order, the same sums in the same order, and no contraction (this file is
 * built with -ffp-contract=off, libavx/CMakeLists.txt), so every output - the Ward merges, the
 * tables - is libref's byte for byte. The vector forms (a dot per lane group, a histogram select
 * over the bucket scores as qsa_select has) are the next step; this row exists so a host serves
 * LSA without the reference library.
 */
#include "avx_vec.h"
#include "avx_common.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace avx;

namespace {

int64_t cu_at(const RadTensor* cu, int64_t i) { return ldi(cu, offlin(cu, i)); }

int64_t lsa_seq(const RadTensor* cu, int64_t n_seq, int64_t m) {
    for (int64_t s = 0; s < n_seq; ++s)
        if (m >= cu_at(cu, s) && m < cu_at(cu, s + 1)) return s;
    return -1;
}


float round_bf16(float x) { return rad_bf16_to_f32(rad_f32_to_bf16(x)); }

/* F.normalize: x / max(|x|, 1e-12), the norm an ascending f32 sum of squares. */
void normalize(std::vector<float>& x) {
    float ss = 0.0f;
    for (float v : x) ss += v * v;
    const float d = std::fmax(std::sqrt(ss), 1e-12f);
    for (float& v : x) v = v / d;
}

/* Whether the trailing dimensions after the first are dense, so page `p` is `numel / shape[0]`
 * contiguous values from `p * stride[0]`. */
bool pages_dense(const RadTensor* t) {
    int64_t acc = 1;
    for (int i = (int)t->rank - 1; i >= 1; --i) {
        if (t->shape[i] != 1 && t->stride[i] != acc) return false;
        acc *= t->shape[i];
    }
    return true;
}

int64_t page_values(const RadTensor* t) { return t->shape[0] > 0 ? numel(t) / t->shape[0] : 0; }

/* The descriptor of one cluster block inside its page: `buckets` centroids of `n` values, then
 * `buckets` x `members` pool offsets within the block (-1 past a bucket's size). */
int64_t desc_values(int64_t buckets, int64_t n, int64_t members) { return buckets * (n + members); }

}  /* namespace */

/* ------------------------------------------------------------------ lsa_index */
/* For every row m, from the attention's own q and k after their norms and rope:
 *
 *   key[j]   = k[m][h][rot + i]                         j = h * (head_dim - rot) + i
 *   query[j] = bf16(sum over g < heads / kv_heads of q[m][h * G + g][rot + i])   (f32 sum)
 *   ik[m]    = normalize(bf16(key . wk^T)),   iq[m] = normalize(bf16(query . wq^T))
 *
 * each dot an ascending f32 sum rounded to bf16 as a bf16 GEMM writes it, the norm in f32. The
 * maps are [n, kv_heads * (head_dim - rot)], a projection's [N][K]. */
AVX_KERNEL(lsa_index) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *q = &a->t[0], *k = &a->t[1], *wq = &a->t[2], *wk = &a->t[3];
    const RadTensor *iq = &a->t[4], *ik = &a->t[5];
    const int64_t n = p_int(a, "n", 0), heads = p_int(a, "heads", 0);
    const int64_t kv = p_int(a, "kv_heads", 0), hd = p_int(a, "head_dim", 0);
    const int64_t rot = p_int(a, "rot", 0);
    if (n <= 0 || heads <= 0 || kv <= 0 || hd <= 0 || rot < 0 || rot >= hd || heads % kv)
        return RAD_E_SHAPE;
    const int64_t free = hd - rot, K = kv * free, G = heads / kv;
    if (wq->rank != 2 || wk->rank != 2 || wq->shape[0] != n || wq->shape[1] != K ||
        wk->shape[0] != n || wk->shape[1] != K)
        return RAD_E_SHAPE;
    if (iq->rank != 2 || ik->rank != 2 || iq->shape[1] != n || ik->shape[1] != n) return RAD_E_SHAPE;
    const int64_t M = iq->shape[0];
    if (M <= 0 || ik->shape[0] < M || rows_of(q, heads * hd) < M || rows_of(k, kv * hd) < M)
        return RAD_E_SHAPE;
    const int64_t q_rs = row_stride(q, heads * hd), q_cs = laststride(q);
    const int64_t k_rs = row_stride(k, kv * hd), k_cs = laststride(k);
    const int64_t wq_rs = wq->stride[0], wq_cs = wq->stride[1];
    const int64_t wk_rs = wk->stride[0], wk_cs = wk->stride[1];
    AVX_PARALLEL_FOR_IF(M * n * K >= AVX_PAR_MIN && M > 1)
    for (int64_t m = 0; m < M; ++m) {
        std::vector<float> key((size_t)K), query((size_t)K);
        for (int64_t h = 0; h < kv; ++h)
            for (int64_t i = 0; i < free; ++i) {
                const int64_t j = h * free + i;
                key[(size_t)j] = round_bf16(ldt(k, m * k_rs + (h * hd + rot + i) * k_cs));
                float s = 0.0f;
                for (int64_t g = 0; g < G; ++g)
                    s += ldt(q, m * q_rs + ((h * G + g) * hd + rot + i) * q_cs);
                query[(size_t)j] = round_bf16(s);
            }
        std::vector<float> yk((size_t)n), yq((size_t)n);
        for (int64_t c = 0; c < n; ++c) {
            float dk = 0.0f, dq = 0.0f;
            for (int64_t j = 0; j < K; ++j) {
                dk += key[(size_t)j] * round_bf16(ldt(wk, c * wk_rs + j * wk_cs));
                dq += query[(size_t)j] * round_bf16(ldt(wq, c * wq_rs + j * wq_cs));
            }
            yk[(size_t)c] = round_bf16(dk);
            yq[(size_t)c] = round_bf16(dq);
        }
        normalize(yk);
        normalize(yq);
        for (int64_t c = 0; c < n; ++c) {
            stt(ik, m * ik->stride[0] + c * ik->stride[1], yk[(size_t)c]);
            stt(iq, m * iq->stride[0] + c * iq->stride[1], yq[(size_t)c]);
        }
    }
    return RAD_OK;
}

/* ------------------------------------------------------------------ lsa_key_store */
/* kv[bt[s][pos / block_size]][0][pos % block_size] = ik[m] for every row a sequence holds; a row
 * whose page id is negative writes nothing. A page holds `block_size` index keys of `n` values. */
AVX_KERNEL(lsa_key_store) {
    if (!rad_args_have(a, 5)) return RAD_E_INVAL;
    const RadTensor *ik = &a->t[0], *po = &a->t[1], *cu = &a->t[2], *bt = &a->t[3];
    const RadTensor *kv = &a->t[4];
    const int64_t n = p_int(a, "n", 0), bs = p_int(a, "block_size", 0);
    if (n <= 0 || bs <= 0) return RAD_E_SHAPE;
    if (ik->rank != 2 || ik->shape[1] != n || bt->rank != 2 || kv->rank < 2) return RAD_E_SHAPE;
    const int64_t M = ik->shape[0], n_seq = numel(cu) - 1, pages = kv->shape[0];
    if (M <= 0 || n_seq <= 0 || numel(po) < M || bt->shape[0] < n_seq) return RAD_E_SHAPE;
    if (page_values(kv) != bs * n) return RAD_E_SHAPE;
    if (!pages_dense(kv)) return RAD_E_STRIDE;
    std::vector<std::pair<int64_t, int64_t>> w;   /* (row, value offset) */
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = lsa_seq(cu, n_seq, m);
        if (s < 0) continue;
        const int64_t p = ldi(po, offlin(po, m));
        if (p < 0 || p / bs >= bt->shape[1]) return RAD_E_INVAL;
        const int64_t pg = ldi(bt, s * bt->stride[0] + (p / bs) * bt->stride[1]);
        if (pg < 0) continue;
        if (pg >= pages) return RAD_E_INVAL;
        w.push_back({ m, pg * kv->stride[0] + (p % bs) * n });
    }
    for (const auto& [m, off] : w)
        for (int64_t c = 0; c < n; ++c)
            stt(kv, off + c, ldt(ik, m * ik->stride[0] + c * ik->stride[1]));
    return RAD_OK;
}

/* ------------------------------------------------------------------ lsa_cluster */
namespace {

/* Ward's merge of one block's pools to `buckets` buckets of at most `cap` keys (latent-sparse-
 * attention core/ward.py, `ward_rounds`): each round every alive bucket finds its cheapest allowed
 * partner, the lower index on a tie, where cost(a, b) = |S_a| + |S_b| - sqrt(|S_a|^2 + |S_b|^2 +
 * 2 S_a.S_b); every mutual pair merges into its lower index, the cheapest pairs first when more
 * would merge than the block has left. Returns each pool's bucket, numbered in order of its first
 * pool, or an empty vector when the cap leaves the block short of its count. */
std::vector<int64_t> ward(const std::vector<std::vector<float>>& pool, int64_t pool_keys,
                          int64_t buckets, int64_t cap) {
    const int64_t P = (int64_t)pool.size(), n = P ? (int64_t)pool[0].size() : 0;
    std::vector<std::vector<float>> S = pool;
    std::vector<int64_t> size((size_t)P, pool_keys), parent((size_t)P);
    std::vector<char> alive((size_t)P, 1);
    std::iota(parent.begin(), parent.end(), 0);
    int64_t n_alive = P;
    for (int round = 0; round < 256 && n_alive > buckets; ++round) {
        std::vector<float> sq((size_t)P, 0.0f);
        std::vector<std::vector<float>> gram((size_t)P, std::vector<float>((size_t)P, 0.0f));
        for (int64_t x = 0; x < P; ++x)
            for (int64_t y = 0; y < P; ++y) {
                float d = 0.0f;
                for (int64_t i = 0; i < n; ++i) d += S[(size_t)x][(size_t)i] * S[(size_t)y][(size_t)i];
                gram[(size_t)x][(size_t)y] = d;
            }
        for (int64_t x = 0; x < P; ++x) sq[(size_t)x] = gram[(size_t)x][(size_t)x];
        std::vector<float> best_cost((size_t)P, INFINITY);
        std::vector<int64_t> best((size_t)P, 0);
        for (int64_t x = 0; x < P; ++x) {
            if (!alive[(size_t)x]) continue;
            const float ax = std::sqrt(std::fmax(sq[(size_t)x], 0.0f));
            for (int64_t y = 0; y < P; ++y) {
                if (y == x || !alive[(size_t)y] || size[(size_t)x] + size[(size_t)y] > cap) continue;
                const float merged = std::sqrt(std::fmax(
                    sq[(size_t)x] + sq[(size_t)y] + 2.0f * gram[(size_t)x][(size_t)y], 0.0f));
                const float cost = ax + std::sqrt(std::fmax(sq[(size_t)y], 0.0f)) - merged;
                if (cost < best_cost[(size_t)x]) { best_cost[(size_t)x] = cost; best[(size_t)x] = y; }
            }
        }
        std::vector<int64_t> mutual;
        for (int64_t x = 0; x < P; ++x)
            if (alive[(size_t)x] && best_cost[(size_t)x] < INFINITY &&
                best[(size_t)best[(size_t)x]] == x && x < best[(size_t)x])
                mutual.push_back(x);
        if (mutual.empty()) break;
        std::stable_sort(mutual.begin(), mutual.end(), [&](int64_t u, int64_t v) {
            return best_cost[(size_t)u] < best_cost[(size_t)v];
        });
        const int64_t left = n_alive - buckets;
        if ((int64_t)mutual.size() > left) mutual.resize((size_t)left);
        for (int64_t x : mutual) {
            const int64_t y = best[(size_t)x];
            for (int64_t i = 0; i < n; ++i) S[(size_t)x][(size_t)i] += S[(size_t)y][(size_t)i];
            size[(size_t)x] += size[(size_t)y];
            alive[(size_t)y] = 0;
            parent[(size_t)y] = x;
            --n_alive;
        }
    }
    if (n_alive != buckets) return {};
    std::vector<int64_t> number((size_t)P, -1), label((size_t)P);
    int64_t next = 0;
    for (int64_t x = 0; x < P; ++x)
        if (alive[(size_t)x]) number[(size_t)x] = next++;
    for (int64_t x = 0; x < P; ++x) {
        int64_t r = x;
        while (parent[(size_t)r] != r) r = parent[(size_t)r];
        label[(size_t)x] = number[(size_t)r];
    }
    return label;
}

}  /* namespace */

/* Every cluster block this batch closes: sequence s with positions p0 .. p1 closes blocks
 * p0 / block .. (p1 + 1) / block - 1, at most `work` in all, in sequence order. A block's pool i is
 * the f32 sum of the `ratio` index keys of its key page bt_key[s][c * block / ratio + i]; the pools
 * merge by Ward to `buckets` buckets of at most `cap` keys, and the block's descriptor page
 * bt_desc[s][c] gets each bucket's centroid, normalize(sum of its pools), then its pools in
 * ascending order (offsets within the block, -1 past its size). A closed block whose descriptor
 * page is negative is skipped; one the cap leaves short of `buckets` is an error. */
AVX_KERNEL(lsa_cluster) {
    if (!rad_args_have(a, 6)) return RAD_E_INVAL;
    const RadTensor *po = &a->t[0], *cu = &a->t[1], *key = &a->t[2], *btk = &a->t[3];
    const RadTensor *btd = &a->t[4], *desc = &a->t[5];
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), block = p_int(a, "block", 0);
    const int64_t buckets = p_int(a, "buckets", 0), cap = p_int(a, "cap", 0);
    const int64_t work = p_int(a, "work", 0);
    if (n <= 0 || ratio <= 0 || block <= 0 || block % ratio || buckets <= 0 || cap < ratio ||
        cap % ratio || work <= 0)
        return RAD_E_SHAPE;
    const int64_t P = block / ratio, members = cap / ratio;
    if (buckets > P) return RAD_E_SHAPE;
    if (btk->rank != 2 || btd->rank != 2 || key->rank < 2 || desc->rank < 2) return RAD_E_SHAPE;
    const int64_t n_seq = numel(cu) - 1;
    /* Zero rows is not a call (docs/OPS.md): refused even when no sequence would close a block. */
    if (numel(po) <= 0 || n_seq <= 0 || btk->shape[0] < n_seq || btd->shape[0] < n_seq)
        return RAD_E_SHAPE;
    if (page_values(key) != ratio * n || page_values(desc) < desc_values(buckets, n, members))
        return RAD_E_SHAPE;
    if (!pages_dense(key) || !pages_dense(desc)) return RAD_E_STRIDE;

    struct Item { int64_t s, c, page; };
    std::vector<Item> items;
    for (int64_t s = 0; s < n_seq; ++s) {
        const int64_t t0 = cu_at(cu, s), t1 = cu_at(cu, s + 1);
        if (t1 <= t0) continue;
        if (t0 < 0 || numel(po) < t1) return RAD_E_SHAPE;
        const int64_t p0 = ldi(po, offlin(po, t0)), p1 = ldi(po, offlin(po, t1 - 1));
        if (p0 < 0 || p1 < p0) return RAD_E_INVAL;
        for (int64_t c = p0 / block; c <= (p1 + 1) / block - 1; ++c) {
            if ((int64_t)items.size() >= work) break;
            if (c >= btd->shape[1] || (c + 1) * P > btk->shape[1]) return RAD_E_SHAPE;
            const int64_t dp = ldi(btd, s * btd->stride[0] + c * btd->stride[1]);
            if (dp < 0) continue;
            if (dp >= desc->shape[0]) return RAD_E_INVAL;
            for (int64_t i = 0; i < P; ++i) {
                const int64_t kp = ldi(btk, s * btk->stride[0] + (c * P + i) * btk->stride[1]);
                if (kp < 0 || kp >= key->shape[0]) return RAD_E_INVAL;
            }
            items.push_back({ s, c, dp });
        }
    }
    std::vector<int> failed(items.size(), 0);
    AVX_PARALLEL_FOR_IF(items.size() > 1)
    for (int64_t w = 0; w < (int64_t)items.size(); ++w) {
        const Item& it = items[(size_t)w];
        std::vector<std::vector<float>> pool((size_t)P, std::vector<float>((size_t)n, 0.0f));
        for (int64_t i = 0; i < P; ++i) {
            const int64_t kp = ldi(btk, it.s * btk->stride[0] + (it.c * P + i) * btk->stride[1]);
            for (int64_t r = 0; r < ratio; ++r)
                for (int64_t d = 0; d < n; ++d)
                    pool[(size_t)i][(size_t)d] += ldt(key, kp * key->stride[0] + r * n + d);
        }
        const std::vector<int64_t> label = ward(pool, ratio, buckets, cap);
        if (label.empty()) { failed[(size_t)w] = 1; continue; }
        const int64_t base = it.page * desc->stride[0];
        std::vector<int64_t> filled((size_t)buckets, 0);
        for (int64_t b = 0; b < buckets; ++b)
            for (int64_t j = 0; j < members; ++j) stt(desc, base + buckets * n + b * members + j, -1.0f);
        std::vector<std::vector<float>> sum((size_t)buckets, std::vector<float>((size_t)n, 0.0f));
        for (int64_t i = 0; i < P; ++i) {
            const int64_t b = label[(size_t)i];
            for (int64_t d = 0; d < n; ++d) sum[(size_t)b][(size_t)d] += pool[(size_t)i][(size_t)d];
            stt(desc, base + buckets * n + b * members + filled[(size_t)b]++, (float)i);
        }
        for (int64_t b = 0; b < buckets; ++b) {
            normalize(sum[(size_t)b]);
            for (int64_t d = 0; d < n; ++d) stt(desc, base + b * n + d, sum[(size_t)b][(size_t)d]);
        }
    }
    for (int f : failed)
        if (f) return RAD_E_INVAL;
    return RAD_OK;
}

/* ------------------------------------------------------------------ lsa_select */
/* The page table one query's sparse attention reads (docs/LSA.md, "The table contract"). For row
 * m at position p of sequence s, the clustered blocks are c < p / block. Each of their buckets
 * scores bf16(iq[m]) . centroid in f32; the best min(top_p, buckets) are picked, a tie to the lower
 * bucket (c * buckets + b). Walking them best first, every pool page g = c * block / ratio + i of a
 * picked bucket with sink / ratio <= g < ws / ratio is taken, ws = max(0, p + 1 - window), until
 * top_k / ratio pages are taken. The sink pages below ws / ratio and the window pages ws / ratio ..
 * p / ratio join them, and the union, ascending, is written as page ids bt[s][g] -- so the partial
 * page is last -- then -1 to width + 1. nsel counts the entries and seqused is
 * (nsel - 1) * ratio + p % ratio + 1. A row no sequence holds writes an empty table. */
AVX_KERNEL(lsa_select) {
    if (!rad_args_have(a, 9)) return RAD_E_INVAL;
    const RadTensor *iq = &a->t[0], *desc = &a->t[1], *btd = &a->t[2], *bt = &a->t[3];
    const RadTensor *po = &a->t[4], *cu = &a->t[5], *se = &a->t[6], *ns = &a->t[7], *su = &a->t[8];
    const int64_t n = p_int(a, "n", 0), ratio = p_int(a, "ratio", 0), block = p_int(a, "block", 0);
    const int64_t buckets = p_int(a, "buckets", 0), cap = p_int(a, "cap", 0);
    const int64_t window = p_int(a, "window", 0), sink = p_int(a, "sink", 0);
    const int64_t top_p = p_int(a, "top_p", 0), top_k = p_int(a, "top_k", 0);
    const int64_t width = p_int(a, "width", 0);
    if (n <= 0 || ratio <= 0 || block <= 0 || block % ratio || buckets <= 0 || cap < ratio ||
        cap % ratio || window <= 0 || sink < 0 || sink % ratio || top_p <= 0 || top_k <= 0 ||
        top_k % ratio || width <= 0)
        return RAD_E_SHAPE;
    const int64_t P = block / ratio, members = cap / ratio;
    const int64_t sink_pages = sink / ratio, budget = top_k / ratio;
    if (width < sink_pages + budget + (window + ratio - 1) / ratio + 1) return RAD_E_SHAPE;
    if (iq->rank != 2 || iq->shape[1] != n || bt->rank != 2 || btd->rank != 2 || se->rank != 2)
        return RAD_E_SHAPE;
    const int64_t M = iq->shape[0], n_seq = numel(cu) - 1;
    if (M <= 0 || n_seq <= 0 || bt->shape[0] < n_seq || btd->shape[0] < n_seq) return RAD_E_SHAPE;
    if (se->shape[0] < M || se->shape[1] < width + 1 || numel(ns) < M || numel(su) < M ||
        numel(po) < M)
        return RAD_E_SHAPE;
    if (desc->rank < 2 || page_values(desc) < desc_values(buckets, n, members)) return RAD_E_SHAPE;
    if (!pages_dense(desc)) return RAD_E_STRIDE;
    const int64_t o_rs = se->stride[0], o_cs = se->stride[1];
    AVX_PARALLEL_FOR_IF(M > 1)
    for (int64_t m = 0; m < M; ++m) {
        const int64_t s = lsa_seq(cu, n_seq, m);
        std::vector<int64_t> pages;
        int64_t p = 0;
        if (s >= 0) {
            p = ldi(po, offlin(po, m));
            const int64_t closed = std::min(p / block, btd->shape[1]);
            std::vector<float> q((size_t)n);
            for (int64_t d = 0; d < n; ++d)
                q[(size_t)d] = round_bf16(ldt(iq, m * iq->stride[0] + d * iq->stride[1]));
            /* Every clustered bucket's score, by its global id c * buckets + b. */
            std::vector<float> score;
            std::vector<int64_t> base;
            for (int64_t c = 0; c < closed; ++c) {
                const int64_t dp = ldi(btd, s * btd->stride[0] + c * btd->stride[1]);
                base.push_back(dp < 0 ? -1 : dp * desc->stride[0]);
                for (int64_t b = 0; b < buckets; ++b) {
                    float d = 0.0f;
                    if (dp >= 0)
                        for (int64_t i = 0; i < n; ++i)
                            d += q[(size_t)i] * ldt(desc, dp * desc->stride[0] + b * n + i);
                    score.push_back(dp < 0 ? -INFINITY : d);
                }
            }
            std::vector<int64_t> order(score.size());
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&](int64_t u, int64_t v) { return score[(size_t)u] > score[(size_t)v]; });
            if ((int64_t)order.size() > top_p) order.resize((size_t)top_p);
            const int64_t ws = std::max<int64_t>(0, p + 1 - window), wpage = ws / ratio;
            for (int64_t g = 0; g < std::min(sink_pages, wpage); ++g) pages.push_back(g);
            int64_t taken = 0;
            for (int64_t id : order) {
                if (taken >= budget) break;
                const int64_t c = id / buckets, b = id % buckets;
                if (base[(size_t)c] < 0) continue;
                for (int64_t j = 0; j < members && taken < budget; ++j) {
                    const float off = ldt(desc, base[(size_t)c] + buckets * n + b * members + j);
                    if (off < 0.0f) break;
                    const int64_t g = c * P + (int64_t)off;
                    if (g < sink_pages || g >= wpage) continue;
                    pages.push_back(g);
                    ++taken;
                }
            }
            for (int64_t g = wpage; g <= p / ratio; ++g) pages.push_back(g);
            std::sort(pages.begin(), pages.end());
        }
        int64_t out = 0;
        for (int64_t g : pages) {
            if (g >= bt->shape[1]) break;
            const int64_t id = ldi(bt, s * bt->stride[0] + g * bt->stride[1]);
            if (id < 0) continue;
            st_int_dt(se->dtype, se->data, m * o_rs + (out++) * o_cs, id);
        }
        for (int64_t i = out; i <= width; ++i) st_int_dt(se->dtype, se->data, m * o_rs + i * o_cs, -1);
        st_int_dt(ns->dtype, ns->data, offlin(ns, m), out);
        st_int_dt(su->dtype, su->data, offlin(su, m), out == 0 ? 0 : (out - 1) * ratio + p % ratio + 1);
    }
    return RAD_OK;
}
