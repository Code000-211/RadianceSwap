/* rad_lsa.h -- Latent Sparse Attention's indexer, D-LSA (Flat), for a gated full-attention layer.
 *
 * docs/LSA.md is the design and docs/OPS.md ("Sparse attention (LSA)") the four ops. In short: a
 * learned map takes each token's position-free key lanes to a 64-wide unit index key, a closed
 * cluster block of 512 tokens is merged by Ward into 64 buckets whose centroids are cached in one
 * descriptor page, and each query scores the centroids of the blocks before its own, expands the
 * best buckets to their 4-token pages and adds the sink and window pages. The result is a page
 * table in QSA's contract (docs/QSA.md, "The table contract"), so the attention is AttnGatedFP8's
 * sparse path unchanged: this header fills `qsa_sel`, `qsa_nsel` and `qsa_sequ`.
 *
 * Unlike QSA's indexer, LSA reads the attention's own q and k after their norms and rope, so it
 * runs inside the attention block's step, between the KV store and the attention, through the
 * block's `pre_read` hook (set by declare()). Both gated blocks carry one: AttnGatedFP8 and
 * AttnGatedBF16.
 *
 * The map is two [n, kv_heads * (head_dim - rot)] bf16 weights a layer, merged into the container
 * (tools/lsa/export_map.py, then rad-convert --add) as
 *
 *     model.language_model.layers.<L>.self_attn.lsa.q_map.weight
 *     model.language_model.layers.<L>.self_attn.lsa.k_map.weight
 *
 * Two KV groups hold its state, both full, paged and dense-read:
 *
 *     kv_lsa_key    block = the attention's page (4)   one index key a token, n bf16
 *     kv_lsa_desc   block = cluster block (512)       one descriptor a closed block
 *
 * Each is declared immediately before the op whose `block_size` sizes it (Builder::find_kv_consumer),
 * once, on the first LSA layer, and bound to every LSA layer.
 */
#ifndef RAD_LSA_H
#define RAD_LSA_H

#include "rad_arch.h"

namespace rad {
namespace arch {

struct LsaIndexer {
    struct Config {
        const char* q_map = nullptr;    /* checkpoint names of the layer's two maps */
        const char* k_map = nullptr;
        int64_t n       = 64;           /* index dims */
        int64_t rot     = 0;            /* the rotated prefix of a head, left out of the index */
        int64_t ratio   = 4;            /* keys a pool, which is the attention's page */
        int64_t block   = 512;          /* keys a cluster block */
        int64_t buckets = 64;           /* buckets a cluster block */
        int64_t cap     = 16;           /* most keys a bucket */
        int64_t window  = 512;
        int64_t sink    = 4;
        int64_t top_p   = 1024;         /* buckets a query picks */
        int64_t top_k   = 4096;         /* long-range keys a query reads */
        rad_kvgroup kv_attn = 0;        /* the attention's group, whose table `sel` indexes */
    };

    /* Within-step scratch shared by every LSA layer, declared by the plugin. */
    struct Wire {
        rad_buf iq   = 0;   /* [max_tok, n]         f32 index queries */
        rad_buf ik   = 0;   /* [max_tok, n]         bf16 index keys */
        rad_buf sel  = 0;   /* [max_tok, width + 1] i32, the attention's table, one row a query */
        rad_buf nsel = 0;   /* [max_tok]            i32 */
        rad_buf sequ = 0;   /* [max_tok]            i32 */
    };

    /* The table's width, sink pages + budget pages + window pages, and the spare column. */
    static int64_t width(const Config& c) {
        return c.sink / c.ratio + c.top_k / c.ratio + (c.window + c.ratio - 1) / c.ratio + 1;
    }
    /* Below this reach every page is selected and the table is the sequence's own, so the dense
     * attention reads the same keys (docs/LSA.md, "The table contract"). */
    static int64_t exact_to(const Config& c) { return c.top_k + c.window + c.sink; }

    Geom        g{};
    Config      c{};
    Wire        w{};
    int         layer = 0;
    rad_buf     b_q = 0, b_k = 0;      /* the block's normed, rotated q and k */
    rad_weight  w_q = 0, w_k = 0;
    rad_kvgroup kv_key = 0, kv_desc = 0;
    rad_op      op_index = 0, op_store = 0, op_cluster = 0, op_select = 0;

    bool on() const { return op_select != 0; }

    /* Declares the layer's indexer after its attention block and wires the block to it: the hook,
     * the selection buffers, the table's width and the exactness bound. `key` and `desc` are the
     * model's two groups, created on the first call. */
    template <class Block>
    int declare(RadBuilder* b, Names& nm, const Geom& geom, int l, const Config& cfg,
                const Wire& wire, Block& block, rad_kvgroup* key, rad_kvgroup* desc);
    void step(RadCtx* ctx, const RadBatch* batch) const;

    static void hook(const void* self, RadCtx* ctx, const RadBatch* batch) {
        static_cast<const LsaIndexer*>(self)->step(ctx, batch);
    }
};

template <class Block>
inline int LsaIndexer::declare(RadBuilder* b, Names& nm, const Geom& geom, int l,
                               const Config& cfg, const Wire& wire, Block& block,
                               rad_kvgroup* key, rad_kvgroup* desc) {
    g = geom; c = cfg; w = wire; layer = l; b_q = block.w.q; b_k = block.w.k;
    if (!c.q_map || !c.k_map || !c.kv_attn || !w.iq || !w.ik || !w.sel || !w.nsel || !w.sequ)
        return RAD_E_INVAL;
    /* The index reads every KV head's lanes, which one rank holds only without tensor parallelism:
     * under it each rank's partial projection would need an all-reduce (docs/LSA.md, open work). */
    if (g.world > 1 || g.n_head_kv != g.n_head_kv_all) {
        rad_note(b, "lsa: the indexer needs every KV head on one rank, so tensor parallelism is "
                    "refused");
        return RAD_E_UNSUPPORTED;
    }
    if (c.n <= 0 || (c.n % 2) || c.rot < 0 || c.rot >= g.head_dim || c.ratio <= 0 ||
        c.block % c.ratio || c.buckets <= 0 || c.cap % c.ratio || c.cap < 2 * c.block / c.buckets ||
        c.sink % c.ratio || c.top_k % c.ratio || c.top_p <= 0 || c.window <= 0)
        return RAD_E_INVAL;
    const int64_t K = g.n_head_kv * (g.head_dim - c.rot);
    const int64_t W = width(c);

    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.lsa_q_map.weight", l), c.q_map));
    RAD_ARCH_TRY(map_copy(b, nm.f("blk.%d.lsa_k_map.weight", l), c.k_map));
    w_q = decl_w(b, nm.f("blk.%d.lsa_q_map.weight", l), RAD_BF16, { c.n, K },
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    w_k = decl_w(b, nm.f("blk.%d.lsa_k_map.weight", l), RAD_BF16, { c.n, K },
                 RAD_ACCESS_PER_TOKEN, RAD_SHARD_NONE, grp_layer(l));
    if (!w_q || !w_k) return RAD_E_INVAL;

    op_index = rw(b, RAD_OP(b, "lsa_index",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.n),
                                RAD_INT("heads", g.n_head), RAD_INT("kv_heads", g.n_head_kv),
                                RAD_INT("head_dim", g.head_dim), RAD_INT("rot", c.rot),
                                RAD_STR("dtype", g.dtype)),
                     RAD_WEIGHTS(w_q, w_k)),
                 {b_q, b_k}, {w.iq, w.ik});

    /* The index-key group: one KV head of n / 2 lanes, so a page of `ratio` tokens holds `ratio`
     * keys of n values; its page is the attention's, so a block index means the same in both. */
    if (*key == 0) {
        RadKVGroupDecl d{};
        d.kind      = RAD_KV_FULL;
        d.dtype     = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim  = c.n / 2;
        *key = rad_decl_kv_group(b, nm.f("kv_lsa_key"), &d);
        if (!*key) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_kv_group_dense_read(b, *key));
    }
    kv_key = *key;
    op_store = rw(b, RAD_OP(b, "lsa_key_store",
                     RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.n),
                                RAD_INT("block_size", c.ratio), RAD_INT("seqs", g.max_seqs),
                                RAD_STR("dtype", "bf16")),
                     RAD_NOWEIGHTS),
                 {w.ik}, {});
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv_key));

    /* The descriptor group: a page of `block` tokens holds `buckets` centroids of n values and
     * their pools, buckets * (n + cap / ratio) bf16 values (the pool offsets are small integers,
     * exact in bf16). */
    if (*desc == 0) {
        const int64_t values = c.buckets * (c.n + c.cap / c.ratio);
        RadKVGroupDecl d{};
        d.kind      = RAD_KV_FULL;
        d.dtype     = RAD_BF16;
        d.n_head_kv = 1;
        d.head_dim  = (values + 2 * c.block - 1) / (2 * c.block);
        *desc = rad_decl_kv_group(b, nm.f("kv_lsa_desc"), &d);
        if (!*desc) return RAD_E_INVAL;
        RAD_ARCH_TRY(rad_kv_group_dense_read(b, *desc));
    }
    kv_desc = *desc;
    /* A batch closes at most one block a sequence, plus one per `block` tokens of the step. */
    const int64_t work = g.max_tok / c.block + g.max_seqs;
    op_cluster = rw(b, RAD_OP(b, "lsa_cluster",
                       RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.n),
                                  RAD_INT("ratio", c.ratio), RAD_INT("block", c.block),
                                  RAD_INT("buckets", c.buckets), RAD_INT("cap", c.cap),
                                  RAD_INT("seqs", g.max_seqs), RAD_INT("work", work),
                                  RAD_INT("block_size", c.block), RAD_STR("dtype", "bf16")),
                       RAD_NOWEIGHTS),
                   {}, {});
    RAD_ARCH_TRY(rad_bind_layer_kv(b, l, kv_desc));

    const int64_t pages = g.max_ctx / c.ratio + 2, cblocks = g.max_ctx / c.block + 2;
    op_select = rw(b, RAD_OP(b, "lsa_select",
                      RAD_PARAMS(RAD_RANGE("M", 1, g.max_tok), RAD_INT("n", c.n),
                                 RAD_INT("ratio", c.ratio), RAD_INT("block", c.block),
                                 RAD_INT("buckets", c.buckets), RAD_INT("cap", c.cap),
                                 RAD_INT("window", c.window), RAD_INT("sink", c.sink),
                                 RAD_INT("top_p", c.top_p), RAD_INT("top_k", c.top_k),
                                 RAD_INT("width", W), RAD_INT("blocks", pages),
                                 RAD_INT("cblocks", cblocks), RAD_INT("seqs", g.max_seqs),
                                 RAD_STR("dtype", "bf16")),
                      RAD_NOWEIGHTS),
                  {w.iq}, {w.sel, w.nsel, w.sequ});

    block.pre_read      = &LsaIndexer::hook;
    block.pre_read_self = this;
    return RAD_OK;
}

inline void LsaIndexer::step(RadCtx* ctx, const RadBatch* batch) const {
    if (!on()) return;
    const int64_t T = batch->n_tok, S = batch->n_seq;
    const RadKVGroupBatch* kb = kv_batch(batch, kv_key);
    const RadKVGroupBatch* db = kv_batch(batch, kv_desc);
    const RadKVGroupBatch* ab = kv_batch(batch, c.kv_attn);
    if (!kb || !db || !ab || !kb->block_table || !db->block_table || !ab->block_table) {
        rad_step_fail(ctx, "lsa: the step has no block table for the indexer's groups");
        return;
    }
    const int32_t* pos = batch->positions;
    const int32_t* cu  = batch->cu_seqlens;
    RAD_ISSUE_N(ctx, op_index, T,
                brows(b_q, T), brows(b_k, T), RAD_W(w_q), RAD_W(w_k),
                brows(w.iq, T), brows(w.ik, T));
    RAD_ISSUE_N(ctx, op_store, T,
                brows(w.ik, T), praw(pos, RAD_I32, T), praw(cu, RAD_I32, S + 1),
                praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch),
                kv_cache(kv_key, layer));
    /* The blocks this step closes are clustered before any query of the step selects: a query
     * reads only blocks closed before its own, some of them closed earlier in this chunk. */
    RAD_ISSUE_N(ctx, op_cluster, T,
                praw(pos, RAD_I32, T), praw(cu, RAD_I32, S + 1), kv_cache(kv_key, layer),
                praw2(kb->block_table, RAD_I32, S, kb->block_table_pitch),
                praw2(db->block_table, RAD_I32, S, db->block_table_pitch),
                kv_cache(kv_desc, layer));
    RAD_ISSUE_N(ctx, op_select, T,
                brows(w.iq, T), kv_cache(kv_desc, layer),
                praw2(db->block_table, RAD_I32, S, db->block_table_pitch),
                praw2(ab->block_table, RAD_I32, S, ab->block_table_pitch),
                praw(pos, RAD_I32, T), praw(cu, RAD_I32, S + 1),
                brows(w.sel, T), brows(w.nsel, T), brows(w.sequ, T));
}

}  /* namespace arch */
}  /* namespace rad */

#endif /* RAD_LSA_H */
