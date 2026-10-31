/* group_planner.h -- the group planner: how the expert VRAM of one GPU group is divided between its
 * active models (spec #1, ticket #14; GLOSSARY.md: Gruppen-Heat).
 *
 * Today every Engine plans its expert slots alone. Engine::plan resolves one weights budget
 * (mem/vram_budget.h), Planner::assign_expert_tiered fills as many slab slots as that budget holds
 * in rank order, and heat.h / mover.h re-rank within those slots at run time. With several
 * engines on one card each one resolves its budget inside a fixed share of the free VRAM
 * (--gpu-headroom-mib, core/main.cpp), so its experts get what the share left.
 *
 * This file is the decision above that and nothing else. It does not place anything: it hands each
 * engine a byte budget (and the slot count per slab class that budget buys), and the engine's own
 * Planner and Mover place within it exactly as before. There is no second placement algorithm.
 *
 * ---- the decision -------------------------------------------------------------------------------
 *
 * Every model has a fixed share of the card's VRAM (core/main.cpp), which caps its KV cache and
 * static weights; the expert bytes that share left it are its offer. The planner moves expert
 * bytes between models, but never grants a model more than its offer, so no model passes its
 * share and a model started early cannot take the room of a later one.
 *
 * Per rank, from the group's expert pool P (the offers of the joined models):
 *   1. Every active model keeps a minimum: max(its expert floor, `min_share` of the equal share
 *      P / n), capped at its whole expert plane. With `to_start` > 0, a model not yet
 *      started reserves the same fraction of the equal share.
 *   2. What is left goes to the hottest units of the group, each model up to its offer. A unit's
 *      worth is the heat the model
 *      reports for it times its bytes (the traffic a resident slot saves), per byte of slot it
 *      costs: heat * bytes / slot_bytes. That is comparable across models whose experts differ in
 *      size. Equal worth shares the room in proportion to bytes (all-zero heat, as at a cold
 *      start, splits by expert-plane size).
 * Ranks are decided independently from their own shards; the engine takes the smallest budget
 * (one split serves every rank, vram_budget.h).
 *
 * A group of one engine is the existing behaviour: it gets back the budget it offered, unchanged.
 *
 * ---- where it acts ------------------------------------------------------------------------------
 *
 * join() runs in Engine::plan, right after the single-engine resolve: the engine offers what it
 * resolved and starts with min(offer, target). Targets of engines already running are recomputed
 * on every join, leave and update, and exposed by target().
 *
 * A running engine acts on its target at a step boundary (ticket #15, Engine::group_tick): it
 * reports its heat with update(), reads target(), and states the cap per slab class to its mover
 * (group_resident_cap, Mover::set_resident_cap), which gives slots back or takes them again.
 * What a shrink frees is logical: the engine's slab stays one reservation in its own weights
 * pool, so the bytes are not another engine's to use (Mover::set_resident_cap).
 *
 * ---- elastic slabs (ticket #35, docs/swap/elastic-slab.md) -------------------------------------
 *
 * A model whose slab is elastic (RankFacts::elastic) is not held to its offer. The group's expert
 * pool is the sum of the members' offers, and the heat-based allocation above may give a hot model
 * more than its offer, a cold one less (never below its minimum). The target is what the model is
 * entitled to; what it may take now is group_gate's: a growth is paid out of the bytes no member
 * holds, so it waits for the giver's pages to go back to the card. The model reports what it holds
 * (update_held) and the planner recomputes; the first tick after a physical release raises the
 * taker's target, and its mover commits the pages (Mover::set_resident_cap).
 *
 * ---- KV partitions (ticket #17, docs/swap/kv-lend.md) -------------------------------------------
 *
 * A model's KV partition is its pool's carve. When a partition needs more than it has, it claims
 * the difference from the group's expert pool, and the existing allocation above decides which
 * models' experts go: the pool the experts share shrinks by the grant, so the coldest units of any
 * model (the claimant's own included) give way and every model keeps its minimum. The planner only
 * ever adds to a partition: a partition's held bytes (running sequences, prefix cache) are the
 * floor of its want, and no output names a byte of any partition to give up.
 *
 * A partition claims only what its pool can physically back (KvPartition::growable). No pool can
 * today (the slab sits in the weights pool, docs/swap/slot-release.md), so every engine reports 0
 * and a claim is never granted: evicting experts for bytes the claimant cannot use would only cost
 * decode time. The expert slab of ticket #35 raises `growable`.
 */
#pragma once
#include "heat.h"
#include "planner.h"

#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace rad {

/* The expert units of one slab class whose heat fell in one bucket. `heat` is the bucket's mean
 * per-unit heat (activations, decayed), `bytes` the weight bytes of its units and `slot_bytes`
 * what they cost in the slab. Units of a bin are interchangeable. */
struct HeatBin {
    float   heat = 0.0f;
    int64_t units = 0;
    int64_t bytes = 0;
    int64_t slot_bytes = 0;
    int32_t slab_class = 0;
};

/* One rank's KV partition of a model, in bytes of that rank's pool. All zero: no claim. */
struct KvPartition {
    int64_t carve = 0;      /* the pool's carve (KVManager::pool_bytes_carved, summed over groups) */
    int64_t held = 0;       /* held by sequences or the prefix cache (pool_bytes_used): never given up */
    int64_t want = 0;       /* what the cache wants in hand: held plus its buffer or a refused request */
    int64_t cap = 0;        /* the most it can ever use (KVManager::ceiling) */
    int64_t growable = 0;   /* bytes the pool can back beyond `carve` today; 0 until #35 */
};

/* One rank's side of an engine's report. */
struct RankFacts {
    int64_t offer = 0;   /* expert bytes the single-engine resolver gave it: VramBudget::experts */
    /* The model's slab is elastic (ticket #35, docs/swap/elastic-slab.md): it can hold more than
     * its offer when another model has physically given bytes up, and it gives bytes back to the
     * card on a shrink. Such a model may be allocated up to the group's whole expert pool, but is
     * granted growth only out of bytes that no member holds (group_gate). Otherwise the offer is
     * its ceiling, as before. */
    bool    elastic = false;
    int64_t held = 0;    /* expert bytes its slab has pages behind now (Mover::slab_held_bytes) */
    int64_t floor = 0;   /* VramFacts::expert_floor: below it the planner refuses experts */
    std::vector<HeatBin> bins;   /* its expert plane; the ceiling is their slot bytes */
    KvPartition kv;
};
using GroupFacts = std::vector<RankFacts>;   /* by rank */

struct RankBudget {
    int64_t bytes = 0;
    std::map<int32_t, int64_t> class_units;   /* slots per slab class; empty for a sole engine */
    /* Bytes its KV partition may take on beyond its carve, paid out of the expert pool (the other
     * budgets already exclude them). Never negative: a partition is not shrunk here. */
    int64_t kv_grow = 0;
};
using GroupBudget = std::vector<RankBudget>;   /* by rank */

struct GroupMember {
    std::string id;
    GroupFacts  facts;
};

/* The pure decision. `pool[r]` is rank r's expert bytes; `pending` is how many more engines will
 * join and reserve a minimum; `min_share` is the fraction of the equal share every model keeps
 * (0 to 1). One budget per member, in order. A sole member with nothing pending gets its `offer`. */
std::vector<GroupBudget> group_allocate(const std::vector<int64_t>& pool,
                                        const std::vector<GroupMember>& members, int pending,
                                        double min_share);

/* What a model may take now, from what the heat-based allocation wants for it. An elastic
 * model's growth is paid only out of the pool's free bytes, those that no member holds: a
 * member that must give bytes up (its want is below what it holds) is told so at once, and the
 * bytes a taker is granted exist only after the giver's pages went back to the card (the giver
 * reports a lower `held`). A grant not yet used (`prev`, the budgets handed out last time) stays
 * reserved while the member still wants it, so two takers are never granted the same free bytes.
 * `want` is group_allocate's result; `prev` is by member (empty: none given yet). Non-elastic
 * members pass through. VRAM is never planned twice: per rank, the sum of what elastic members
 * hold or are granted does not exceed the pool. */
std::vector<GroupBudget> group_gate(const std::vector<int64_t>& pool,
                                    const std::vector<GroupMember>& members,
                                    std::vector<GroupBudget> want,
                                    const std::vector<GroupBudget>& prev);

/* A running engine's expert plane as bins: its conditional units with a slab slot, by class and by
 * the power of two their heat is near. Without a heat engine running every unit has heat 0. `scale`
 * multiplies every heat first (an idle engine's cooling, Engine::group_tick). */
std::vector<HeatBin> group_heat_bins(const Plan& plan, const HeatEngine& heat, float scale = 1.0f);

/* The slab classes of a plan whose units the group planner counts (group_heat_bins). */
std::vector<int32_t> group_classes(const Plan& plan);

/* How many resident units of slab class `slab_class` the target `b` lets an engine keep, for
 * Mover::set_resident_cap; -1 for no limit. A sole engine's target names no classes and equals
 * its offer; a target that names some grants the others nothing. */
int64_t group_resident_cap(const RankBudget& b, int32_t slab_class);

/* The process-wide planner, one per GPU group. Thread-safe. Engines join when they plan (the
 * default model at startup, a cold model when its first request activates it) and leave when
 * they end. */
class GroupPlanner {
public:
    /* `to_start` is how many engines have yet to join: each reserves a minimum until it does.
     * Manifest mode passes 0: every model's share is fixed, so a model not started yet takes
     * nothing from the pool. */
    explicit GroupPlanner(int to_start = 1, double min_share = 0.5)
        : pending_(to_start), min_share_(min_share) {}

    /* Registers (or re-registers) an engine and returns the budget it starts with: the smaller of
     * its offer and its target, per rank. */
    GroupBudget join(const std::string& id, GroupFacts facts);
    /* Unregisters an engine; the others' targets are recomputed. */
    void leave(const std::string& id);
    /* A running engine's new heat summary, one bin list per rank; targets are recomputed. */
    void update(const std::string& id, const std::vector<std::vector<HeatBin>>& bins);
    /* The same for one rank, which reports on its own: the other ranks' bins stay as they were. */
    void update(const std::string& id, size_t rank, std::vector<HeatBin> bins);
    /* How many expert bytes a rank's slab has pages behind now; targets are recomputed. This is the
     * giver's receipt: only after it falls can a taker be granted the bytes. The bytes are the
     * movable slots the bins count (Mover::slab_held_bytes), so they compare with the targets.
     * `offer` >= 0 replaces the offer join() took from the single-engine resolver, which also
     * covers the slots that cannot move (Mover::slab_offer_bytes); the first report states it,
     * and until then the member's whole offer is reserved as held. Ignored for an unknown id or
     * rank. */
    void update_held(const std::string& id, size_t rank, int64_t held, int64_t offer = -1);
    /* A rank's KV partition; targets are recomputed. Ignored for an unknown id or rank. */
    void update_kv(const std::string& id, size_t rank, const KvPartition& kv);
    /* What the engine is entitled to now (empty when it is not registered). */
    GroupBudget target(const std::string& id) const;

    /* One rank of a member, for status and metrics: the target (bytes), what its slab has pages
     * behind (held, the physical commit) and the offer the pool counts for it. */
    struct RankView { int64_t target = 0, committed = 0, offer = 0; bool elastic = false; };
    std::vector<RankView> view(const std::string& id) const;

private:
    void recompute();

    mutable std::mutex mu_;
    int    pending_;
    double min_share_;
    std::vector<GroupMember> members_;
    std::vector<GroupBudget> granted_;   /* by member: what each started with */
    std::vector<GroupBudget> targets_;   /* by member */
    std::vector<int64_t>     pool_;      /* by rank */
    std::map<std::string, GroupBudget> given_;   /* by member: the last targets handed out (group_gate) */
};

}  /* namespace rad */
