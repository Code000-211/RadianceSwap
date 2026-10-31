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

/* One rank's side of an engine's report. */
struct RankFacts {
    int64_t offer = 0;   /* expert bytes the single-engine resolver gave it: VramBudget::experts */
    int64_t floor = 0;   /* VramFacts::expert_floor: below it the planner refuses experts */
    std::vector<HeatBin> bins;   /* its expert plane; the ceiling is their slot bytes */
};
using GroupFacts = std::vector<RankFacts>;   /* by rank */

struct RankBudget {
    int64_t bytes = 0;
    std::map<int32_t, int64_t> class_units;   /* slots per slab class; empty for a sole engine */
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

/* A running engine's expert plane as bins: its conditional units with a slab slot, by class and by
 * the power of two their heat is near. Without a heat engine running every unit has heat 0. */
std::vector<HeatBin> group_heat_bins(const Plan& plan, const HeatEngine& heat);

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
    /* What the engine is entitled to now (empty when it is not registered). */
    GroupBudget target(const std::string& id) const;

private:
    void recompute();

    mutable std::mutex mu_;
    int    pending_;
    double min_share_;
    std::vector<GroupMember> members_;
    std::vector<GroupBudget> granted_;   /* by member: what each started with */
    std::vector<GroupBudget> targets_;   /* by member */
    std::vector<int64_t>     pool_;      /* by rank */
};

}  /* namespace rad */
