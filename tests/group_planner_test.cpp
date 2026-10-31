/* group_planner_test.cpp -- the group planner (core/place/group_planner.h): how the expert VRAM of
 * a GPU group is divided between its active models.
 *
 *   Heat        the hotter model's units fill the pool first
 *   Minimum     a cold model keeps its minimum share and its expert floor
 *   Bytes       worth is heat times bytes per slot byte, so models with different expert sizes
 *               compare
 *   Recompute   targets follow every join, leave and heat update
 *   Sharding    ranks (World Size 2) are decided from their own shards
 *   Single      one engine gets back exactly what it offered
 *   Shares      no model is granted or owed more than it offered (its fixed VRAM share)
 *   Cold start  a model activated later joins the running group; leaving frees its room
 *   KV claims   a partition that needs room takes it from the experts of any model (ticket #17)
 *   Elastic     a model with an elastic slab may hold more than its offer, and is granted the
 *               bytes only after the giver physically released them (ticket #35)
 *
 * Pure host arithmetic: no device, no model.
 */
#include "rad_test.h"

#include "place/group_planner.h"

using namespace rad;

namespace {

constexpr int64_t MiB = 1024 * 1024;

/* `units` units of `unit` MiB at `heat` each, in slab class `cls`. */
HeatBin bin(int64_t units, int64_t unit, float heat, int32_t cls = 0) {
    return { heat, units, units * unit * MiB, units * unit * MiB, cls };
}

/* One rank's report: a pool offer, an expert floor and bins. */
GroupFacts one_rank(int64_t offer_mib, std::vector<HeatBin> bins, int64_t floor_mib = 0) {
    GroupFacts f(1);
    f[0].offer = offer_mib * MiB;
    f[0].floor = floor_mib * MiB;
    f[0].bins = std::move(bins);
    return f;
}

std::vector<GroupBudget> allocate(int64_t pool_mib, std::vector<GroupFacts> facts, int pending = 0,
                                  double min_share = 0.5) {
    std::vector<GroupMember> members;
    for (size_t i = 0; i < facts.size(); ++i) members.push_back({ "m" + std::to_string(i), facts[i] });
    return group_allocate({ pool_mib * MiB }, members, pending, min_share);
}

}  /* namespace */

TEST(a_sole_engine_gets_back_the_budget_it_offered) {
    /* Offers beyond the plane and below it alike: the group changes nothing for one engine. */
    for (int64_t offer : { 3, 12, 64 }) {
        GroupPlanner g(1);
        const GroupBudget b = g.join("only", one_rank(offer, { bin(8, 2, 5.0f) }));
        REQUIRE_EQ(b.size(), (size_t)1);
        CHECK_EQ(b[0].bytes, offer * MiB);
        CHECK(b[0].class_units.empty());
    }
}

TEST(the_hotter_model_gets_the_slots) {
    /* Two models of eight 2 MiB experts; a 20 MiB pool holds ten. B's experts are twice as hot. */
    const auto got = allocate(20, { one_rank(20, { bin(8, 2, 1.0f) }),
                                    one_rank(20, { bin(8, 2, 2.0f) }) });
    /* Each keeps its minimum, 5 MiB rounded down to whole units: 2 units. The rest, by heat. */
    CHECK_EQ(got[1][0].bytes, (int64_t)16 * MiB);   /* all eight */
    CHECK_EQ(got[0][0].bytes, (int64_t)4 * MiB);    /* its two minimum units; the pool is spent */
    CHECK_EQ(got[1][0].class_units.at(0), (int64_t)8);
    CHECK_EQ(got[0][0].class_units.at(0), (int64_t)2);
}

TEST(a_cold_model_keeps_its_minimum_share) {
    /* B is all cold, A very hot, and A wants the whole pool. B still holds half the equal share. */
    const auto got = allocate(40, { one_rank(40, { bin(20, 2, 9.0f) }),
                                    one_rank(40, { bin(20, 2, 0.0f) }) });
    CHECK_EQ(got[1][0].bytes, (int64_t)10 * MiB);   /* 0.5 * (40 / 2) */
    CHECK_EQ(got[0][0].bytes, (int64_t)30 * MiB);
}

TEST(the_minimum_is_at_least_the_expert_floor) {
    /* The floor is what the host pool cannot hold: below it the planner refuses experts. */
    const auto got = allocate(40, { one_rank(40, { bin(20, 2, 9.0f) }),
                                    one_rank(40, { bin(20, 2, 0.0f) }, /*floor*/ 16) });
    CHECK_EQ(got[1][0].bytes, (int64_t)16 * MiB);
    CHECK_EQ(got[0][0].bytes, (int64_t)24 * MiB);
}

TEST(the_minimum_never_exceeds_a_models_whole_plane) {
    /* A 4 MiB plane keeps 4 MiB, not the 10 MiB minimum; the surplus goes to the other model. */
    const auto got = allocate(40, { one_rank(40, { bin(2, 2, 0.0f) }),
                                    one_rank(40, { bin(20, 2, 1.0f) }) });
    CHECK_EQ(got[0][0].bytes, (int64_t)4 * MiB);
    CHECK_EQ(got[1][0].bytes, (int64_t)36 * MiB);
}

TEST(a_pending_engine_reserves_its_minimum) {
    /* The first of two engines: it is alone but not the group, so it leaves the other's minimum. */
    const auto got = allocate(40, { one_rank(40, { bin(20, 2, 1.0f) }) }, /*pending*/ 1);
    CHECK_EQ(got[0][0].bytes, (int64_t)30 * MiB);   /* 40 - 0.5 * (40 / 2) */
}

TEST(worth_is_heat_times_bytes_per_slot_so_models_with_different_experts_compare) {
    /* A: 4 MiB experts, heat 2. B: 1 MiB experts, heat 3. 12 MiB, no minimum. */
    {
        const auto got = allocate(12, { one_rank(12, { bin(4, 4, 2.0f) }),
                                        one_rank(12, { bin(10, 1, 3.0f) }) }, 0, 0.0);
        CHECK_EQ(got[1][0].bytes, (int64_t)10 * MiB);   /* every B unit is worth more per byte */
        CHECK_EQ(got[0][0].bytes, (int64_t)0);          /* 2 MiB left: no whole 4 MiB slot */
    }
    /* Padding is not weight: A's 4 MiB slots hold 3.5 MiB of expert, worth 3 * 3.5 / 4 = 2.625
     * a byte against B's 2.8. */
    {
        HeatBin padded = { 3.0f, 4, 14 * MiB, 16 * MiB, 0 };
        const auto got = allocate(12, { one_rank(12, { padded }),
                                        one_rank(12, { bin(10, 1, 2.8f) }) }, 0, 0.0);
        CHECK_EQ(got[1][0].bytes, (int64_t)10 * MiB);
        CHECK_EQ(got[0][0].bytes, (int64_t)0);
    }
    /* Reverse the heats and the big experts win the bytes. */
    {
        const auto got = allocate(12, { one_rank(12, { bin(4, 4, 3.0f) }),
                                        one_rank(12, { bin(10, 1, 2.0f) }) }, 0, 0.0);
        CHECK_EQ(got[0][0].bytes, (int64_t)12 * MiB);
        CHECK_EQ(got[1][0].bytes, (int64_t)0);
    }
}

TEST(equal_worth_shares_the_room_in_proportion_to_bytes) {
    /* Both cold, as at a cold start: the pool splits by plane size. */
    const auto got = allocate(24, { one_rank(24, { bin(8, 2, 0.0f) }),
                                    one_rank(24, { bin(16, 2, 0.0f) }) }, 0, 0.0);
    CHECK_EQ(got[0][0].bytes, (int64_t)8 * MiB);
    CHECK_EQ(got[1][0].bytes, (int64_t)16 * MiB);
}

TEST(classes_of_one_model_are_filled_hottest_first_and_counted) {
    /* One rank holding an uneven slice: two slab classes of different strides. */
    const auto got = allocate(17, { one_rank(17, { bin(4, 2, 1.0f, 0), bin(4, 3, 4.0f, 1) }),
                                    one_rank(17, { bin(1, 1, 0.5f) }) }, 0, 0.0);
    CHECK_EQ(got[0][0].class_units.at(1), (int64_t)4);   /* the 3 MiB class is hotter: 12 MiB */
    CHECK_EQ(got[0][0].class_units.at(0), (int64_t)2);   /* 4 of the 5 MiB left; B takes the last */
    CHECK_EQ(got[1][0].bytes, (int64_t)1 * MiB);
}

TEST(budgets_are_recomputed_when_an_engine_joins_and_leaves) {
    GroupPlanner g(2);
    /* A starts first, with the card to itself, and leaves room for the engine to come. */
    const GroupBudget a = g.join("a", one_rank(40, { bin(20, 2, 1.0f) }));
    CHECK_EQ(a[0].bytes, (int64_t)30 * MiB);
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)30 * MiB);

    /* B joins the 10 MiB that is left, offering it. The pool is what A holds plus B's offer; the
     * group is complete. B is hotter, but its offer of 10 is its ceiling; A keeps the rest. */
    const GroupBudget b = g.join("b", one_rank(10, { bin(20, 2, 3.0f) }));
    CHECK_EQ(b[0].bytes, (int64_t)10 * MiB);   /* its target (hotter) is larger, but only 10 is free */
    CHECK_EQ(g.target("a")[0].bytes + g.target("b")[0].bytes, (int64_t)40 * MiB);
    CHECK_EQ(g.target("b")[0].bytes, (int64_t)10 * MiB);
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)30 * MiB);

    /* B leaves: A is alone again and entitled to what it was granted. */
    g.leave("b");
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)40 * MiB);
    CHECK(g.target("b").empty());
}

TEST(an_engine_starting_with_less_free_than_its_target_gets_only_what_is_free) {
    GroupPlanner g(2);
    g.join("a", one_rank(40, { bin(20, 2, 0.0f) }));   /* holds 30: 10 stay for the engine to come */
    /* Equal planes and no heat would split the 50 MiB the pool now is in half, but B offers 20 and
     * that is its ceiling; A is owed the rest. A running engine acts on its target at a step
     * boundary (Engine::group_tick); the planner just records what it is owed. */
    const GroupBudget b = g.join("b", one_rank(20, { bin(20, 2, 0.0f) }));
    CHECK_EQ(b[0].bytes, (int64_t)20 * MiB);
    CHECK(g.target("b")[0].bytes <= (int64_t)20 * MiB);
    CHECK(g.target("a")[0].bytes > g.target("b")[0].bytes);
    CHECK(g.target("a")[0].bytes + g.target("b")[0].bytes >= (int64_t)48 * MiB);
}

TEST(no_model_is_granted_more_than_it_offered_however_hot) {
    /* A is hot and wants the whole 40 MiB pool, but its share left it 10 MiB for experts; B's 30
     * MiB offer is the other ceiling. The pool's surplus stays unclaimed. */
    const auto got = allocate(40, { one_rank(10, { bin(20, 2, 9.0f) }),
                                    one_rank(30, { bin(20, 2, 0.1f) }) });
    CHECK_EQ(got[0][0].bytes, (int64_t)10 * MiB);
    CHECK_EQ(got[1][0].bytes, (int64_t)30 * MiB);
    /* A cold neighbour cannot take more than it offered either: A gives up bytes, B stops at 30. */
    const auto cold = allocate(40, { one_rank(10, { bin(20, 2, 0.1f) }),
                                     one_rank(30, { bin(20, 2, 9.0f) }) });
    CHECK(cold[0][0].bytes <= (int64_t)10 * MiB);
    CHECK_EQ(cold[1][0].bytes, (int64_t)30 * MiB);

    /* Through the planner, targets of running engines respect the ceiling too. */
    GroupPlanner g(0);
    g.join("a", one_rank(10, { bin(20, 2, 9.0f) }));
    g.join("b", one_rank(30, { bin(20, 2, 0.1f) }));
    CHECK(g.target("a")[0].bytes <= (int64_t)10 * MiB);
    CHECK(g.target("b")[0].bytes <= (int64_t)30 * MiB);
}

TEST(a_model_activated_later_joins_the_running_group_and_leaves_when_it_ends) {
    /* Manifest mode: shares are fixed, so nothing is reserved for a model not started (0). */
    GroupPlanner g(0);
    const GroupBudget a = g.join("default", one_rank(20, { bin(20, 2, 1.0f) }));
    CHECK_EQ(a[0].bytes, (int64_t)20 * MiB);   /* alone, it keeps its whole share */
    CHECK_EQ(g.target("cold").size(), (size_t)0);

    /* The cold model's first request activates it: it joins with its own share. */
    const GroupBudget c = g.join("cold", one_rank(20, { bin(20, 2, 4.0f) }));
    CHECK(c[0].bytes > 0 && c[0].bytes <= (int64_t)20 * MiB);
    CHECK_EQ(g.target("default")[0].bytes + g.target("cold")[0].bytes, (int64_t)40 * MiB);
    CHECK(g.target("cold")[0].bytes <= (int64_t)20 * MiB);

    /* It ends: it is gone and the default model is alone again. */
    g.leave("cold");
    CHECK(g.target("cold").empty());
    CHECK_EQ(g.target("default")[0].bytes, (int64_t)20 * MiB);

    /* The last one leaves too: an empty group, and a model may activate again. */
    g.leave("default");
    CHECK(g.target("default").empty());
    CHECK_EQ(g.join("cold", one_rank(20, { bin(20, 2, 4.0f) }))[0].bytes, (int64_t)20 * MiB);
}

TEST(a_heat_update_moves_the_targets) {
    GroupPlanner g(1);
    g.join("a", one_rank(30, { bin(20, 2, 1.0f) }));
    GroupPlanner h(2);
    h.join("a", one_rank(40, { bin(20, 2, 1.0f) }));   /* holds 30; 10 stay for the engine to come */
    h.join("b", one_rank(40, { bin(20, 2, 1.0f) }));   /* the pool is 70 MiB, each offered 40 */
    const int64_t before = h.target("a")[0].bytes;
    h.update("a", { { bin(20, 2, 20.0f) } });
    CHECK(h.target("a")[0].bytes > before);
    CHECK_EQ(h.target("a")[0].bytes, (int64_t)40 * MiB);   /* up to its offer, not past it */
    CHECK_EQ(h.target("a")[0].bytes + h.target("b")[0].bytes, (int64_t)70 * MiB);
}

TEST(each_rank_is_decided_from_its_own_shard_at_world_size_2) {
    /* Each rank holds half of every expert. Equal shards: the ranks agree. */
    auto shard = [](float heat, int64_t rank_pool_mib) {
        GroupFacts f(2);
        for (GroupFacts::iterator r = f.begin(); r != f.end(); ++r) {
            r->offer = rank_pool_mib * MiB;
            r->bins = { bin(8, 1, heat) };
        }
        return f;
    };
    std::vector<GroupMember> m = { { "a", shard(1.0f, 10) }, { "b", shard(2.0f, 10) } };
    auto got = group_allocate({ 10 * MiB, 10 * MiB }, m, 0, 0.5);
    for (size_t i = 0; i < 2; ++i) CHECK_EQ(got[i][0].bytes, got[i][1].bytes);
    CHECK_EQ(got[1][0].bytes, (int64_t)8 * MiB);
    CHECK_EQ(got[0][0].bytes, (int64_t)2 * MiB);

    /* A card with less free VRAM budgets less, and the engine takes the smaller (the one split
     * every rank shares). */
    got = group_allocate({ 10 * MiB, 6 * MiB }, m, 0, 0.5);
    CHECK(got[1][1].bytes < got[1][0].bytes);
    CHECK(got[0][1].bytes + got[1][1].bytes <= (int64_t)6 * MiB);

    /* Through the planner: both ranks are registered, and the sole engine keeps its offers. */
    GroupPlanner g(1);
    GroupFacts f = shard(1.0f, 10);
    f[1].offer = 6 * MiB;
    const GroupBudget b = g.join("only", f);
    REQUIRE_EQ(b.size(), (size_t)2);
    CHECK_EQ(b[0].bytes, (int64_t)10 * MiB);
    CHECK_EQ(b[1].bytes, (int64_t)6 * MiB);
}

TEST(heat_bins_group_a_plans_expert_units_by_class_and_heat) {
    /* One routed layer of eight experts; experts 0 and 1 are routed, the rest are not. */
    Plan p;
    for (int e = 0; e < 8; ++e) {
        MoveUnit mu;
        mu.layer = 0;
        mu.expert = e;
        mu.slab_class = 0;
        mu.bytes = 2 * MiB - 4096;
        mu.slot_bytes = 2 * MiB;
        mu.access = RAD_ACCESS_CONDITIONAL;
        mu.movable = true;
        mu.tier = e < 4 ? Tier::VRAM : Tier::HostPinned;
        p.units.push_back(mu);
    }
    HeatEngine h;
    CHECK_OK(h.init(p, HeatParams{}));
    CHECK(h.enabled());
    std::vector<int32_t> counts(8, 0);
    counts[0] = 6;
    counts[1] = 6;
    h.observe(0, counts.data(), 8);

    const std::vector<HeatBin> bins = group_heat_bins(p, h);
    REQUIRE_EQ(bins.size(), (size_t)2);
    int64_t units = 0, slot = 0, bytes = 0;
    const HeatBin* hot = nullptr;
    for (const HeatBin& b : bins) {
        units += b.units;
        slot += b.slot_bytes;
        bytes += b.bytes;
        if (b.heat > 0.0f) hot = &b;
    }
    CHECK_EQ(units, (int64_t)8);
    CHECK_EQ(slot, 8 * 2 * MiB);
    CHECK_EQ(bytes, 8 * (2 * MiB - 4096));
    REQUIRE(hot != nullptr);
    CHECK_EQ(hot->units, (int64_t)2);
    CHECK_EQ(hot->heat, 6.0f);

    /* The bins are what the planner consumes. */
    GroupFacts f(1);
    f[0].offer = 16 * MiB;
    f[0].bins = bins;
    GroupFacts other(1);
    other[0].offer = 16 * MiB;
    other[0].bins = { bin(8, 2, 0.5f) };
    const auto got = group_allocate({ 16 * MiB }, { { "a", f }, { "b", other } }, 0, 0.0);
    /* The routed pair is hotter than b's units, which are hotter than a's cold six. */
    CHECK_EQ(got[0][0].class_units.at(0), (int64_t)2);
    CHECK_EQ(got[1][0].class_units.at(0), (int64_t)6);
}

TEST(a_rank_reports_its_own_heat_and_the_others_keep_theirs) {
    GroupPlanner g(0);
    GroupFacts two(2);
    for (RankFacts& r : two) { r.offer = 10 * MiB; r.bins = { bin(8, 1, 1.0f) }; }
    g.join("a", two);
    g.join("b", two);
    const int64_t r1 = g.target("a")[1].bytes;
    g.update("a", 0, { bin(8, 1, 50.0f) });
    CHECK_EQ(g.target("a")[1].bytes, r1);                    /* rank 1 reported nothing */
    g.update("a", 5, { bin(8, 1, 50.0f) });                  /* no such rank: ignored */
    g.update("none", 0, { bin(8, 1, 50.0f) });
    CHECK_EQ(g.target("a")[1].bytes, r1);
}

TEST(a_target_states_a_resident_cap_per_slab_class) {
    /* A sole engine: its target names no class and equals its offer, so nothing is capped. */
    GroupPlanner g(1);
    const GroupBudget sole = g.join("only", one_rank(20, { bin(8, 2, 1.0f, 3) }));
    CHECK_EQ(group_resident_cap(sole[0], 3), (int64_t)-1);
    CHECK_EQ(group_resident_cap(g.target("only")[0], 3), (int64_t)-1);

    /* A target that names classes grants the others nothing. */
    RankBudget b;
    b.bytes = 6 * MiB;
    b.class_units[0] = 3;
    CHECK_EQ(group_resident_cap(b, 0), (int64_t)3);
    CHECK_EQ(group_resident_cap(b, 1), (int64_t)0);
    /* An empty grant is a cap of zero, not "no limit". */
    CHECK_EQ(group_resident_cap(RankBudget{}, 0), (int64_t)0);

    /* The classes a plan's experts live in; preplaced and non-expert units do not count. */
    Plan p;
    for (int i = 0; i < 6; ++i) {
        MoveUnit mu;
        mu.expert = i < 5 ? i : -1;
        mu.slab_class = i % 2 ? 2 : 0;
        mu.slot_bytes = mu.bytes = MiB;
        mu.access = RAD_ACCESS_CONDITIONAL;
        mu.tier = i == 2 ? Tier::Mapped : Tier::VRAM;
        p.units.push_back(mu);
    }
    const std::vector<int32_t> classes = group_classes(p);
    CHECK_EQ(classes.size(), (size_t)2);
    CHECK(classes == (std::vector<int32_t>{ 0, 2 }));
}

/* ---- KV partitions (ticket #17) ---------------------------------------------------------------- */

/* A model's side with a KV partition: `carve`/`held`/`want`/`cap`/`growable` in MiB. */
GroupFacts with_kv(GroupFacts f, int64_t carve, int64_t held, int64_t want, int64_t cap,
                   int64_t growable) {
    f[0].kv = { carve * MiB, held * MiB, want * MiB, cap * MiB, growable * MiB };
    return f;
}

TEST(a_partition_that_needs_room_takes_it_from_the_experts_of_another_model) {
    /* A's cold slab fills its whole 30 MiB offer; B's 10 MiB of experts are hot. B's KV carve of
     * 8 MiB is full (held) and it wants 14. Without the claim A keeps 30. */
    const GroupFacts a = one_rank(30, { bin(15, 2, 0.1f) });
    const GroupFacts b = one_rank(10, { bin(5, 2, 5.0f) });
    const auto calm = allocate(40, { a, b });
    CHECK_EQ(calm[0][0].bytes, (int64_t)30 * MiB);
    CHECK_EQ(calm[1][0].kv_grow, (int64_t)0);

    const auto got = allocate(40, { a, with_kv(b, 8, 8, 14, 32, 16) });
    CHECK_EQ(got[1][0].kv_grow, (int64_t)6 * MiB);   /* B gets its KV ... */
    CHECK_EQ(got[0][0].bytes, (int64_t)24 * MiB);    /* ... out of A's slab, the colder one */
    CHECK_EQ(got[1][0].bytes, (int64_t)10 * MiB);    /* B's hot experts stay */
    CHECK_EQ(got[0][0].kv_grow, (int64_t)0);
    /* One pool, planned once: experts plus KV growth never pass it. */
    CHECK(got[0][0].bytes + got[1][0].bytes + got[0][0].kv_grow + got[1][0].kv_grow <=
          (int64_t)40 * MiB);
}

TEST(a_claimant_can_evict_its_own_experts_when_they_are_the_coldest) {
    const auto got = allocate(40, { one_rank(20, { bin(10, 2, 5.0f) }),
                                    with_kv(one_rank(20, { bin(10, 2, 0.1f) }), 8, 8, 14, 32, 16) });
    CHECK_EQ(got[1][0].kv_grow, (int64_t)6 * MiB);
    CHECK_EQ(got[0][0].bytes, (int64_t)20 * MiB);   /* the hot slab is untouched */
    CHECK_EQ(got[1][0].bytes, (int64_t)14 * MiB);   /* its own cold experts paid */
}

TEST(a_claim_never_takes_a_models_minimum) {
    /* B claims everything, but A keeps max(floor 16, 0.5 * 20 = 10) MiB of its slab. */
    const auto got = allocate(40, { one_rank(30, { bin(15, 2, 0.1f) }, /*floor*/ 16),
                                    with_kv(one_rank(10, { bin(5, 2, 0.1f) }), 8, 8, 200, 200, 200) });
    CHECK(got[0][0].bytes >= (int64_t)16 * MiB);
    CHECK(got[1][0].bytes >= (int64_t)10 * MiB);   /* its own minimum share, 0.5 * (40 / 2) */
    CHECK(got[1][0].kv_grow <= (int64_t)14 * MiB);
    CHECK(got[0][0].bytes + got[1][0].bytes + got[1][0].kv_grow <= (int64_t)40 * MiB);
}

TEST(running_sequences_are_never_evicted_a_partition_is_only_ever_added_to) {
    /* A holds 20 of its 20 MiB carve and says it wants 5: held is the floor of want, so A claims
     * nothing and no output takes a byte from any partition. */
    const auto got = allocate(40, { with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 20, 20, 5, 32, 12),
                                    with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 8, 6, 12, 32, 12) });
    CHECK_EQ(got[0][0].kv_grow, (int64_t)0);
    CHECK_EQ(got[1][0].kv_grow, (int64_t)4 * MiB);
    CHECK(got[0][0].kv_grow >= 0 && got[1][0].kv_grow >= 0);
    /* A cap under the held bytes cannot make a claim negative or evict anyone's sequence. */
    const auto low = allocate(40, { with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 20, 20, 30, 10, 12),
                                    one_rank(20, { bin(10, 2, 1.0f) }) });
    CHECK_EQ(low[0][0].kv_grow, (int64_t)0);
}

TEST(a_claim_stops_at_what_the_partition_can_use_and_back) {
    /* Wants more than its ceiling: only up to the cap (20 - 8 = 12 MiB) ... */
    auto got = allocate(40, { one_rank(20, { bin(10, 2, 1.0f) }),
                              with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 8, 8, 100, 20, 100) });
    CHECK_EQ(got[1][0].kv_grow, (int64_t)12 * MiB);
    /* ... and only up to what its pool can back. Today that is nothing: no claim, no eviction. */
    got = allocate(40, { one_rank(20, { bin(10, 2, 1.0f) }),
                         with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 8, 8, 100, 20, 0) });
    CHECK_EQ(got[1][0].kv_grow, (int64_t)0);
    CHECK_EQ(got[0][0].bytes, (int64_t)20 * MiB);
    CHECK_EQ(got[1][0].bytes, (int64_t)20 * MiB);
}

TEST(claims_that_do_not_fit_share_what_the_minimums_leave_in_proportion) {
    /* Pool 40, minimums 10 + 10 leave 20; the claims are 30 and 10. */
    const auto got = allocate(40, { with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 4, 4, 34, 64, 64),
                                    with_kv(one_rank(20, { bin(10, 2, 1.0f) }), 4, 4, 14, 64, 64) });
    CHECK_EQ(got[0][0].kv_grow, (int64_t)15 * MiB);
    CHECK_EQ(got[1][0].kv_grow, (int64_t)5 * MiB);
    CHECK(got[0][0].bytes >= (int64_t)10 * MiB && got[1][0].bytes >= (int64_t)10 * MiB);
}

TEST(a_sole_engine_is_unchanged_by_a_kv_report) {
    GroupPlanner g(0);
    const GroupBudget b = g.join("only", with_kv(one_rank(20, { bin(8, 2, 1.0f) }), 8, 8, 99, 99, 99));
    CHECK_EQ(b[0].bytes, (int64_t)20 * MiB);
    g.update_kv("only", 0, { 8 * MiB, 8 * MiB, 99 * MiB, 99 * MiB, 99 * MiB });
    CHECK_EQ(g.target("only")[0].bytes, (int64_t)20 * MiB);
    CHECK_EQ(g.target("only")[0].kv_grow, (int64_t)0);
}

TEST(a_kv_report_moves_the_targets_and_ending_the_need_gives_the_experts_back) {
    GroupPlanner g(0);
    g.join("a", one_rank(30, { bin(15, 2, 0.1f) }));
    g.join("b", one_rank(10, { bin(5, 2, 5.0f) }));
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)30 * MiB);

    g.update_kv("b", 0, { 8 * MiB, 8 * MiB, 14 * MiB, 32 * MiB, 16 * MiB });
    CHECK_EQ(g.target("b")[0].kv_grow, (int64_t)6 * MiB);
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)24 * MiB);
    g.update_kv("none", 0, { MiB, MiB, 99 * MiB, 99 * MiB, 99 * MiB });   /* ignored */
    g.update_kv("b", 3, { MiB, MiB, 99 * MiB, 99 * MiB, 99 * MiB });      /* no such rank */
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)24 * MiB);

    /* B's sequences end: it wants only what it holds again, and A's slab regrows. */
    g.update_kv("b", 0, { 14 * MiB, 0, 0, 32 * MiB, 16 * MiB });
    CHECK_EQ(g.target("b")[0].kv_grow, (int64_t)0);
    CHECK_EQ(g.target("a")[0].bytes, (int64_t)30 * MiB);
}

TEST(a_claim_is_decided_per_rank_at_world_size_2) {
    auto side = [](int64_t offer_mib, float heat, int64_t want_mib) {
        GroupFacts f(2);
        for (RankFacts& r : f) {
            r.offer = offer_mib * MiB;
            r.bins = { bin(offer_mib / 2, 2, heat) };
        }
        f[1].kv = { 8 * MiB, 8 * MiB, want_mib * MiB, 32 * MiB, 32 * MiB };   /* only rank 1 */
        return f;
    };
    std::vector<GroupMember> m = { { "a", side(30, 0.1f, 0) }, { "b", side(10, 5.0f, 14) } };
    const auto got = group_allocate({ 40 * MiB, 40 * MiB }, m, 0, 0.5);
    CHECK_EQ(got[1][0].kv_grow, (int64_t)0);
    CHECK_EQ(got[1][1].kv_grow, (int64_t)6 * MiB);
    CHECK_EQ(got[0][0].bytes, (int64_t)30 * MiB);
    CHECK_EQ(got[0][1].bytes, (int64_t)24 * MiB);
}

RAD_TEST_MAIN()

/* ---- elastic slabs (ticket #35): the pool is the sum of the offers, growth waits for the giver. */
namespace {
GroupFacts elastic_rank(int64_t offer_mib, std::vector<HeatBin> bins) {
    GroupFacts f = one_rank(offer_mib, std::move(bins));
    f[0].elastic = true;
    return f;
}
int64_t mib(const GroupBudget& b) { return b[0].bytes / MiB; }
}  /* namespace */

TEST(an_elastic_model_may_be_allocated_more_than_its_offer) {
    /* Two 20 MiB offers make a 40 MiB pool. A is hot, B cold: A is allocated past its own offer. */
    std::vector<GroupMember> m = { { "a", elastic_rank(20, { bin(30, 2, 9.0f) }) },
                                   { "b", elastic_rank(20, { bin(30, 2, 0.0f) }) } };
    const auto got = group_allocate({ 40 * MiB }, m, 0, 0.5);
    CHECK_EQ(mib(got[0]), 30);      /* B keeps 0.5 * (40 / 2) = 10 */
    CHECK_EQ(mib(got[1]), 10);
    /* The same facts without the elastic flag stay at the offer, as before. */
    m[0].facts[0].elastic = m[1].facts[0].elastic = false;
    const auto old = group_allocate({ 40 * MiB }, m, 0, 0.5);
    CHECK_EQ(mib(old[0]), 20);
}

TEST(a_taker_is_granted_bytes_only_after_the_giver_released_them) {
    GroupPlanner g(0);
    const GroupBudget a0 = g.join("a", elastic_rank(20, { bin(30, 2, 0.0f) }));
    CHECK_EQ(mib(a0), 20);           /* a starts with its offer */
    const GroupBudget b0 = g.join("b", elastic_rank(20, { bin(30, 2, 0.0f) }));
    CHECK_EQ(mib(b0), 20);           /* and so does b: both hold their offers physically */

    /* a turns hot, b cold. The allocation says 30 / 10, but the bytes are b's until it lets go. */
    g.update("a", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    g.update("b", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 0.0f) } });
    CHECK_EQ(mib(g.target("b")), 10);   /* the giver is told at once */
    CHECK_EQ(mib(g.target("a")), 20);   /* the taker waits: nothing is free */

    /* b's slab gives back 6 MiB of pages: a is granted exactly those. */
    g.update_held("b", 0, 14 * MiB);
    CHECK_EQ(mib(g.target("b")), 10);
    CHECK_EQ(mib(g.target("a")), 26);
    g.update_held("b", 0, 10 * MiB);
    CHECK_EQ(mib(g.target("a")), 30);
    /* a takes the pages: nothing changes, the grant was a's own. */
    g.update_held("a", 0, 30 * MiB);
    CHECK_EQ(mib(g.target("a")), 30);
    CHECK_EQ(mib(g.target("b")), 10);
}

TEST(two_takers_are_never_granted_the_same_free_bytes) {
    GroupPlanner g(0);
    (void)g.join("a", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)g.join("b", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)g.join("c", elastic_rank(20, { bin(30, 2, 0.0f) }));
    /* c is hot and wants most of the 60 MiB pool; a and b are cold at 20 MiB each. */
    g.update("a", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 0.0f) } });
    g.update("b", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 0.0f) } });
    g.update("c", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    CHECK_EQ(mib(g.target("c")), 20);
    /* a gives 6, b gives 4: c may take 10, not the 10 + 10 it would if each grant were counted
     * against the other's release. */
    g.update_held("a", 0, 14 * MiB);
    g.update_held("b", 0, 16 * MiB);
    CHECK_EQ(mib(g.target("c")), 30);
    /* Two takers want the same free bytes: c and d are equally hot, a cold. */
    GroupPlanner h(0);
    (void)h.join("a", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)h.join("c", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)h.join("d", elastic_rank(20, { bin(30, 2, 0.0f) }));
    h.update("c", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    h.update("d", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    h.update_held("a", 0, 18 * MiB);       /* 2 MiB free, and both want more */
    CHECK_EQ(mib(h.target("c")) + mib(h.target("d")), 42);    /* the 2 shared, not 2 + 2 */
    CHECK_EQ(mib(h.target("c")), 21);
    /* What the three hold or are granted never exceeds the pool. */
    CHECK(18 + mib(h.target("c")) + mib(h.target("d")) <= 60);
}

TEST(a_grant_not_yet_used_stays_reserved_while_it_is_wanted) {
    GroupPlanner g(0);
    (void)g.join("a", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)g.join("b", elastic_rank(20, { bin(30, 2, 0.0f) }));
    g.update("a", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 0.0f) } });
    g.update("b", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    g.update_held("a", 0, 10 * MiB);       /* a released 10 */
    CHECK_EQ(mib(g.target("b")), 30);
    /* b has not committed the pages yet; a heat update must not hand the same 10 to anyone. */
    g.update("b", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 8.0f) } });
    CHECK_EQ(mib(g.target("b")), 30);
    g.update_held("b", 0, 24 * MiB);
    CHECK_EQ(mib(g.target("b")), 30);
}

TEST(an_elastic_model_that_leaves_gives_the_pool_back_to_the_others_offers) {
    GroupPlanner g(0);
    (void)g.join("a", elastic_rank(20, { bin(30, 2, 0.0f) }));
    (void)g.join("b", elastic_rank(20, { bin(30, 2, 0.0f) }));
    g.update("a", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) } });
    g.update_held("b", 0, 10 * MiB);
    CHECK_EQ(mib(g.target("a")), 30);
    g.leave("b");
    /* A sole engine is back to its offer: what b's share held is b's again when it returns. */
    CHECK_EQ(mib(g.target("a")), 20);
    CHECK(g.target("a")[0].class_units.empty());
}

TEST(elastic_targets_are_gated_per_rank_at_world_size_2) {
    GroupPlanner g(0);
    GroupFacts fa(2), fb(2);
    for (size_t r = 0; r < 2; ++r) {
        fa[r] = elastic_rank(20, { bin(30, 2, 0.0f) })[0];
        fb[r] = elastic_rank(20, { bin(30, 2, 0.0f) })[0];
    }
    (void)g.join("a", fa);
    (void)g.join("b", fb);
    g.update("a", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 9.0f) }, { bin(30, 2, 9.0f) } });
    g.update("b", std::vector<std::vector<HeatBin>>{ { bin(30, 2, 0.0f) }, { bin(30, 2, 0.0f) } });
    g.update_held("b", 1, 12 * MiB);       /* only rank 1 has released */
    const GroupBudget a = g.target("a");
    CHECK_EQ(a[0].bytes, 20 * MiB);
    CHECK_EQ(a[1].bytes, 28 * MiB);
}

TEST(a_sole_elastic_engine_is_unchanged) {
    GroupPlanner g(1);
    const GroupBudget b = g.join("only", elastic_rank(12, { bin(30, 2, 5.0f) }));
    CHECK_EQ(mib(b), 12);
    CHECK(b[0].class_units.empty());
    g.update_held("only", 0, 4 * MiB);
    CHECK_EQ(mib(g.target("only")), 12);
}
