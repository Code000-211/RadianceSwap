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
     * that is its ceiling; A is owed the rest. A running engine is not shrunk here (ticket #15):
     * its target just records what it is owed. */
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

RAD_TEST_MAIN()
