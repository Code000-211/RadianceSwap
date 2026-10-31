/* host_budget_test.cpp -- the host-RAM budget of the PreparedWeightStore (core/model/weight_store.h,
 * ticket #13): vorbereitete Models, the active models' host pools and the loads' staging together
 * stay under `cache.host_budget_mib`, and when they would not, the prepared model idle longest is
 * evicted and is kalt.
 *
 * The store is driven the way Engine::load_weights and the Launcher drive it: activate, begin,
 * reserve, commit, deactivate. Sizes are bytes, not MiB, since the store counts bytes.
 */
#include "rad_test.h"

#include "model/weight_store.h"

#include <cstring>
#include <string>
#include <vector>

using namespace rad;

namespace {

constexpr int64_t KiB = 1024;

/* One model's activation that fills the store with `bytes` and ends: the model is then prepared
 * (or kalt if the budget refused the copy). Returns whether the copy was made. */
bool fill(PreparedWeightStore& s, const std::string& model, int64_t bytes) {
    s.activate(model);
    s.begin(model, "key");
    uint8_t* p = s.reserve(model, "w0", bytes);
    if (p) std::memset(p, 1, (size_t)bytes);
    s.commit(model);
    s.deactivate(model);
    return p != nullptr;
}

/* The same for a model that is prepared already: a warm activation, which bumps its use. */
void reuse(PreparedWeightStore& s, const std::string& model) {
    s.activate(model);
    s.begin(model, "key");
    CHECK(s.find(model, "w0", s.ram_bytes(model)) != nullptr);
    s.commit(model);
    s.deactivate(model);
}

}  /* namespace */

/* The budget holds at every step: filling past it evicts, and the sum of store, pools and staging
 * never passes it. */
TEST(the_budget_is_never_exceeded_by_the_store) {
    PreparedWeightStore s;
    s.set_budget(10 * KiB);
    for (const char* m : { "a", "b", "c", "d", "e" }) {
        CHECK(fill(s, m, 4 * KiB));
        CHECK(s.stats().used() <= 10 * KiB);
    }
    CHECK_EQ(s.stats().store_bytes, 8 * KiB);   /* two fit, the third evicted the oldest */
    CHECK_EQ(s.stats().evicted_prepared, (int64_t)3);

    /* A pool claim of an active model counts: it leaves room for one copy only. */
    s.activate("p");
    CHECK(s.claim("p", HostUse::Pools, 4 * KiB));
    CHECK(s.stats().used() <= 10 * KiB);
    CHECK_EQ(s.stats().store_bytes, 4 * KiB);
    CHECK(s.claim("p", HostUse::Staging, 2 * KiB));
    CHECK_EQ(s.stats().used(), 10 * KiB);
    CHECK_EQ(s.stats().store_bytes, 4 * KiB);
    /* A copy that does not fit beside them evicts the prepared one and is made; one that cannot
     * fit even with the store empty is refused. */
    CHECK(!s.make_room("q", 7 * KiB));
    CHECK(s.prepared("e"));              /* nothing was evicted for a copy that could not fit */
    CHECK(s.make_room("q", 4 * KiB));
    CHECK(!s.prepared("e"));
    s.deactivate("p");
    CHECK_EQ(s.stats().pool_bytes, (int64_t)0);
    CHECK_EQ(s.stats().staging_bytes, (int64_t)0);
}

/* The model idle the longest goes first; a model that was used again (a warm activation) is
 * younger than one that was not. */
TEST(the_prepared_model_idle_longest_is_evicted_first) {
    PreparedWeightStore s;
    s.set_budget(12 * KiB);
    std::vector<std::string> cold;
    s.set_state_listener([&](const std::string& id, bool prepared, int64_t bytes) {
        CHECK(!prepared);
        CHECK_EQ(bytes, (int64_t)0);
        cold.push_back(id);
    });
    CHECK(fill(s, "a", 4 * KiB));
    CHECK(fill(s, "b", 4 * KiB));
    CHECK(fill(s, "c", 4 * KiB));
    reuse(s, "a");                       /* b is now the oldest */
    CHECK(fill(s, "d", 4 * KiB));
    REQUIRE_EQ(cold.size(), (size_t)1);
    CHECK_EQ(cold[0], std::string("b"));
    CHECK(fill(s, "e", 4 * KiB));
    REQUIRE_EQ(cold.size(), (size_t)2);
    CHECK_EQ(cold[1], std::string("c"));
    CHECK(s.prepared("a"));
    CHECK(s.prepared("d"));
    CHECK(s.prepared("e"));
    CHECK(!s.prepared("b"));
}

/* A model the budget made kalt is still activatable: it loads cold, fills the store again (by
 * evicting another) and is prepared once more. */
TEST(a_cold_model_after_eviction_activates_and_is_prepared_again) {
    PreparedWeightStore s;
    s.set_budget(8 * KiB);
    CHECK(fill(s, "a", 4 * KiB));
    CHECK(fill(s, "b", 4 * KiB));
    CHECK(fill(s, "c", 4 * KiB));
    CHECK(!s.prepared("a"));
    CHECK_EQ(s.ram_bytes("a"), (int64_t)0);
    CHECK(fill(s, "a", 4 * KiB));        /* a cold load: nothing was found, the copy is made */
    CHECK(s.prepared("a"));
    CHECK(!s.prepared("b"));             /* the next oldest gave way */
    CHECK(s.stats().used() <= 8 * KiB);
}

/* An active model is never evicted by a copy being made: when only active copies and pools are
 * left, the activating model does not fill the store (it is kalt once deactivated). */
TEST(an_active_model_is_never_evicted_by_a_new_copy) {
    PreparedWeightStore s;
    s.set_budget(8 * KiB);
    s.activate("a");
    s.begin("a", "key");
    REQUIRE(s.reserve("a", "w0", 4 * KiB) != nullptr);
    s.commit("a");                       /* a is active and holds its copy */
    CHECK(fill(s, "b", 4 * KiB));
    CHECK(s.prepared("b"));
    CHECK(fill(s, "c", 4 * KiB));        /* b is evicted for it, a is not */
    CHECK(!s.prepared("b"));
    CHECK(s.prepared("c"));
    CHECK(s.prepared("a"));
    CHECK_EQ(s.stats().evicted_active_copies, (int64_t)0);

    /* a's copy and c's fill the budget; with c gone nothing is left to drop for d but a's. */
    s.activate("d");
    s.begin("d", "key");
    CHECK(s.reserve("d", "w0", 4 * KiB) != nullptr);   /* c is prepared: it gives way */
    CHECK(!s.prepared("c"));
    CHECK(s.reserve("d", "w1", 4 * KiB) == nullptr);   /* a is active: refused, not evicted */
    CHECK(s.prepared("a"));
    s.abort("d");
    CHECK(s.stats().used() <= 8 * KiB);
    s.deactivate("d");
    CHECK(!s.prepared("d"));             /* d stays kalt */
    s.deactivate("a");
    CHECK(s.prepared("a"));              /* a, deactivated, is prepared with the copy it kept */
}

/* A mandatory claim (an activation's host pool) that the budget cannot give room for drops the
 * prepared models and then the copy of an active model, never another model's pools; the model
 * stays active and is kalt once deactivated. */
TEST(the_copy_of_an_active_model_is_dropped_under_pressure) {
    PreparedWeightStore s;
    s.set_budget(10 * KiB);
    std::vector<std::string> cold;
    s.set_state_listener([&](const std::string& id, bool, int64_t) { cold.push_back(id); });
    s.activate("a");
    s.begin("a", "key");
    REQUIRE(s.reserve("a", "w0", 6 * KiB) != nullptr);
    s.commit("a");
    CHECK(s.claim("a", HostUse::Pools, 2 * KiB));
    CHECK(fill(s, "p", 2 * KiB));        /* 6 + 2 + 2: the budget is full */
    CHECK_EQ(s.stats().used(), 10 * KiB);

    s.activate("b");
    CHECK(s.claim("b", HostUse::Pools, 3 * KiB));      /* p goes first, then a's copy */
    CHECK(!s.prepared("p"));
    CHECK(!s.prepared("a"));
    CHECK_EQ(s.stats().evicted_prepared, (int64_t)1);
    CHECK_EQ(s.stats().evicted_active_copies, (int64_t)1);
    CHECK_EQ(s.stats().used(), 5 * KiB);   /* a's pool (2) and b's (3) stay */
    REQUIRE_EQ(cold.size(), (size_t)1);
    CHECK_EQ(cold[0], std::string("p"));   /* only prepared models are announced */

    /* The pools of active models are never evicted: a claim that cannot be met even so is
     * recorded and reported. */
    CHECK(!s.claim("b", HostUse::Staging, 20 * KiB));
    CHECK_EQ(s.stats().over_budget_claims, (int64_t)1);
    CHECK_EQ(s.stats().pool_bytes, 5 * KiB);
    s.deactivate("a");
    CHECK(!s.prepared("a"));             /* a is kalt now */
}

/* A host check that refused (evict_one) takes prepared models first, then active copies, then
 * the requester's own, and nothing that is loading. */
TEST(evict_one_orders_its_victims_and_spares_a_model_that_is_loading) {
    PreparedWeightStore s;
    CHECK(fill(s, "p", KiB));
    s.activate("x");
    s.begin("x", "key");
    REQUIRE(s.reserve("x", "w0", KiB) != nullptr);
    s.commit("x");                       /* x: active, copy */
    s.activate("me");
    s.begin("me", "key");
    REQUIRE(s.reserve("me", "w0", KiB) != nullptr);
    s.commit("me");
    s.begin("me", "key");                /* me is loading again: not evictable */
    CHECK(s.evict_one("me", false));             /* p goes */
    CHECK(!s.prepared("p"));
    CHECK(!s.evict_one("me", false));            /* only active copies are left */
    CHECK(s.evict_one("me", true));              /* x's */
    CHECK(!s.prepared("x"));
    CHECK(!s.evict_one("me", true));             /* me's own is loading */
    CHECK(s.prepared("me"));
    s.abort("me");
    CHECK(s.evict_one("me", true));
    CHECK(!s.prepared("me"));
}

/* Without a budget nothing is evicted by the store itself: the Engine's host checks decide. */
TEST(no_budget_means_the_store_evicts_nothing_by_itself) {
    PreparedWeightStore s;
    for (const char* m : { "a", "b", "c" }) CHECK(fill(s, m, 64 * KiB));
    CHECK(s.make_room("d", INT64_MAX / 2));
    CHECK_EQ(s.stats().evicted_prepared, (int64_t)0);
    CHECK_EQ(s.stats().budget_bytes, (int64_t)0);
    CHECK_EQ(s.ram_bytes(), 192 * KiB);
}

/* Page-cache backing is not RAM backing: only what reserve() allocated is the store's, and a
 * model with no entry (a mapped or file-tier weight) has no bytes in it. */
TEST(only_ram_backing_is_counted) {
    PreparedWeightStore s;
    s.set_budget(8 * KiB);
    s.activate("m");
    s.begin("m", "key");
    s.commit("m");                       /* every weight of m was mapped: nothing reserved */
    s.deactivate("m");
    CHECK(!s.prepared("m"));
    CHECK_EQ(s.stats().used(), (int64_t)0);
}

RAD_TEST_MAIN()
