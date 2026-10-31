/* hostmem_test.cpp -- the memory cgroup limits bring-up reads before locking a table into host
 * memory (core/engine_hostmem.h), against cgroup files written into a temporary tree.
 *
 * A misread limit either refuses a copy that fits or locks one that does not, and under Docker's
 * --memory the second ends in the OOM killer partway through loading.
 */
#include "rad_test.h"

#include "engine_hostmem.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace rad;

namespace {

struct TmpTree {
    std::string root;
    TmpTree() {
        std::string t = radtest::tmp_path("tree.XXXXXX");
        root = mkdtemp(t.data()) ? t : "";
    }
    void put(const std::string& rel, const std::string& text) const {
        std::filesystem::create_directories(std::filesystem::path(root + rel).parent_path());
        std::ofstream(root + rel) << text;
    }
};

constexpr int64_t GiB = (int64_t)1 << 30;

}  /* namespace */

/* v2: the limit less the memory reclaim cannot give back, which is the usage less its file pages.
 * "max" is no limit, and so is a missing file. */
TEST(a_v2_cgroups_room_is_its_limit_less_unreclaimable_usage) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/a/memory.max", "8589934592\n");
    t.put("/a/memory.current", "3221225472\n");
    t.put("/a/memory.stat", "anon 1073741824\nfile 1073741824\nkernel 4096\n");
    int64_t room = -1;
    CHECK(cgroup_room(t.root + "/a", true, &room));
    CHECK_EQ(room, 8 * GiB - (3 * GiB - 1 * GiB));

    t.put("/b/memory.max", "max\n");
    t.put("/b/memory.current", "1024\n");
    CHECK(!cgroup_room(t.root + "/b", true, &room));
    CHECK(!cgroup_room(t.root + "/missing", true, &room));

    /* Usage past the limit leaves no room rather than a negative one. */
    t.put("/c/memory.max", "1073741824\n");
    t.put("/c/memory.current", "2147483648\n");
    CHECK(cgroup_room(t.root + "/c", true, &room));
    CHECK_EQ(room, (int64_t)0);
}

/* Page-cache backing is not RAM backing (ticket #13): a cgroup whose usage is all file pages (a
 * model's container read through the page cache, mapped weights) has its whole limit left for the
 * host budget, since reclaim gives those pages back; the same usage as anonymous memory (a
 * PreparedWeightStore copy, a host pool) takes its share. */
TEST(page_cache_usage_leaves_the_whole_limit_and_anonymous_usage_does_not) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/cache/memory.max", "4294967296\n");
    t.put("/cache/memory.current", "3221225472\n");
    t.put("/cache/memory.stat", "anon 0\nfile 3221225472\n");
    t.put("/anon/memory.max", "4294967296\n");
    t.put("/anon/memory.current", "3221225472\n");
    t.put("/anon/memory.stat", "anon 3221225472\nfile 0\n");
    int64_t room = -1;
    CHECK(cgroup_room(t.root + "/cache", true, &room));
    CHECK_EQ(room, 4 * GiB);
    CHECK(cgroup_room(t.root + "/anon", true, &room));
    CHECK_EQ(room, 1 * GiB);
}

/* v1: the same from usage_in_bytes less total_cache (cache when there is no total_cache), and
 * the near-INT64_MAX value v1 reports for no limit is no limit. */
TEST(a_v1_cgroups_room_reads_its_cache_and_its_no_limit_sentinel) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/a/memory.limit_in_bytes", "4294967296\n");
    t.put("/a/memory.usage_in_bytes", "2147483648\n");
    t.put("/a/memory.stat", "cache 536870912\nrss 1073741824\ntotal_cache 1073741824\n");
    int64_t room = -1;
    CHECK(cgroup_room(t.root + "/a", false, &room));
    CHECK_EQ(room, 4 * GiB - (2 * GiB - 1 * GiB));

    t.put("/b/memory.limit_in_bytes", "4294967296\n");
    t.put("/b/memory.usage_in_bytes", "2147483648\n");
    t.put("/b/memory.stat", "cache 536870912\nrss 1073741824\n");
    CHECK(cgroup_room(t.root + "/b", false, &room));
    CHECK_EQ(room, 4 * GiB - (2 * GiB - GiB / 2));

    t.put("/c/memory.limit_in_bytes", "9223372036854771712\n");
    t.put("/c/memory.usage_in_bytes", "1024\n");
    CHECK(!cgroup_room(t.root + "/c", false, &room));
}

/* Every level from the process's own cgroup up to the root is a limit of its own: each one with
 * a limit is visited, leaf first, a level without one is passed over, and the walk stops when
 * asked. */
TEST(the_cgroup_walk_visits_every_limited_level_up_to_the_root) {
    TmpTree t;
    REQUIRE(!t.root.empty());
    t.put("/memory.max", "max\n");                       /* the root: no limit */
    t.put("/memory.current", "1\n");
    t.put("/docker/memory.max", "4294967296\n");
    t.put("/docker/memory.current", "1073741824\n");
    t.put("/docker/memory.stat", "file 0\n");
    t.put("/docker/ctr/memory.max", "max\n");            /* the container's own: none */
    t.put("/docker/ctr/memory.current", "1\n");
    t.put("/docker/ctr/app/memory.max", "17179869184\n");
    t.put("/docker/ctr/app/memory.current", "0\n");
    std::vector<std::pair<std::string, int64_t>> seen;
    cgroup_levels(t.root, "/docker/ctr/app", true, [&](const std::string& dir, int64_t room) {
        seen.emplace_back(dir, room);
        return true;
    });
    REQUIRE_EQ(seen.size(), (size_t)2);
    CHECK_EQ(seen[0].first, t.root + "/docker/ctr/app");
    CHECK_EQ(seen[0].second, 16 * GiB);
    CHECK_EQ(seen[1].first, t.root + "/docker");
    CHECK_EQ(seen[1].second, 3 * GiB);

    int visits = 0;
    cgroup_levels(t.root, "/docker/ctr/app", true, [&](const std::string&, int64_t) {
        ++visits;
        return false;
    });
    CHECK_EQ(visits, 1);

    /* The root itself, named "/". */
    seen.clear();
    t.put("/memory.max", "2147483648\n");
    cgroup_levels(t.root, "/", true, [&](const std::string& dir, int64_t room) {
        seen.emplace_back(dir, room);
        return true;
    });
    REQUIRE_EQ(seen.size(), (size_t)1);
    CHECK_EQ(seen[0].first, t.root);
}

RAD_TEST_MAIN()
