#include "group_planner.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace rad {

namespace {

int64_t unit_slot(const HeatBin& b) { return b.units > 0 ? b.slot_bytes / b.units : 0; }

/* Traffic a resident slot saves per byte of slab it costs. */
double worth(const HeatBin& b) {
    return b.slot_bytes > 0 ? (double)b.heat * (double)b.bytes / (double)b.slot_bytes : 0.0;
}

/* A bin as the allocation consumes it: units not yet granted. */
struct Left {
    const HeatBin* bin;
    size_t         member;
    int64_t        units;
    int64_t        slot;
    double         worth;
    size_t         seq;   /* creation order: the tie-break, so the order is deterministic */
};

bool hotter(const Left& a, const Left& b) {
    return a.worth != b.worth ? a.worth > b.worth : a.seq < b.seq;
}

/* What a model can hold at most: its whole plane, and never more than its offer, the room its
 * fixed VRAM share left for experts. */
int64_t ceiling_of(const RankFacts& f) {
    int64_t c = 0;
    for (const HeatBin& b : f.bins)
        if (unit_slot(b) > 0) c += b.slot_bytes;
    return std::min(c, f.offer);
}

/* One rank of group_allocate. `facts[i]` is member i's side. */
std::vector<RankBudget> allocate_rank(int64_t pool, const std::vector<const RankFacts*>& facts,
                                      int pending, double min_share) {
    const size_t n = facts.size();
    std::vector<RankBudget> out(n);
    if (n == 0) return out;   /* the last engine left */
    if (n == 1 && pending == 0) {
        out[0].bytes = facts[0]->offer;
        return out;
    }
    pool = std::max<int64_t>(pool, 0);

    const int64_t share = (int64_t)(std::clamp(min_share, 0.0, 1.0) *
                                    (double)(pool / (int64_t)(n + (size_t)pending)));
    const int64_t avail = std::max<int64_t>(pool - (int64_t)pending * share, 0);

    std::vector<int64_t> keep(n);
    int64_t keep_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        keep[i] = std::min(std::max(facts[i]->floor, share), ceiling_of(*facts[i]));
        keep_sum += keep[i];
    }
    /* The minimums do not fit: scale them down together rather than serve the first in order. */
    if (keep_sum > avail)
        for (int64_t& k : keep) k = (int64_t)((__int128)k * avail / keep_sum);

    std::vector<std::vector<Left>> own(n);
    size_t seq = 0;
    for (size_t i = 0; i < n; ++i) {
        for (const HeatBin& b : facts[i]->bins)
            if (unit_slot(b) > 0)
                own[i].push_back({ &b, i, b.units, unit_slot(b), worth(b), seq++ });
        std::sort(own[i].begin(), own[i].end(), hotter);
    }

    int64_t granted = 0;
    /* The units of `l` its model can still take under its offer. */
    auto room = [&](const Left& l) {
        return std::min(l.units, (ceiling_of(*facts[l.member]) - out[l.member].bytes) / l.slot);
    };
    auto take = [&](Left& l, int64_t units) {
        units = std::min(units, room(l));
        if (units <= 0) return;
        l.units -= units;
        out[l.member].bytes += units * l.slot;
        out[l.member].class_units[l.bin->slab_class] += units;
        granted += units * l.slot;
    };

    /* 1. each model's minimum, from its own hottest units. */
    for (size_t i = 0; i < n; ++i)
        for (Left& l : own[i])
            take(l, std::min(l.units, (keep[i] - out[i].bytes) / l.slot));

    /* 2. the rest by worth. A tier of equal worth that does not fit whole is shared in proportion
     * to its bytes, each share rounded down to whole units. */
    std::vector<Left*> all;
    for (auto& v : own)
        for (Left& l : v) all.push_back(&l);
    std::sort(all.begin(), all.end(), [](const Left* a, const Left* b) { return hotter(*a, *b); });
    /* A model that reaches its offer drops out of its tier, and the others share what it leaves:
     * go round again until nothing moves. */
    for (int64_t before = -1; before != granted && granted < avail;) {
        before = granted;
        for (size_t lo = 0; lo < all.size();) {
            size_t hi = lo;
            int64_t tier = 0;
            while (hi < all.size() && all[hi]->worth == all[lo]->worth) {
                if (room(*all[hi]) > 0) tier += all[hi]->units * all[hi]->slot;
                ++hi;
            }
            const int64_t remaining = avail - granted;
            if (remaining <= 0) break;
            for (size_t k = lo; k < hi; ++k) {
                Left& l = *all[k];
                if (tier <= remaining) {
                    take(l, l.units);
                } else {
                    const int64_t bytes = (int64_t)((__int128)remaining * (l.units * l.slot) / tier);
                    take(l, std::min(l.units, bytes / l.slot));
                }
            }
            lo = hi;
        }
    }
    return out;
}

}  /* namespace */

std::vector<GroupBudget> group_allocate(const std::vector<int64_t>& pool,
                                        const std::vector<GroupMember>& members, int pending,
                                        double min_share) {
    std::vector<GroupBudget> out(members.size(), GroupBudget(pool.size()));
    for (size_t r = 0; r < pool.size(); ++r) {
        std::vector<const RankFacts*> facts;
        static const RankFacts none;
        for (const GroupMember& m : members) facts.push_back(r < m.facts.size() ? &m.facts[r] : &none);
        std::vector<RankBudget> got = allocate_rank(pool[r], facts, pending, min_share);
        for (size_t i = 0; i < members.size(); ++i) out[i][r] = std::move(got[i]);
    }
    return out;
}

namespace {
bool counted(const MoveUnit& mu) {
    return mu.access == RAD_ACCESS_CONDITIONAL && mu.expert >= 0 && mu.slab_class >= 0 &&
           !mu.preplaced();
}
}  /* namespace */

std::vector<int32_t> group_classes(const Plan& plan) {
    std::set<int32_t> k;
    for (const MoveUnit& mu : plan.units)
        if (counted(mu)) k.insert(mu.slab_class);
    return { k.begin(), k.end() };
}

int64_t group_resident_cap(const RankBudget& b, int32_t slab_class) {
    if (b.class_units.empty()) return b.bytes > 0 ? -1 : 0;
    const auto it = b.class_units.find(slab_class);
    return it == b.class_units.end() ? 0 : it->second;
}

std::vector<HeatBin> group_heat_bins(const Plan& plan, const HeatEngine& heat) {
    struct Acc { double heat = 0; HeatBin bin; };
    std::map<std::pair<int32_t, int>, Acc> acc;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (!counted(mu)) continue;
        const float h = heat.enabled() ? heat.heat((int32_t)u) : 0.0f;
        /* Octaves of heat; bucket 0 is cold. */
        const int bucket = h > 0.0f ? std::clamp(std::ilogb(h) + 24, 1, 63) : 0;
        Acc& a = acc[{ mu.slab_class, bucket }];
        a.heat += h;
        a.bin.slab_class = mu.slab_class;
        a.bin.units += 1;
        a.bin.bytes += mu.bytes;
        a.bin.slot_bytes += mu.slot_bytes;
    }
    std::vector<HeatBin> out;
    for (auto& kv : acc) {
        kv.second.bin.heat = (float)(kv.second.heat / (double)kv.second.bin.units);
        out.push_back(kv.second.bin);
    }
    return out;
}

void GroupPlanner::recompute() {
    targets_ = group_allocate(pool_, members_, pending_, min_share_);
}

GroupBudget GroupPlanner::join(const std::string& id, GroupFacts facts) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < members_.size(); ++i)
        if (members_[i].id == id) {
            members_.erase(members_.begin() + (std::ptrdiff_t)i);
            granted_.erase(granted_.begin() + (std::ptrdiff_t)i);
            break;
        }
    /* The pool is what the running engines hold plus what this one was offered. */
    pool_.assign(facts.size(), 0);
    for (size_t r = 0; r < facts.size(); ++r) {
        pool_[r] = facts[r].offer;
        for (const GroupBudget& g : granted_)
            if (r < g.size()) pool_[r] += g[r].bytes;
    }
    members_.push_back({ id, std::move(facts) });
    if (pending_ > 0) --pending_;
    recompute();

    const GroupBudget& want = targets_.back();
    GroupBudget got(want.size());
    for (size_t r = 0; r < want.size(); ++r) {
        const int64_t offer = members_.back().facts[r].offer;
        got[r].bytes = std::min(want[r].bytes, offer);
        if (want[r].bytes <= offer) got[r].class_units = want[r].class_units;
    }
    granted_.push_back(got);
    return got;
}

void GroupPlanner::leave(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < members_.size(); ++i)
        if (members_[i].id == id) {
            members_.erase(members_.begin() + (std::ptrdiff_t)i);
            granted_.erase(granted_.begin() + (std::ptrdiff_t)i);
            break;
        }
    recompute();
}

void GroupPlanner::update(const std::string& id, const std::vector<std::vector<HeatBin>>& bins) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id)
            for (size_t r = 0; r < bins.size() && r < m.facts.size(); ++r) m.facts[r].bins = bins[r];
    recompute();
}

void GroupPlanner::update(const std::string& id, size_t rank, std::vector<HeatBin> bins) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id && rank < m.facts.size()) m.facts[rank].bins = std::move(bins);
    recompute();
}

GroupBudget GroupPlanner::target(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < members_.size() && i < targets_.size(); ++i)
        if (members_[i].id == id) return targets_[i];
    return {};
}

}  /* namespace rad */
