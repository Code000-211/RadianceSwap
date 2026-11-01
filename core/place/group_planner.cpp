#include "group_planner.h"

#include <algorithm>
#include <cmath>
#include <map>
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
 * fixed VRAM share left for experts. An elastic model's bound is the group's pool: it may hold
 * bytes another model's share left (group_gate decides when). */
int64_t ceiling_of(const RankFacts& f, int64_t pool) {
    int64_t c = 0;
    for (const HeatBin& b : f.bins)
        if (unit_slot(b) > 0) c += b.slot_bytes;
    return std::min(c, f.elastic ? std::max(f.offer, pool) : f.offer);
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
    int64_t avail = std::max<int64_t>(pool - (int64_t)pending * share, 0);

    std::vector<int64_t> keep(n);
    int64_t keep_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        keep[i] = std::min(std::max(facts[i]->floor, share), ceiling_of(*facts[i], pool));
        keep_sum += keep[i];
    }

    /* KV claims come off the pool before the experts see it. Only what the minimums leave over is
     * claimable, shared in proportion to the claims when it falls short. */
    std::vector<int64_t> claim(n, 0);
    int64_t claimed = 0;
    for (size_t i = 0; i < n; ++i) {
        const KvPartition& k = facts[i]->kv;
        const int64_t cap  = std::max(k.cap, k.held);
        const int64_t want = std::clamp(k.want, k.held, cap);
        claim[i] = std::min(std::max<int64_t>(want - k.carve, 0), std::max<int64_t>(k.growable, 0));
        claimed += claim[i];
    }
    if (claimed > 0) {
        const int64_t grant = std::min(claimed, std::max<int64_t>(avail - keep_sum, 0));
        for (size_t i = 0; i < n; ++i)
            out[i].kv_grow = (int64_t)((__int128)claim[i] * grant / claimed);
        avail -= grant;
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
        return std::min(l.units, (ceiling_of(*facts[l.member], pool) - out[l.member].bytes) / l.slot);
    };
    auto take = [&](Left& l, int64_t units) {
        units = std::min(units, room(l));
        if (units <= 0) return;
        l.units -= units;
        out[l.member].bytes += units * l.slot;
        out[l.member].class_units[l.bin->slab_class] += units;
        granted += units * l.slot;
    };

    /* 1. each model's minimum, from its own hottest units. A model's bins of equal worth (its
     * experts' halves in different slab classes, or a cold start) share the minimum in proportion
     * to their bytes instead of the first in order taking all of it: an expert is useful only
     * with every half resident, so the classes of a model fill together. */
    for (size_t i = 0; i < n; ++i) {
        std::vector<Left>& v = own[i];
        for (size_t lo = 0; lo < v.size();) {
            size_t hi = lo;
            int64_t tier = 0;
            while (hi < v.size() && v[hi].worth == v[lo].worth) {
                if (room(v[hi]) > 0) tier += v[hi].units * v[hi].slot;
                ++hi;
            }
            const int64_t remaining = keep[i] - out[i].bytes;
            for (size_t k = lo; k < hi && remaining > 0; ++k) {
                Left& l = v[k];
                if (tier <= remaining) take(l, l.units);
                else take(l, (int64_t)((__int128)remaining * (l.units * l.slot) / tier) / l.slot);
            }
            lo = hi;
        }
    }

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

    /* An elastic model's classes move together. An expert is two halves in two slab classes (gate_up,
     * down), each with its own heat buckets, and the worth order above would grow one class while
     * the other shrinks; the slots each class gives or takes are physical pages, so that churn is
     * paid in commits and copies for no gain. The model's bytes are cut across its classes in
     * proportion to the classes' planes instead. */
    for (size_t i = 0; i < n; ++i) {
        if (!facts[i]->elastic || out[i].bytes <= 0) continue;
        std::map<int32_t, std::pair<int64_t, int64_t>> plane;   /* class -> units, slot bytes */
        int64_t total = 0;
        for (const HeatBin& b : facts[i]->bins)
            if (unit_slot(b) > 0) {
                auto& p = plane[b.slab_class];
                p.first += b.units;
                p.second = unit_slot(b);
                total += b.units * unit_slot(b);
            }
        if (total <= 0 || plane.size() < 2) continue;
        const int64_t bytes = out[i].bytes;
        out[i].bytes = 0;
        for (const auto& kv : plane) {
            const int64_t units = std::min<int64_t>(
                kv.second.first,
                (int64_t)((__int128)bytes * (kv.second.first * kv.second.second) / total) /
                    kv.second.second);
            out[i].class_units[kv.first] = units;
            out[i].bytes += units * kv.second.second;
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

std::vector<GroupBudget> group_gate(const std::vector<int64_t>& pool,
                                    const std::vector<GroupMember>& members,
                                    std::vector<GroupBudget> want,
                                    const std::vector<GroupBudget>& prev,
                                    const std::vector<int64_t>& static_pending) {
    for (size_t r = 0; r < pool.size(); ++r) {
        const size_t n = members.size();
        std::vector<int64_t> reserved(n, 0), extra(n, 0), kv_extra(n, 0);
        int64_t wanted = 0, taken = 0, kv_wanted = 0;
        std::vector<char> grows(n, 0);
        for (size_t i = 0; i < n && i < want.size(); ++i) {
            if (r >= want[i].size()) continue;
            const int64_t t = want[i][r].bytes;
            const RankFacts* f = r < members[i].facts.size() ? &members[i].facts[r] : nullptr;
            /* A KV partition holds `grown` bytes of the pool. A claim above that is paid only out of
             * the free bytes, like an elastic slab's growth; the part granted and not yet backed
             * (`prev`) stays reserved while it is still wanted. At or below `grown` the partition is
             * told to give chunks back, and holds them until it reports them returned. */
            const int64_t kv_want = want[i][r].kv_grow;
            const int64_t grown = f ? std::max<int64_t>(f->kv.grown, 0) : 0;
            if (kv_want > 0 || grown > 0) {
                const int64_t before = i < prev.size() && r < prev[i].size() ? prev[i][r].kv_grow : 0;
                const int64_t kv_res = kv_want <= grown ? grown : std::max(grown, std::min(before, kv_want));
                kv_extra[i] = std::max<int64_t>(kv_want - kv_res, 0);
                if (kv_want > grown) want[i][r].kv_grow = kv_res;
                taken += kv_res;
                kv_wanted += kv_extra[i];
            }
            if (!f || !f->elastic) { reserved[i] = t; taken += t; continue; }
            const int64_t held = std::max<int64_t>(f->held, 0);
            if (t <= held) { reserved[i] = held; taken += held; continue; }
            const int64_t before = i < prev.size() && r < prev[i].size() ? prev[i][r].bytes : 0;
            reserved[i] = std::max(held, std::min(before, t));
            extra[i] = t - reserved[i];
            grows[i] = 1;
            taken += reserved[i];
            wanted += extra[i];
        }
        /* The order of the free bytes: statics, then the KV partitions' claims, then the experts' growth.
         * Statics are read every token; a partition that holds waiting sequences stalls them all when it
         * cannot grow, and an expert is only a cache line. Each step takes from what the one before it
         * left, so no byte is planned twice. */
        const int64_t statics = r < static_pending.size() ? static_pending[r] : 0;
        int64_t free_bytes = std::max<int64_t>(pool[r] - taken - statics, 0);
        /* The experts were already cut by the claim (group_allocate). */
        const int64_t kv_free = free_bytes;
        for (size_t i = 0; i < n && i < want.size(); ++i) {
            if (r >= want[i].size() || kv_extra[i] <= 0) continue;
            const int64_t give = kv_wanted <= kv_free ? kv_extra[i]
                                                     : (int64_t)((__int128)kv_extra[i] * kv_free / kv_wanted);
            want[i][r].kv_grow += give;
            free_bytes -= give;
        }
        for (size_t i = 0; i < n && i < want.size(); ++i) {
            if (r >= want[i].size() || !grows[i]) continue;
            RankBudget& b = want[i][r];
            const int64_t t = b.bytes;
            const int64_t give = wanted <= free_bytes ? extra[i]
                                                     : (int64_t)((__int128)extra[i] * free_bytes / wanted);
            b.bytes = reserved[i] + give;
            /* The units follow the bytes, class by class. */
            if (b.bytes < t && t > 0)
                for (auto& kv : b.class_units) kv.second = (int64_t)((__int128)kv.second * b.bytes / t);
        }
    }
    return want;
}

std::vector<int64_t> group_statics(const std::vector<int64_t>& free,
                                   const std::vector<GroupMember>& members,
                                   const std::vector<GroupBudget>& prev,
                                   std::vector<GroupBudget>* budgets) {
    const size_t n = members.size();
    std::vector<int64_t> pending(free.size(), 0);
    budgets->resize(n);
    for (size_t r = 0; r < free.size(); ++r) {
        int64_t avail = std::max<int64_t>(free[r], 0);
        std::vector<int64_t> reserved(n, 0);
        for (size_t i = 0; i < n; ++i) {
            if ((*budgets)[i].size() <= r) (*budgets)[i].resize(free.size());
            const RankFacts* f = r < members[i].facts.size() ? &members[i].facts[r] : nullptr;
            const RankBudget* p = i < prev.size() && r < prev[i].size() ? &prev[i][r] : nullptr;
            if (!f) continue;
            const int64_t held = std::max<int64_t>(f->held, 0);
            /* Bytes an elastic member was granted and has not backed yet are spoken for. */
            if (f->elastic && p) avail -= std::max<int64_t>(p->bytes - held, 0);
            /* So are the bytes a KV partition was granted and has not backed yet. */
            if (p) avail -= std::max<int64_t>(p->kv_grow - f->kv.grown, 0);
            if (p) reserved[i] = std::clamp(p->statics_cap - f->statics.held, (int64_t)0, f->statics.left);
            avail -= reserved[i];
        }
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const float ha = r < members[a].facts.size() ? members[a].facts[r].statics.heat : 0.0f;
            const float hb = r < members[b].facts.size() ? members[b].facts[r].statics.heat : 0.0f;
            return ha > hb;
        });
        for (size_t i : order) {
            if (r >= members[i].facts.size()) continue;
            const StaticFacts& s = members[i].facts[r].statics;
            const int64_t give = std::clamp(s.left - reserved[i], (int64_t)0, std::max<int64_t>(avail, 0));
            avail -= give;
            (*budgets)[i][r].statics_cap = s.held + reserved[i] + give;
            pending[r] += reserved[i] + give;
        }
    }
    return pending;
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

std::vector<HeatBin> group_heat_bins(const Plan& plan, const HeatEngine& heat, float scale) {
    struct Acc { double heat = 0; HeatBin bin; };
    std::map<std::pair<int32_t, int>, Acc> acc;
    for (size_t u = 0; u < plan.units.size(); ++u) {
        const MoveUnit& mu = plan.units[u];
        if (!counted(mu)) continue;
        const float h = heat.enabled() ? heat.heat((int32_t)u) * scale : 0.0f;
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
    /* With an elastic member the pool is what the members were offered: the bytes a model's share
     * left unclaimed are free for any member to grow into. */
    std::vector<int64_t> pool = pool_;
    for (size_t r = 0; r < pool.size(); ++r) {
        bool elastic = false;
        int64_t offered = 0;
        for (const GroupMember& m : members_)
            if (r < m.facts.size()) {
                elastic = elastic || m.facts[r].elastic;
                offered += m.facts[r].offer;
            }
        if (elastic) pool[r] = offered;
    }
    std::vector<GroupBudget> prev;
    for (const GroupMember& m : members_) {
        const auto it = given_.find(m.id);
        prev.push_back(it == given_.end() ? GroupBudget{} : it->second);
    }
    /* Statics first: their grants come off the free bytes before an expert may grow (group_gate). */
    std::vector<GroupBudget> statics(members_.size(), GroupBudget(pool.size()));
    std::vector<int64_t> free = free_;
    free.resize(pool.size(), 0);
    const std::vector<int64_t> static_pending = group_statics(free, members_, prev, &statics);
    targets_ = group_gate(pool, members_, group_allocate(pool, members_, pending_, min_share_), prev,
                          static_pending);
    for (size_t i = 0; i < targets_.size(); ++i)
        for (size_t r = 0; r < targets_[i].size() && r < statics[i].size(); ++r)
            targets_[i][r].statics_cap = statics[i][r].statics_cap;
    given_.clear();
    for (size_t i = 0; i < members_.size() && i < targets_.size(); ++i)
        given_[members_[i].id] = targets_[i];
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
    given_.erase(id);
    /* It holds nothing until its pools are made: what it starts with is its first report. */
    for (RankFacts& f : facts) f.held = 0;
    members_.push_back({ id, std::move(facts) });
    if (pending_ > 0) --pending_;
    recompute();

    const GroupBudget& want = targets_.back();
    GroupBudget got(want.size());
    for (size_t r = 0; r < want.size(); ++r) {
        const int64_t offer = members_.back().facts[r].offer;
        got[r].bytes = std::min(want[r].bytes, offer);
        if (want[r].bytes <= offer) got[r].class_units = want[r].class_units;
        members_.back().facts[r].held = got[r].bytes;
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
    given_.erase(id);
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

void GroupPlanner::update_held(const std::string& id, size_t rank, int64_t held, int64_t offer) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id && rank < m.facts.size()) {
            m.facts[rank].held = held;
            if (offer >= 0) m.facts[rank].offer = offer;
        }
    recompute();
}

void GroupPlanner::update_arena_spent(const std::string& id, size_t rank) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id && rank < m.facts.size()) m.facts[rank].arena_spent = true;
}

void GroupPlanner::update_kv(const std::string& id, size_t rank, const KvPartition& kv) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id && rank < m.facts.size()) m.facts[rank].kv = kv;
    recompute();
}

void GroupPlanner::update_statics(const std::string& id, size_t rank, const StaticFacts& s) {
    std::lock_guard<std::mutex> lk(mu_);
    for (GroupMember& m : members_)
        if (m.id == id && rank < m.facts.size()) m.facts[rank].statics = s;
    recompute();
}

void GroupPlanner::update_free(size_t rank, int64_t bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_.size() <= rank) free_.resize(rank + 1, 0);
    free_[rank] = bytes;
    recompute();
}

std::vector<GroupPlanner::RankView> GroupPlanner::view(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<RankView> out;
    for (size_t i = 0; i < members_.size() && i < targets_.size(); ++i) {
        if (members_[i].id != id) continue;
        for (size_t r = 0; r < members_[i].facts.size() && r < targets_[i].size(); ++r)
            out.push_back({ targets_[i][r].bytes, members_[i].facts[r].held, members_[i].facts[r].offer,
                            members_[i].facts[r].elastic, members_[i].facts[r].statics,
                            targets_[i][r].kv_grow, members_[i].facts[r].kv.grown,
                            members_[i].facts[r].arena_spent });
    }
    return out;
}

GroupBudget GroupPlanner::target(const std::string& id) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < members_.size() && i < targets_.size(); ++i)
        if (members_[i].id == id) return targets_[i];
    return {};
}

}  /* namespace rad */
