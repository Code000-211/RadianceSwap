/* tests/collective_sequencer_test.cpp -- unit tests for CollectiveSequencer.
 *
 * Verifies:
 * - Single-engine fast path: zero overhead, no blocking.
 * - Multi-engine, multi-rank: identical collective order across all ranks.
 * - No deadlock under heavy concurrent thread arrival races.
 * - Correct lifecycle (register, unregister, abort).
 */
#include "rad_test.h"
#include "runtime/collective_sequencer.h"
#include "device/device.h"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

namespace {

/* Mock stream pointer: non-null distinct pointers so rad_event_wait/record are tested on host */
RadStream fake_stream(int rank, int id) {
    return reinterpret_cast<RadStream>((uintptr_t)(0x1000 + rank * 0x100 + id));
}

}  /* namespace */

TEST(single_engine_fast_path) {
    rad::CollectiveSequencer seq;
    const void* engine1 = reinterpret_cast<const void*>(0x1111);

    CHECK(seq.is_single_engine());
    CHECK_EQ(seq.active_engines(), 0u);

    /* Single engine registered: still single-engine fast-path */
    seq.register_engine(engine1, 2);
    CHECK(seq.is_single_engine());
    CHECK_EQ(seq.active_engines(), 1u);

    /* before_collective and after_collective return immediately with 0 */
    uint64_t t0 = seq.before_collective(engine1, 0, 2, fake_stream(0, 0));
    CHECK_EQ(t0, 0u);
    seq.after_collective(engine1, 0, 2, fake_stream(0, 0), t0);

    uint64_t t1 = seq.before_collective(engine1, 1, 2, fake_stream(1, 0));
    CHECK_EQ(t1, 0u);
    seq.after_collective(engine1, 1, 2, fake_stream(1, 0), t1);

    seq.unregister_engine(engine1);
    CHECK(seq.is_single_engine());
    CHECK_EQ(seq.active_engines(), 0u);
}

TEST(multi_engine_two_ranks_identical_order) {
    rad::CollectiveSequencer seq;
    const void* engineA = reinterpret_cast<const void*>(0xAAAA);
    const void* engineB = reinterpret_cast<const void*>(0xBBBB);

    seq.register_engine(engineA, 2);
    seq.register_engine(engineB, 2);
    CHECK(!seq.is_single_engine());
    CHECK_EQ(seq.active_engines(), 2u);

    constexpr int kStepsPerEngine = 100;
    constexpr int kTotalCollectives = kStepsPerEngine * 2;

    std::vector<const void*> rank0_order;
    std::vector<const void*> rank1_order;
    std::mutex m0, m1;

    std::atomic<bool> start{ false };

    auto run_rank = [&](const void* owner, int rank, std::vector<const void*>& out, std::mutex& out_m) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < kStepsPerEngine; ++i) {
            uint64_t ticket = seq.before_collective(owner, rank, 2, nullptr);
            {
                std::lock_guard<std::mutex> lk(out_m);
                out.push_back(owner);
            }
            seq.after_collective(owner, rank, 2, nullptr, ticket);
        }
    };

    /* Spawn 4 threads: Engine A (rank 0, rank 1), Engine B (rank 0, rank 1) */
    std::thread tA0(run_rank, engineA, 0, std::ref(rank0_order), std::ref(m0));
    std::thread tA1(run_rank, engineA, 1, std::ref(rank1_order), std::ref(m1));
    std::thread tB0(run_rank, engineB, 0, std::ref(rank0_order), std::ref(m0));
    std::thread tB1(run_rank, engineB, 1, std::ref(rank1_order), std::ref(m1));

    start.store(true, std::memory_order_release);

    tA0.join();
    tA1.join();
    tB0.join();
    tB1.join();

    REQUIRE_EQ(rank0_order.size(), (size_t)kTotalCollectives);
    REQUIRE_EQ(rank1_order.size(), (size_t)kTotalCollectives);

    /* Assert that every single collective in sequence is executed in the IDENTICAL order on both ranks */
    for (size_t i = 0; i < (size_t)kTotalCollectives; ++i) {
        CHECK_EQ(rank0_order[i], rank1_order[i]);
    }

    seq.unregister_engine(engineA);
    CHECK(seq.is_single_engine());
    seq.unregister_engine(engineB);
    CHECK(seq.is_single_engine());
}

TEST(multi_engine_four_ranks_interleaved_heavy) {
    rad::CollectiveSequencer seq;
    const void* engineA = reinterpret_cast<const void*>(0x10);
    const void* engineB = reinterpret_cast<const void*>(0x20);
    const void* engineC = reinterpret_cast<const void*>(0x30);

    seq.register_engine(engineA, 4);
    seq.register_engine(engineB, 4);
    seq.register_engine(engineC, 4);
    CHECK(!seq.is_single_engine());
    CHECK_EQ(seq.active_engines(), 3u);

    constexpr int kSteps = 50;
    constexpr int kTotal = kSteps * 3;

    std::vector<const void*> rank_order[4];
    std::mutex rank_mu[4];

    std::atomic<bool> start{ false };

    auto run_engine_rank = [&](const void* owner, int rank) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        for (int i = 0; i < kSteps; ++i) {
            uint64_t t = seq.before_collective(owner, rank, 4, nullptr);
            {
                std::lock_guard<std::mutex> lk(rank_mu[rank]);
                rank_order[rank].push_back(owner);
            }
            seq.after_collective(owner, rank, 4, nullptr, t);
        }
    };

    std::vector<std::thread> th;
    th.reserve(12);
    for (const void* eng : { engineA, engineB, engineC }) {
        for (int r = 0; r < 4; ++r) {
            th.emplace_back(run_engine_rank, eng, r);
        }
    }

    start.store(true, std::memory_order_release);
    for (auto& t : th) t.join();

    for (int r = 0; r < 4; ++r) {
        REQUIRE_EQ(rank_order[r].size(), (size_t)kTotal);
    }

    for (size_t i = 0; i < (size_t)kTotal; ++i) {
        const void* expected = rank_order[0][i];
        CHECK_EQ(rank_order[1][i], expected);
        CHECK_EQ(rank_order[2][i], expected);
        CHECK_EQ(rank_order[3][i], expected);
    }

    seq.reset();
}

TEST(abort_releases_waiting_threads) {
    rad::CollectiveSequencer seq;
    const void* engineA = reinterpret_cast<const void*>(0x11);
    const void* engineB = reinterpret_cast<const void*>(0x22);

    seq.register_engine(engineA, 2);
    seq.register_engine(engineB, 2);

    std::atomic<bool> thread_started{ false };
    std::atomic<bool> thread_done{ false };

    /* Rank 1 thread of Engine B: will block waiting for a ticket from Rank 0 */
    std::thread t([&] {
        thread_started.store(true, std::memory_order_release);
        (void)seq.before_collective(engineB, 1, 2, nullptr);
        thread_done.store(true, std::memory_order_release);
    });

    while (!thread_started.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(!thread_done.load(std::memory_order_acquire));

    /* Abort must unblock the waiting thread immediately without hanging */
    seq.abort();
    t.join();
    CHECK(thread_done.load(std::memory_order_acquire));

    seq.reset();
}

RAD_TEST_MAIN()
