#include <gtest/gtest.h>

#include <string>

#include "VersionedPipelineTestKit.hpp"
#include "MortonSpec.hpp"
#include "MortonSortMixVersion.hpp"
#include "MortonSortReferenceVersion.hpp"
#include "../header/BarnesHutParticleSystem.h"
#include "../header/EventSystem.h"
#include "../header/Bounds.hpp"
#include "../header/MortonEncoder.h"
#include "../header/morton_sort_mix.hpp"

namespace vptk = VersionedPipelineTestKit;
using Mix = SortV1;

static MortonSpec::Output collect_output(const MortonSpec::State& st) {
    MortonSpec::Output out;
    const std::size_t N = st.input.x.size();
    out.indices = BHAccess::morton_indices_const(*st.bh);
    out.keys    = BHAccess::morton_keys_const(*st.bh);
    if (out.indices.size() > N) out.indices.resize(N);
    if (out.keys.size() > N) out.keys.resize(N);
    return out;
}

static void build_bh(MortonSpec::State& st,
                     const MortonSpec::Input& in,
                     bool enable_threading = false) {
    st.input = in;
    const std::size_t N = in.x.size();

    BarnesHutParticleSystem::Config cfg{};
    cfg.enable_threading = enable_threading;    

    st.bh = std::make_unique<BarnesHutParticleSystem>(N, st.bus, cfg);

    st.bh->set_boundary(in.world.min_x, in.world.max_x,
                        in.world.min_y, in.world.max_y);

    BHAccess::pos_x(*st.bh) = in.x;
    BHAccess::pos_y(*st.bh) = in.y;

    BHAccess::count(*st.bh) = N;
    BHAccess::morton_indices(*st.bh).resize(N);
    BHAccess::morton_keys(*st.bh).resize(N);
    BHAccess::indices_filled(*st.bh) = 0;
}


static MortonSpec::Output encode_then_sort_reference(const MortonSpec::Input& in) {
    const std::size_t N = in.x.size();
    MortonSpec::Output out;
    out.keys.resize(N);
    out.indices.resize(N);
    std::iota(out.indices.begin(), out.indices.end(), 0);

    MortonEncoder::encode_morton_keys(in.x.data(), in.y.data(),
                                      out.keys.data(), N, in.world,
                                      /* enable_threading */ false);

    std::stable_sort(out.indices.begin(), out.indices.end(),
        [&](std::size_t a, std::size_t b) {
            auto ka = out.keys[a], kb = out.keys[b];
            if (ka != kb) return ka < kb;
            return a < b; // tie-breaker
        });

    std::vector<std::uint64_t> sortedKeys(N);
    for (std::size_t k=0; k<N; ++k) sortedKeys[k] = out.keys[out.indices[k]];
    out.keys.swap(sortedKeys);
    return out;
}

// =============== EnsureIdxV1 =================

TEST(MortonPrimes_Ensure, FillsPermutationFromZero) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(33, 42u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    EnsureIdxV1::run(*st.bh, in.x.size());

    ASSERT_EQ(BHAccess::indices_filled(*st.bh), in.x.size());
    auto idx = BHAccess::morton_indices_const(*st.bh);
    std::vector<std::size_t> perm(idx.begin(), idx.begin() + in.x.size());
    std::sort(perm.begin(), perm.end());
    for (std::size_t i = 0; i < in.x.size(); ++i) EXPECT_EQ(perm[i], i);

    // positions untouched
    auto& xs = BHAccess::pos_x(*st.bh);
    auto& ys = BHAccess::pos_y(*st.bh);
    for (std::size_t i=0;i<in.x.size();++i){
        EXPECT_FLOAT_EQ(xs[i], in.x[i]);
        EXPECT_FLOAT_EQ(ys[i], in.y[i]);
    }
}

TEST(MortonPrimes_Ensure, ResumesWhenFilledLtN) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(16, 1u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    // Simulate partial pre-fill: [0..k)
    auto& idx = BHAccess::morton_indices(*st.bh);
    const std::size_t k = 5;
    std::iota(idx.begin(), idx.begin()+k, 0);
    BHAccess::indices_filled(*st.bh) = k;

    EnsureIdxV1::run(*st.bh, in.x.size());

    EXPECT_EQ(BHAccess::indices_filled(*st.bh), in.x.size());
    // Now expect idx == [0,1,2,3,4, 5,6,7,...,15]
    for (std::size_t i=0;i<in.x.size();++i) EXPECT_EQ(idx[i], i);
}

TEST(MortonPrimes_Ensure, ReinitWhenFilledGeN) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(8, 2u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    // Mark as "overfilled": should rewrite 0..N-1
    BHAccess::indices_filled(*st.bh) = in.x.size() + 10;
    auto& idx = BHAccess::morton_indices(*st.bh);
    std::fill(idx.begin(), idx.end(), 999u);

    EnsureIdxV1::run(*st.bh, in.x.size());

    EXPECT_EQ(BHAccess::indices_filled(*st.bh), in.x.size());
    for (std::size_t i=0;i<in.x.size();++i) EXPECT_EQ(idx[i], i);
}


// ---------- EncodeKeysV1 ----------

TEST(MortonPrimes_Encode, KeysMatchReference_Grid) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(1000, 7u, MortonSpec::Dist::Grid);
    build_bh(st, in);

    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);

    auto keys = BHAccess::morton_keys_const(*st.bh);
    std::vector<std::uint64_t> ref(in.x.size());
    MortonEncoder::encode_morton_keys(in.x.data(), in.y.data(),
                                      ref.data(), in.x.size(), in.world,
                                      /*enable_threading*/ false);
    for (std::size_t i=0;i<in.x.size();++i) EXPECT_EQ(keys[i], ref[i]);
}

TEST(MortonPrimes_Encode, ThreadingFlagDoesNotChangeKeys) {
    MortonSpec::State st1, st2;
    auto in = MortonSpec::gen_input_dist(4096, 9u, MortonSpec::Dist::Uniform);
    build_bh(st1, in, /*enable_threading=*/false);  
    build_bh(st2, in, /*enable_threading=*/true);  


    EnsureIdxV1::run(*st1.bh, in.x.size());
    EnsureIdxV1::run(*st2.bh, in.x.size());
    EncodeKeysV1::run(*st1.bh, in.world);
    EncodeKeysV1::run(*st2.bh, in.world);

    auto k1 = BHAccess::morton_keys_const(*st1.bh);
    auto k2 = BHAccess::morton_keys_const(*st2.bh);
    ASSERT_EQ(k1.size(), k2.size());
    for (std::size_t i=0;i<in.x.size();++i) EXPECT_EQ(k1[i], k2[i]);
}

// ---------- RadixSortV1 ----------

TEST(MortonPrimes_Sort, SortsByKey_PermutationOK) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(2000, 99u, MortonSpec::Dist::Duplicates); // big tie buckets
    build_bh(st, in);

    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);

    auto keys_before = BHAccess::morton_keys_const(*st.bh);
    RadixSortV1::run(*st.bh);

    auto idx = BHAccess::morton_indices_const(*st.bh);
    ASSERT_GE(idx.size(), in.x.size());

    // Permutation
    std::vector<std::size_t> perm(idx.begin(), idx.begin()+in.x.size());
    std::sort(perm.begin(), perm.end());
    for (std::size_t i=0;i<in.x.size();++i) EXPECT_EQ(perm[i], i);

    // Nondecreasing by the pre-encoded keys
    for (std::size_t k=1;k<in.x.size();++k) {
        auto prev = keys_before[idx[k-1]];
        auto curr = keys_before[idx[k]];
        EXPECT_LE(prev, curr);
    }
}


// ---------- Sort_by_morton_key ----------
TEST(MortonImpl, EarlyReturnOnZeroCount) {
    MortonSpec::State st;
    MortonSpec::Input in;
    in.world = geom::AABBf{-1.f,-1.f,1.f,1.f};
    // empty x,y -> N = 0
    build_bh(st, in);

    // Pre-mark some sentinel values we can check after
    BHAccess::indices_filled(*st.bh) = 123;

    // Should do nothing (and not crash)
    sort_by_morton_key_impl<Mix>(*st.bh);

    // Count is still zero; indices_filled unchanged
    EXPECT_EQ(BHAccess::count(*st.bh), 0u);
    EXPECT_EQ(BHAccess::indices_filled(*st.bh), 123u);
    // Keys/indices vectors can be any size here; we only require “no crash”.
}

static void build_bh_with_capacity(MortonSpec::State& st,
                                   const MortonSpec::Input& in,
                                   std::size_t capacity,
                                   bool enable_threading=false) {
    st.input = in;
    BarnesHutParticleSystem::Config cfg{}; cfg.enable_threading = enable_threading;
    st.bh = std::make_unique<BarnesHutParticleSystem>(capacity, st.bus, cfg);
    st.bh->set_boundary(in.world.min_x, in.world.max_x, in.world.min_y, in.world.max_y);
    BHAccess::pos_x(*st.bh) = in.x;
    BHAccess::pos_y(*st.bh) = in.y;
    BHAccess::count(*st.bh) = in.x.size();
    BHAccess::morton_indices(*st.bh).resize(in.x.size());
    BHAccess::morton_keys(*st.bh).clear();          // < N to force resize
    BHAccess::indices_filled(*st.bh) = 0;
}

TEST(MortonImpl, ResizesKeysToCapacityWhenTooSmall) {
    auto in = MortonSpec::gen_input_dist(100, 1u, MortonSpec::Dist::Uniform);
    MortonSpec::State st;
    const std::size_t capacity = 512; // > N
    build_bh_with_capacity(st, in, capacity);

    ASSERT_LT(BHAccess::morton_keys(*st.bh).size(), in.x.size());

    sort_by_morton_key_impl<Mix>(*st.bh);

    // Contract: size becomes max_particles (capacity), not just N
    EXPECT_EQ(BHAccess::morton_keys(*st.bh).size(), capacity);

    // And pipeline did the job
    auto out = collect_output(st);
    MortonSpec::check_invariants(in, out);
}

TEST(MortonImpl, UsesBHBoundsForEncoding) {
    // Start with uniform points, then set custom bounds on the BH
    auto in = MortonSpec::gen_input_dist(1000, 12u, MortonSpec::Dist::Grid);
    MortonSpec::State stA, stB;

    // Build both BHs with SAME positions but DIFFERENT bounds than in.world
    geom::AABBf custom{ -2.f, -3.f, 4.f, 5.f };
    {
        build_bh(stA, in);
        stA.bh->set_boundary(custom.min_x, custom.max_x, custom.min_y, custom.max_y);
        BHAccess::morton_keys(*stA.bh).clear(); // exercise resize path sometimes
    }
    {
        build_bh(stB, in);
        stB.bh->set_boundary(custom.min_x, custom.max_x, custom.min_y, custom.max_y);
    }

    // Run wrapper (grabs bounds from BH)
    sort_by_morton_key_impl<Mix>(*stA.bh);

    // Manual 3-stage using the SAME custom bounds
    EnsureIdxV1::run(*stB.bh, in.x.size());
    EncodeKeysV1::run(*stB.bh, custom);
    RadixSortV1::run(*stB.bh);

    auto A = collect_output(stA);
    auto B = collect_output(stB);

    // Keys-only: tie order may differ
    EXPECT_EQ(A.keys, B.keys);
    // And invariants still hold
    EXPECT_NO_THROW(MortonSpec::check_invariants(in, A));
}

TEST(MortonImpl, InvariantsAcrossDistributions_AndSizes) {
    for (auto N : {0uz, 1uz, 2uz, 8uz, 31uz, 32uz, 33uz, 1000uz}) {
        for (auto d : {MortonSpec::Dist::Uniform,
                       MortonSpec::Dist::Duplicates,
                       MortonSpec::Dist::Grid}) {
            MortonSpec::State st;
            auto in = MortonSpec::gen_input_dist(N, 42u, d);
            build_bh(st, in);

            sort_by_morton_key_impl<Mix>(*st.bh);

            auto out = collect_output(st);
            EXPECT_NO_THROW(MortonSpec::check_invariants(in, out)) << "N="<<N;
        }
    }
}

TEST(MortonImpl, KeysMatchReference_KeysOnly) {
    for (auto d : {MortonSpec::Dist::Uniform,
                   MortonSpec::Dist::Duplicates,
                   MortonSpec::Dist::Grid}) {
        MortonSpec::State st;
        auto in = MortonSpec::gen_input_dist(5000, 99u, d);
        build_bh(st, in);

        sort_by_morton_key_impl<Mix>(*st.bh);

        auto A = collect_output(st);
        auto B = encode_then_sort_reference(in); // uses in.world
        EXPECT_EQ(A.keys, B.keys);
    }
}

TEST(MortonImpl, DoesNotMutatePositions_FillsIndices) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(128, 5u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    auto xs0 = BHAccess::pos_x(*st.bh);
    auto ys0 = BHAccess::pos_y(*st.bh);

    sort_by_morton_key_impl<Mix>(*st.bh);

    // positions unchanged
    auto& xs = BHAccess::pos_x(*st.bh);
    auto& ys = BHAccess::pos_y(*st.bh);
    ASSERT_EQ(xs.size(), xs0.size());
    for (std::size_t i=0;i<xs.size();++i) {
        EXPECT_FLOAT_EQ(xs[i], xs0[i]);
        EXPECT_FLOAT_EQ(ys[i], ys0[i]);
    }

    // filled == N and indices is a permutation
    const std::size_t N = in.x.size();
    EXPECT_EQ(BHAccess::indices_filled(*st.bh), N);
    auto idx = BHAccess::morton_indices_const(*st.bh);
    std::vector<std::size_t> perm(idx.begin(), idx.begin()+N);
    std::sort(perm.begin(), perm.end());
    for (std::size_t i=0;i<N;++i) EXPECT_EQ(perm[i], i);
}

static int morton_total_bits() {
    return static_cast<int>(BHAccess::morton_total_bits());
}

static int morton_max_depth() {              // number of 2-bit levels
    return morton_total_bits() / 2;
}

// ---------- Prime 1: CalcParamsV1 ----------
TEST(MortonPrimes_Params, ShiftAndMaskFormulaAcrossDepths) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(1, 1u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    const int total = morton_total_bits();
    const int maxd  = morton_max_depth();      // valid depths: [0, maxd-1]

    std::vector<int> depths = {0, 1, std::max(0, maxd/2 - 1), maxd-1};
    for (int d : depths) {
        auto p = CalcParamsV1::run(*st.bh, d);
        const int expected_shift = total - 2 * (d + 1);
        const uint64_t expected_mask = (3ULL << expected_shift);

        EXPECT_EQ(p.level_shift, expected_shift) << "depth=" << d;
        EXPECT_EQ(p.mask, expected_mask) << "depth=" << d;
        EXPECT_GE(p.level_shift, 0) << "depth=" << d;     // last level should be shift==0
    }
}

// ---------- Prime 2: ExtractQuadV1 ----------
TEST(MortonPrimes_Extract, ConsistentWithBitMath_Depth0and1) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(256, 7u, MortonSpec::Dist::Grid);
    build_bh(st, in);

    // prepare sorted state 
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    // test at two depths: top (0) and next (1)
    for (int depth : {0, 1}) {
        auto params = CalcParamsV1::run(*st.bh, depth);

        auto idx  = BHAccess::morton_indices_const(*st.bh);
        auto keys = BHAccess::morton_keys_const(*st.bh);

        // sample across the range (all also fine, but sampling keeps it quick)
        for (std::size_t k = 0; k < idx.size(); k += 17) {
            const std::size_t g = idx[k];                    // global index
            const auto key = keys[g];
            const int expected = static_cast<int>((key & params.mask) >> params.level_shift);
            const int got      = ExtractQuadV1::run(*st.bh, k, params);

            ASSERT_GE(expected, 0); ASSERT_LE(expected, 3);
            EXPECT_EQ(got, expected) << "k=" << k << " depth=" << depth;
        }
    }
}

// ---------- Prime 3: ZToChildV1 ----------
TEST(MortonPrimes_ZToChild, MappingIsCorrectForAllQuadrants) {
    // Spec: z order [SW(0), NW(1), SE(2), NE(3)] -> canonical [SW, SE, NW, NE] == [0,2,1,3]
    static const int expect[4] = {0, 2, 1, 3};
    for (int z = 0; z < 4; ++z) {
        EXPECT_EQ(ZToChildV1::run(z), expect[z]) << "z=" << z;
    }
}

// ---------- Prime 4: SeqEndV1 ----------
TEST(MortonPrimes_SeqEnd, FindsEndOfRun_TopLevel) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(1024, 11u, MortonSpec::Dist::Grid);
    build_bh(st, in);

    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    const std::size_t N = in.x.size();
    auto idx  = BHAccess::morton_indices_const(*st.bh);
    auto keys = BHAccess::morton_keys_const(*st.bh);

    // Depth 0 groups by top-level quadrant; we’ll find the first run starting at 0
    auto params  = CalcParamsV1::run(*st.bh, /*depth=*/0);
    const int q0 = ExtractQuadV1::run(*st.bh, /*particle_index=*/0, params);

    const std::size_t end = SeqEndV1::run(*st.bh, /*start=*/0, /*last=*/N-1, q0, params);

    // 1) All elements in [0, end) have the same quadrant
    for (std::size_t k = 0; k < end; ++k) {
        const int q = ExtractQuadV1::run(*st.bh, k, params);
        EXPECT_EQ(q, q0) << "k=" << k;
    }
    // 2) Boundary condition: end is either N or the first index with different quadrant
    if (end < N) {
        const int q_end = ExtractQuadV1::run(*st.bh, end, params);
        EXPECT_NE(q_end, q0);
    }
    // Sanity: end is at least 1 (SeqEnd contract returns start+1 minimum)
    EXPECT_GE(end, 1u);
}

TEST(MortonPrimes_SeqEnd, EdgeCases_StartEqualsLast_And_LastIndex) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(64, 12u, MortonSpec::Dist::Grid);
    build_bh(st, in);

    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    const std::size_t N = in.x.size();
    auto params = CalcParamsV1::run(*st.bh, /*depth=*/1);

    // start==last -> should return last+1
    {
        const std::size_t start = 10, last = 10;
        const int q = ExtractQuadV1::run(*st.bh, start, params);
        const std::size_t end = SeqEndV1::run(*st.bh, start, last, q, params);
        EXPECT_EQ(end, last + 1);
    }

    // starting at last-1 through last
    {
        const std::size_t start = N - 2, last = N - 1;
        const int q = ExtractQuadV1::run(*st.bh, start, params);
        const std::size_t end = SeqEndV1::run(*st.bh, start, last, q, params);
        EXPECT_TRUE(end == last || end == last + 1);
    }
}

// Prime 5: InitRangesV1
TEST(MortonPrimes_Ranges, InitRanges_ReturnsSentinels) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(4, 1u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    auto ranges = InitRangesV1::run(*st.bh);
    ASSERT_EQ(ranges.size(), 4u);
    for (size_t c = 0; c < ranges.size(); ++c) {
        EXPECT_EQ(ranges[c].first,  SIZE_MAX) << "slot " << c;
        EXPECT_EQ(ranges[c].second, SIZE_MAX) << "slot " << c;
    }
}

// Prime 6: IsValidV1
TEST(MortonPrimes_Ranges, IsValid_Basic) {
    EXPECT_TRUE(IsValidV1::run(0,0));
    EXPECT_TRUE(IsValidV1::run(0,5));
    EXPECT_TRUE(IsValidV1::run(5,5));
    EXPECT_FALSE(IsValidV1::run(7,6)); // first > last
}

// Prime 7: SetRangeV1
TEST(MortonPrimes_Ranges, SetRange_ModifiesOnlyChosenSlot) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(10, 2u, MortonSpec::Dist::Uniform);
    build_bh(st, in);

    auto ranges = InitRangesV1::run(*st.bh);
    // set child slot 2
    SetRangeV1::run(ranges, /*child_slot=*/2, /*start=*/3, /*end=*/7);

    // only slot 2 should change
    for (int c = 0; c < 4; ++c) {
        if (c == 2) {
            EXPECT_EQ(ranges[c].first,  3u);
            EXPECT_EQ(ranges[c].second, 7u);
        } else {
            EXPECT_EQ(ranges[c].first,  SIZE_MAX);
            EXPECT_EQ(ranges[c].second, SIZE_MAX);
        }
    }

    SetRangeV1::run(ranges, /*child_slot=*/0, /*start=*/0, /*end=*/2);
    EXPECT_EQ(ranges[0].first, 0u);
    EXPECT_EQ(ranges[0].second, 2u);
    EXPECT_EQ(ranges[2].first, 3u);
    EXPECT_EQ(ranges[2].second,7u);
    EXPECT_EQ(ranges[1].first, SIZE_MAX);
    EXPECT_EQ(ranges[3].first, SIZE_MAX);
}

// Prime 8: ValidateV1 
TEST(MortonPrimes_Ranges, Validate_CorrectSplitTrue) {
    const size_t first = 0, last = 11;
    RangeArr ranges = InitRangesV1::run(*((MortonSpec::State{}).bh.get())); 
    SetRangeV1::run(ranges, 0, 0, 2);   // size 3
    SetRangeV1::run(ranges, 1, 3, 5);   // size 3
    SetRangeV1::run(ranges, 2, 6, 11);  // size 6  → total 12 == last-first+1
    EXPECT_TRUE(ValidateV1::run(ranges, first, last));
}

static RangeArr compute_expected_ranges(const MortonSpec::State& st,
                                        size_t first, size_t last, int depth)
{
    RangeArr exp = InitRangesV1::run(*st.bh);

    // params for this depth
    auto params = CalcParamsV1::run(*st.bh, depth);

    const auto idx  = BHAccess::morton_indices_const(*st.bh);
    const auto keys = BHAccess::morton_keys_const(*st.bh);

    if (first > last) return exp; // empty

    // Start first run
    size_t run_start = first;
    auto key0 = keys[idx[first]];
    int zq_prev = static_cast<int>((key0 & params.mask) >> params.level_shift);
    int child_prev = ZToChildV1::run(zq_prev);

    for (size_t i = first + 1; i <= last; ++i) {
        auto key = keys[idx[i]];
        int zq = static_cast<int>((key & params.mask) >> params.level_shift);
        if (zq != zq_prev) {
            // close previous run
            auto &slot = exp[child_prev];
            // at a fixed depth over a contiguous [first..last], each child should appear at most once
            EXPECT_EQ(slot.first,  SIZE_MAX) << "child appears twice at depth " << depth;
            EXPECT_EQ(slot.second, SIZE_MAX) << "child appears twice at depth " << depth;
            slot = { run_start, i - 1 };
            // start new
            run_start = i;
            zq_prev = zq;
            child_prev = ZToChildV1::run(zq_prev);
        }
    }
    // close final run
    auto &slot = exp[child_prev];
    EXPECT_EQ(slot.first,  SIZE_MAX);
    EXPECT_EQ(slot.second, SIZE_MAX);
    slot = { run_start, last };

    return exp;
}

TEST(MortonRangeSplit, InvalidRange_ReturnsSentinels) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(16, 1u, MortonSpec::Dist::Uniform);
    build_bh(st, in);
    // prepare sorted state 
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    using RS = RangeSplitPrimes</*defaults*/>;
    auto ranges = RS::split_morton_range_CT(*st.bh, /*first*/ 10, /*last*/ 5, /*depth*/ 0);

    for (int c=0;c<4;++c) {
        EXPECT_EQ(ranges[c].first,  SIZE_MAX);
        EXPECT_EQ(ranges[c].second, SIZE_MAX);
    }
}

TEST(MortonRangeSplit, NZero_EarlyOut_NoCrash) {
    MortonSpec::State st;
    MortonSpec::Input in;
    in.world = geom::AABBf{-1.f,-1.f,1.f,1.f};
    build_bh(st, in);               // N == 0

    using RS = RangeSplitPrimes</*defaults*/>;
    auto ranges = RS::split_morton_range_CT(*st.bh, 1, 0, 0);  // <-- empty range

    for (int c=0; c<4; ++c) {
        EXPECT_EQ(ranges[c].first,  SIZE_MAX);
        EXPECT_EQ(ranges[c].second, SIZE_MAX);
    }
}


TEST(MortonRangeSplit, FullArray_Depth0_MatchesExpected) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(1024, 11u, MortonSpec::Dist::Grid);
    build_bh(st, in);
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    const size_t N = in.x.size();
    using RS = RangeSplitPrimes</*defaults*/>;
    auto got = RS::split_morton_range_CT(*st.bh, 0, N ? N-1 : 0, /*depth*/ 0);
    auto exp = compute_expected_ranges(st, 0, N ? N-1 : 0, /*depth*/ 0);

    // Compare slot-by-slot (SIZE_MAX sentinels allowed)
    for (int c=0;c<4;++c) {
        EXPECT_EQ(got[c].first,  exp[c].first)  << "slot " << c;
        EXPECT_EQ(got[c].second, exp[c].second) << "slot " << c;
    }

    EXPECT_TRUE(ValidateV1::run(got, 0, N ? N-1 : 0));
}
TEST(MortonRangeSplit, Subrange_CrossesBoundaries_Depth0) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(512, 23u, MortonSpec::Dist::Grid);
    build_bh(st, in);
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    const size_t N = in.x.size();
    // choose a subrange well inside [0..N-1]
    const size_t first = N/8;
    const size_t last  = N - N/7;

    using RS = RangeSplitPrimes</*defaults*/>;
    auto got = RS::split_morton_range_CT(*st.bh, first, last, /*depth*/ 0);
    auto exp = compute_expected_ranges(st, first, last, /*depth*/ 0);

    for (int c=0;c<4;++c) {
        EXPECT_EQ(got[c].first,  exp[c].first)  << "slot " << c;
        EXPECT_EQ(got[c].second, exp[c].second) << "slot " << c;
    }
    EXPECT_TRUE(ValidateV1::run(got, first, last));
}
TEST(MortonRangeSplit, Nested_ChildRange_Depth1) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(1024, 5u, MortonSpec::Dist::Grid);
    build_bh(st, in);
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    using RS = RangeSplitPrimes</*defaults*/>;

    // depth 0 split of the full span
    auto top = RS::split_morton_range_CT(*st.bh, 0, in.x.size()-1, /*depth*/ 0);

    // pick the first present child slot
    int chosen = -1;
    for (int c=0;c<4;++c) {
        if (top[c].first != SIZE_MAX) { chosen = c; break; }
    }
    ASSERT_NE(chosen, -1);
    auto [first, last] = top[chosen];

    // Now split that child range at depth 1
    auto got = RS::split_morton_range_CT(*st.bh, first, last, /*depth*/ 1);
    auto exp = compute_expected_ranges(st, first, last, /*depth*/ 1);

    for (int c=0;c<4;++c) {
        EXPECT_EQ(got[c].first,  exp[c].first)  << "slot " << c;
        EXPECT_EQ(got[c].second, exp[c].second) << "slot " << c;
    }
    EXPECT_TRUE(ValidateV1::run(got, first, last));
}
// Fake mapper: swap SE(2) <-> NW(1)
struct ZToChildSwap12 {
    static int run(int zq) {
        static constexpr int map[4] = {0, 2, 1, 3}; // baseline
        int child = map[zq];
        if (child == 1) return 2;
        if (child == 2) return 1;
        return child;
    }
};

TEST(MortonRangeSplit, Plugin_PrimeSwap_ChangesSlotsButCovers) {
    MortonSpec::State st;
    auto in = MortonSpec::gen_input_dist(2048, 3u, MortonSpec::Dist::Grid);
    build_bh(st, in);
    EnsureIdxV1::run(*st.bh, in.x.size());
    EncodeKeysV1::run(*st.bh, in.world);
    RadixSortV1::run(*st.bh);

    using RS_Default = RangeSplitPrimes</*CalcParams*/CalcParamsV1, /*Extract*/ExtractQuadV1,
                                        /*SeqEnd*/SeqEndV1, /*Init*/InitRangesV1,
                                        /*ZToChild*/ZToChildV1, /*IsValid*/IsValidV1,
                                        /*SetRange*/SetRangeV1, /*Validate*/ValidateV1>;

    using RS_Swap = RangeSplitPrimes</*CalcParams*/CalcParamsV1, /*Extract*/ExtractQuadV1,
                                     /*SeqEnd*/SeqEndV1, /*Init*/InitRangesV1,
                                     /*ZToChild*/ZToChildSwap12, /*IsValid*/IsValidV1,
                                     /*SetRange*/SetRangeV1, /*Validate*/ValidateV1>;

    auto a = RS_Default::split_morton_range_CT(*st.bh, 0, in.x.size()-1, 0);
    auto b = RS_Swap::split_morton_range_CT(*st.bh, 0, in.x.size()-1, 0);

    // Coverage still correct
    EXPECT_TRUE(ValidateV1::run(a, 0, in.x.size()-1));
    EXPECT_TRUE(ValidateV1::run(b, 0, in.x.size()-1));

    // Slot assignments differ (at least one slot moved) unless a quadrant is empty
    bool any_diff = false;
    for (int c=0;c<4;++c)
        if (a[c] != b[c] && a[c].first != SIZE_MAX && b[c].first != SIZE_MAX)
            any_diff = true;
    EXPECT_TRUE(any_diff);
}

