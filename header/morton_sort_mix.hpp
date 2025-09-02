#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <numeric>      
#include <array>
#include <cstddef>
#include "Bounds.hpp"
#include "MortonEncoder.h"

#include "BarnesHutParticleSystem.h"   

struct BHAccess {
    using BH = BarnesHutParticleSystem;

    static std::vector<size_t>&        morton_indices(BH& s)   { return s.morton_indices_; }
    static std::vector<uint64_t>&      morton_keys(BH& s)      { return s.morton_keys_; }
    static std::vector<float>&         pos_x(BH& s)            { return s.positions_x_; }
    static std::vector<float>&         pos_y(BH& s)            { return s.positions_y_; }
    static size_t&                     indices_filled(BH& s)   { return s.indices_filled_; }
    static size_t&                     count(BH& s)            { return s.particle_count_; }
    static const BH::Config&           config(const BH& s)     { return s.config_; }
    static void                        radix_sort(BH& s)       { s.radix_sort_indices(); }
    static float                       minx(const BH& s)       { return s.bounds_min_x_; }
    static float                       miny(const BH& s)       { return s.bounds_min_y_; }
    static float                       maxx(const BH& s)       { return s.bounds_max_x_; }
    static float                       maxy(const BH& s)       { return s.bounds_max_y_; }
    static size_t                      max_particles(const BH& s){ return s.max_particles_; }

    static const std::vector<size_t>&   morton_indices_const(const BH& s) { return s.morton_indices_; }
    static const std::vector<uint64_t>& morton_keys_const(const BH& s)    { return s.morton_keys_; }
    static constexpr int                morton_total_bits()                { return BH::MORTON_TOTAL_BITS; }
};

// prime
struct EnsureIdxV1 {
  static inline void run(BHAccess::BH& s, size_t N) {
    auto& idx = BHAccess::morton_indices(s);
    auto& filled = BHAccess::indices_filled(s);
    if (filled < N) {
      std::iota(idx.begin() + filled, idx.begin() + N, filled);
    } else {
      std::iota(idx.begin(), idx.begin() + N, 0);
    }
    filled = N;
  }
};

// Prime
struct EncodeKeysV1 {
  static inline void run(BHAccess::BH& s, const geom::AABBf& world) {
    MortonEncoder::encode_morton_keys(
      BHAccess::pos_x(s).data(),
      BHAccess::pos_y(s).data(),
      BHAccess::morton_keys(s).data(),
      BHAccess::count(s),
      world,
      BHAccess::config(s).enable_threading
    );
  }
};

// PRIME
struct RadixSortV1 {
  static inline void run(BHAccess::BH& s) { BHAccess::radix_sort(s); }
};

template<class Ensure, class Encode, class Radix>
struct Primes {
  static inline void ensure(BHAccess::BH& s, size_t N)           { Ensure::run(s, N); }
  static inline void encode(BHAccess::BH& s, const geom::AABBf& b){ Encode::run(s, b); }
  static inline void sort  (BHAccess::BH& s)                     { Radix::run(s); }
};

using SortV1 = Primes<EnsureIdxV1, EncodeKeysV1, RadixSortV1>;

template<class Mix>
inline void sort_by_morton_key_impl(BHAccess::BH& s) {
  const size_t N = BHAccess::count(s);
  if (!N) return;

  if (BHAccess::morton_keys(s).size() < N)
    BHAccess::morton_keys(s).resize(BHAccess::max_particles(s));

  const geom::AABBf world{ BHAccess::minx(s), BHAccess::miny(s),
                           BHAccess::maxx(s), BHAccess::maxy(s) };

  Mix::ensure(s, N);
  Mix::encode(s, world);
  Mix::sort(s);
}

// ===================================================================
// MORTON RANGE SPLITTING
// ===================================================================

// Type aliases for compile-time injection pattern
struct MortonLevelParams { 
    int level_shift; 
    uint64_t mask; 
};
using MLP = MortonLevelParams;
using Range    = std::pair<size_t, size_t>;
using RangeArr = std::array<Range, 4>;
using Self     = BarnesHutParticleSystem;

// Prime function 1: Calculate bit extraction parameters for a tree level
// Contract: depth >= 0 -> valid shift and mask for 2-bit quadrant extraction
struct CalcParamsV1 {
    struct MortonLevelParams { int level_shift; uint64_t mask; };
    
    static MortonLevelParams run(const Self& /*s*/, int depth) {
        const int level_shift = BHAccess::morton_total_bits() - 2 * (depth + 1);
        const uint64_t mask = 3ULL << level_shift;
        return {level_shift, mask};
    }
};

// Prime function 2: Extract quadrant from Morton key  
// Contract: valid particle_index -> quadrant in Z-order [0,3]
struct ExtractQuadV1 {
    using Params = CalcParamsV1::MortonLevelParams;
    
    static int run(const Self& s, size_t particle_index, const Params& params) {
        const auto& indices = BHAccess::morton_indices_const(s);
        const auto& keys = BHAccess::morton_keys_const(s);
        const size_t global_index = indices[particle_index];
        const uint64_t morton_key = keys[global_index];
        return static_cast<int>((morton_key & params.mask) >> params.level_shift);
    }
};

// Prime function 3: Convert Z-order quadrant to canonical child slot
// Contract: z_quadrant in [0,3] -> canonical child [0,3] for [SW, SE, NW, NE]
struct ZToChildV1 {
    static int run(int z_quadrant) {
        static constexpr int z_to_child[4] = {0, 2, 1, 3}; // SW, NW, SE, NE -> SW, SE, NW, NE
        return z_to_child[z_quadrant];
    }
};

// Prime function 4: Find end of contiguous quadrant sequence
// Contract: start <= last, valid quadrant -> first index where quadrant changes
struct SeqEndV1 {
    using Params = CalcParamsV1::MortonLevelParams;
    
    static size_t run(const Self& s, size_t start, size_t last, int expected_quadrant, const Params& params) {
        size_t end = start + 1;
        while (end <= last) {
            const int current_quadrant = ExtractQuadV1::run(s, end, params);
            if (current_quadrant != expected_quadrant) {
                break;
            }
            ++end;
        }
        return end;
    }
};

// Prime function 5: Initialize empty range array
// Contract: returns 4-element array with sentinel values {SIZE_MAX, SIZE_MAX}
struct InitRangesV1 {
    static RangeArr run(const Self& /*s*/) {
        RangeArr ranges;
        for (auto& r : ranges) {
            r = {SIZE_MAX, SIZE_MAX};
        }
        return ranges;
    }
};

// Prime function 6: Validate range bounds
// Contract: deterministic bounds check first <= last
struct IsValidV1 {
    static bool run(size_t first, size_t last) {
        return first <= last;
    }
};

// Prime function 7: Set range in result array
// Contract: child_slot in [0,3] -> modify only ranges[child_slot]
struct SetRangeV1 {
    static void run(RangeArr& ranges, int child_slot, size_t range_start, size_t range_end) {
        ranges[child_slot] = {range_start, range_end};
    }
};

// Prime function 8: Validate split completeness (debug only)
// Contract: sum of range sizes equals expected total
struct ValidateV1 {
    #ifndef NDEBUG
    static bool run(const RangeArr& ranges, size_t first, size_t last) {
        size_t total = 0;
        for (const auto& [range_start, range_end] : ranges) {
            if (range_start != SIZE_MAX) {
                total += (range_end - range_start + 1);
            }
        }
        return total == (last - first + 1);
    }
    #else
    static bool run(const RangeArr& /*ranges*/, size_t /*first*/, size_t /*last*/) {
        return true; // No-op in release builds
    }
    #endif
};

template<
  class CalcParams  = CalcParamsV1,
  class ExtractQuad = ExtractQuadV1, 
  class SeqEnd      = SeqEndV1,
  class InitRanges  = InitRangesV1,
  class ZToChild    = ZToChildV1,
  class IsValid     = IsValidV1,
  class SetRange    = SetRangeV1,
  class Validate    = ValidateV1
>
struct RangeSplitPrimes {
    using Params = typename CalcParams::MortonLevelParams;
    
    static RangeArr split_morton_range_CT(const Self& s, size_t first, size_t last, int depth) {
        // Prime 6: validate input range
        if (!IsValid::run(first, last)) {
            // Prime 5: empty result  
            return InitRanges::run(s);
        }
        
        // Prime 5: init result
        RangeArr ranges = InitRanges::run(s);
        
        // Prime 1: level params
        const Params params = CalcParams::run(s, depth);
        
        size_t i = first;
        while (i <= last) {
            // Prime 2: quadrant at i
            const int zq = ExtractQuad::run(s, i, params);
            
            // Prime 3: map to canonical child slot
            const int child = ZToChild::run(zq);
            
            // Prime 4: find end of this quadrant run
            const size_t end_excl = SeqEnd::run(s, i, last, zq, params);
            const size_t end_incl = end_excl - 1;
            
            // Prime 7: write the range
            SetRange::run(ranges, child, i, end_incl);
            
            i = end_excl;
        }
        
        // Prime 8: debug completeness check
        assert(Validate::run(ranges, first, last));
        
        return ranges;
    }
};

using RangeSplitV1 = RangeSplitPrimes<>;

template<class RangeSplit = RangeSplitV1>
inline RangeArr split_morton_range_impl(const Self& s, size_t first, size_t last, int depth) {
    return RangeSplit::split_morton_range_CT(s, first, last, depth);
}

struct ZToChildIdentity {
    static int run(int z_quadrant) { return z_quadrant; }
};

struct ExtractQuadFake {
    using Params = CalcParamsV1::MortonLevelParams;
    static int run(const Self& /*s*/, size_t idx, const Params& /*params*/) {
        return static_cast<int>(idx & 3); // Alternating 0,1,2,3 pattern
    }
};

struct ValidateNoOp {
    static bool run(const RangeArr& /*ranges*/, size_t /*first*/, size_t /*last*/) {
        return true;
    }
};

using RangeSplitIdentityMap = RangeSplitPrimes<CalcParamsV1, ExtractQuadV1, SeqEndV1, InitRangesV1,
                                               ZToChildIdentity, IsValidV1, SetRangeV1, ValidateV1>;

using RangeSplitFakeExtract = RangeSplitPrimes<CalcParamsV1, ExtractQuadFake, SeqEndV1, InitRangesV1,
                                               ZToChildV1, IsValidV1, SetRangeV1, ValidateV1>;

using RangeSplitNoValidate  = RangeSplitPrimes<CalcParamsV1, ExtractQuadV1, SeqEndV1, InitRangesV1,
                                               ZToChildV1, IsValidV1, SetRangeV1, ValidateNoOp>;
