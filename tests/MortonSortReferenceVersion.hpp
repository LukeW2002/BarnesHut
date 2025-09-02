#pragma once
#include <string_view>
#include <algorithm>
#include "MortonSpec.hpp"
#include <vector>
#include <numeric>   
#include "MortonEncoder.h"


struct MortonSortReferenceSystem {
    static std::string_view name() { return "reference::encode_then_sort"; }

    static void load(MortonSpec::State& state, const MortonSpec::Input& in) {
        state.input = in;
    }

    static MortonSpec::Output execute(MortonSpec::State& state) {
        const auto& in = state.input;
        const std::size_t N = in.x.size();

        MortonSpec::Output out;
        out.keys.resize(N);
        out.indices.resize(N);
        std::iota(out.indices.begin(), out.indices.end(), 0);

        MortonEncoder::encode_morton_keys(in.x.data(), in.y.data(),
                                          out.keys.data(), N, in.world,
                                          /*enable_threading*/ false);

        std::stable_sort(out.indices.begin(), out.indices.end(),
            [&](std::size_t a, std::size_t b) {
                const auto ka = out.keys[a], kb = out.keys[b];
                if (ka != kb) return ka < kb;
                return a < b; 
            });

        // Repack keys in the sorted order, to match the mix’s “keys aligned with indices[k]”
        std::vector<std::uint64_t> sortedKeys(N);
        for (std::size_t k=0;k<N;++k) sortedKeys[k] = out.keys[out.indices[k]];
        out.keys.swap(sortedKeys);

        return out;
    }
};

using MortonSortReferenceVersion = VersionedPipelineTestKit::VersionFromSystem<MortonSpec, MortonSortReferenceSystem>;

static_assert( VersionedPipelineTestKit::VersionFor<MortonSortReferenceVersion, MortonSpec> );

