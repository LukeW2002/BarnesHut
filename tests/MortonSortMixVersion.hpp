#pragma once
#include <string_view>
#include "../header/morton_sort_mix.hpp"    
#include "MortonSpec.hpp"      

using Mix = SortV1;

struct MortonSortMixSystem{
    static std::string_view name() {
        return "morton_sort_mix::SortV1";
    }

    static void load(MortonSpec::State& state, const MortonSpec::Input& in){
        state.input = in;
        const std::size_t N = in.x.size();

        state.bh = std::make_unique<BarnesHutParticleSystem>(N, state.bus);
        state.bh->set_boundary(in.world.min_x, in.world.max_x, in.world.min_y, in.world.max_y);
        auto& bx = BHAccess::pos_x(*state.bh);
        auto& by = BHAccess::pos_y(*state.bh);

        bx = in.x;
        by = in.y;

        BHAccess::count(*state.bh) = N;
        BHAccess::morton_indices(*state.bh).resize(N);  
        BHAccess::morton_keys(*state.bh).resize(N);     
        BHAccess::indices_filled(*state.bh) = 0;        
    }

    static MortonSpec::Output execute(MortonSpec::State& state) {
        const std::size_t N = state.input.x.size();

        Mix::ensure(*state.bh, N);
        Mix::encode(*state.bh, state.input.world);
        Mix::sort(*state.bh);

        MortonSpec::Output out;
        out.indices = BHAccess::morton_indices_const(*state.bh);
        out.keys    = BHAccess::morton_keys_const(*state.bh);
        if (out.indices.size() > N) out.indices.resize(N);
        if (out.keys.size()    > N) out.keys.resize(N);
        return out;
  }
};
using MortonSortMixVersion = VersionedPipelineTestKit::VersionFromSystem<MortonSpec, MortonSortMixSystem>;
