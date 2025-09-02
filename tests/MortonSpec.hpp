#pragma once
#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>

#include "../header/BarnesHutParticleSystem.h"
#include "../header/EventSystem.h"
#include "../header/Bounds.hpp"
#include "../header/MortonEncoder.h"
#include "VersionedPipelineTestKit.hpp" 


struct MortonSpec {
    struct Input {
        std::vector<float> x,y;
        geom::AABBf world;
    };

    struct State {
        EventBus bus;
        std::unique_ptr<BarnesHutParticleSystem> bh;
        Input input;
    };
    struct Output {
        std::vector<std::size_t> indices;
        std::vector<std::uint64_t> keys;

        bool operator==(const Output& o) const {
            return indices == o.indices && keys == o.keys;
        }
    };

    static Input gen_input(std::size_t n, unsigned seed) {
        geom::AABBf world{-1.f, -1.f, 1.f, 1.f};

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dx(world.min_x, world.max_x);
        std::uniform_real_distribution<float> dy(world.min_y, world.max_y);

        Input in;
        in.x.resize(n);
        in.y.resize(n);

        for (std::size_t i = 0; i<n; ++i) { 
            in.x[i] = dx(rng);
            in.y[i] = dy(rng);
        }
        return in;
    }
    enum class Dist { Uniform, Grid, Line, Clustered, Duplicates };

    static Input gen_input_dist(std::size_t n, unsigned seed, Dist d) {
        Input in = gen_input(n, seed); // Uniform baseline (already done)
        switch (d) {
          case Dist::Uniform: break;
          case Dist::Duplicates: {
            if (n) { in.x.assign(n, 0.25f); in.y.assign(n, 0.5f); }
            break;
          }
          case Dist::Grid: {
            // Make a w×h grid inside world; simple row-major fill
            std::size_t w = std::max<std::size_t>(1, std::floor(std::sqrt(n)));
            std::size_t h = (w ? (n + w - 1)/w : 0);
            float dx = (in.world.max_x - in.world.min_x) / std::max<std::size_t>(1, w-1);
            float dy = (in.world.max_y - in.world.min_y) / std::max<std::size_t>(1, h-1);
            for (std::size_t i=0, k=0; i<h && k<n; ++i)
              for (std::size_t j=0; j<w && k<n; ++j, ++k) {
                in.x[k] = in.world.min_x + j*dx;
                in.y[k] = in.world.min_y + i*dy;
              }
            break;
          }
          default: break;
        }
        return in;
    }

    static void check_invariants(const Input in, const Output& out){
        const std::size_t N = in.x.size();
        // Input should equal output
        if (out.indices.size() != N || out.keys.size() != N){
            throw std::runtime_error("Sizes dont match N");
        }

        // Indices is a permutation of N, and up to N
        std::vector<std::size_t> perm = out.indices;
        std::sort(perm.begin(), perm.end());
        for (std::size_t i=0; i<N; ++i){
            if (perm[i] != i){
                throw std::runtime_error("indices is not permutation");
            }
        }
        
        // Keys non decreasing in the order of indices
        for (std::size_t k=1; k<N; ++k){
            if (out.keys[k-1] > out.keys[k]){
                throw std::runtime_error("Keys not nondecreasing");
            }
        }

    }
};
static_assert(VersionedPipelineTestKit::PipelineSpec<MortonSpec>);
