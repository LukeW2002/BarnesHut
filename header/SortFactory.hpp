// SortFactory.hpp
#pragma once
#include <cstddef>
#include <algorithm>
#include <numeric>
#include "Bounds.hpp"  

struct MortonTag {};
struct RadixTag  {};
struct StdSortTag {}; //example


struct MortonPolicy {
    template<class PS>
    static void encode(PS& ps, const geom::AABBf& world, std::size_t N) noexcept {
        MortonEncoder::encode_morton_keys(
            ps.positions_x_.data(),
            ps.positions_y_.data(),
            ps.morton_keys_.data(),
            N,
            world,
            ps.config_.enable_threading
        );
    }
};

struct RadixPolicy {
    template<class PS>
    static void sort(PS& ps, std::size_t /*N*/) noexcept {
        ps.radix_sort_indices();
    }
};

struct StdSortPolicy {
    template<class PS>
    static void sort(PS& ps, std::size_t N) {
        auto* idx  = ps.morton_indices_.data();
        auto* keys = ps.morton_keys_.data();
        std::iota(idx, idx + N, 0u);
        std::sort(idx, idx + N, [&](auto a, auto b){ return keys[a] < keys[b]; });
    }
};

template<class EncoderPolicy, class SortPolicy>
struct SortFactory {
    template<class PS>
    static void run(PS& ps) {
        const std::size_t N = ps.particle_count_;
        ps.ensure_keys_capacity(N);
        ps.ensure_indices_upto(N);

        const geom::AABBf world{
            static_cast<float>(ps.bounds_min_x_),
            static_cast<float>(ps.bounds_min_y_),
            static_cast<float>(ps.bounds_max_x_),
            static_cast<float>(ps.bounds_max_y_)
        };

        EncoderPolicy::encode(ps, world, N);
        SortPolicy::sort(ps, N);
    }
};

