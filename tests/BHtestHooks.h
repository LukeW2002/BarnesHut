// tests/BHTestHooks.h
#pragma once
#ifndef BH_TESTING
#define BH_TESTING
#endif

#include "BarnesHutParticleSystem.h"
#include <cstdint>
#include <vector>
#include <array>     // <-- add
#include <utility>   // <-- add
#include <limits>    // <-- add for SIZE_MAX


struct BHTestHooks {
    struct Snapshot {
        std::vector<BarnesHutParticleSystem::QuadTreeNode> nodes;
        uint32_t root;
        std::vector<uint32_t> leaf_offset, leaf_count, leaf_idx, particle_leaf_slot;
        std::vector<float> leaf_x, leaf_y, leaf_m;
        double min_x, max_x, min_y, max_y;
        size_t N;
    };

    static Snapshot snapshot(const BarnesHutParticleSystem& s) {
        Snapshot out;
        out.nodes = s.tree_nodes_;
        out.root = s.root_node_index_;
        out.leaf_offset = s.leaf_offset_;
        out.leaf_count = s.leaf_count_;
        out.leaf_idx = s.leaf_idx_;
        out.particle_leaf_slot = s.particle_leaf_slot_;
        out.leaf_x = s.leaf_pos_x_;
        out.leaf_y = s.leaf_pos_y_;
        out.leaf_m = s.leaf_mass_;
        out.min_x = s.bounds_min_x_;
        out.max_x = s.bounds_max_x_;
        out.min_y = s.bounds_min_y_;
        out.max_y = s.bounds_max_y_;
        out.N = s.particle_count_;
        return out;
    }


    // IMPORTANT: take a node index instead of the private type to keep tests decoupled
    static void leaf_neon_at(const BarnesHutParticleSystem& s,
                             uint32_t node_index,
                             int i_local, float px, float py, float gi,
                             float& fx, float& fy,
                             const float* leaf_x, const float* leaf_y, const float* leaf_m)
    {
        const auto& node = s.tree_nodes_[node_index];

        // Use the same origin as the main traversal: root COM
        const float ox = s.tree_nodes_[s.root_node_index_].com_x;
        const float oy = s.tree_nodes_[s.root_node_index_].com_y;

        // Center the particle coordinates once
        const float px_c = px - ox;
        const float py_c = py - oy;

        // Call the centered NEON kernel (returns forces *without* G_GALACTIC scaling)
        s.process_leaf_forces_neon_centered(node, i_local, px_c, py_c, gi,
                                            fx, fy, ox, oy,
                                            leaf_x, leaf_y, leaf_m);
    } // <-- properly close this function

    // ---------- Self-contained Morton testing helpers ----------
    struct MortonTest {
        using BH = BarnesHutParticleSystem;
        using Range = std::pair<size_t,size_t>;
        using RangeArr = std::array<Range,4>;

        struct MortonLevelParams {
            uint64_t mask;
            int      level_shift;
        };

        // If your keys are 42 bits (as your tests assume), depth 0 shift is 40.
        static constexpr int kMortonBits = 42;

        static MortonLevelParams level_params(const BH&, int depth) {
            int shift = (kMortonBits - 2) - 2*depth;
            if (shift < 0) shift = 0;
            return MortonLevelParams{ (3ULL << shift), shift };
        }

        static int extract_quadrant(const BH& s, size_t i, const MortonLevelParams& p) {
            return int((s.morton_keys_[i] & p.mask) >> p.level_shift);
        }

        // Return end position (one past last), as your tests expect.
        static size_t find_sequence_end(const BH& s, size_t start, size_t last, int q,
                                        const MortonLevelParams& p)
        {
            size_t i = start;
            while (i <= last && extract_quadrant(s, i, p) == q) ++i;
            return i;
        }

        // End-to-end: encode + radix
        static void sort_by_morton(BH& s) { s.sort_by_morton_key(); }

        // Radix-only (expects keys prefilled)
        static void radix_sort_only(BH& s) { s.radix_sort_indices(); }

        // Accessors
        static const std::vector<size_t>&   get_indices(const BH& s) { return s.morton_indices_; }
        static const std::vector<uint64_t>& get_keys(const BH& s)    { return s.morton_keys_;    }

        // Seed exact arrays for controlled tests
        static void set_morton_arrays(BH& s,
                                      std::vector<uint64_t> keys,
                                      std::vector<size_t>   indices,
                                      size_t N_override = SIZE_MAX)
        {
            s.morton_keys_    = std::move(keys);
            s.morton_indices_ = std::move(indices);
            s.particle_count_ = (N_override==SIZE_MAX)? s.morton_indices_.size() : N_override;
        }

        // Composite: split [first,last] at given depth (z-order → canonical mapping)
        static RangeArr split_range(const BH& s, size_t first, size_t last, int depth) {
            RangeArr out{ { {SIZE_MAX,SIZE_MAX},{SIZE_MAX,SIZE_MAX},{SIZE_MAX,SIZE_MAX},{SIZE_MAX,SIZE_MAX} } };
            if (first > last) return out;

            const MortonLevelParams p = level_params(s, depth);
            size_t pos = first;
            static const int z_to_child[4] = {0, 2, 1, 3};

            for (int z = 0; z < 4 && pos <= last; ++z) {
                const size_t end_excl = find_sequence_end(s, pos, last, z, p);
                if (end_excl > pos) {
                    const int child = z_to_child[z];
                    out[child] = {pos, end_excl - 1};
                    pos = end_excl;
                }
            }
            return out;
        }

    };
};

