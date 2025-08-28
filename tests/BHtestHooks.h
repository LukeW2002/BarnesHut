// tests/BHTestHooks.h
#pragma once
#ifndef BH_TESTING
#define BH_TESTING
#endif

#include "BarnesHutParticleSystem.h"
#include <cstdint>
#include <vector>

struct BHTestHooks {
    struct Snapshot {
        std::vector<BarnesHutParticleSystem::QuadTreeNode> nodes;
        uint32_t root;
        std::vector<uint32_t> leaf_offset, leaf_count, leaf_idx, particle_leaf_slot;
        std::vector<float> leaf_x, leaf_y, leaf_m;
        double min_x, max_x, min_y, max_y;
        size_t N;
    };

    // Copy out a cheap snapshot of internal state for verification
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
}
};

