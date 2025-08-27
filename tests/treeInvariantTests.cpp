#ifndef BH_TESTING
#define BH_TESTING
#endif
#include <gtest/gtest.h>
#include "BarnesHutParticleSystem.h"
#include "EventSystem.h"
#include "BHTestHooks.h"
#include <random>
#include <queue>

using BH = BarnesHutParticleSystem;

static BH make_sys(size_t N, BH::Config cfg, EventBus& bus, uint32_t seed=123) {
  BH sys(N, bus, cfg);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-1.f,1.f);
  for (size_t i=0;i<N;i++) {
    sys.add_particle({d(rng), d(rng)}, {0,0}, 1.0f, {1,1,1});
  }
  sys.set_boundary(-1,1,-1,1);
  sys.update(1e-6f); // triggers build
  return sys;
}

TEST(Tree, TerminatesAndRespectsLimits) {
  EventBus bus;
  BH::Config cfg; 
  cfg.max_particles_per_leaf=16; 
  cfg.tree_depth_limit=20;
  auto sys = make_sys(4096, cfg, bus);

#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);
  ASSERT_NE(S.root, (uint32_t)UINT32_MAX);
  ASSERT_FALSE(S.nodes.empty());

  // a) Depth and leaf criterion
  for (auto& n : S.nodes) {
    ASSERT_LE(n.depth, cfg.tree_depth_limit);
    if (n.is_leaf) {
      uint32_t cnt = (n.leaf_first==UINT32_MAX) ? 0 : (n.leaf_last - n.leaf_first + 1);
      ASSERT_TRUE(cnt <= cfg.max_particles_per_leaf || n.depth==cfg.tree_depth_limit);
    }
  }

  // b) Children indices valid; no out-of-range
  for (auto& n : S.nodes) {
    for (int k=0;k<4;k++) {
      uint32_t c = n.children[k];
      ASSERT_TRUE(c==UINT32_MAX || c < S.nodes.size());
    }
  }
#endif
}

TEST(Tree, BijectionAndLeafBounds) {
  EventBus bus;
  BH::Config cfg;
  auto sys = make_sys(2000, cfg, bus);
  
#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);
  std::vector<char> seen(S.N, 0);
  size_t total=0;

  for (size_t l=0; l<S.leaf_offset.size(); ++l) {
    uint32_t off = S.leaf_offset[l];
    uint32_t cnt = S.leaf_count[l];
    total += cnt;

    // each slot maps to exactly one global index, and particle_leaf_slot points back
    for (uint32_t t=0; t<cnt; ++t) {
      uint32_t slot = off + t;
      uint32_t gi = S.leaf_idx[slot];
      ASSERT_LT(gi, S.N);
      ASSERT_EQ(S.particle_leaf_slot[gi], slot);
      ASSERT_FALSE(seen[gi]) << "duplicate particle in leaves";
      seen[gi]=1;
      
      // AABB check: particle must lie inside its node bounds
      uint32_t leaf_idx = UINT32_MAX;
      for (uint32_t i = 0; i < S.nodes.size(); ++i) {
        const auto& n = S.nodes[i];
        if (n.is_leaf && n.leaf_first == off && n.leaf_last == off + cnt - 1) {
          leaf_idx = i;
          break;
        }
      }
      ASSERT_NE(leaf_idx, (uint32_t)UINT32_MAX);
      const auto& leaf = S.nodes[leaf_idx];
      float x = S.leaf_x[slot], y = S.leaf_y[slot];
      ASSERT_GE(x, leaf.min_x);
      ASSERT_LE(x, leaf.max_x);
      ASSERT_GE(y, leaf.min_y);
      ASSERT_LE(y, leaf.max_y);
    }
  }
  ASSERT_EQ(total, S.N);       // every particle appears exactly once
  for (size_t i=0;i<S.N;i++) ASSERT_TRUE(seen[i]);
#endif
}

TEST(Tree, MassCOMAndBoundRadius) {
  EventBus bus;
  BH::Config cfg;
  auto sys = make_sys(1500, cfg, bus);
  
#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);

  // Recompute COM/mass for leaves and check bound_r
  for (auto& n : S.nodes) {
    if (!n.is_leaf) continue;
    if (n.leaf_first==UINT32_MAX) continue;
    
    uint32_t off=n.leaf_first, cnt = n.leaf_last - off + 1;
    double m=0, wx=0, wy=0, maxd2=0;
    
    for (uint32_t t=0;t<cnt;++t) {
      float mm = S.leaf_m[off+t];
      float x  = S.leaf_x[off+t];
      float y  = S.leaf_y[off+t];
      m += mm; wx += mm*x; wy += mm*y;
    }
    
    double cx = (m>0)? wx/m : 0.0, cy=(m>0)? wy/m : 0.0;
    for (uint32_t t=0;t<cnt;++t) {
      double dx=S.leaf_x[off+t]-cx, dy=S.leaf_y[off+t]-cy;
      maxd2 = std::max(maxd2, dx*dx+dy*dy);
    }
    double br = std::sqrt(maxd2);
    
    EXPECT_NEAR(n.total_mass, m, 1e-4);
    EXPECT_NEAR(n.com_x, cx, std::max(1.0, std::abs(cx))*1e-4);
    EXPECT_NEAR(n.com_y, cy, std::max(1.0, std::abs(cy))*1e-4);
    EXPECT_NEAR(n.bound_r, br, br*2e-3); // small tolerance for fast rsqrt
  }
#endif
}

TEST(Tree, DegenerateDuplicatesStopAtLimit) {
  EventBus bus; 
  BH::Config cfg; 
  cfg.max_particles_per_leaf=1; 
  cfg.tree_depth_limit=8;
  BH sys(1000,bus,cfg);
  
  // CRITICAL TEST: All particles at exactly the same position
  for (int i=0;i<1000;i++) {
    sys.add_particle({0,0},{0,0},1,{1,1,1});
  }
  sys.set_boundary(-1,1,-1,1); 
  sys.update(1e-6f);

#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);
  uint16_t maxd=0; 
  for (auto& n:S.nodes) {
    maxd = std::max<uint16_t>(maxd, n.depth);
  }
  EXPECT_EQ(maxd, cfg.tree_depth_limit);
  
  // Tree should not crash or loop infinitely
  ASSERT_GT(S.nodes.size(), 0);
  ASSERT_LT(S.nodes.size(), 10000); // Sanity check - shouldn't create millions of nodes
#endif
}

TEST(Tree, ParentBoundsContainChildren) {
  EventBus bus;
  BH::Config cfg;
  auto sys = make_sys(800, cfg, bus);
  
#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);
  
  for (auto& parent : S.nodes) {
    if (parent.is_leaf) continue; // Skip leaves
    
    for (int i = 0; i < 4; ++i) {
      if (parent.children[i] == UINT32_MAX) continue;
      
      ASSERT_LT(parent.children[i], S.nodes.size());
      const auto& child = S.nodes[parent.children[i]];
      
      // Parent bounds must contain child bounds
      EXPECT_LE(parent.min_x, child.min_x) << "Parent min_x doesn't contain child";
      EXPECT_GE(parent.max_x, child.max_x) << "Parent max_x doesn't contain child";
      EXPECT_LE(parent.min_y, child.min_y) << "Parent min_y doesn't contain child";
      EXPECT_GE(parent.max_y, child.max_y) << "Parent max_y doesn't contain child";
    }
  }
#endif
}

TEST(Tree, NoOrphanNodes) {
  EventBus bus;
  BH::Config cfg;
  auto sys = make_sys(500, cfg, bus);
  
#ifdef BH_TESTING
  auto S = BHTestHooks::snapshot(sys);
  
  // Mark all reachable nodes from root
  std::vector<bool> reachable(S.nodes.size(), false);
  std::queue<uint32_t> to_visit;
  
  if (S.root != UINT32_MAX && S.root < S.nodes.size()) {
    to_visit.push(S.root);
    reachable[S.root] = true;
  }
  
  while (!to_visit.empty()) {
    uint32_t idx = to_visit.front();
    to_visit.pop();
    
    const auto& node = S.nodes[idx];
    if (!node.is_leaf) {
      for (int i = 0; i < 4; ++i) {
        if (node.children[i] != UINT32_MAX && node.children[i] < S.nodes.size()) {
          if (!reachable[node.children[i]]) {
            reachable[node.children[i]] = true;
            to_visit.push(node.children[i]);
          }
        }
      }
    }
  }
  
  // All nodes should be reachable from root
  for (size_t i = 0; i < S.nodes.size(); ++i) {
    EXPECT_TRUE(reachable[i]) << "Node " << i << " is orphaned (not reachable from root)";
  }
#endif
}
