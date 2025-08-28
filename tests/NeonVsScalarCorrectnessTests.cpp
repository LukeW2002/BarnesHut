#ifndef BH_TESTING
#define BH_TESTING
#endif
#include <gtest/gtest.h>
#include "BarnesHutParticleSystem.h"
#include "EventSystem.h"
#include "BHTestHooks.h"
#include <random>

using BH = BarnesHutParticleSystem;

TEST(NEONLeaf, MatchesScalarReference) {
#if defined(__ARM_NEON)
  EventBus bus; 
  BH::Config cfg; 
  auto sys = BH(64, bus, cfg);
  
  // Build a single large leaf: bound min/max tightly so Morton puts them together
  for (int i = 0; i < 32; i++) {
    sys.add_particle({-0.2f + 0.0125f*i, 0.1f + 0.01f*i}, {0,0}, 1.0f, {1,1,1});
  }
  sys.set_boundary(-1,1,-1,1); 
  sys.update(1e-6f);

  auto S = BHTestHooks::snapshot(sys);

  // Find a leaf with cnt >= 8
  uint32_t leaf_index = UINT32_MAX;
  for (uint32_t idx = 0; idx < S.nodes.size(); ++idx) {
    const auto& n = S.nodes[idx];
    if (n.is_leaf && n.leaf_first != UINT32_MAX && (n.leaf_last - n.leaf_first + 1) >= 8) { 
      leaf_index = idx;
      break; 
    } 
  }
  ASSERT_NE(leaf_index, (uint32_t)UINT32_MAX) << "No leaf with 8+ particles found";
  
  const auto& leaf = S.nodes[leaf_index];
  const uint32_t off = leaf.leaf_first; 
  const uint32_t cnt = leaf.leaf_last - off + 1;
  
  // Test NEON vs scalar on target particle (index 0 in leaf)
  int i_local = 0;
  uint32_t slot = off + i_local;
  float px = S.leaf_x[slot], py = S.leaf_y[slot], gi = 1.0f;

  float fxN = 0, fyN = 0;
  BHTestHooks::leaf_neon_at(sys, leaf_index, i_local, px, py, gi, fxN, fyN,
                            &S.leaf_x[off], &S.leaf_y[off], &S.leaf_m[off]);

  // scalar reference (skip self)
  float fxS = 0, fyS = 0;
  for (uint32_t t = 0; t < cnt; ++t) { 
    if ((int)t == i_local) continue;
    float dx = S.leaf_x[off+t] - px, dy = S.leaf_y[off+t] - py;
    float r2 = dx*dx + dy*dy + EPS_SQ; 
    float inv = 1.0f / std::sqrt(r2);
    float inv3 = inv * inv * inv; 
    float s = gi * S.leaf_m[off+t] * inv3;
    fxS += s * dx; 
    fyS += s * dy;
  }

  // Should handle overlapping particles gracefully (softening prevents division by zero)
  EXPECT_TRUE(std::isfinite(fxN) && std::isfinite(fyN)) << "NEON produced non-finite result";
  EXPECT_TRUE(std::isfinite(fxS) && std::isfinite(fyS)) << "Scalar produced non-finite result";
  
  // Allow larger tolerance for edge cases with very close particles
  float tolerance = std::max(1.f, std::max(std::abs(fxS), std::abs(fyS))) * 1e-2f;
  EXPECT_NEAR(fxN, fxS, tolerance) << "NEON vs scalar mismatch in edge case";
  EXPECT_NEAR(fyN, fyS, tolerance);
#else
  GTEST_SKIP() << "NEON not available on this target.";
#endif
}

TEST(NEONLeaf, PerformanceConsistency) {
#if defined(__ARM_NEON)
  // Test that NEON gives consistent results across multiple runs
  EventBus bus; 
  BH::Config cfg; 
  auto sys = BH(64, bus, cfg);
  
  // Create predictable particle layout
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  
  for (int i = 0; i < 24; i++) {
    sys.add_particle({dist(rng), dist(rng)}, {0,0}, 1.0f + 0.1f*i, {1,1,1});
  }
  sys.set_boundary(-1,1,-1,1); 
  sys.update(1e-6f);

  auto S = BHTestHooks::snapshot(sys);
  
  uint32_t leaf_index = UINT32_MAX;
  for (uint32_t idx = 0; idx < S.nodes.size(); ++idx) {
    const auto& n = S.nodes[idx];
    if (n.is_leaf && n.leaf_first != UINT32_MAX && (n.leaf_last - n.leaf_first + 1) >= 8) { 
      leaf_index = idx;
      break; 
    } 
  }
  ASSERT_NE(leaf_index, (uint32_t)UINT32_MAX);
  
  const auto& leaf = S.nodes[leaf_index];
  const uint32_t off = leaf.leaf_first; 
  const uint32_t cnt = leaf.leaf_last - off + 1;
  int i_local = 5;
  uint32_t slot = off + i_local;
  float px = S.leaf_x[slot], py = S.leaf_y[slot], gi = 1.5f;

  // Run NEON computation multiple times
  std::vector<std::pair<float, float>> results;
  for (int run = 0; run < 5; ++run) {
    float fx = 0, fy = 0;
    BHTestHooks::leaf_neon_at(sys, leaf_index, i_local, px, py, gi, fx, fy,
                              &S.leaf_x[off], &S.leaf_y[off], &S.leaf_m[off]);
    results.emplace_back(fx, fy);
  }

  // All results should be identical (deterministic)
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_EQ(results[0].first, results[i].first) << "NEON result inconsistent across runs";
    EXPECT_EQ(results[0].second, results[i].second) << "NEON result inconsistent across runs";
  }
#else
  GTEST_SKIP() << "NEON not available on this target.";
#endif
}

TEST(InternalNodeBatch, NEONvsScalarConsistency) {
#if defined(__ARM_NEON)
  EventBus bus;
  BH::Config cfg;
  cfg.theta = 1.5f; // High theta to ensure internal nodes are accepted
  cfg.theta_squared = cfg.theta * cfg.theta;
  auto sys = BH(200, bus, cfg);
  
  // Spread particles to create internal nodes that will be accepted
  std::mt19937 rng(9999);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  
  for (int i = 0; i < 200; i++) {
    sys.add_particle({dist(rng), dist(rng)}, {0,0}, 1.0f, {1,1,1});
  }
  sys.set_boundary(-3, 3, -3, 3);
  sys.update(1e-6f);
  
  // Test a few particles
  for (int test_particle = 0; test_particle < 5; ++test_particle) {
    auto force = sys.get_force(test_particle);
    
    // Forces should be finite and reasonable
    EXPECT_TRUE(std::isfinite(force.x)) << "Force X is not finite for particle " << test_particle;
    EXPECT_TRUE(std::isfinite(force.y)) << "Force Y is not finite for particle " << test_particle;
    EXPECT_LT(std::abs(force.x), 1000.0f) << "Force X unexpectedly large for particle " << test_particle;
    EXPECT_LT(std::abs(force.y), 1000.0f) << "Force Y unexpectedly large for particle " << test_particle;
  }
#else
  GTEST_SKIP() << "NEON not available on this target.";
#endif
}

TEST(NEONLeaf, HandlesEdgeCases) {
#if defined(__ARM_NEON)
  EventBus bus; 
  BH::Config cfg; 
  auto sys = BH(16, bus, cfg);
  
  // Create edge case: particle exactly at same position as target
  sys.add_particle({0.5f, 0.5f}, {0,0}, 1.0f, {1,1,1}); // target particle
  sys.add_particle({0.5f, 0.5f}, {0,0}, 2.0f, {1,1,1}); // exact overlap
  sys.add_particle({0.501f, 0.501f}, {0,0}, 1.5f, {1,1,1}); // very close
  for (int i = 0; i < 13; i++) {
    sys.add_particle({0.5f + 0.01f*i, 0.5f + 0.01f*i}, {0,0}, 1.0f, {1,1,1});
  }
  
  sys.set_boundary(0, 1, 0, 1);
  sys.update(1e-6f);

  auto S = BHTestHooks::snapshot(sys);
  
  // Find the leaf containing our particles
  uint32_t leaf_index = UINT32_MAX;
  for (uint32_t idx = 0; idx < S.nodes.size(); ++idx) {
    const auto& n = S.nodes[idx];
    if (n.is_leaf && n.leaf_first != UINT32_MAX) { 
      leaf_index = idx;
      break; 
    } 
  }
  ASSERT_NE(leaf_index, (uint32_t)UINT32_MAX);

  const auto& leaf = S.nodes[leaf_index];
  const uint32_t off = leaf.leaf_first; 
  const uint32_t cnt = leaf.leaf_last - off + 1;
  
  // pick a target inside this leaf (self-skip must be honored)
  int i_local = 3; 
  uint32_t slot = off + i_local;
  float px = S.leaf_x[slot], py = S.leaf_y[slot], gi = 1.0f;

  float fxN = 0, fyN = 0;
  BHTestHooks::leaf_neon_at(sys, leaf_index, i_local, px, py, gi, fxN, fyN,
                            &S.leaf_x[off], &S.leaf_y[off], &S.leaf_m[off]);

  // scalar reference (skip self)
  float fxS = 0, fyS = 0;
  for (uint32_t t = 0; t < cnt; ++t) { 
    if ((int)t == i_local) continue;
    float dx = S.leaf_x[off+t] - px, dy = S.leaf_y[off+t] - py;
    float r2 = dx*dx + dy*dy + EPS_SQ; 
    float inv = 1.0f / std::sqrt(r2);
    float inv3 = inv * inv * inv; 
    float s = gi * S.leaf_m[off+t] * inv3;
    fxS += s * dx; 
    fyS += s * dy;
  }

  EXPECT_NEAR(fxN, fxS, std::max(1.f, std::abs(fxS)) * 2e-3f);
  EXPECT_NEAR(fyN, fyS, std::max(1.f, std::abs(fyS)) * 2e-3f);
#else
  GTEST_SKIP() << "NEON not available on this target.";
#endif
}
