#ifndef BH_TESTING
#define BH_TESTING
#endif
#include <gtest/gtest.h>
#include "BarnesHutParticleSystem.h"
#include "EventSystem.h"
#include "BHTestHooks.h"
#include <random>

using BH = BarnesHutParticleSystem;

static void brute_forces(const std::vector<float>& x, const std::vector<float>& y,
                         const std::vector<float>& m,
                         std::vector<float>& fx, std::vector<float>& fy) {
  const size_t N = x.size(); 
  const float eps2 = EPS_SQ;
  
  for (size_t i = 0; i < N; i++) {
    float fxi = 0, fyi = 0;
    for (size_t j = 0; j < N; j++) { 
      if (i == j) continue;
      float dx = x[j] - x[i], dy = y[j] - y[i];
      float r2 = dx*dx + dy*dy + eps2;
      float inv = 1.0f / std::sqrt(r2);
      float inv3 = inv * inv * inv;
      float s = G_GALACTIC * m[i] * m[j] * inv3;
      fxi += s * dx; 
      fyi += s * dy;
    }
    fx[i] = fxi; 
    fy[i] = fyi;
  }
}

static BH make(BH::Config cfg, size_t N, uint32_t seed=7) {
  EventBus bus; 
  BH sys(N, bus, cfg);
  std::mt19937 rng(seed); 
  std::normal_distribution<float> d(0, 0.4f);
  
  for (size_t i = 0; i < N; i++) {
    sys.add_particle({d(rng), d(rng)}, {0,0}, 1.0f, {1,1,1});
  }
  sys.set_boundary(-2,2,-2,2); 
  sys.update(1e-6f); 
  return sys;
}

static float max_rel_err(const std::vector<float>& ax, const std::vector<float>& ay,
                         const std::vector<float>& bx, const std::vector<float>& by) {
  float e = 0;
  for (size_t i = 0; i < ax.size(); ++i) {
    float da = ax[i] - bx[i], db = ay[i] - by[i];
    float nb = std::hypot(bx[i], by[i]); 
    if (nb < 1e-7f) nb = 1.0f;
    e = std::max(e, std::hypot(da, db) / nb);
  }
  return e;
}

TEST(Traversal, ThetaZeroMatchesBruteForceSmallN) {
  BH::Config cfg; 
  cfg.theta = 1e-3f; 
  cfg.theta_squared = cfg.theta * cfg.theta;
  auto sys = make(cfg, 128);

  // Collect forces from BH (single step already computed in update())
  auto X = sys.get_positions_x(), Y = sys.get_positions_y(), M = sys.get_masses();
  std::vector<float> fx_bf(X.size()), fy_bf(Y.size());
  brute_forces(X, Y, M, fx_bf, fy_bf);

  std::vector<float> fx_bh(X.size()), fy_bh(Y.size());
  for (size_t i = 0; i < X.size(); ++i) { 
    auto f = sys.get_force(i); 
    fx_bh[i] = f.x; 
    fy_bh[i] = f.y; 
  }

  EXPECT_LT(max_rel_err(fx_bh, fy_bh, fx_bf, fy_bf), 0.02f); // ≤2% at tiny θ
}

TEST(Traversal, ErrorMonotonicWithTheta) {
  // same scene, two theta values
  BH::Config cfgA; 
  cfgA.theta = 0.5f; 
  cfgA.theta_squared = cfgA.theta * cfgA.theta;
  auto sysA = make(cfgA, 256);
  
  BH::Config cfgB = cfgA; 
  cfgB.theta = 1.0f; 
  cfgB.theta_squared = 1.0f;
  auto sysB = make(cfgB, 256);

  auto X = sysA.get_positions_x(), Y = sysA.get_positions_y(), M = sysA.get_masses();
  std::vector<float> fx_bf(X.size()), fy_bf(Y.size());
  brute_forces(X, Y, M, fx_bf, fy_bf);

  auto errA = [&] { 
    std::vector<float> fx(X.size()), fy(Y.size());
    for (size_t i = 0; i < X.size(); ++i) { 
      auto f = sysA.get_force(i); 
      fx[i] = f.x; 
      fy[i] = f.y; 
    }
    return max_rel_err(fx, fy, fx_bf, fy_bf);
  }();
  
  auto errB = [&] { 
    std::vector<float> fx(X.size()), fy(Y.size());
    for (size_t i = 0; i < X.size(); ++i) { 
      auto f = sysB.get_force(i); 
      fx[i] = f.x; 
      fy[i] = f.y; 
    }
    return max_rel_err(fx, fy, fx_bf, fy_bf);
  }();
  
  EXPECT_LE(errA, errB + 0.01f); // allow a tiny slack for FP noise
}

TEST(Traversal, TranslationEquivariance) {
  BH::Config cfg; 
  auto sys = make(cfg, 300);
  auto X = sys.get_positions_x(), Y = sys.get_positions_y();
  
  std::vector<float> base_fx(X.size()), base_fy(Y.size());
  for (size_t i = 0; i < X.size(); ++i) { 
    auto f = sys.get_force(i); 
    base_fx[i] = f.x; 
    base_fy[i] = f.y; 
  }

  // Shift everything by (+3,+5), rebuild, compare
  EventBus bus; 
  BH sys2(X.size(), bus, cfg);
  for (size_t i = 0; i < X.size(); ++i) {
    sys2.add_particle({X[i] + 3, Y[i] + 5}, {0,0}, 1.0f, {1,1,1});
  }
  sys2.set_boundary(-2+3, 2+3, -2+5, 2+5); 
  sys2.update(1e-6f);

  float e = 0;
  for (size_t i = 0; i < X.size(); ++i) { 
    auto f = sys2.get_force(i);
    e = std::max(e, std::hypot(f.x - base_fx[i], f.y - base_fy[i]));
  }
  EXPECT_LT(e, 5e-3f);
}

TEST(Traversal, UniformScalingEquivariance) {
  BH::Config cfg;
  auto sys1 = make(cfg, 200);
  auto X = sys1.get_positions_x(), Y = sys1.get_positions_y(), M = sys1.get_masses();
  
  std::vector<float> fx1(X.size()), fy1(Y.size());
  for (size_t i = 0; i < X.size(); ++i) { 
    auto f = sys1.get_force(i); 
    fx1[i] = f.x; 
    fy1[i] = f.y; 
  }

  // Scale everything by factor 2.0
  const float scale = 2.0f;
  EventBus bus;
  BH sys2(X.size(), bus, cfg);
  for (size_t i = 0; i < X.size(); ++i) {
    sys2.add_particle({X[i] * scale, Y[i] * scale}, {0,0}, M[i], {1,1,1});
  }
  sys2.set_boundary(-2*scale, 2*scale, -2*scale, 2*scale);
  sys2.update(1e-6f);

  std::vector<float> fx2(X.size()), fy2(Y.size());
  for (size_t i = 0; i < X.size(); ++i) { 
    auto f = sys2.get_force(i); 
    fx2[i] = f.x; 
    fy2[i] = f.y; 
  }

  // Forces should scale by 1/scale² = 1/4
  const float expected_scale = 1.0f / (scale * scale);
  float max_error = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    float expected_fx = fx1[i] * expected_scale;
    float expected_fy = fy1[i] * expected_scale;
    float error_fx = std::abs(fx2[i] - expected_fx);
    float error_fy = std::abs(fy2[i] - expected_fy);
    max_error = std::max(max_error, std::max(error_fx, error_fy));
  }
  
  // Allow for some numerical error
  EXPECT_LT(max_error, 1e-4f) << "Uniform scaling invariance violated";
}

TEST(Traversal, ReproducibilityTest) {
  // Build same system twice - should get identical results
  BH::Config cfg;
  cfg.theta = 0.8f;
  cfg.theta_squared = cfg.theta * cfg.theta;
  
  auto sys1 = make(cfg, 500, 42); // Fixed seed
  auto sys2 = make(cfg, 500, 42); // Same seed
  
  auto X1 = sys1.get_positions_x(), Y1 = sys1.get_positions_y();
  auto X2 = sys2.get_positions_x(), Y2 = sys2.get_positions_y();
  
  // Positions should be identical
  for (size_t i = 0; i < X1.size(); ++i) {
    EXPECT_EQ(X1[i], X2[i]) << "Position X mismatch at particle " << i;
    EXPECT_EQ(Y1[i], Y2[i]) << "Position Y mismatch at particle " << i;
  }
  
  // Forces should be identical
  for (size_t i = 0; i < X1.size(); ++i) {
    auto f1 = sys1.get_force(i);
    auto f2 = sys2.get_force(i);
    EXPECT_EQ(f1.x, f2.x) << "Force X mismatch at particle " << i;
    EXPECT_EQ(f1.y, f2.y) << "Force Y mismatch at particle " << i;
  }
}
