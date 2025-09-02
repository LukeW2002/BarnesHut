// benchmarks/morton_bench.cpp
#ifndef BH_TESTING
#define BH_TESTING
#endif
#include <benchmark/benchmark.h>
#include "BarnesHutParticleSystem.h"
#include "BHTestHooks.h"
#include <random>
#include <algorithm>

using BH = BarnesHutParticleSystem;
using MT = BHTestHooks::MortonTest;

// Utility to create predictable particle distributions
class ParticleSeeder {
public:
    static void uniform_random(BH& sys, size_t N, uint32_t seed = 123) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (size_t i = 0; i < N; ++i) {
            sys.add_particle({dist(rng), dist(rng)}, {0,0}, 1.0f, {1,1,1});
        }
        sys.set_boundary(-1.2f, 1.2f, -1.2f, 1.2f);
    }
    
    static void grid_pattern(BH& sys, size_t N) {
        const int side = static_cast<int>(std::ceil(std::sqrt(N)));
        const float step = 2.0f / side;
        
        for (int i = 0; i < side && sys.get_particle_count() < N; ++i) {
            for (int j = 0; j < side && sys.get_particle_count() < N; ++j) {
                sys.add_particle({-1.0f + i*step, -1.0f + j*step}, {0,0}, 1.0f, {1,1,1});
            }
        }
        sys.set_boundary(-1.2f, 1.2f, -1.2f, 1.2f);
    }
    
    static void clustered(BH& sys, size_t N, uint32_t seed = 456) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> cluster_center(-0.8f, 0.8f);
        std::normal_distribution<float> cluster_spread(0.0f, 0.05f);
        std::uniform_int_distribution<int> cluster_choice(0, 3);
        
        // Create 4 clusters
        std::array<std::pair<float, float>, 4> centers;
        for (auto& center : centers) {
            center = {cluster_center(rng), cluster_center(rng)};
        }
        
        for (size_t i = 0; i < N; ++i) {
            int cluster = cluster_choice(rng);
            float x = centers[cluster].first + cluster_spread(rng);
            float y = centers[cluster].second + cluster_spread(rng);
            sys.add_particle({x, y}, {0,0}, 1.0f, {1,1,1});
        }
        sys.set_boundary(-1.2f, 1.2f, -1.2f, 1.2f);
    }
};

// ===========================================================================================
// MORTON SORT BENCHMARKS
// ===========================================================================================

static void BM_SortByMortonKey_Uniform(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    ParticleSeeder::uniform_random(sys, st.range(0));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        MT::sort_by_morton(sys);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
    st.SetBytesProcessed(st.iterations() * st.range(0) * (sizeof(uint64_t) + sizeof(size_t)));
}
BENCHMARK(BM_SortByMortonKey_Uniform)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_SortByMortonKey_Grid(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    ParticleSeeder::grid_pattern(sys, st.range(0));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        MT::sort_by_morton(sys);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_SortByMortonKey_Grid)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_SortByMortonKey_Clustered(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    ParticleSeeder::clustered(sys, st.range(0));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        MT::sort_by_morton(sys);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_SortByMortonKey_Clustered)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

// ===========================================================================================
// RADIX SORT ISOLATION BENCHMARKS  
// ===========================================================================================

static void BM_RadixSortOnly_PrecomputedKeys(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    ParticleSeeder::uniform_random(sys, st.range(0));
    
    // Generate keys once
    MT::sort_by_morton(sys);
    
    for (auto _ : st) {
        // Reset indices to identity (fast)
        auto& indices = const_cast<std::vector<size_t>&>(MT::get_indices(sys));
        std::iota(indices.begin(), indices.begin() + sys.get_particle_count(), 0u);
        
        benchmark::DoNotOptimize(sys);
        MT::radix_sort_only(sys);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_RadixSortOnly_PrecomputedKeys)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

// Test the small-N std::sort fallback path
static void BM_RadixSortOnly_SmallN(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    ParticleSeeder::uniform_random(sys, st.range(0));
    MT::sort_by_morton(sys); // Generate keys
    
    for (auto _ : st) {
        auto& indices = const_cast<std::vector<size_t>&>(MT::get_indices(sys));
        std::iota(indices.begin(), indices.begin() + sys.get_particle_count(), 0u);
        
        benchmark::DoNotOptimize(sys);
        MT::radix_sort_only(sys);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_RadixSortOnly_SmallN)->RangeMultiplier(2)->Range(32, 4096)->Unit(benchmark::kMicrosecond);

// ===========================================================================================
// MORTON RANGE SPLITTING BENCHMARKS
// ===========================================================================================

static void BM_SplitMortonRange_Balanced(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    const size_t N = st.range(0);
    
    // Create balanced quadrant distribution
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // Create pattern with all 4 quadrants equally represented
    const auto params = MT::level_params(sys, 0);
    for (size_t i = 0; i < N; ++i) {
        int quad = static_cast<int>(i % 4);
        keys[i] = (uint64_t(quad) << params.level_shift);
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        auto ranges = MT::split_range(sys, 0, N-1, 0);
        benchmark::DoNotOptimize(ranges);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_SplitMortonRange_Balanced)->RangeMultiplier(2)->Range(1<<8, 1<<18)->Unit(benchmark::kNanosecond);

static void BM_SplitMortonRange_Skewed(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    const size_t N = st.range(0);
    
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // Create skewed distribution (90% in quadrant 2, 10% spread across others)
    const auto params = MT::level_params(sys, 0);
    for (size_t i = 0; i < N; ++i) {
        int quad = (i < N * 9 / 10) ? 2 : static_cast<int>(i % 4);
        keys[i] = (uint64_t(quad) << params.level_shift);
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        auto ranges = MT::split_range(sys, 0, N-1, 0);
        benchmark::DoNotOptimize(ranges);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_SplitMortonRange_Skewed)->RangeMultiplier(2)->Range(1<<8, 1<<18)->Unit(benchmark::kNanosecond);

static void BM_SplitMortonRange_SingleQuadrant(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    const size_t N = st.range(0);
    
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // All particles in single quadrant (worst case for tree balancing)
    const auto params = MT::level_params(sys, 0);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = (uint64_t(1) << params.level_shift); // All in quadrant 1
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        auto ranges = MT::split_range(sys, 0, N-1, 0);
        benchmark::DoNotOptimize(ranges);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_SplitMortonRange_SingleQuadrant)->RangeMultiplier(2)->Range(1<<8, 1<<18)->Unit(benchmark::kNanosecond);

// ===========================================================================================
// PRIME FUNCTION MICROBENCHMARKS
// ===========================================================================================

static void BM_ExtractQuadrant(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(1000, bus, cfg);
    
    const size_t N = st.range(0);
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // Random quadrants
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> quad_dist(0, 3);
    const auto params = MT::level_params(sys, 0);
    
    for (size_t i = 0; i < N; ++i) {
        int quad = quad_dist(rng);
        keys[i] = (uint64_t(quad) << params.level_shift);
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    size_t idx = 0;
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        int quad = MT::extract_quadrant(sys, idx % N, params);
        benchmark::DoNotOptimize(quad);
        idx = (idx + 1) % N; // Cycle through indices
    }
    
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK(BM_ExtractQuadrant)->Arg(1000)->Unit(benchmark::kNanosecond);

static void BM_FindSequenceEnd_WorstCase(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    const size_t N = st.range(0);
    
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // Worst case: all same quadrant (must scan entire range)
    const auto params = MT::level_params(sys, 0);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = (uint64_t(2) << params.level_shift);
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        size_t end = MT::find_sequence_end(sys, 0, N-1, 2, params);
        benchmark::DoNotOptimize(end);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
}
BENCHMARK(BM_FindSequenceEnd_WorstCase)->RangeMultiplier(2)->Range(1<<8, 1<<16)->Unit(benchmark::kNanosecond);

static void BM_FindSequenceEnd_BestCase(benchmark::State& st) {
    EventBus bus; 
    BH::Config cfg; 
    BH sys(st.range(0), bus, cfg);
    
    const size_t N = st.range(0);
    
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    std::iota(indices.begin(), indices.end(), 0u);
    
    // Best case: alternating quadrants (early termination)
    const auto params = MT::level_params(sys, 0);
    for (size_t i = 0; i < N; ++i) {
        int quad = static_cast<int>(i % 4);
        keys[i] = (uint64_t(quad) << params.level_shift);
    }
    
    MT::set_morton_arrays(sys, std::move(keys), std::move(indices));
    
    for (auto _ : st) {
        benchmark::DoNotOptimize(sys);
        size_t end = MT::find_sequence_end(sys, 0, N-1, 0, params); // Will find end at index 1
        benchmark::DoNotOptimize(end);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations());
}
BENCHMARK(BM_FindSequenceEnd_BestCase)->RangeMultiplier(2)->Range(1<<8, 1<<16)->Unit(benchmark::kNanosecond);

// ===========================================================================================
// MEMORY ACCESS PATTERN BENCHMARKS
// ===========================================================================================

static void BM_SequentialKeyAccess(benchmark::State& st) {
    const size_t N = st.range(0);
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    
    // Fill with predictable data
    std::iota(indices.begin(), indices.end(), 0u);
    for (size_t i = 0; i < N; ++i) {
        keys[i] = i * 12345; // Some pattern
    }
    
    volatile uint64_t sum = 0; // Prevent optimization
    
    for (auto _ : st) {
        // Sequential access pattern (cache-friendly)
        for (size_t i = 0; i < N; ++i) {
            sum += keys[indices[i]];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
    st.SetBytesProcessed(st.iterations() * st.range(0) * sizeof(uint64_t));
}
BENCHMARK(BM_SequentialKeyAccess)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

static void BM_RandomKeyAccess(benchmark::State& st) {
    const size_t N = st.range(0);
    std::vector<uint64_t> keys(N);
    std::vector<size_t> indices(N);
    
    // Fill with data and shuffle indices for random access
    std::iota(indices.begin(), indices.end(), 0u);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (size_t i = 0; i < N; ++i) {
        keys[i] = i * 12345;
    }
    
    volatile uint64_t sum = 0;
    
    for (auto _ : st) {
        // Random access pattern (cache-unfriendly)
        for (size_t i = 0; i < N; ++i) {
            sum += keys[indices[i]];
        }
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    
    st.SetItemsProcessed(st.iterations() * st.range(0));
    st.SetBytesProcessed(st.iterations() * st.range(0) * sizeof(uint64_t));
}
BENCHMARK(BM_RandomKeyAccess)->RangeMultiplier(2)->Range(1<<10, 1<<20)->Unit(benchmark::kMicrosecond);

// Custom main with configuration reporting
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    
    // Report system configuration
    std::cout << "\nMorton Operations Benchmark Suite\n";
    std::cout << "==================================\n";
    
    EventBus test_bus;
    BH::Config test_cfg;
    BH test_sys(16, test_bus, test_cfg);
    
    std::cout << "Configuration:\n";
    std::cout << "  MORTON_TOTAL_BITS: 42\n";
    std::cout << "  Radix sort threshold: 2048 particles\n";
    std::cout << "  Max particles per leaf: " << test_cfg.max_particles_per_leaf << "\n";
    std::cout << "  Threading enabled: " << (test_cfg.enable_threading ? "yes" : "no") << "\n";
    
#if defined(__ARM_NEON)
    std::cout << "  NEON support: available\n";
#else
    std::cout << "  NEON support: not available\n";
#endif
    
#ifdef _OPENMP
    std::cout << "  OpenMP support: available\n";
#else
    std::cout << "  OpenMP support: not available\n";
#endif
    
    std::cout << "\nRunning benchmarks...\n\n";
    
    ::benchmark::RunSpecifiedBenchmarks();
    
    std::cout << "\nBenchmark Notes:\n";
    std::cout << "================\n";
    std::cout << "- Times < 2048 particles use std::sort fallback\n";
    std::cout << "- Times >= 2048 particles use 4-pass radix sort\n";
    std::cout << "- Range splitting is O(N) linear scan\n";
    std::cout << "- Cache effects dominate at large N\n";
    std::cout << "- Use 'xcrun xctrace record --template \"Time Profiler\"' for detailed profiling\n";
    
    return 0;
}
