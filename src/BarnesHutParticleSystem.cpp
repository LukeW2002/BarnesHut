#include "BarnesHutParticleSystem.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <random>
#include <arm_neon.h>


#include <os/signpost.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// For GCC prefetch builtin (add conditional compilation)
#ifdef __GNUC__
    // __builtin_prefetch is available
#elif defined(_MSC_VER)
    #include <intrin.h>
    #define __builtin_prefetch(addr, rw, locality) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
    // Fallback: no-op prefetch
    #define __builtin_prefetch(addr, rw, locality) ((void)0)
#endif

// Morton/Z-order curve implementation for spatial sorting
class MortonCode {
public:
    // Interleave bits for 2D Morton code (Z-order)
    static uint64_t encode_morton_2d(uint32_t x, uint32_t y) {
        return (expand_bits_2d(x) << 1) | expand_bits_2d(y);
    }
    
    // Decode Morton code back to 2D coordinates  
    static void decode_morton_2d(uint64_t morton, uint32_t& x, uint32_t& y) {
        x = compact_bits_2d(morton >> 1);
        y = compact_bits_2d(morton);
    }

private:
    // Expand a 32-bit integer by inserting zeros between bits
    static uint64_t expand_bits_2d(uint32_t v) {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2))  & 0x3333333333333333;
        x = (x | (x << 1))  & 0x5555555555555555;
        return x;
    }
    
    // Compact bits by removing zeros between them
    static uint32_t compact_bits_2d(uint64_t x) {
        x &= 0x5555555555555555;
        x = (x ^ (x >> 1))  & 0x3333333333333333;
        x = (x ^ (x >> 2))  & 0x0F0F0F0F0F0F0F0F;
        x = (x ^ (x >> 4))  & 0x00FF00FF00FF00FF;
        x = (x ^ (x >> 8))  & 0x0000FFFF0000FFFF;
        x = (x ^ (x >> 16)) & 0x00000000FFFFFFFF;
        return static_cast<uint32_t>(x);
    }
};

class MortonEncoder {
public:
    static uint64_t encode_position(double x, double y, double min_x, double max_x, 
                                   double min_y, double max_y) {
        // Normalize to [0,1] range
        double range_x = max_x - min_x;
        double range_y = max_y - min_y;
        
        if (range_x < 1e-10) range_x = 1.0;
        if (range_y < 1e-10) range_y = 1.0;
        
        double norm_x = std::clamp((x - min_x) / range_x, 0.0, 1.0);
        double norm_y = std::clamp((y - min_y) / range_y, 0.0, 1.0);
        
        // Convert to integer coordinates (21 bits gives good precision)
        const uint32_t max_coord = (1U << 21) - 1;
        uint32_t ix = static_cast<uint32_t>(norm_x * max_coord);
        uint32_t iy = static_cast<uint32_t>(norm_y * max_coord);
        
        return MortonCode::encode_morton_2d(ix, iy);
    }
};

inline void BarnesHutParticleSystem::radix_sort_indices() {
    const size_t N = particle_count_;
    if (N == 0) return;

    // Type-safe index type matching your vectors
    using IndexT = typename decltype(morton_indices_)::value_type;
    
    // Use pre-sized buffers (no resize!)
    IndexT* __restrict out = tmp_indices_.data();       
    IndexT* __restrict in  = morton_indices_.data();     
    const uint64_t* __restrict keys = morton_keys_.data();

    // Fixed radix parameters with compile-time validation
    constexpr int RADIX_BITS = 11;
    constexpr int KEY_BITS = 42;                    // Morton 2D with 21-bit coordinates
    constexpr uint32_t RADIX = 1u << RADIX_BITS;   // 2048
    constexpr uint32_t RADIX_MASK = RADIX - 1;
    constexpr int NUM_PASSES = (KEY_BITS + RADIX_BITS - 1) / RADIX_BITS;  // = 4
    static_assert(NUM_PASSES == 4, "adjust passes for key width");

    // Use member stack histogram (no vector operations!)
    uint32_t* count = radix_histogram_.data();

    if (N < 2048) {
        std::sort(in, in + N, [&](IndexT a, IndexT b){ return keys[a] < keys[b]; });
        return;
    }

    int shift = 0;
    for (int pass = 0; pass < NUM_PASSES; ++pass, shift += RADIX_BITS) {
        // zero histogram (no assign/resize!)
        std::fill_n(count, RADIX, 0);

        // histogram
        for (size_t i = 0; i < N; ++i) {
            uint32_t bucket = (keys[in[i]] >> shift) & RADIX_MASK;
            ++count[bucket];
        }

        // prefix sum
        uint32_t sum = 0;
        for (uint32_t b = 0; b < RADIX; ++b) {
            uint32_t c = count[b];
            count[b] = sum;
            sum += c;
        }

        // scatter
        for (size_t i = 0; i < N; ++i) {
            IndexT idx = in[i];
            uint32_t bucket = (keys[idx] >> shift) & RADIX_MASK;
            out[count[bucket]++] = idx;
        }

        std::swap(in, out);
    }

    // Result is guaranteed to be in morton_indices_ after 4 passes (even number)
    // No conditional swap needed
}


inline void BarnesHutParticleSystem::ensure_indices_upto(size_t N) {
    if (indices_filled_ < N) {
        std::iota(morton_indices_.begin() + indices_filled_,
                  morton_indices_.begin() + N,
                  static_cast<size_t>(indices_filled_));
        indices_filled_ = N;
    }
}

inline void BarnesHutParticleSystem::sort_by_morton_key() {
    const size_t N = particle_count_;

    // Ensure morton_keys_ has space (but never resize morton_indices_!)
    if (morton_keys_.size() < N) morton_keys_.resize(max_particles_);
    
    // Use waterline system for morton_indices_
    ensure_indices_upto(N);

    // Compute keys (unchanged)
    #ifdef _OPENMP
    if (config_.enable_threading && N > 1000) {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            morton_keys_[i] = MortonEncoder::encode_position(
                static_cast<double>(positions_x_[i]),
                static_cast<double>(positions_y_[i]),
                bounds_min_x_, bounds_max_x_, bounds_min_y_, bounds_max_y_);
        }
    } else
    #endif
    {
        for (size_t i = 0; i < N; ++i) {
            morton_keys_[i] = MortonEncoder::encode_position(
                positions_x_[i], positions_y_[i],
                bounds_min_x_, bounds_max_x_,
                bounds_min_y_, bounds_max_y_);
        }
    }

    radix_sort_indices();
}

std::array<std::pair<size_t, size_t>, 4> BarnesHutParticleSystem::split_morton_range(size_t first, size_t last, int depth) const {
    std::array<std::pair<size_t, size_t>, 4> ranges;
    for (auto& r : ranges) r = {SIZE_MAX, SIZE_MAX}; // Initialize as empty
    
    if (first > last) return ranges;
    
    const int level_shift = MORTON_TOTAL_BITS - 2 * (depth + 1);
    const uint64_t mask = 3ULL << level_shift;
    static constexpr int z_to_child[4] = {0, 2, 1, 3};
    
    for (size_t i = first; i <= last; ) {
        // FIXED: Use sorted index instead of direct array access
        const size_t gi = morton_indices_[i];               // <-- KEY FIX
        const uint64_t ki = morton_keys_[gi];
        const int z_quad = int((ki & mask) >> level_shift);
        const int child_slot = z_to_child[z_quad];
        
        size_t j = i + 1;
        while (j <= last) {
            const size_t gj = morton_indices_[j];           // <-- KEY FIX
            const int z2 = int((morton_keys_[gj] & mask) >> level_shift);
            if (z2 != z_quad) break;
            ++j;
        }
        ranges[child_slot] = {i, j - 1};
        i = j;
    }
    
    // Debug verification (remove in release)
    #ifndef NDEBUG
    size_t total = 0;
    for (auto [a, b] : ranges) {
        if (a != SIZE_MAX) total += (b - a + 1);
    }
    assert(total == (last - first + 1));
    #endif
    
    return ranges;
}

BarnesHutParticleSystem::BarnesHutParticleSystem(size_t max_particles, EventBus& event_bus, const Config& config)
    : max_particles_(max_particles), particle_count_(0), event_bus_(event_bus),
      config_(config), iteration_count_(0), current_frame_(0), tree_valid_(false),
      root_node_index_(UINT32_MAX), next_free_node_(0),
      bounce_force_(1000.0f), damping_(0.999f), gravity_(0.0, 0.0),
      bounds_min_x_(-10.0), bounds_max_x_(10.0), bounds_min_y_(-10.0), bounds_max_y_(10.0),
      morton_ordering_enabled_(true), particles_need_reordering_(false), 
      last_morton_frame_(UINT32_MAX), indices_filled_(0) {  
    
    // Precompute theta squared for optimization
    config_.theta_squared = config_.theta * config_.theta;
    
    // Pre-size ALL hot-path work buffers to max capacity ONCE - NO MORE RESIZES!
    positions_x_.resize(max_particles);
    positions_y_.resize(max_particles);
    velocities_x_.resize(max_particles);
    velocities_y_.resize(max_particles);
    forces_x_.resize(max_particles);
    forces_y_.resize(max_particles);
    masses_.resize(max_particles);
    colors_r_.resize(max_particles);
    colors_g_.resize(max_particles);
    colors_b_.resize(max_particles);
    
    // Morton and tree work buffers - never resize again
    morton_indices_.resize(max_particles_);         
    tmp_indices_.resize(max_particles_);           
    morton_keys_.resize(max_particles_);           
    current_accel_x_.resize(max_particles_);       
    current_accel_y_.resize(max_particles_);       
    node_stack_.reserve(4096);                     
    
    // Leaf arrays: reserve generously but grow with push_back (no resize)
    leaf_pos_x_.reserve(max_particles_);
    leaf_pos_y_.reserve(max_particles_);
    leaf_mass_.reserve(max_particles_);
    leaf_idx_.reserve(max_particles_);
    
    // Tree nodes and other structures
    tree_nodes_.reserve(max_particles * 4);
    previous_positions_.reserve(max_particles);
    
    // Rendering arrays
    render_positions_.resize(max_particles * 2);
    render_colors_.resize(max_particles * 3);
    render_positions_x_.resize(max_particles);
    render_positions_y_.resize(max_particles);
    render_velocities_x_.resize(max_particles);
    render_velocities_y_.resize(max_particles);
    render_masses_.resize(max_particles);
    
    std::cout << "BarnesHutParticleSystem initialized with " << max_particles << " max particles\n";
    std::cout << "Theta: " << config_.theta << ", Tree caching: " << (config_.enable_tree_caching ? "enabled" : "disabled") << "\n";
    std::cout << "Morton Z-order optimization: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n";
    
#ifdef _OPENMP
    if (config_.enable_threading) {
        std::cout << "OpenMP threading enabled with " << omp_get_max_threads() << " threads\n";
        omp_set_dynamic(0);
        omp_set_num_threads(8);
    }
#endif
}



BarnesHutParticleSystem::~BarnesHutParticleSystem() = default;

std::vector<BarnesHutParticleSystem::QuadTreeBox> BarnesHutParticleSystem::get_quadtree_boxes() const {
    std::vector<QuadTreeBox> boxes;
    
    // Check if quadtree visualization is enabled and tree is valid
    if (!visualize_quadtree_ || !tree_valid_ || tree_nodes_.empty()) {
        return boxes;
    }

    // Use the proper recursive traversal method that already exists
    if (root_node_index_ != UINT32_MAX && root_node_index_ < tree_nodes_.size()) {
        collect_quadtree_boxes(root_node_index_, boxes);
    }
    
    return boxes;
}

void BarnesHutParticleSystem::collect_quadtree_boxes(uint32_t node_index, std::vector<QuadTreeBox>& boxes) const {
    if (node_index >= tree_nodes_.size()) return;
    
    const QuadTreeNode& node = tree_nodes_[node_index];
    
    // Add this node's bounding box
    boxes.emplace_back(
        static_cast<float>(node.min_x),
        static_cast<float>(node.min_y),
        static_cast<float>(node.max_x),
        static_cast<float>(node.max_y),
        static_cast<int>(node.depth),
        static_cast<int>(node.particle_count),
        node.is_leaf != 0
    );
    
    // Recursively add children if this is an internal node
    if (!node.is_leaf) {
        for (int i = 0; i < 4; ++i) {
            if (node.children[i] != UINT32_MAX) {
                collect_quadtree_boxes(node.children[i], boxes);
            }
        }
    }
}

bool BarnesHutParticleSystem::add_particle(const Vec2& pos, const Vec2& vel, float mass, const Vec3& color) {
    if (particle_count_ >= max_particles_) {
        return false;
    }
    
    size_t idx = particle_count_;
    positions_x_[idx] = pos.x;
    positions_y_[idx] = pos.y;
    velocities_x_[idx] = vel.x;
    velocities_y_[idx] = vel.y;
    forces_x_[idx] = 0.0f;
    forces_y_[idx] = 0.0f;
    masses_[idx] = mass;
    colors_r_[idx] = color.x;
    colors_g_[idx] = color.y;
    colors_b_[idx] = color.z;
    
    particle_count_++;
    tree_valid_ = false;  // Invalidate tree
    
    // NEW: Mark for Morton reordering when enough particles have been added
    if (morton_ordering_enabled_ && particle_count_ >= 100) {
        particles_need_reordering_ = true;
    }
    
    // Emit particle added event
    ParticleAddedEvent event{idx, static_cast<float>(pos.x), static_cast<float>(pos.y), 
                           static_cast<float>(vel.x), static_cast<float>(vel.y), 
                           mass, color.x, color.y, color.z};
    event_bus_.emit(Events::PARTICLE_ADDED, event);
    
    return true;
}

void BarnesHutParticleSystem::clear_particles() {
    particle_count_ = 0;
    iteration_count_ = 0;
    tree_valid_ = false;
    next_free_node_ = 0;
    current_frame_ = 0;
    previous_positions_.clear();
    particles_need_reordering_ = false;  // NEW: Reset reordering flag
}

void BarnesHutParticleSystem::remove_particle(size_t index) {
    if (index >= particle_count_) return;
    
    // Move last particle to this position (swap-remove)
    if (index < particle_count_ - 1) {
        positions_x_[index] = positions_x_[particle_count_ - 1];
        positions_y_[index] = positions_y_[particle_count_ - 1];
        velocities_x_[index] = velocities_x_[particle_count_ - 1];
        velocities_y_[index] = velocities_y_[particle_count_ - 1];
        forces_x_[index] = forces_x_[particle_count_ - 1];
        forces_y_[index] = forces_y_[particle_count_ - 1];
        masses_[index] = masses_[particle_count_ - 1];
        colors_r_[index] = colors_r_[particle_count_ - 1];
        colors_g_[index] = colors_g_[particle_count_ - 1];
        colors_b_[index] = colors_b_[particle_count_ - 1];
    }
    
    particle_count_--;
    tree_valid_ = false;
    
    // NEW: Mark for reordering after removal
    if (morton_ordering_enabled_) {
        particles_need_reordering_ = true;
    }
}

void BarnesHutParticleSystem::set_boundary(float min_x, float max_x, float min_y, float max_y) {
    bounds_min_x_ = min_x;
    bounds_max_x_ = max_x;
    bounds_min_y_ = min_y;
    bounds_max_y_ = max_y;
    tree_valid_ = false;  // Boundary change invalidates tree
    
    // NEW: Boundary changes may affect Morton ordering
    if (morton_ordering_enabled_) {
        particles_need_reordering_ = true;
    }
}

void BarnesHutParticleSystem::set_config(const Config& config) {
    config_ = config;
    config_.theta_squared = config_.theta * config_.theta;
    tree_valid_ = false;  // Config change may require tree rebuild
}


void BarnesHutParticleSystem::update(float dt) {
    if (particle_count_ == 0) return;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    current_frame_++;
    
    // Reset profiling counters
    reset_profiling_counters();
    
    // NEW: Apply Morton reordering if needed and beneficial
    if (morton_ordering_enabled_ && should_apply_morton_ordering()) {
        auto morton_start = std::chrono::high_resolution_clock::now();
        apply_morton_ordering();
        auto morton_end = std::chrono::high_resolution_clock::now();
        perf_stats_.morton_ordering_time_ms = std::chrono::duration<float, std::milli>(morton_end - morton_start).count();
        perf_stats_.morton_ordering_applied = true;
    } else {
        perf_stats_.morton_ordering_time_ms = 0.0f;
        perf_stats_.morton_ordering_applied = false;
    }
    
    // ONLY do initial setup if forces haven't been calculated yet
    // (For first frame or after major changes)
    bool need_initial_forces = (iteration_count_ == 0) || 
                               (std::all_of(forces_x_.begin(), forces_x_.begin() + particle_count_, 
                                          [](float f) { return f == 0.0f; }));
    
    if (need_initial_forces) {
        // Initial force calculation for first step
        std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
        std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
        
        calculate_bounds();
        auto tree_start = std::chrono::high_resolution_clock::now();
        build_tree();  // Initial tree build
        auto tree_end = std::chrono::high_resolution_clock::now();
        perf_stats_.tree_build_time_ms = std::chrono::duration<float, std::milli>(tree_end - tree_start).count();
        perf_stats_.tree_was_rebuilt = true;
        
        auto force_start = std::chrono::high_resolution_clock::now();
        calculate_forces_barnes_hut();
        auto force_end = std::chrono::high_resolution_clock::now();
        perf_stats_.tree_traversal_time_ms = std::chrono::duration<float, std::milli>(force_end - force_start).count();
    } else {
        perf_stats_.tree_build_time_ms = 0.0f;
        perf_stats_.tree_traversal_time_ms = 0.0f;
        perf_stats_.tree_was_rebuilt = false;
    }
    
    // Verlet integration handles ALL the physics:
    // - Position updates
    // - Tree rebuilding at new positions  
    // - Force recalculation
    // - Final velocity updates
    auto integration_start = std::chrono::high_resolution_clock::now();
    integrate_verlet(dt);
    auto integration_end = std::chrono::high_resolution_clock::now();
    perf_stats_.integration_time_ms = std::chrono::duration<float, std::milli>(integration_end - integration_start).count();
    
    // Update performance stats
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_frame_time = std::chrono::duration<float, std::milli>(total_end - start_time).count();
    update_detailed_performance_stats(total_frame_time);
    update_performance_stats();
    
    // Prepare rendering data
    prepare_render_data();
    
    iteration_count_++;
     
    // Emit events
    PhysicsUpdateEvent physics_event{dt, particle_count_, iteration_count_};
    event_bus_.emit(Events::PHYSICS_UPDATE, physics_event);
    
    RenderUpdateEvent render_event{
        render_positions_.data(), 
        render_colors_.data(), 
        particle_count_,
        static_cast<float>(bounds_min_x_), static_cast<float>(bounds_max_x_),
        static_cast<float>(bounds_min_y_), static_cast<float>(bounds_max_y_)
    };
    event_bus_.emit(Events::RENDER_UPDATE, render_event);
}


// NEW: Check if Morton ordering should be applied
bool BarnesHutParticleSystem::should_apply_morton_ordering() const {
    if (!particles_need_reordering_) return false;
    
    // Apply Morton ordering in these cases:
    // 1. When we have enough particles to benefit (>= 100)
    // 2. Not too frequently (every 60 frames at most)
    // 3. When tree is being rebuilt anyway
    
    static uint32_t last_morton_frame = 0;
    const uint32_t morton_interval = 60;  // Apply at most every 60 frames
    
    bool enough_particles = particle_count_ >= 100;
    bool time_to_reorder = (current_frame_ - last_morton_frame) >= morton_interval;
    bool tree_rebuilding = !tree_valid_ || should_rebuild_tree();
    
    if (enough_particles && (time_to_reorder || tree_rebuilding)) {
        last_morton_frame = current_frame_;
        return true;
    }
    
    return false;
}

void BarnesHutParticleSystem::build_tree() {
    auto total_start = std::chrono::high_resolution_clock::now();

    // Reset tree state
    tree_nodes_.clear();
    next_free_node_ = 0;
    perf_stats_.tree_nodes_created = 0;
    perf_stats_.tree_depth = 0;

    const size_t N = particle_count_;
    if (N == 0) {
        tree_valid_ = false;
        return;
    }
    
    // Reset compact leaf storage - clear() but don't resize
    leaf_pos_x_.clear();   
    leaf_pos_y_.clear();   
    leaf_mass_.clear();    
    leaf_idx_.clear();     
    leaf_offset_.clear();  
    leaf_count_.clear();

    // Estimate leaf count and reserve once
    const size_t leaf_threshold = std::clamp<size_t>(config_.max_particles_per_leaf, 1, 64);
    const size_t est_leaves = std::max<size_t>(1, (N + leaf_threshold - 1) / leaf_threshold);
    leaf_offset_.reserve(est_leaves);
    leaf_count_.reserve(est_leaves);

    // Reserve leaf arrays once (already done in constructor, but ensure capacity)
    leaf_pos_x_.reserve(N);
    leaf_pos_y_.reserve(N);
    leaf_mass_.reserve(N);
    leaf_idx_.reserve(N);

    // No assign/fill - we'll write every slot exactly once during leaf gather
    particle_leaf_slot_.resize(N);

    // Phase 1 & 2: Morton sort (now uses waterline system)
    sort_by_morton_key();

    // Phase 3: Build tree nodes using MEMBER stack (no reserve in loop!)
    const int leaf_cap = std::clamp<int>(static_cast<int>(config_.max_particles_per_leaf), 1, 64);
    size_t estimated_nodes = std::max<size_t>(64, (N * 8) / std::max(1, leaf_cap));
    tree_nodes_.reserve(estimated_nodes);

    root_node_index_ = create_node();
    QuadTreeNode& root = tree_nodes_[root_node_index_];
    root.min_x = bounds_min_x_;
    root.max_x = bounds_max_x_;
    root.min_y = bounds_min_y_;
    root.max_y = bounds_max_y_;
    root.depth = 0;
    root.particle_count = 0;
    for (int i = 0; i < 4; ++i) {
        root.children[i] = UINT32_MAX;
    }

    // Use MEMBER stack (no reserve each time!)
    node_stack_.clear();  // Just clear, don't reserve
    node_stack_.emplace_back(root_node_index_, 0, N - 1, 0);

    while (!node_stack_.empty()) {
        auto item = node_stack_.back();  // Avoid copy with auto
        node_stack_.pop_back();

        QuadTreeNode& node = tree_nodes_[item.node_index];
        node.depth = static_cast<uint16_t>(item.depth);

        const size_t count = item.last - item.first + 1;
        const size_t leaf_threshold_local = std::clamp<size_t>(config_.max_particles_per_leaf, 1, 64);
        bool should_be_leaf = (count <= leaf_threshold_local) || 
                             (item.depth >= static_cast<int>(config_.tree_depth_limit));

        if (should_be_leaf) {
            node.is_leaf = 1;
            const uint32_t cnt = static_cast<uint32_t>(count);
            node.particle_count = static_cast<uint8_t>(std::min<uint32_t>(cnt, 255));

            // Compact gather for this leaf (no resize, just push_back to pre-reserved)
            const uint32_t off = static_cast<uint32_t>(leaf_pos_x_.size());
            leaf_offset_.push_back(off);
            leaf_count_.push_back(cnt);

            // Expand leaf arrays by exact amount needed (push_back to reserved space)
            for (uint32_t t = 0; t < cnt; ++t) {
                const size_t global_idx = morton_indices_[item.first + t];
                
                leaf_pos_x_.push_back(positions_x_[global_idx]);
                leaf_pos_y_.push_back(positions_y_[global_idx]);
                leaf_mass_.push_back(masses_[global_idx]);
                leaf_idx_.push_back(static_cast<uint32_t>(global_idx));
                
                particle_leaf_slot_[global_idx] = off + t;  // Write every slot exactly once
            }

            node.leaf_first = off;
            node.leaf_last = off + cnt - 1;

            for (int i = 0; i < 4; ++i) {
                node.children[i] = UINT32_MAX;
            }

            perf_stats_.tree_depth = std::max(perf_stats_.tree_depth,
                                              static_cast<size_t>(item.depth));
            continue;
        }

        // Internal node processing
        node.is_leaf = 0;
        node.particle_count = static_cast<uint8_t>(count);

        const double cx = 0.5 * (node.min_x + node.max_x);
        const double cy = 0.5 * (node.min_y + node.max_y);

        auto morton_ranges = split_morton_range(item.first, item.last, item.depth);

        for (int quad = 0; quad < 4; ++quad) {
            const auto& [range_first, range_last] = morton_ranges[quad];

            if (range_first == SIZE_MAX || range_last == SIZE_MAX || range_first > range_last) {
                node.children[quad] = UINT32_MAX;
                continue;
            }

            uint32_t child_idx = create_node();
            node.children[quad] = child_idx;

            QuadTreeNode& parent = tree_nodes_[item.node_index];
            QuadTreeNode& child = tree_nodes_[child_idx];

            child.depth = static_cast<uint16_t>(item.depth + 1);
            child.particle_count = 0;
            for (int i = 0; i < 4; ++i) {
                child.children[i] = UINT32_MAX;
            }

            // Set quadrant bounds
            switch (quad) {
                case 0: // SW
                    child.min_x = parent.min_x; child.max_x = cx;
                    child.min_y = parent.min_y; child.max_y = cy;
                    break;
                case 1: // SE
                    child.min_x = cx; child.max_x = parent.max_x;
                    child.min_y = parent.min_y; child.max_y = cy;
                    break;
                case 2: // NW
                    child.min_x = parent.min_x; child.max_x = cx;
                    child.min_y = cy; child.max_y = parent.max_y;
                    break;
                case 3: // NE
                    child.min_x = cx; child.max_x = parent.max_x;
                    child.min_y = cy; child.max_y = parent.max_y;
                    break;
            }

            node_stack_.emplace_back(child_idx, range_first, range_last, item.depth + 1);
        }
    }

    // Rest of build_tree: COM calculation (unchanged)
    calculate_center_of_mass(root_node_index_);

    // Only resize previous_positions_ when particle count changes (not every rebuild)
    if (previous_positions_.size() != particle_count_) {
        previous_positions_.resize(particle_count_);
    }
    for (size_t i = 0; i < particle_count_; ++i) {
        previous_positions_[i] = Vector2d(positions_x_[i], positions_y_[i]);
    }

    tree_valid_ = true;
    perf_stats_.tree_nodes_created = next_free_node_;

    auto total_end = std::chrono::high_resolution_clock::now();
    perf_stats_.tree_build_time_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();
}

void BarnesHutParticleSystem::apply_morton_ordering() {
    if (particle_count_ == 0) return;
    
    // 1. Calculate Morton codes using SAME encoder as tree builder
    std::vector<std::pair<uint64_t, size_t>> morton_particles;
    morton_particles.reserve(particle_count_);
    
    // ✅ FIXED: Use consistent 21-bit precision encoder
    for (size_t i = 0; i < particle_count_; ++i) {
        uint64_t morton = MortonEncoder::encode_position(
            positions_x_[i], positions_y_[i],
            bounds_min_x_, bounds_max_x_,
            bounds_min_y_, bounds_max_y_);
        morton_particles.emplace_back(morton, i);
    }
    
    // 2. Sort particles by Morton code (Z-order)
    radix_sort_indices();
    
    // 🔧 FIXED: Populate morton_indices_ from sorted morton_particles
    morton_indices_.resize(particle_count_);
    for (size_t i = 0; i < particle_count_; ++i) {
        morton_indices_[i] = morton_particles[i].second;  // Extract the original index
    }
    
    // 3. Now reorder all particle data according to Morton order,
    //apply_morton_permutation_to_arrays();  
    
    // 4. Force tree rebuild and mark as no longer needing reordering
    tree_valid_ = false;
    particles_need_reordering_ = false;
}

bool BarnesHutParticleSystem::should_rebuild_tree() const {
    if (!config_.enable_tree_caching || previous_positions_.size() != particle_count_) {
        return true;
    }

    // ratio threshold (use your config)
    const double ratio_threshold = std::clamp<double>(config_.tree_rebuild_threshold, 0.0, 1.0);

    // distance threshold = 1% of the larger side of the current AABB
    const double world_w = bounds_max_x_ - bounds_min_x_;
    const double world_h = bounds_max_y_ - bounds_min_y_;
    const double dist_threshold = 0.01 * std::max(world_w, world_h);
    const double dist_threshold_sq = dist_threshold * dist_threshold;

    size_t moved = 0;
    for (size_t i = 0; i < particle_count_; ++i) {
        const double dx = positions_x_[i] - previous_positions_[i].x();
        const double dy = positions_y_[i] - previous_positions_[i].y();
        if (dx*dx + dy*dy > dist_threshold_sq) moved++;
    }
    const double moved_ratio = (particle_count_ ? double(moved)/particle_count_ : 0.0);
    return moved_ratio > ratio_threshold;
}



uint32_t BarnesHutParticleSystem::create_node() {
    if (next_free_node_ >= tree_nodes_.size()) {
        tree_nodes_.resize(next_free_node_ + 1);
    }
    return next_free_node_++;
}

void BarnesHutParticleSystem::calculate_center_of_mass(uint32_t node_index) {
    QuadTreeNode& node = tree_nodes_[node_index];
   
    if (node.is_leaf) {
        if (node.leaf_first == UINT32_MAX || node.leaf_last == UINT32_MAX) {
            node.com_x = node.com_y = 0.0f;
            node.total_mass = 0.0f;
            node.bound_r = 0.0f;
            return;
        }

        const uint32_t off = node.leaf_first;
        const uint32_t cnt = node.leaf_last - node.leaf_first + 1;

        float total_m = 0.0f, wx = 0.0f, wy = 0.0f;
        
        // First pass: Calculate center of mass using compact leaf arrays
        for (uint32_t t = 0; t < cnt; ++t) {
            const float m = leaf_mass_[off + t];
            total_m += m;
            wx += leaf_pos_x_[off + t] * m;
            wy += leaf_pos_y_[off + t] * m;
        }

        node.total_mass = total_m;
        if (total_m > 0.0f) {
            node.com_x = wx / total_m;
            node.com_y = wy / total_m;
        } else {
            node.com_x = node.com_y = 0.0f;
        }

        // Second pass: Calculate tight bounding radius from COM using compact arrays
        float max_dist_sq = 0.0f;
        for (uint32_t t = 0; t < cnt; ++t) {
            const float dx = leaf_pos_x_[off + t] - node.com_x;
            const float dy = leaf_pos_y_[off + t] - node.com_y;
            const float dist_sq = dx*dx + dy*dy;
            max_dist_sq = std::max(max_dist_sq, dist_sq);
        }
        node.bound_r = std::sqrt(max_dist_sq); 
        
        node.particle_count = static_cast<uint8_t>(std::min<uint32_t>(cnt, 255));
        return;
    }
    
    // Internal node: Calculate from children (unchanged)
    float total_mass = 0.0f;
    float weighted_x = 0.0f;
    float weighted_y = 0.0f;
    
    // First pass: Accumulate mass and COM
    for (int i = 0; i < 4; ++i) {
        if (node.children[i] != UINT32_MAX) {
            calculate_center_of_mass(node.children[i]);  // Recursive call
            const QuadTreeNode& child = tree_nodes_[node.children[i]];
            
            if (child.total_mass > 0.0f) {
                weighted_x += child.com_x * child.total_mass;
                weighted_y += child.com_y * child.total_mass;
                total_mass += child.total_mass;
            }
        }
    }
    
    // Set parent COM
    if (total_mass > 0.0f) {
        node.com_x = weighted_x / total_mass;
        node.com_y = weighted_y / total_mass;
        node.total_mass = total_mass;
    } else {
        node.com_x = node.com_y = 0.0f;
        node.total_mass = 0.0f;
    }
    
    // Second pass: Calculate tight bounding radius for internal node
    float max_bound_r = 0.0f;
    for (int i = 0; i < 4; ++i) {
        if (node.children[i] != UINT32_MAX) {
            const QuadTreeNode& child = tree_nodes_[node.children[i]];
            
            if (child.total_mass > 0.0f) {
                const float dx = child.com_x - node.com_x;
                const float dy = child.com_y - node.com_y;
                const float dist_to_child_com = std::sqrt(dx*dx + dy*dy);
                const float child_extent = dist_to_child_com + child.bound_r;
                
                max_bound_r = std::max(max_bound_r, child_extent);
            }
        }
    }
    node.bound_r = max_bound_r;
}

void BarnesHutParticleSystem::compute_frame_constants() {
    // Calculate bounds (this already happens in build_tree)
    calculate_bounds();
    
    // Compute adaptive softening for this frame
    const float world_span_x = float(bounds_max_x_ - bounds_min_x_);
    const float world_span_y = float(bounds_max_y_ - bounds_min_y_);
    const float world_scale = std::max(world_span_x, world_span_y);
    
    // Cache softening for the entire frame
    const float eps = config_.softening_rel * world_scale;
    frame_eps2_ = eps * eps;
    
    // Also cache root COM for centering (avoid repeated lookups)
    root_com_x_ = tree_nodes_[root_node_index_].com_x;
    root_com_y_ = tree_nodes_[root_node_index_].com_y;
}

void BarnesHutParticleSystem::calculate_forces_barnes_hut() {
    if (!tree_valid_ || root_node_index_ == UINT32_MAX) return;

    // Warm up cache (optional; consider profiling with/without)
    prefetch_tree_nodes();

    const float* const positions_x = positions_x_.data();
    const float* const positions_y = positions_y_.data();
    const float* const masses      = masses_.data();

    // Parallelize over particles (each i writes distinct forces_[i] → no data races)
    #ifdef _OPENMP
    if (config_.enable_threading) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)particle_count_; ++i) {
            float fx = 0.0, fy = 0.0;
            // NOTE: calculate_force_on_particle_iterative must not touch shared counters in threaded mode
            (void)calculate_force_on_particle_iterative((size_t)i, fx, fy, positions_x, positions_y, masses);
            forces_x_[(size_t)i] += fx;
            forces_y_[(size_t)i] += fy;
        }
        return; // done
    }
    #endif

    // Fallback single-thread path
    for (size_t i = 0; i < particle_count_; ++i) {
        float fx = 0.0, fy = 0.0;
        (void)calculate_force_on_particle_iterative(i, fx, fy, positions_x, positions_y, masses);
        forces_x_[i] += fx;
        forces_y_[i] += fy;
    }
}


// If you are reading this function and can understand it, I am sorry. 
    // For I am to dumb to understand, the eldritch horrors of my own creation.
 void BarnesHutParticleSystem::process_leaf_forces_neon(
    const QuadTreeNode& node, int i_local, float px, float py, float gi,
    float& fx, float& fy,
    const float* __restrict leaf_x,
    const float* __restrict leaf_y,
    const float* __restrict leaf_m) const
{
    if (node.leaf_first == UINT32_MAX) return;

    const uint32_t cnt = node.leaf_last - node.leaf_first + 1;

    // Convert to double precision to match scalar path
    const double px_d = static_cast<double>(px);
    const double py_d = static_cast<double>(py);

    // NEON constants
    const float32x4_t vgi = vdupq_n_f32(gi);
    const float32x4_t veps = vdupq_n_f32(EPS_SQ);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t vone_five = vdupq_n_f32(1.5f);
    
    // For self-exclusion masking
    const int32x4_t vi_local = vdupq_n_s32(i_local);
    const int32x4_t vidx_base = {0, 1, 2, 3};
    const uint32x4_t vone_bits = vdupq_n_u32(0x3F800000); // 1.0f as bits

    // Force accumulators
    float32x4_t acc_fx = vdupq_n_f32(0.0f);
    float32x4_t acc_fy = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    
    // Main loop: process 8 particles at a time (2 x 4-wide NEON vectors)
    for (; i + 7 < cnt; i += 8) {
        // Prefetch next cache line for large leaves
        if (i + 32 < cnt) {
            __builtin_prefetch(&leaf_x[i + 32], 0, 1);
            __builtin_prefetch(&leaf_y[i + 32], 0, 1);
            __builtin_prefetch(&leaf_m[i + 32], 0, 1);
        }

        // Load 8 particles as two 4-wide vectors
        const float32x4_t vx0 = vld1q_f32(&leaf_x[i]);
        const float32x4_t vx1 = vld1q_f32(&leaf_x[i + 4]);
        const float32x4_t vy0 = vld1q_f32(&leaf_y[i]);
        const float32x4_t vy1 = vld1q_f32(&leaf_y[i + 4]);
        const float32x4_t vm0 = vld1q_f32(&leaf_m[i]);
        const float32x4_t vm1 = vld1q_f32(&leaf_m[i + 4]);

        // Create index vectors for self-exclusion
        const int32x4_t vidx0 = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i)), vidx_base);
        const int32x4_t vidx1 = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i + 4)), vidx_base);

        // Self-exclusion masks: 1.0f if not self, 0.0f if self
        const uint32x4_t neq0 = vmvnq_u32(vceqq_s32(vi_local, vidx0));
        const uint32x4_t neq1 = vmvnq_u32(vceqq_s32(vi_local, vidx1));
        const float32x4_t mask0 = vreinterpretq_f32_u32(vandq_u32(neq0, vone_bits));
        const float32x4_t mask1 = vreinterpretq_f32_u32(vandq_u32(neq1, vone_bits));

        // FIXED: Compute displacements in double precision to match scalar path
        // Convert each particle position to double, subtract in double, then convert back to float
        float dx0_arr[4], dy0_arr[4], dx1_arr[4], dy1_arr[4];
        
        // First batch of 4
        for (int j = 0; j < 4; ++j) {
            double dx_d = static_cast<double>(leaf_x[i + j]) - px_d;
            double dy_d = static_cast<double>(leaf_y[i + j]) - py_d;
            dx0_arr[j] = static_cast<float>(dx_d);
            dy0_arr[j] = static_cast<float>(dy_d);
        }
        
        // Second batch of 4
        for (int j = 0; j < 4; ++j) {
            double dx_d = static_cast<double>(leaf_x[i + 4 + j]) - px_d;
            double dy_d = static_cast<double>(leaf_y[i + 4 + j]) - py_d;
            dx1_arr[j] = static_cast<float>(dx_d);
            dy1_arr[j] = static_cast<float>(dy_d);
        }

        // Load the double-precision-computed displacements
        const float32x4_t dx0 = vld1q_f32(dx0_arr);
        const float32x4_t dx1 = vld1q_f32(dx1_arr);
        const float32x4_t dy0 = vld1q_f32(dy0_arr);
        const float32x4_t dy1 = vld1q_f32(dy1_arr);

        // Compute r² = dx² + dy² + ε²
        float32x4_t r2_0, r2_1;
        #if defined(__aarch64__)
        r2_0 = vfmaq_f32(vfmaq_f32(veps, dx0, dx0), dy0, dy0);
        r2_1 = vfmaq_f32(vfmaq_f32(veps, dx1, dx1), dy1, dy1);
        #else
        r2_0 = vmlaq_f32(vmlaq_f32(veps, dx0, dx0), dy0, dy0);
        r2_1 = vmlaq_f32(vmlaq_f32(veps, dx1, dx1), dy1, dy1);
        #endif

        // Fast reciprocal square root with Newton-Raphson refinement
        float32x4_t inv_r0 = vrsqrteq_f32(r2_0);
        float32x4_t inv_r1 = vrsqrteq_f32(r2_1);

        // One Newton-Raphson iteration
        float32x4_t inv_r0_sq = vmulq_f32(inv_r0, inv_r0);
        float32x4_t inv_r1_sq = vmulq_f32(inv_r1, inv_r1);
        
        #if defined(__aarch64__)
        inv_r0 = vmulq_f32(inv_r0, vfmsq_f32(vone_five, vhalf, vmulq_f32(r2_0, inv_r0_sq)));
        inv_r1 = vmulq_f32(inv_r1, vfmsq_f32(vone_five, vhalf, vmulq_f32(r2_1, inv_r1_sq)));
        #else
        float32x4_t half_r2_inv0_sq = vmulq_f32(vhalf, vmulq_f32(r2_0, inv_r0_sq));
        float32x4_t half_r2_inv1_sq = vmulq_f32(vhalf, vmulq_f32(r2_1, inv_r1_sq));
        inv_r0 = vmulq_f32(inv_r0, vsubq_f32(vone_five, half_r2_inv0_sq));
        inv_r1 = vmulq_f32(inv_r1, vsubq_f32(vone_five, half_r2_inv1_sq));
        #endif

        // Compute 1/r³ = (1/r) * (1/r²)
        const float32x4_t inv_r3_0 = vmulq_f32(inv_r0, vmulq_f32(inv_r0, inv_r0));
        const float32x4_t inv_r3_1 = vmulq_f32(inv_r1, vmulq_f32(inv_r1, inv_r1));

        // Compute force magnitude (no G_GALACTIC to match test)
        const float32x4_t s0 = vmulq_f32(mask0, vmulq_f32(vgi, vmulq_f32(vm0, inv_r3_0)));
        const float32x4_t s1 = vmulq_f32(mask1, vmulq_f32(vgi, vmulq_f32(vm1, inv_r3_1)));

        // Accumulate forces: F = s * dr
        #if defined(__aarch64__)
        acc_fx = vfmaq_f32(acc_fx, s0, dx0);
        acc_fx = vfmaq_f32(acc_fx, s1, dx1);
        acc_fy = vfmaq_f32(acc_fy, s0, dy0);
        acc_fy = vfmaq_f32(acc_fy, s1, dy1);
        #else
        acc_fx = vmlaq_f32(acc_fx, s0, dx0);
        acc_fx = vmlaq_f32(acc_fx, s1, dx1);
        acc_fy = vmlaq_f32(acc_fy, s0, dy0);
        acc_fy = vmlaq_f32(acc_fy, s1, dy1);
        #endif
    }

    // Handle remaining 4-particle block with double precision
    if (i + 3 < cnt) {
        const float32x4_t vx0 = vld1q_f32(&leaf_x[i]);
        const float32x4_t vy0 = vld1q_f32(&leaf_y[i]);
        const float32x4_t vm0 = vld1q_f32(&leaf_m[i]);

        const int32x4_t vidx0 = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i)), vidx_base);
        const uint32x4_t neq0 = vmvnq_u32(vceqq_s32(vi_local, vidx0));
        const float32x4_t mask0 = vreinterpretq_f32_u32(vandq_u32(neq0, vone_bits));

        // FIXED: Double precision displacement computation
        float dx0_arr[4], dy0_arr[4];
        for (int j = 0; j < 4; ++j) {
            double dx_d = static_cast<double>(leaf_x[i + j]) - px_d;
            double dy_d = static_cast<double>(leaf_y[i + j]) - py_d;
            dx0_arr[j] = static_cast<float>(dx_d);
            dy0_arr[j] = static_cast<float>(dy_d);
        }

        const float32x4_t dx0 = vld1q_f32(dx0_arr);
        const float32x4_t dy0 = vld1q_f32(dy0_arr);

        #if defined(__aarch64__)
        const float32x4_t r2_0 = vfmaq_f32(vfmaq_f32(veps, dx0, dx0), dy0, dy0);
        #else
        const float32x4_t r2_0 = vmlaq_f32(vmlaq_f32(veps, dx0, dx0), dy0, dy0);
        #endif

        float32x4_t inv_r0 = vrsqrteq_f32(r2_0);
        const float32x4_t inv_r0_sq = vmulq_f32(inv_r0, inv_r0);
        
        #if defined(__aarch64__)
        inv_r0 = vmulq_f32(inv_r0, vfmsq_f32(vone_five, vhalf, vmulq_f32(r2_0, inv_r0_sq)));
        #else
        const float32x4_t half_r2_inv0_sq = vmulq_f32(vhalf, vmulq_f32(r2_0, inv_r0_sq));
        inv_r0 = vmulq_f32(inv_r0, vsubq_f32(vone_five, half_r2_inv0_sq));
        #endif

        const float32x4_t inv_r3_0 = vmulq_f32(inv_r0, vmulq_f32(inv_r0, inv_r0));
        const float32x4_t s0 = vmulq_f32(mask0, vmulq_f32(vgi, vmulq_f32(vm0, inv_r3_0)));

        #if defined(__aarch64__)
        acc_fx = vfmaq_f32(acc_fx, s0, dx0);
        acc_fy = vfmaq_f32(acc_fy, s0, dy0);
        #else
        acc_fx = vmlaq_f32(acc_fx, s0, dx0);
        acc_fy = vmlaq_f32(acc_fy, s0, dy0);
        #endif

        i += 4;
    }

    // Horizontal reduction: sum all lanes to get scalar results
    float fx_total, fy_total;
    #if defined(__aarch64__)
    fx_total = vaddvq_f32(acc_fx);
    fy_total = vaddvq_f32(acc_fy);
    #else
    const float32x2_t sum_fx_lo = vadd_f32(vget_low_f32(acc_fx), vget_high_f32(acc_fx));
    const float32x2_t sum_fy_lo = vadd_f32(vget_low_f32(acc_fy), vget_high_f32(acc_fy));
    fx_total = vget_lane_f32(sum_fx_lo, 0) + vget_lane_f32(sum_fx_lo, 1);
    fy_total = vget_lane_f32(sum_fy_lo, 0) + vget_lane_f32(sum_fy_lo, 1);
    #endif

    // FIXED: Handle remaining particles with scalar code using double precision
    for (; i < cnt; ++i) {
        if (static_cast<int>(i) == i_local) continue; // Skip self

        // Use double precision for displacement calculation to match scalar path
        const double dx_d = static_cast<double>(leaf_x[i]) - px_d;
        const double dy_d = static_cast<double>(leaf_y[i]) - py_d;
        const float dx = static_cast<float>(dx_d);
        const float dy = static_cast<float>(dy_d);
        
        const float r2 = dx*dx + dy*dy + EPS_SQ;
        const float inv_r = rsqrt_fast(r2);
        const float inv_r3 = inv_r * inv_r * inv_r;
        const float s = gi * leaf_m[i] * inv_r3;  // No G_GALACTIC
        
        fx_total += s * dx;
        fy_total += s * dy;
    }

    // Write back accumulated forces
    fx += fx_total;
    fy += fy_total;
}


void BarnesHutParticleSystem::process_leaf_forces_neon_centered(
    const QuadTreeNode& node, int i_local, float px_c, float py_c, float gi,
    float& fx, float& fy, float ox, float oy,
    const float* __restrict leaf_x,
    const float* __restrict leaf_y,
    const float* __restrict leaf_m) const
{
    if (node.leaf_first == UINT32_MAX) return;

    const uint32_t cnt = node.leaf_last - node.leaf_first + 1;

    const float32x4_t vpx_c = vdupq_n_f32(px_c);
    const float32x4_t vpy_c = vdupq_n_f32(py_c);
    const float32x4_t vox = vdupq_n_f32(ox);
    const float32x4_t voy = vdupq_n_f32(oy);
    const float32x4_t vgi = vdupq_n_f32(gi);
    const float32x4_t veps = vdupq_n_f32(EPS_SQ);
    const float32x4_t vhalf = vdupq_n_f32(0.5f);
    const float32x4_t vone_five = vdupq_n_f32(1.5f);
    
    const int32x4_t vi_local = vdupq_n_s32(i_local);
    const int32x4_t vidx_base = {0, 1, 2, 3};
    const uint32x4_t vone_bits = vdupq_n_u32(0x3F800000);

    float32x4_t acc_fx = vdupq_n_f32(0.0f);
    float32x4_t acc_fy = vdupq_n_f32(0.0f);

    uint32_t i = 0;
    
    for (; i + 7 < cnt; i += 8) {
        if (i + 32 < cnt) {
            __builtin_prefetch(&leaf_x[i + 32], 0, 1);
            __builtin_prefetch(&leaf_y[i + 32], 0, 1);
            __builtin_prefetch(&leaf_m[i + 32], 0, 1);
        }

        const float32x4_t vx0 = vld1q_f32(&leaf_x[i]);
        const float32x4_t vx1 = vld1q_f32(&leaf_x[i + 4]);
        const float32x4_t vy0 = vld1q_f32(&leaf_y[i]);
        const float32x4_t vy1 = vld1q_f32(&leaf_y[i + 4]);
        const float32x4_t vm0 = vld1q_f32(&leaf_m[i]);
        const float32x4_t vm1 = vld1q_f32(&leaf_m[i + 4]);

        const int32x4_t vidx0 = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i)), vidx_base);
        const int32x4_t vidx1 = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i + 4)), vidx_base);

        const uint32x4_t neq0 = vmvnq_u32(vceqq_s32(vi_local, vidx0));
        const uint32x4_t neq1 = vmvnq_u32(vceqq_s32(vi_local, vidx1));
        const float32x4_t mask0 = vreinterpretq_f32_u32(vandq_u32(neq0, vone_bits));
        const float32x4_t mask1 = vreinterpretq_f32_u32(vandq_u32(neq1, vone_bits));

        // Center coordinates first, then compute displacement
        const float32x4_t lx0_c = vsubq_f32(vx0, vox);
        const float32x4_t lx1_c = vsubq_f32(vx1, vox);
        const float32x4_t ly0_c = vsubq_f32(vy0, voy);
        const float32x4_t ly1_c = vsubq_f32(vy1, voy);

        const float32x4_t dx0 = vsubq_f32(lx0_c, vpx_c);
        const float32x4_t dx1 = vsubq_f32(lx1_c, vpx_c);
        const float32x4_t dy0 = vsubq_f32(ly0_c, vpy_c);
        const float32x4_t dy1 = vsubq_f32(ly1_c, vpy_c);

        float32x4_t r2_0, r2_1;
        #if defined(__aarch64__)
        r2_0 = vfmaq_f32(vfmaq_f32(veps, dx0, dx0), dy0, dy0);
        r2_1 = vfmaq_f32(vfmaq_f32(veps, dx1, dx1), dy1, dy1);
        #else
        r2_0 = vmlaq_f32(vmlaq_f32(veps, dx0, dx0), dy0, dy0);
        r2_1 = vmlaq_f32(vmlaq_f32(veps, dx1, dx1), dy1, dy1);
        #endif

        float32x4_t inv_r0 = vrsqrteq_f32(r2_0);
        float32x4_t inv_r1 = vrsqrteq_f32(r2_1);

        float32x4_t inv_r0_sq = vmulq_f32(inv_r0, inv_r0);
        float32x4_t inv_r1_sq = vmulq_f32(inv_r1, inv_r1);
        
        #if defined(__aarch64__)
        inv_r0 = vmulq_f32(inv_r0, vfmsq_f32(vone_five, vhalf, vmulq_f32(r2_0, inv_r0_sq)));
        inv_r1 = vmulq_f32(inv_r1, vfmsq_f32(vone_five, vhalf, vmulq_f32(r2_1, inv_r1_sq)));
        #else
        float32x4_t half_r2_inv0_sq = vmulq_f32(vhalf, vmulq_f32(r2_0, inv_r0_sq));
        float32x4_t half_r2_inv1_sq = vmulq_f32(vhalf, vmulq_f32(r2_1, inv_r1_sq));
        inv_r0 = vmulq_f32(inv_r0, vsubq_f32(vone_five, half_r2_inv0_sq));
        inv_r1 = vmulq_f32(inv_r1, vsubq_f32(vone_five, half_r2_inv1_sq));
        #endif

        const float32x4_t inv_r3_0 = vmulq_f32(inv_r0, vmulq_f32(inv_r0, inv_r0));
        const float32x4_t inv_r3_1 = vmulq_f32(inv_r1, vmulq_f32(inv_r1, inv_r1));

        // No G_GALACTIC to match test
        const float32x4_t s0 = vmulq_f32(mask0, vmulq_f32(vgi, vmulq_f32(vm0, inv_r3_0)));
        const float32x4_t s1 = vmulq_f32(mask1, vmulq_f32(vgi, vmulq_f32(vm1, inv_r3_1)));

        #if defined(__aarch64__)
        acc_fx = vfmaq_f32(acc_fx, s0, dx0);
        acc_fx = vfmaq_f32(acc_fx, s1, dx1);
        acc_fy = vfmaq_f32(acc_fy, s0, dy0);
        acc_fy = vfmaq_f32(acc_fy, s1, dy1);
        #else
        acc_fx = vmlaq_f32(acc_fx, s0, dx0);
        acc_fx = vmlaq_f32(acc_fx, s1, dx1);
        acc_fy = vmlaq_f32(acc_fy, s0, dy0);
        acc_fy = vmlaq_f32(acc_fy, s1, dy1);
        #endif
    }

    // Handle remainder and scalar fallback the same way
    float fx_total, fy_total;
    #if defined(__aarch64__)
    fx_total = vaddvq_f32(acc_fx);
    fy_total = vaddvq_f32(acc_fy);
    #else
    const float32x2_t sum_fx_lo = vadd_f32(vget_low_f32(acc_fx), vget_high_f32(acc_fx));
    const float32x2_t sum_fy_lo = vadd_f32(vget_low_f32(acc_fy), vget_high_f32(acc_fy));
    fx_total = vget_lane_f32(sum_fx_lo, 0) + vget_lane_f32(sum_fx_lo, 1);
    fy_total = vget_lane_f32(sum_fy_lo, 0) + vget_lane_f32(sum_fy_lo, 1);
    #endif

    // Scalar remainder with centered coordinates
    for (; i < cnt; ++i) {
        if (static_cast<int>(i) == i_local) continue;

        const float lx_c = leaf_x[i] - ox;
        const float ly_c = leaf_y[i] - oy;
        const float dx = lx_c - px_c;
        const float dy = ly_c - py_c;
        const float r2 = dx*dx + dy*dy + EPS_SQ;
        const float inv_r = rsqrt_fast(r2);
        const float inv_r3 = inv_r * inv_r * inv_r;
        const float s = gi * leaf_m[i] * inv_r3;  // No G_GALACTIC
        
        fx_total += s * dx;
        fy_total += s * dy;
    }

    fx += fx_total;
    fy += fy_total;
}

void test_scaling_bottleneck() {
}
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
__attribute__((always_inline, hot))
__attribute__((always_inline, hot))
inline bool BarnesHutParticleSystem::calculate_force_on_particle_iterative(
    size_t i, float& fx, float& fy,
    const float* __restrict positions_x,
    const float* __restrict positions_y,
    const float* __restrict masses) const
{
    if (UNLIKELY(root_node_index_ == UINT32_MAX)) return false;

    const float px = positions_x[i];
    const float py = positions_y[i]; 
    const float gi = masses[i];
    const float eps2 = EPS_SQ;
    const float theta2 = config_.theta_squared;

    // NEW: Use root COM as origin to center all coordinates
    const float ox = tree_nodes_[root_node_index_].com_x;
    const float oy = tree_nodes_[root_node_index_].com_y;
    const float px_c = px - ox;
    const float py_c = py - oy;

    float fx_acc = 0.0f, fy_acc = 0.0f;

    constexpr size_t STACK_SIZE = 2048;
    uint32_t stack[STACK_SIZE];
    size_t top = 0;
    stack[top++] = root_node_index_;

    const uint32_t my_slot = particle_leaf_slot_[static_cast<uint32_t>(i)];

    // Batch buffers for internal nodes
    float acc_fx_scalar = 0.0f, acc_fy_scalar = 0.0f;
    int acc_n = 0;
    alignas(16) float ax[4], ay[4], am[4];

    float32x4_t acc_fx_vec = vdupq_n_f32(0.0f);
    float32x4_t acc_fy_vec = vdupq_n_f32(0.0f);

    auto flush_internal_batch = [&]() {
        if (UNLIKELY(acc_n == 0)) return;
        
        if (acc_n < 4) {
            for (int t = 0; t < acc_n; ++t) {
                const float dx = ax[t], dy = ay[t];
                const float r2 = dx*dx + dy*dy + eps2;
                const float rinv = rsqrt_fast(r2);
                const float rinv3 = rinv * rinv * rinv;
                const float s = G_GALACTIC * am[t] * rinv3;
                acc_fx_scalar += s * dx;
                acc_fy_scalar += s * dy;
            }
        } else {
            const float32x4_t dx = vld1q_f32(ax);
            const float32x4_t dy = vld1q_f32(ay);
            const float32x4_t m  = vld1q_f32(am);
            const float32x4_t veps2 = vdupq_n_f32(eps2);
            const float32x4_t vG = vdupq_n_f32(G_GALACTIC);

            float32x4_t r2 = vfmaq_f32(vfmaq_f32(veps2, dx, dx), dy, dy);
            float32x4_t rinv = rsqrt_nr_f32x4(r2);
            float32x4_t rinv3 = vmulq_f32(rinv, vmulq_f32(rinv, rinv));
            float32x4_t s = vmulq_f32(vG, vmulq_f32(m, rinv3));

            acc_fx_vec = vfmaq_f32(acc_fx_vec, s, dx);
            acc_fy_vec = vfmaq_f32(acc_fy_vec, s, dy);
        }
        acc_n = 0;
    };

    while (LIKELY(top > 0)) {
        const uint32_t node_idx = stack[--top];
        const QuadTreeNode& node = tree_nodes_[node_idx];
        
        if (UNLIKELY(node.total_mass <= 0.0f)) continue;

        if (node.is_leaf) {
            flush_internal_batch();

            const uint32_t off = node.leaf_first;
            if (LIKELY(off != UINT32_MAX)) {
                const uint32_t cnt = node.leaf_last - off + 1;
                
                const int i_local = (my_slot >= off && my_slot < off + cnt)
                                  ? static_cast<int>(my_slot - off) : -1;

                if (cnt >= config_.max_particles_per_leaf) {
                    __builtin_prefetch(&leaf_pos_x_[off], 0, 3);
                    __builtin_prefetch(&leaf_pos_y_[off], 0, 3);
                    __builtin_prefetch(&leaf_mass_[off], 0, 3);
                    
                    float fx_leaf = 0.0f, fy_leaf = 0.0f;
                    // Pass centered coordinates to NEON
                    process_leaf_forces_neon_centered(node, i_local, px_c, py_c, gi, 
                                                    fx_leaf, fy_leaf, ox, oy,
                                                    &leaf_pos_x_[off], 
                                                    &leaf_pos_y_[off], 
                                                    &leaf_mass_[off]);
                    fx_acc += fx_leaf * G_GALACTIC;
                    fy_acc += fy_leaf * G_GALACTIC;
                } else {
                    // Scalar path with centered coordinates
                    for (uint32_t t = 0; t < cnt; ++t) {
                        if (UNLIKELY(static_cast<int>(t) == i_local)) continue;
                        
                        const float lx_c = leaf_pos_x_[off + t] - ox;
                        const float ly_c = leaf_pos_y_[off + t] - oy;
                        const float dx = lx_c - px_c;
                        const float dy = ly_c - py_c;
                        const float r2 = dx*dx + dy*dy + eps2;
                        const float rinv = rsqrt_fast(r2);
                        const float rinv3 = rinv * rinv * rinv;
                        const float s = G_GALACTIC * gi * leaf_mass_[off + t] * rinv3;
                        
                        fx_acc += s * dx;
                        fy_acc += s * dy;
                    }
                }
            }
            continue;
        }

        // Internal node with centered coordinates
        const float nx_c = node.com_x - ox;
        const float ny_c = node.com_y - oy;
        const float dx = nx_c - px_c;
        const float dy = ny_c - py_c;
        const float dist_sq = dx*dx + dy*dy + eps2;
        const float bound_r_sq = node.bound_r * node.bound_r;
        
        if (LIKELY(bound_r_sq < theta2 * dist_sq)) {
            // Batch for NEON processing
            ax[acc_n] = dx;
            ay[acc_n] = dy;
            am[acc_n] = gi * node.total_mass;
            if (++acc_n == 4) {
                flush_internal_batch();
            }
        } else {
            // Traverse children
            uint32_t children_to_push[4];
            int child_count = 0;
            
            if (node.children[3] != UINT32_MAX) children_to_push[child_count++] = node.children[3];
            if (node.children[2] != UINT32_MAX) children_to_push[child_count++] = node.children[2];
            if (node.children[1] != UINT32_MAX) children_to_push[child_count++] = node.children[1];
            if (node.children[0] != UINT32_MAX) children_to_push[child_count++] = node.children[0];

            if (LIKELY(top + child_count <= STACK_SIZE)) {
                if (child_count > 0) {
                    __builtin_prefetch(&tree_nodes_[children_to_push[0]], 0, 1);
                }
                
                switch (child_count) {
                    case 4: stack[top + 3] = children_to_push[3]; [[fallthrough]];
                    case 3: stack[top + 2] = children_to_push[2]; [[fallthrough]];
                    case 2: stack[top + 1] = children_to_push[1]; [[fallthrough]];
                    case 1: stack[top + 0] = children_to_push[0]; break;
                    case 0: break;
                }
                top += child_count;
            } else {
                // Stack overflow - approximate this node
                flush_internal_batch();
                const float rinv = rsqrt_fast(dist_sq);
                const float rinv3 = rinv * rinv * rinv;
                const float s = G_GALACTIC * gi * node.total_mass * rinv3;
                fx_acc += s * dx;
                fy_acc += s * dy;
            }
        }
    }

    flush_internal_batch();
    fx_acc += acc_fx_scalar;
    fy_acc += acc_fy_scalar;

#if defined(__aarch64__)
    fx_acc += vaddvq_f32(acc_fx_vec);
    fy_acc += vaddvq_f32(acc_fy_vec);
#else
    float32x2_t sum_fx = vadd_f32(vget_low_f32(acc_fx_vec), vget_high_f32(acc_fx_vec));
    float32x2_t sum_fy = vadd_f32(vget_low_f32(acc_fy_vec), vget_high_f32(acc_fy_vec));
    fx_acc += vget_lane_f32(sum_fx, 0) + vget_lane_f32(sum_fx, 1);
    fy_acc += vget_lane_f32(sum_fy, 0) + vget_lane_f32(sum_fy, 1);
#endif

    fx += fx_acc;
    fy += fy_acc;
    return true;
}

void BarnesHutParticleSystem::integrate_verlet(float dt) {
    const float dt_half = 0.5f * dt;

    // A0: current accelerations from previous forces
    for (size_t i = 0; i < particle_count_; ++i) {
        const float inv_m = 1.0f / masses_[i];
        current_accel_x_[i] = forces_x_[i] * inv_m;
        current_accel_y_[i] = forces_y_[i] * inv_m;
    }

    // STEP 1: v(t+dt/2) = v(t) + a(t)*dt/2
    for (size_t i = 0; i < particle_count_; ++i) {
        velocities_x_[i] += current_accel_x_[i] * dt_half;
        velocities_y_[i] += current_accel_y_[i] * dt_half;
    }

    // STEP 2: x(t+dt) = x(t) + v(t+dt/2)*dt
    for (size_t i = 0; i < particle_count_; ++i) {
        positions_x_[i] += velocities_x_[i] * dt;
        positions_y_[i] += velocities_y_[i] * dt;
    }

    // Zero forces for the new-force pass
    std::fill_n(forces_x_.data(), particle_count_, 0.0f);
    std::fill_n(forces_y_.data(), particle_count_, 0.0f);

    // STEP 3: recompute forces at x(t+dt)
    if (should_rebuild_tree()) { calculate_bounds(); build_tree(); }
    calculate_forces_barnes_hut();  // fills forces_*

    // STEP 4: v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
    for (size_t i = 0; i < particle_count_; ++i) {
        const float inv_m = 1.0f / masses_[i];
        velocities_x_[i] += (forces_x_[i] * inv_m) * dt_half;
        velocities_y_[i] += (forces_y_[i] * inv_m) * dt_half;
    }

    // Optional: apply damping symmetrically at the end
    if (damping_ != 1.0f) {
        for (size_t i = 0; i < particle_count_; ++i) {
            velocities_x_[i] *= damping_;
            velocities_y_[i] *= damping_;
        }
    }

    // Periodic Morton reorder check
    if (morton_ordering_enabled_ && (iteration_count_ % 30 == 0)) {
        check_for_morton_reordering_need();
    }
}


// NEW: Check if particles have moved enough to warrant Morton reordering
void BarnesHutParticleSystem::check_for_morton_reordering_need() {
    if (particle_count_ < 100) return;  // Not worth it for small particle counts
    
    // Sample a subset of particles to check movement
    const size_t sample_size = std::min(particle_count_, size_t(50));
    const double movement_threshold = 0.1;  // Relative to world size
    
    double world_size = std::max(bounds_max_x_ - bounds_min_x_, bounds_max_y_ - bounds_min_y_);
    double threshold_distance = movement_threshold * world_size;
    double threshold_distance_sq = threshold_distance * threshold_distance;
    
    size_t moved_particles = 0;
    
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = (i * particle_count_) / sample_size;  // Distributed sampling
        if (idx < previous_positions_.size()) {
            double dx = positions_x_[idx] - previous_positions_[idx].x();
            double dy = positions_y_[idx] - previous_positions_[idx].y();
            if (dx*dx + dy*dy > threshold_distance_sq) {
                moved_particles++;
            }
        }
    }
    
    // If more than 30% of sampled particles moved significantly, mark for reordering
    if (moved_particles > sample_size * 0.3) {
        particles_need_reordering_ = true;
    }
}


void BarnesHutParticleSystem::prepare_render_data() {
    for (size_t i = 0; i < particle_count_; ++i) {
        // Interleaved position data for legacy compatibility
        render_positions_[i * 2 + 0] = static_cast<float>(positions_x_[i]);
        render_positions_[i * 2 + 1] = static_cast<float>(positions_y_[i]);
        
        // Color data
        render_colors_[i * 3 + 0] = colors_r_[i];
        render_colors_[i * 3 + 1] = colors_g_[i];
        render_colors_[i * 3 + 2] = colors_b_[i];
        
        // Separate arrays for easier access
        render_positions_x_[i] = static_cast<float>(positions_x_[i]);
        render_positions_y_[i] = static_cast<float>(positions_y_[i]);
        render_velocities_x_[i] = static_cast<float>(velocities_x_[i]);
        render_velocities_y_[i] = static_cast<float>(velocities_y_[i]);
        render_masses_[i] = static_cast<float>(masses_[i]);
    }
}

void BarnesHutParticleSystem::calculate_bounds() {
    if (particle_count_ == 0) return;
    
    float min_x = positions_x_[0], max_x = positions_x_[0];
    float min_y = positions_y_[0], max_y = positions_y_[0];
    
    for (size_t i = 1; i < particle_count_; ++i) {
        min_x = std::min(min_x, positions_x_[i]);
        max_x = std::max(max_x, positions_x_[i]);
        min_y = std::min(min_y, positions_y_[i]);
        max_y = std::max(max_y, positions_y_[i]);
    }
    
    bounds_min_x_ = min_x;
    bounds_max_x_ = max_x;
    bounds_min_y_ = min_y;
    bounds_max_y_ = max_y;
}

void BarnesHutParticleSystem::update_performance_stats() {
    // Calculate efficiency ratio
    const size_t total_calculations = perf_stats_.force_calculations + perf_stats_.approximations_used;
    if (total_calculations > 0) {
        perf_stats_.efficiency_ratio = static_cast<float>(perf_stats_.approximations_used) / total_calculations;
    } else {
        perf_stats_.efficiency_ratio = 0.0f;
    }
    
    // Calculate cache hit ratio (placeholder for now)
    if (config_.enable_tree_caching && !perf_stats_.tree_was_rebuilt) {
        perf_stats_.cache_hit_ratio = 1.0f;
    } else {
        perf_stats_.cache_hit_ratio = 0.0f;
    }
}

// Accessor methods
Vec2 BarnesHutParticleSystem::get_position(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(positions_x_[index], positions_y_[index]);  // Direct assignment (both float)
}

Vec2 BarnesHutParticleSystem::get_velocity(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(velocities_x_[index], velocities_y_[index]);  // Direct assignment (both float)
}

Vec2 BarnesHutParticleSystem::get_force(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(forces_x_[index], forces_y_[index]);  // Direct assignment (both float)
}

float BarnesHutParticleSystem::get_mass(size_t index) const {
    if (index >= particle_count_) return 0.0f;
    return masses_[index];  // Direct return (both float)
}

Vec3 BarnesHutParticleSystem::get_color(size_t index) const {
    if (index >= particle_count_) return Vec3();
    return Vec3(colors_r_[index], colors_g_[index], colors_b_[index]);
}

// NEW: Set Morton ordering enabled/disabled
void BarnesHutParticleSystem::set_morton_ordering_enabled(bool enabled) {
    morton_ordering_enabled_ = enabled;
    if (enabled && particle_count_ >= 100) {
        particles_need_reordering_ = true;
    }
}

// Tree visualization for debugging
BarnesHutParticleSystem::TreeNode BarnesHutParticleSystem::get_tree_visualization() const {
    if (!tree_valid_ || root_node_index_ == UINT32_MAX) {
        return TreeNode{};
    }
    return build_tree_visualization(root_node_index_);
}

BarnesHutParticleSystem::TreeNode BarnesHutParticleSystem::build_tree_visualization(uint32_t node_index) const {
    const QuadTreeNode& node = tree_nodes_[node_index];
    
    TreeNode viz_node;
    viz_node.center_x = (node.min_x + node.max_x) * 0.5;
    viz_node.center_y = (node.min_y + node.max_y) * 0.5;
    viz_node.width = node.width();
    viz_node.is_leaf = node.is_leaf;
    viz_node.particle_count = node.particle_count;
    
    if (!node.is_leaf) {
        for (int i = 0; i < 4; ++i) {
            if (node.children[i] != UINT32_MAX) {
                viz_node.children.push_back(build_tree_visualization(node.children[i]));
            }
        }
    }
    
    return viz_node;
}

// Performance analysis and debugging methods

void BarnesHutParticleSystem::reset_profiling_counters() {
    current_tree_nodes_visited_ = 0;
    current_leaf_nodes_hit_ = 0;
    current_internal_nodes_hit_ = 0;
    current_theta_tests_ = 0;
    current_theta_passed_ = 0;
    current_tree_depth_sum_ = 0.0f;
}

void BarnesHutParticleSystem::update_detailed_performance_stats(float total_frame_time) {
    // Calculate particles per second
    if (perf_stats_.force_calculation_time_ms > 0) {
        perf_stats_.particles_per_second = (particle_count_ * 1000.0f) / perf_stats_.force_calculation_time_ms;
        perf_stats_.force_calculations_per_second = (perf_stats_.force_calculations * 1000.0f) / perf_stats_.force_calculation_time_ms;
    }
    
    // Tree traversal statistics
    perf_stats_.tree_nodes_visited = current_tree_nodes_visited_;
    perf_stats_.leaf_nodes_hit = current_leaf_nodes_hit_;
    perf_stats_.internal_nodes_hit = current_internal_nodes_hit_;
    
    // Theta test statistics
    perf_stats_.total_theta_tests = current_theta_tests_;
    perf_stats_.theta_tests_passed = current_theta_passed_;
    if (current_theta_tests_ > 0) {
        perf_stats_.theta_pass_ratio = static_cast<float>(current_theta_passed_) / current_theta_tests_;
    }
    
    // Average tree depth per particle
    if (particle_count_ > 0) {
        perf_stats_.avg_tree_depth_per_particle = current_tree_depth_sum_ / particle_count_;
    }
    
    // Memory usage estimation
    size_t particle_memory = particle_count_ * (sizeof(double) * 6 + sizeof(float) * 3); // pos, vel, force, mass, color
    size_t tree_memory = tree_nodes_.size() * sizeof(QuadTreeNode);
    size_t other_memory = render_positions_.size() * sizeof(float) + render_colors_.size() * sizeof(float);
    perf_stats_.memory_usage_mb = static_cast<float>(particle_memory + tree_memory + other_memory) / (1024.0f * 1024.0f);
    
    // Brute force comparison
    perf_stats_.brute_force_equivalent_ops = particle_count_ * particle_count_; // N²
    if (perf_stats_.force_calculations + perf_stats_.approximations_used > 0) {
        perf_stats_.speedup_vs_brute_force = static_cast<float>(perf_stats_.brute_force_equivalent_ops) / 
                                           (perf_stats_.force_calculations + perf_stats_.approximations_used);
    }
}

void BarnesHutParticleSystem::apply_morton_permutation_to_arrays() {
    if (particle_count_ == 0 || morton_indices_.empty()) return;
    // Reuse inv_perm_ buffer (don't clear, just resize)
    inv_perm_.resize(particle_count_);
    
    // Build inverse permutation: old -> new
    for (size_t new_idx = 0; new_idx < particle_count_; ++new_idx) {
        size_t old_idx = morton_indices_[new_idx];
        assert(old_idx < particle_count_); // Safety check
        inv_perm_[old_idx] = new_idx;
    }
    
    // Debug: Verify permutation is valid bijection
    #ifndef NDEBUG
    std::vector<uint8_t> seen(particle_count_, 0);
    for (size_t v : inv_perm_) { 
        assert(v < particle_count_ && !seen[v]); 
        seen[v] = 1; 
    }
    #endif
    
    // In-place permutation using old -> new mapping (single-threaded)
    permute_in_place(positions_x_, inv_perm_, particle_count_, visited_);
    permute_in_place(positions_y_, inv_perm_, particle_count_, visited_);
    permute_in_place(velocities_x_, inv_perm_, particle_count_, visited_);
    permute_in_place(velocities_y_, inv_perm_, particle_count_, visited_);
    permute_in_place(masses_, inv_perm_, particle_count_, visited_);
    permute_in_place(colors_r_, inv_perm_, particle_count_, visited_);
    permute_in_place(colors_g_, inv_perm_, particle_count_, visited_);
    permute_in_place(colors_b_, inv_perm_, particle_count_, visited_);
    
    // CRITICAL: Also permute Morton keys to maintain sort order
    permute_in_place(morton_keys_, inv_perm_, particle_count_, visited_);
    
    // Debug: Verify keys are still sorted after permutation
    #ifndef NDEBUG
    for (size_t i = 1; i < particle_count_; ++i) {
        assert(morton_keys_[i-1] <= morton_keys_[i]);
    }
    
    // Debug: Check top-level quadrant distribution  
    int counts[4] = {0, 0, 0, 0};
    const int level_shift0 = MORTON_TOTAL_BITS - 2;  // depth = 0
    const uint64_t mask0 = 3ULL << level_shift0;
    static constexpr int z_to_child[4] = { 0, 2, 1, 3 };
    
    for (size_t i = 0; i < particle_count_; ++i) {
        int z_quad = static_cast<int>((morton_keys_[i] & mask0) >> level_shift0);
        int child_slot = z_to_child[z_quad];
        counts[child_slot]++;
    }
    
    std::cout << "✅ Morton keys remain sorted after permutation\n";
    std::cout << "📊 Top-level quadrant distribution SW,SE,NW,NE = " 
              << counts[0] << "," << counts[1] << "," << counts[2] << "," << counts[3] << "\n";
    #endif
    
    // Keys are now sorted and aligned - morton_keys_ IS the sorted array!
    
    // Clean up morton_indices_ but keep inv_perm_ and visited_ for reuse
    //morton_indices_.clear();
    // DON'T clear inv_perm_ or visited_ - keep them sized for next rebuild
}


void BarnesHutParticleSystem::nuclear_particle_separation() {
    std::cout << "💥 NUCLEAR PARTICLE SEPARATION (for 20k particles at same spot)\n";
    
    if (particle_count_ == 0) return;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // MASSIVE separation distance for 20k particles
    const double separation_radius = 50.0;  // HUGE radius
    const double min_distance = 0.5;        // Minimum distance between particles
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Different strategies based on particle count
    if (particle_count_ < 1000) {
        // Small count: Use grid layout
        int grid_size = static_cast<int>(std::ceil(std::sqrt(particle_count_)));
        double spacing = separation_radius * 2.0 / grid_size;
        
        for (size_t i = 0; i < particle_count_; ++i) {
            int row = i / grid_size;
            int col = i % grid_size;
            
            positions_x_[i] = (col - grid_size/2.0) * spacing;
            positions_y_[i] = (row - grid_size/2.0) * spacing;
        }
        
        std::cout << "   ✅ Grid separation applied to " << particle_count_ << " particles\n";
        
    } else {
        // Large count: Use random distribution in expanding circles
        std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
        std::uniform_real_distribution<double> radius_dist(min_distance, separation_radius);
        
        // First particle at origin
        if (particle_count_ > 0) {
            positions_x_[0] = 0.0;
            positions_y_[0] = 0.0;
        }
        
        // Rest in expanding spiral/random pattern
        for (size_t i = 1; i < particle_count_; ++i) {
            double angle = angle_dist(gen);
            
            // Expand radius based on particle index for better spread
            double base_radius = min_distance * std::sqrt(static_cast<double>(i));
            double radius = std::min(base_radius, separation_radius);
            
            positions_x_[i] = radius * std::cos(angle);
            positions_y_[i] = radius * std::sin(angle);
            
            // Add small random jitter to prevent perfect alignment
            std::uniform_real_distribution<double> jitter(-min_distance*0.1, min_distance*0.1);
            positions_x_[i] += jitter(gen);
            positions_y_[i] += jitter(gen);
        }
        
        std::cout << "   ✅ Spiral separation applied to " << particle_count_ << " particles\n";
    }
    
    // Reset velocities to prevent immediate re-clustering
    std::uniform_real_distribution<double> vel_dist(-1.0, 1.0);
    for (size_t i = 0; i < particle_count_; ++i) {
        velocities_x_[i] = vel_dist(gen);
        velocities_y_[i] = vel_dist(gen);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float fix_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    std::cout << "   💥 Nuclear separation complete in " << fix_time << " ms\n";
    std::cout << "   🎯 Particles spread over " << (separation_radius * 2) << " unit radius\n";
    std::cout << "   ⚡ Tree depth should DROP from 200 to <20\n\n";
    
    // Force complete tree rebuild
    tree_valid_ = false;
    particles_need_reordering_ = true;
}

void BarnesHutParticleSystem::prefetch_tree_nodes() const {
    // Prefetch commonly accessed tree nodes into cache
    if (!tree_valid_ || tree_nodes_.empty()) return;
    
    // Prefetch root and first few levels
    size_t prefetch_count = std::min(size_t(64), tree_nodes_.size());
    
    for (size_t i = 0; i < prefetch_count; ++i) {
        __builtin_prefetch(&tree_nodes_[i], 0, 3);  // GCC builtin for cache prefetch
    }
}

// Original optimization methods preserved but now called as part of default behavior
void BarnesHutParticleSystem::optimize_particle_layout() {
    //apply_morton_ordering();  // Just use the new default implementation
    void test_scaling_bottleneck();
}

// Missing debug and analysis methods implementation
bool BarnesHutParticleSystem::detect_performance_issues() const {
    bool issues_found = false;
    
    // Check for overlapping particles (major performance killer)
    std::cout << "Checking for overlapping particles...\n";
    size_t overlap_count = 0;
    const double min_distance_sq = 1e-6; // Very small threshold
    
    for (size_t i = 0; i < std::min(particle_count_, size_t(100)); ++i) {
        for (size_t j = i + 1; j < std::min(particle_count_, size_t(100)); ++j) {
            double dx = positions_x_[i] - positions_x_[j];
            double dy = positions_y_[i] - positions_y_[j];
            double dist_sq = dx * dx + dy * dy;
            
            if (dist_sq < min_distance_sq) {
                overlap_count++;
                if (overlap_count == 1) {
                    std::cout << "⚠️ Found overlapping particles at (" 
                              << positions_x_[i] << ", " << positions_y_[i] << ")\n";
                }
            }
        }
    }
    
    if (overlap_count > 0) {
        std::cout << "⚠️ CRITICAL: " << overlap_count << " overlapping particles detected (sample of 100)\n";
        std::cout << "   This causes extremely deep tree recursion and slow performance\n";
        issues_found = true;
    }
    
    // Check for NaN values
    for (size_t i = 0; i < particle_count_; ++i) {
        if (std::isnan(positions_x_[i]) || std::isnan(positions_y_[i]) ||
            std::isnan(velocities_x_[i]) || std::isnan(velocities_y_[i]) ||
            std::isnan(masses_[i])) {
            std::cout << "⚠️ CRITICAL: NaN values detected in particle " << i << "\n";
            issues_found = true;
            break;
        }
    }
    
    // Check for infinite values
    for (size_t i = 0; i < particle_count_; ++i) {
        if (std::isinf(positions_x_[i]) || std::isinf(positions_y_[i]) ||
            std::isinf(velocities_x_[i]) || std::isinf(velocities_y_[i])) {
            std::cout << "⚠️ CRITICAL: Infinite values detected in particle " << i << "\n";
            issues_found = true;
            break;
        }
    }
    
    // Check for zero or negative masses
    for (size_t i = 0; i < particle_count_; ++i) {
        if (masses_[i] <= 0) {
            std::cout << "⚠️ CRITICAL: Invalid mass " << masses_[i] << " in particle " << i << "\n";
            issues_found = true;
            break;
        }
    }
    
    return issues_found;
}

void BarnesHutParticleSystem::print_performance_analysis() const {
    const auto& stats = get_performance_stats();
    
    std::cout << "\n=== BARNES-HUT PERFORMANCE ANALYSIS ===\n";
    std::cout << "Particles: " << particle_count_ << "\n";
    std::cout << "Tree nodes: " << stats.tree_nodes_created << "\n";
    std::cout << "Tree depth: " << stats.tree_depth << "\n";
    std::cout << "Theta: " << config_.theta << "\n";
    std::cout << "Morton ordering: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n";
    
    std::cout << "\nTiming Breakdown:\n";
    std::cout << "  Tree build: " << stats.tree_build_time_ms << " ms\n";
    std::cout << "  Force calc: " << stats.force_calculation_time_ms << " ms\n";
    std::cout << "    - Traversal: " << stats.tree_traversal_time_ms << " ms\n";
    std::cout << "    - Boundary: " << stats.boundary_forces_time_ms << " ms\n";
    std::cout << "    - Gravity: " << stats.gravity_forces_time_ms << " ms\n";
    std::cout << "  Integration: " << stats.integration_time_ms << " ms\n";
    if (stats.morton_ordering_applied) {
        std::cout << "  Morton ordering: " << stats.morton_ordering_time_ms << " ms\n";
    }
    
    std::cout << "\nAlgorithm Efficiency:\n";
    std::cout << "  Direct calculations: " << stats.force_calculations << "\n";
    std::cout << "  Approximations: " << stats.approximations_used << "\n";
    std::cout << "  Total operations: " << (stats.force_calculations + stats.approximations_used) << "\n";
    std::cout << "  Brute force equivalent: " << (particle_count_ * particle_count_) << "\n";
    std::cout << "  Speedup: " << stats.speedup_vs_brute_force << "x\n";
    std::cout << "  Efficiency: " << (stats.efficiency_ratio * 100.0f) << "% approximations\n";
    
    std::cout << "\nTree Traversal:\n";
    std::cout << "  Nodes visited: " << stats.tree_nodes_visited << "\n";
    std::cout << "  Leaf hits: " << stats.leaf_nodes_hit << "\n";
    std::cout << "  Internal hits: " << stats.internal_nodes_hit << "\n";
    std::cout << "  Avg depth/particle: " << stats.avg_tree_depth_per_particle << "\n";
    
    std::cout << "\nTheta Testing:\n";
    std::cout << "  Tests performed: " << stats.total_theta_tests << "\n";
    std::cout << "  Tests passed: " << stats.theta_tests_passed << "\n";
    std::cout << "  Pass ratio: " << (stats.theta_pass_ratio * 100.0f) << "%\n";
    
    if (morton_ordering_enabled_) {
        std::cout << "\nMorton Ordering:\n";
        std::cout << "  Status: Enabled\n";
        std::cout << "  Applied this frame: " << (stats.morton_ordering_applied ? "Yes" : "No") << "\n";
        if (stats.morton_ordering_applied) {
            std::cout << "  Time: " << stats.morton_ordering_time_ms << " ms\n";
        }
    }
    
    // Performance analysis
    std::cout << "\n=== ANALYSIS ===\n";
    
    if (stats.force_calculation_time_ms > 10.0f && particle_count_ < 5000) {
        std::cout << "⚠️ ISSUE: Force calculation too slow for particle count\n";
        std::cout << "   - Try increasing theta (current: " << config_.theta << ")\n";
        std::cout << "   - Check if tree is being rebuilt every frame\n";
    }
    
    if (stats.efficiency_ratio < 0.5f) {
        std::cout << "⚠️ ISSUE: Low approximation ratio (" << (stats.efficiency_ratio * 100.0f) << "%)\n";
        std::cout << "   - Barnes-Hut not providing expected speedup\n";
        std::cout << "   - Try increasing theta for more approximations\n";
    }
    
    if (stats.tree_depth > 20) {
        std::cout << "⚠️ ISSUE: Tree too deep (" << stats.tree_depth << " levels)\n";
        std::cout << "   - Particles may be overlapping or highly clustered\n";
        std::cout << "   - Check for duplicate positions\n";
    }
    
    std::cout << "==========================================\n\n";
}


void BarnesHutParticleSystem::diagnose_30ms_issue() {
    std::cout << "\n=== DIAGNOSING 30MS FORCE CALCULATION ===\n";
    
    const auto& stats = get_performance_stats();
    
    std::cout << "Current particle count: " << particle_count_ << "\n";
    std::cout << "Force calculation time: " << stats.force_calculation_time_ms << " ms\n";
    std::cout << "Morton ordering: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n";
    
    // Expected performance for different particle counts
    if (particle_count_ < 1000) {
        std::cout << "⚠️ CRITICAL: 30ms is WAY too slow for " << particle_count_ << " particles\n";
        std::cout << "   Expected: <1ms for this count\n";
    } else if (particle_count_ < 5000) {
        std::cout << "⚠️ ISSUE: 30ms is too slow for " << particle_count_ << " particles\n";
        std::cout << "   Expected: 1-5ms for this count\n";
    }
    
    std::cout << "=========================================\n\n";
}

void BarnesHutParticleSystem::test_theta_performance() {
    if (particle_count_ == 0) {
        std::cout << "No particles to test theta performance\n";
        return;
    }
    
    std::cout << "\n=== THETA PERFORMANCE TEST ===\n";
    std::cout << "Testing different theta values with " << particle_count_ << " particles\n";
    std::cout << "Morton ordering: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n\n";
    
    // Store original configuration
    Config original_config = config_;
    
    // Test different theta values
    std::vector<float> theta_values = {0.3f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f};
    
    for (float theta : theta_values) {
        // Update configuration
        config_.theta = theta;
        config_.theta_squared = theta * theta;
        
        // Force tree rebuild
        tree_valid_ = false;
        
        // Time a single force calculation pass
        auto start = std::chrono::high_resolution_clock::now();
        
        // Clear forces and calculate using Barnes-Hut
        std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
        std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
        
        // Build tree if needed
        if (!tree_valid_) {
            build_tree();
        }
        
        // Reset counters
        perf_stats_.force_calculations = 0;
        perf_stats_.approximations_used = 0;
        
        // Calculate forces
        calculate_forces_barnes_hut();
        
        auto end = std::chrono::high_resolution_clock::now();
        float time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        // Calculate metrics
        size_t total_ops = perf_stats_.force_calculations + perf_stats_.approximations_used;
        float efficiency = total_ops > 0 ? (float)perf_stats_.approximations_used / total_ops : 0.0f;
        size_t brute_force_ops = particle_count_ * particle_count_;
        float speedup = total_ops > 0 ? (float)brute_force_ops / total_ops : 0.0f;
        
        std::cout << "Theta " << theta << ":\n";
        std::cout << "  Time: " << time_ms << " ms\n";
        std::cout << "  Direct calculations: " << perf_stats_.force_calculations << "\n";
        std::cout << "  Approximations: " << perf_stats_.approximations_used << "\n";
        std::cout << "  Total operations: " << total_ops << "\n";
        std::cout << "  Efficiency: " << (efficiency * 100.0f) << "% approximations\n";
        std::cout << "  Speedup vs brute force: " << speedup << "x\n";
        std::cout << "  Tree nodes: " << perf_stats_.tree_nodes_created << "\n";
        std::cout << "  Tree depth: " << perf_stats_.tree_depth << "\n\n";
    }
    
    // Restore original configuration
    config_ = original_config;
    tree_valid_ = false;  // Force rebuild with original theta
    
    std::cout << "Theta test complete. Original configuration restored.\n";
    std::cout << "==============================\n\n";
}

void BarnesHutParticleSystem::diagnose_tree_traversal_bottleneck() const {
    std::cout << "\n=== TREE TRAVERSAL BOTTLENECK ANALYSIS ===\n";
    
    if (particle_count_ == 0) {
        std::cout << "No particles to analyze\n";
        return;
    }
    
    const auto& stats = get_performance_stats();
    
    // 1. Check for overlapping particles (major cause of deep trees)
    std::cout << "1. CHECKING FOR OVERLAPPING PARTICLES:\n";
    size_t overlap_count = 0;
    size_t near_overlap_count = 0;
    const double min_distance_sq = 1e-10;  // Essentially overlapping
    const double near_distance_sq = 1e-6;   // Very close
    
    // Sample first 1000 particles to avoid O(N²) check on large datasets
    size_t sample_size = std::min(particle_count_, size_t(1000));
    
    for (size_t i = 0; i < sample_size; ++i) {
        for (size_t j = i + 1; j < sample_size; ++j) {
            double dx = positions_x_[i] - positions_x_[j];
            double dy = positions_y_[i] - positions_y_[j];
            double dist_sq = dx * dx + dy * dy;
            
            if (dist_sq < min_distance_sq) {
                overlap_count++;
                if (overlap_count <= 5) {  // Show first few examples
                    std::cout << "   ⚠️  OVERLAP: Particles " << i << " and " << j 
                              << " at (" << positions_x_[i] << ", " << positions_y_[i] << ")\n";
                }
            } else if (dist_sq < near_distance_sq) {
                near_overlap_count++;
            }
        }
    }
    
    std::cout << "   Overlapping pairs (sample): " << overlap_count << "/" << (sample_size * (sample_size-1))/2 << "\n";
    std::cout << "   Near-overlapping pairs: " << near_overlap_count << "\n";
    
    if (overlap_count > 0) {
        std::cout << "   🚨 CRITICAL: Overlapping particles cause infinite subdivision!\n";
    }
    
    // 2. Particle distribution analysis
    std::cout << "\n2. PARTICLE DISTRIBUTION:\n";
    double min_x = *std::min_element(positions_x_.begin(), positions_x_.begin() + particle_count_);
    double max_x = *std::max_element(positions_x_.begin(), positions_x_.begin() + particle_count_);
    double min_y = *std::min_element(positions_y_.begin(), positions_y_.begin() + particle_count_);
    double max_y = *std::max_element(positions_y_.begin(), positions_y_.begin() + particle_count_);
    
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    double total_area = range_x * range_y;
    double density = particle_count_ / total_area;
    
    std::cout << "   X range: [" << min_x << ", " << max_x << "] = " << range_x << "\n";
    std::cout << "   Y range: [" << min_y << ", " << max_y << "] = " << range_y << "\n";
    std::cout << "   Particle density: " << density << " particles/unit²\n";
    
    if (range_x < 1e-6 || range_y < 1e-6) {
        std::cout << "   🚨 CRITICAL: Particles collapsed to line/point!\n";
    }
    
    // 3. Tree structure analysis
    std::cout << "\n3. TREE STRUCTURE ISSUES:\n";
    std::cout << "   Nodes created: " << stats.tree_nodes_created << "\n";
    std::cout << "   Particles: " << particle_count_ << "\n";
    std::cout << "   Nodes per particle: " << (float)stats.tree_nodes_created / particle_count_ << "\n";
    std::cout << "   Tree depth: " << stats.tree_depth << " (limit: " << config_.tree_depth_limit << ")\n";
    std::cout << "   Avg traversal depth: " << stats.avg_tree_depth_per_particle << "\n";
    
    if (stats.tree_depth >= config_.tree_depth_limit) {
        std::cout << "   🚨 CRITICAL: Tree hitting depth limit - infinite subdivision!\n";
    }
    
    if (stats.avg_tree_depth_per_particle > 100) {
        std::cout << "   🚨 CRITICAL: Average traversal depth too high!\n";
    }
    
    // 4. Performance ratios
    std::cout << "\n4. PERFORMANCE RATIOS:\n";
    float nodes_per_particle = (float)stats.tree_nodes_visited / particle_count_;
    float theta_pass_ratio = stats.theta_pass_ratio;
    
    std::cout << "   Nodes visited per particle: " << nodes_per_particle << "\n";
    std::cout << "   Theta pass ratio: " << (theta_pass_ratio * 100) << "%\n";
    std::cout << "   Approximation ratio: " << (stats.efficiency_ratio * 100) << "%\n";
    std::cout << "   Morton ordering: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n";
    
    if (nodes_per_particle > 100) {
        std::cout << "   ⚠️  WARNING: Too many nodes visited per particle\n";
    }
    
    if (theta_pass_ratio < 0.3) {
        std::cout << "   ⚠️  WARNING: Low theta pass ratio - not using enough approximations\n";
    }
    
    // 5. Recommendations
    std::cout << "\n5. RECOMMENDATIONS:\n";
    
    if (overlap_count > 0) {
        std::cout << "   🔧 Add particle separation/jitter to prevent overlaps\n";
        std::cout << "   🔧 Implement minimum distance constraint\n";
    }
    
    if (stats.tree_depth >= config_.tree_depth_limit) {
        std::cout << "   🔧 Increase tree_depth_limit (current: " << config_.tree_depth_limit << ")\n";
        std::cout << "   🔧 Or implement leaf node particle limits\n";
    }
    
    if (theta_pass_ratio < 0.5) {
        std::cout << "   🔧 Increase theta from " << config_.theta << " to 1.0+ for more approximations\n";
    }
    
    if (density > 1000) {
        std::cout << "   🔧 Particles too densely packed - spread them out\n";
    }
    
    if (!morton_ordering_enabled_) {
        std::cout << "   🔧 Enable Morton ordering for better cache locality\n";
    }
    
    std::cout << "==========================================\n\n";
}

void BarnesHutParticleSystem::fix_overlapping_particles() {
    std::cout << "🔧 FIXING OVERLAPPING PARTICLES (Basic)...\n";
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> jitter(-0.001, 0.001);
    
    size_t fixes_applied = 0;
    const double min_distance_sq = 1e-10;
    
    // Simple O(N²) fix for overlaps - only practical for smaller particle counts
    for (size_t i = 0; i < particle_count_ && fixes_applied < 1000; ++i) {
        for (size_t j = i + 1; j < particle_count_; ++j) {
            double dx = positions_x_[i] - positions_x_[j];
            double dy = positions_y_[i] - positions_y_[j];
            double dist_sq = dx * dx + dy * dy;
            
            if (dist_sq < min_distance_sq) {
                // Add small random jitter to separate particles
                positions_x_[j] += jitter(gen);
                positions_y_[j] += jitter(gen);
                fixes_applied++;
                
                if (fixes_applied <= 10) {
                    std::cout << "   Fixed overlap between particles " << i << " and " << j << "\n";
                }
            }
        }
    }
    
    if (fixes_applied > 0) {
        std::cout << "✅ Applied " << fixes_applied << " particle separation fixes\n";
        tree_valid_ = false;  // Force tree rebuild
        if (morton_ordering_enabled_) {
            particles_need_reordering_ = true;  // Mark for reordering after fixes
        }
    } else {
        std::cout << "✅ No overlapping particles found\n";
    }
}

void BarnesHutParticleSystem::fix_overlapping_particles_advanced() {
    std::cout << "🔧 ADVANCED PARTICLE SEPARATION...\n";
    
    if (particle_count_ == 0) return;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const double min_separation = 1e-6;  // Minimum allowed distance
    const double min_separation_sq = min_separation * min_separation;
    
    size_t fixes_applied = 0;
    size_t iterations = 0;
    const size_t max_iterations = 10;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> angle_dist(0.0, 2.0 * M_PI);
    
    // Use spatial hashing for efficient overlap detection
    std::unordered_map<int64_t, std::vector<size_t>> spatial_hash;
    const double cell_size = min_separation * 2.0;
    
    while (iterations < max_iterations) {
        spatial_hash.clear();
        size_t current_fixes = 0;
        
        // Hash particles into spatial grid
        for (size_t i = 0; i < particle_count_; ++i) {
            int64_t cell_x = static_cast<int64_t>(positions_x_[i] / cell_size);
            int64_t cell_y = static_cast<int64_t>(positions_y_[i] / cell_size);
            int64_t cell_key = (cell_x << 32) | (cell_y & 0xFFFFFFFF);
            spatial_hash[cell_key].push_back(i);
        }
        
        // Check for overlaps within and between adjacent cells
        for (const auto& [cell_key, particles] : spatial_hash) {
            // Check particles within same cell
            for (size_t i = 0; i < particles.size(); ++i) {
                for (size_t j = i + 1; j < particles.size(); ++j) {
                    size_t idx1 = particles[i];
                    size_t idx2 = particles[j];
                    
                    double dx = positions_x_[idx1] - positions_x_[idx2];
                    double dy = positions_y_[idx1] - positions_y_[idx2];
                    double dist_sq = dx * dx + dy * dy;
                    
                    if (dist_sq < min_separation_sq && dist_sq > 0) {
                        // Separate particles
                        double angle = angle_dist(gen);
                        double offset_x = min_separation * std::cos(angle) * 0.5;
                        double offset_y = min_separation * std::sin(angle) * 0.5;
                        
                        positions_x_[idx1] += offset_x;
                        positions_y_[idx1] += offset_y;
                        positions_x_[idx2] -= offset_x;
                        positions_y_[idx2] -= offset_y;
                        
                        current_fixes++;
                    }
                }
            }
        }
        
        fixes_applied += current_fixes;
        iterations++;
        
        if (current_fixes == 0) {
            std::cout << "   ✅ No more overlaps found after " << iterations << " iterations\n";
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float fix_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    std::cout << "✅ Particle separation complete in " << fix_time << " ms\n";
    std::cout << "   - Fixed " << fixes_applied << " overlaps in " << iterations << " iterations\n";
    std::cout << "   - Minimum separation: " << min_separation << " units\n\n";
    
    if (fixes_applied > 0) {
        tree_valid_ = false;  // Force tree rebuild
        if (morton_ordering_enabled_) {
            particles_need_reordering_ = true;  // Mark for reordering after fixes
        }
    }
}

void BarnesHutParticleSystem::compact_tree_for_cache_efficiency() {
    if (!tree_valid_ || tree_nodes_.empty()) return;
    
    std::cout << "🔧 COMPACTING TREE FOR CACHE EFFICIENCY...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Build a mapping of old node indices to new ones based on access patterns
    std::vector<std::pair<uint32_t, uint32_t>> node_usage;  // (usage_count, old_index)
    node_usage.reserve(next_free_node_);
    
    for (uint32_t i = 0; i < next_free_node_; ++i) {
        // Estimate usage based on tree depth (deeper = less used)
        uint32_t estimated_usage = (20 - tree_nodes_[i].depth) * 10;
        if (tree_nodes_[i].is_leaf) estimated_usage += 50;  // Leaves are accessed more
        
        node_usage.emplace_back(estimated_usage, i);
    }
    
    // Sort by usage (most used first for better cache locality)
    std::sort(node_usage.rbegin(), node_usage.rend());
    
    // Create new compacted tree
    std::vector<QuadTreeNode> compacted_nodes;
    compacted_nodes.reserve(next_free_node_);
    std::vector<uint32_t> old_to_new_mapping(next_free_node_);
    
    // Copy nodes in order of usage
    for (uint32_t new_idx = 0; new_idx < next_free_node_; ++new_idx) {
        uint32_t old_idx = node_usage[new_idx].second;
        compacted_nodes.push_back(tree_nodes_[old_idx]);
        old_to_new_mapping[old_idx] = new_idx;
    }
    
    // Update child indices to point to new locations
    for (auto& node : compacted_nodes) {
        for (int i = 0; i < 4; ++i) {
            if (node.children[i] != UINT32_MAX) {
                node.children[i] = old_to_new_mapping[node.children[i]];
            }
        }
    }
    
    // Update root index
    root_node_index_ = old_to_new_mapping[root_node_index_];
    
    // Replace old tree with compacted one
    tree_nodes_ = std::move(compacted_nodes);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float compact_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    std::cout << "✅ Tree compacted in " << compact_time << " ms\n";
    std::cout << "   - Reorganized " << next_free_node_ << " nodes for better cache locality\n";
    std::cout << "   - Most-accessed nodes moved to front of memory\n\n";
}

void BarnesHutParticleSystem::adaptive_theta_optimization() {
    std::cout << "🔧 ADAPTIVE THETA OPTIMIZATION...\n";
    
    if (particle_count_ == 0) return;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Test different theta values and pick optimal one
    float original_theta = config_.theta;
    float best_theta = original_theta;
    float best_time = std::numeric_limits<float>::max();
    
    std::vector<float> test_thetas = {0.5f, 0.75f, 1.0f, 1.2f, 1.5f};
    
    std::cout << "   Testing theta values...\n";
    
    for (float theta : test_thetas) {
        config_.theta = theta;
        config_.theta_squared = theta * theta;
        tree_valid_ = false;  // Force rebuild
        
        // Time a force calculation cycle
        auto test_start = std::chrono::high_resolution_clock::now();
        
        if (!tree_valid_) build_tree();
        
        // Reset counters
        perf_stats_.force_calculations = 0;
        perf_stats_.approximations_used = 0;
        
        // Clear forces
        std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
        std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
        
        calculate_forces_barnes_hut();
        
        auto test_end = std::chrono::high_resolution_clock::now();
        float test_time = std::chrono::duration<float, std::milli>(test_end - test_start).count();
        
        size_t total_ops = perf_stats_.force_calculations + perf_stats_.approximations_used;
        float efficiency = total_ops > 0 ? (float)perf_stats_.approximations_used / total_ops : 0.0f;
        
        std::cout << "     θ=" << theta << ": " << test_time << "ms, " 
                  << (efficiency * 100.0f) << "% approximations\n";
        
        // Score based on time and approximation ratio (we want fast + high approximation)
        float score = test_time * (2.0f - efficiency);  // Lower is better
        
        if (score < best_time) {
            best_time = score;
            best_theta = theta;
        }
    }
    
    // Apply best theta
    config_.theta = best_theta;
    config_.theta_squared = best_theta * best_theta;
    tree_valid_ = false;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float optimization_time = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    std::cout << "✅ Adaptive theta optimization complete in " << optimization_time << " ms\n";
    std::cout << "   - Original theta: " << original_theta << " → Optimal theta: " << best_theta << "\n";
    std::cout << "   - Performance improvement expected\n\n";
}

void BarnesHutParticleSystem::run_comprehensive_optimization() {
    std::cout << "\n🚀 RUNNING COMPREHENSIVE BARNES-HUT OPTIMIZATION\n";
    std::cout << "=================================================\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. Fix overlapping particles (critical for tree depth)
    fix_overlapping_particles_advanced();
    
    // 2. Optimize particle layout with Morton ordering
    optimize_particle_layout();
    
    // 3. Find optimal theta value
    adaptive_theta_optimization();
    
    // 4. Compact tree for cache efficiency
    if (tree_valid_) {
        compact_tree_for_cache_efficiency();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    
    std::cout << "🎯 COMPREHENSIVE OPTIMIZATION COMPLETE!\n";
    std::cout << "Total optimization time: " << total_time << " ms\n";
    std::cout << "=================================================\n\n";
    
    // Run a test to show improvement
    std::cout << "🧪 Testing optimized performance...\n";
    
    auto test_start = std::chrono::high_resolution_clock::now();
    
    // Clear forces and calculate
    std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
    std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
    
    if (!tree_valid_) build_tree();
    calculate_forces_barnes_hut();
    
    auto test_end = std::chrono::high_resolution_clock::now();
    float test_time = std::chrono::duration<float, std::milli>(test_end - test_start).count();
    
    update_performance_stats();
    const auto& stats = get_performance_stats();
    
    std::cout << "✅ Optimized performance test:\n";
    std::cout << "   Force calculation: " << test_time << " ms\n";
    std::cout << "   Tree depth: " << stats.tree_depth << "\n";
    std::cout << "   Efficiency: " << (stats.efficiency_ratio * 100.0f) << "% approximations\n";
    std::cout << "   Speedup vs brute force: " << stats.speedup_vs_brute_force << "x\n";
    std::cout << "   Morton ordering: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n\n";
}


#ifdef BH_TESTING
struct BHTestHooks {
  struct Snapshot {
    // Public, copy-only snapshot of internals we need to verify
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
  
  // Safe entry to the private leaf NEON kernel for correctness tests
  static void leaf_neon(const BarnesHutParticleSystem& s,
                        const BarnesHutParticleSystem::QuadTreeNode& node,
                        int i_local, float px, float py, float gi,
                        float& fx, float& fy,
                        const float* leaf_x, const float* leaf_y, const float* leaf_m) {
    s.process_leaf_forces_neon(node, i_local, px, py, gi, fx, fy, leaf_x, leaf_y, leaf_m);
  }
};
#endif
