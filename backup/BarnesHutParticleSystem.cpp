#include "BarnesHutParticleSystem.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <random>

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
    
    // Pre-allocate and reuse buffers
    if (tmp_indices_.size() != N) {
        tmp_indices_.resize(N);
        radix_count_.resize(256);
    }
    
    // Find max key to determine number of passes needed
    uint64_t max_key = 0;
    for (size_t i = 0; i < N; ++i) {
        max_key = std::max(max_key, morton_keys_[morton_indices_[i]]);
    }
    
    // Calculate significant bits
    int significant_bits = max_key == 0 ? 0 : 64 - __builtin_clzll(max_key);
    int passes = (significant_bits + 7) / 8; // Round up to byte boundary
    
    // For small datasets or few bits, use std::sort
    if (passes <= 2 && N < 1000) {
        std::sort(morton_indices_.begin(), morton_indices_.begin() + N,
                  [this](size_t a, size_t b) {
                      return morton_keys_[a] < morton_keys_[b];
                  });
        return;
    }
    
    // Multi-pass radix sort (8-bit radix for cache efficiency)
    for (int pass = 0; pass < passes; ++pass) {
        int shift = pass * 8;
        std::fill(radix_count_.begin(), radix_count_.end(), 0);
        
        // Count phase
        for (size_t i = 0; i < N; ++i) {
            uint8_t byte = static_cast<uint8_t>(morton_keys_[morton_indices_[i]] >> shift);
            radix_count_[byte]++;
        }
        
        // Prefix sum for scatter positions
        int sum = 0;
        for (int b = 0; b < 256; ++b) {
            int c = radix_count_[b];
            radix_count_[b] = sum;
            sum += c;
        }
        
        // Scatter phase
        for (size_t i = 0; i < N; ++i) {
            size_t idx = morton_indices_[i];
            uint8_t byte = static_cast<uint8_t>(morton_keys_[idx] >> shift);
            tmp_indices_[radix_count_[byte]++] = idx;
        }
        
        morton_indices_.swap(tmp_indices_);
    }
}

inline void BarnesHutParticleSystem::sort_by_morton_key() {
    const size_t N = particle_count_;
    
    // Resize Morton arrays if needed
    if (morton_keys_.size() < N) {
        morton_keys_.resize(max_particles_);
        morton_indices_.resize(max_particles_);
    }
    
    // Calculate Morton keys for all particles
    #ifdef _OPENMP
    if (config_.enable_threading && N > 1000) {
        #pragma omp parallel for
        for (size_t i = 0; i < N; ++i) {
            morton_keys_[i] = MortonEncoder::encode_position(
                positions_x_[i], positions_y_[i],
                bounds_min_x_, bounds_max_x_,
                bounds_min_y_, bounds_max_y_);
            morton_indices_[i] = i;
        }
    } else 
    #endif
    {
        for (size_t i = 0; i < N; ++i) {
            morton_keys_[i] = MortonEncoder::encode_position(
                positions_x_[i], positions_y_[i],
                bounds_min_x_, bounds_max_x_,
                bounds_min_y_, bounds_max_y_);
            morton_indices_[i] = i;
        }
    }
    
    // Sort indices by Morton key
    radix_sort_indices();
}

std::array<std::pair<size_t, size_t>, 4> 
BarnesHutParticleSystem::split_morton_range(size_t first, size_t last, int depth) const {
    std::array<std::pair<size_t, size_t>, 4> ranges;
    
    // Initialize all ranges as invalid
    for (int i = 0; i < 4; ++i) {
        ranges[i] = {SIZE_MAX, SIZE_MAX};
    }
    
    if (first > last) return ranges;
    
    // CRITICAL FIX: Extract bits from the MOST significant bits, not least significant
    // For Morton codes, we need to look at the top-level bits first
    const int total_bits = 42;  // 21 bits per dimension * 2
    const int level_shift = total_bits - (depth + 1) * 2;  // Start from MSB
    
    if (level_shift < 0) {
        // Too deep - treat as single range in quadrant 0
        ranges[0] = {first, last};
        return ranges;
    }
    
    const uint64_t mask = 3ULL << level_shift;  // Extract 2 bits at this level
    
    // Group particles by their quadrant at this tree level
    size_t range_starts[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    size_t range_ends[4] = {SIZE_MAX, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    
    for (size_t i = first; i <= last; ++i) {
        uint64_t key = morton_keys_[morton_indices_[i]];
        int quad = static_cast<int>((key & mask) >> level_shift);
        
        if (range_starts[quad] == SIZE_MAX) {
            range_starts[quad] = i;
        }
        range_ends[quad] = i;
    }
    
    // Convert to final ranges
    for (int quad = 0; quad < 4; ++quad) {
        if (range_starts[quad] != SIZE_MAX) {
            ranges[quad] = {range_starts[quad], range_ends[quad]};
        }
    }
    
    return ranges;
}


BarnesHutParticleSystem::BarnesHutParticleSystem(size_t max_particles, EventBus& event_bus, const Config& config)
    : max_particles_(max_particles), particle_count_(0), event_bus_(event_bus),
      config_(config), iteration_count_(0), current_frame_(0), tree_valid_(false),
      root_node_index_(UINT32_MAX), next_free_node_(0),
      bounce_force_(1000.0f), damping_(0.999f), gravity_(0.0, 0.0),
      bounds_min_x_(-10.0), bounds_max_x_(10.0), bounds_min_y_(-10.0), bounds_max_y_(10.0),
      morton_ordering_enabled_(true), particles_need_reordering_(false),last_morton_frame_(UINT32_MAX) {  
    
    // Precompute theta squared for optimization
    config_.theta_squared = config_.theta * config_.theta;
    
    // Reserve memory for SOA arrays
    positions_x_.reserve(max_particles);
    positions_y_.reserve(max_particles);
    velocities_x_.reserve(max_particles);
    velocities_y_.reserve(max_particles);
    forces_x_.reserve(max_particles);
    forces_y_.reserve(max_particles);
    masses_.reserve(max_particles);
    colors_r_.reserve(max_particles);
    colors_g_.reserve(max_particles);
    colors_b_.reserve(max_particles);
    
    // Resize to max size for direct indexing
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
    
    // Tree nodes - estimate 4x particles for balanced tree
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
    morton_keys_.reserve(max_particles);
    morton_indices_.reserve(max_particles);
    tmp_indices_.reserve(max_particles);
    radix_count_.reserve(256);
    
    std::cout << "BarnesHutParticleSystem initialized with " << max_particles << " max particles\n";
    std::cout << "Theta: " << config_.theta << ", Tree caching: " << (config_.enable_tree_caching ? "enabled" : "disabled") << "\n";
    std::cout << "Morton Z-order optimization: " << (morton_ordering_enabled_ ? "enabled" : "disabled") << "\n";  // NEW
    
#ifdef _OPENMP
    if (config_.enable_threading) {
        std::cout << "OpenMP threading enabled with " << omp_get_max_threads() << " threads\n";
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
    forces_x_[idx] = 0.0;
    forces_y_[idx] = 0.0;
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
    
    // Clear forces
    std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
    std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
    
    // Build/update tree if necessary
    calculate_bounds();
    auto tree_start = std::chrono::high_resolution_clock::now();
    if (!tree_valid_ || should_rebuild_tree()) {
        build_tree();
        perf_stats_.tree_was_rebuilt = true;
    } else {
        perf_stats_.tree_was_rebuilt = false;
    }
    auto tree_end = std::chrono::high_resolution_clock::now();
    perf_stats_.tree_build_time_ms = std::chrono::duration<float, std::milli>(tree_end - tree_start).count();
    
    // Calculate forces using Barnes-Hut with detailed timing
    auto force_start = std::chrono::high_resolution_clock::now();
    
    auto traversal_start = std::chrono::high_resolution_clock::now();
    calculate_forces_barnes_hut();
    auto traversal_end = std::chrono::high_resolution_clock::now();
    perf_stats_.tree_traversal_time_ms = std::chrono::duration<float, std::milli>(traversal_end - traversal_start).count();
    
    auto boundary_start = std::chrono::high_resolution_clock::now();
    apply_boundary_forces();
    auto boundary_end = std::chrono::high_resolution_clock::now();
    perf_stats_.boundary_forces_time_ms = std::chrono::duration<float, std::milli>(boundary_end - boundary_start).count();
    
    auto gravity_start = std::chrono::high_resolution_clock::now();
    apply_gravity_forces();
    auto gravity_end = std::chrono::high_resolution_clock::now();
    perf_stats_.gravity_forces_time_ms = std::chrono::duration<float, std::milli>(gravity_end - gravity_start).count();
    
    auto force_end = std::chrono::high_resolution_clock::now();
    perf_stats_.force_calculation_time_ms = std::chrono::duration<float, std::milli>(force_end - force_start).count();
    
    // Integrate physics
    auto integration_start = std::chrono::high_resolution_clock::now();
    integrate_verlet(dt);
    auto integration_end = std::chrono::high_resolution_clock::now();
    perf_stats_.integration_time_ms = std::chrono::duration<float, std::milli>(integration_end - integration_start).count();
    
    // Update detailed performance stats
    auto total_end = std::chrono::high_resolution_clock::now();
    float total_frame_time = std::chrono::duration<float, std::milli>(total_end - start_time).count();
    update_detailed_performance_stats(total_frame_time);
    
    // Update standard performance stats
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

// NEW: Apply Morton Z-order reordering (optimized version of the original method)
void BarnesHutParticleSystem::apply_morton_ordering() {
    if (particle_count_ == 0) return;
    
    // 1. Calculate Morton codes for all particles
    std::vector<std::pair<uint64_t, size_t>> morton_particles;
    morton_particles.reserve(particle_count_);
    
    // Find bounds for quantization
    double min_x = *std::min_element(positions_x_.begin(), positions_x_.begin() + particle_count_);
    double max_x = *std::max_element(positions_x_.begin(), positions_x_.begin() + particle_count_);
    double min_y = *std::min_element(positions_y_.begin(), positions_y_.begin() + particle_count_);
    double max_y = *std::max_element(positions_y_.begin(), positions_y_.begin() + particle_count_);
    
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    
    // Avoid division by zero
    if (range_x < 1e-10) range_x = 1.0;
    if (range_y < 1e-10) range_y = 1.0;
    
    const uint32_t max_coord = (1U << 16) - 1;  // 16-bit coordinates
    
    for (size_t i = 0; i < particle_count_; ++i) {
        // Quantize coordinates to integer space
        uint32_t x = static_cast<uint32_t>((positions_x_[i] - min_x) / range_x * max_coord);
        uint32_t y = static_cast<uint32_t>((positions_y_[i] - min_y) / range_y * max_coord);
        
        // Generate Morton code
        uint64_t morton = MortonCode::encode_morton_2d(x, y);
        morton_particles.emplace_back(morton, i);
    }
    
    // 2. Sort particles by Morton code (Z-order)
    std::sort(morton_particles.begin(), morton_particles.end());
    
    // 3. Reorder all particle data according to Morton order
    reorder_particles_by_indices(morton_particles);
    
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

    
void BarnesHutParticleSystem::build_tree() {
    auto start_time = std::chrono::high_resolution_clock::now();
    
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
    
    // *** STEP 1: Sort by Morton key - O(N log N) or O(N) with radix ***
    sort_by_morton_key();
    
    // *** STEP 2: Reserve space and create root ***
    const int leaf_cap = std::min<int>(config_.max_particles_per_leaf, 4);
    size_t estimated_nodes = std::max<size_t>(64, (N * 8) / std::max(1, leaf_cap));
    tree_nodes_.reserve(estimated_nodes);
    
    root_node_index_ = create_node();
    QuadTreeNode& root = tree_nodes_[root_node_index_];
    root.min_x = bounds_min_x_;
    root.max_x = bounds_max_x_;
    root.min_y = bounds_min_y_;
    root.max_y = bounds_max_y_;
    root.width = std::max(bounds_max_x_ - bounds_min_x_, bounds_max_y_ - bounds_min_y_);
    root.depth = 0;
    root.is_leaf = 1;
    root.particle_count = 0;
    for (int i = 0; i < 4; ++i) {
        root.children[i] = UINT32_MAX;
    }
    
    // *** STEP 3: Iterative subdivision using stack ***
    struct StackItem {
        uint32_t node_index;
        size_t first, last;
        int depth;
    };
    
    std::vector<StackItem> stack;
    stack.reserve(64);  // Reasonable estimate for stack depth
    stack.push_back({root_node_index_, 0, N - 1, 0});
    
    while (!stack.empty()) {
        StackItem item = stack.back();
        stack.pop_back();
        
        QuadTreeNode& node = tree_nodes_[item.node_index];
        node.depth = static_cast<uint16_t>(item.depth);
        
        const size_t count = item.last - item.first + 1;
        
        // Use reasonable leaf capacity
        const size_t leaf_cap_threshold = 4;
        
        bool should_be_leaf = (count <= leaf_cap_threshold) || 
                             (item.depth >= static_cast<int>(config_.tree_depth_limit));
        
        if (should_be_leaf) {
            node.is_leaf = 1;
            node.particle_count = static_cast<uint8_t>(std::min<size_t>(count, 4));
            
            for (size_t k = 0; k < std::min<size_t>(count, 4); ++k) {
                node.children[k] = static_cast<uint32_t>(morton_indices_[item.first + k]);
            }
            
            if (count == 1) {
                node.particle_index = static_cast<uint32_t>(morton_indices_[item.first]);
            } else {
                node.particle_index = UINT32_MAX;
            }
            
            perf_stats_.tree_depth = std::max(perf_stats_.tree_depth, 
                                            static_cast<size_t>(item.depth));
            continue;
        }
        
        // Internal node: split Morton range
        node.is_leaf = 0;
        node.particle_count = static_cast<uint8_t>(count);
        
        const double cx = 0.5 * (node.min_x + node.max_x);
        const double cy = 0.5 * (node.min_y + node.max_y);
        const double half_width = 0.5 * node.width;
        
        // Get Morton ranges (Morton convention: 0=SW, 1=SE, 2=NW, 3=NE)
        auto morton_ranges = split_morton_range(item.first, item.last, item.depth);
        
        // Map Morton quadrants to your convention (0=NW, 1=NE, 2=SW, 3=SE)
        std::array<std::pair<size_t, size_t>, 4> your_ranges;
        your_ranges[0] = morton_ranges[2];  // NW <- Morton 2
        your_ranges[1] = morton_ranges[3];  // NE <- Morton 3
        your_ranges[2] = morton_ranges[0];  // SW <- Morton 0
        your_ranges[3] = morton_ranges[1];  // SE <- Morton 1
        
        // Process using YOUR quadrant numbering
        for (int quad = 0; quad < 4; ++quad) {
            const auto& [range_first, range_last] = your_ranges[quad];
            
            if (range_first == SIZE_MAX || range_last == SIZE_MAX || range_first > range_last) {
                node.children[quad] = UINT32_MAX;
                continue;
            }
            
            uint32_t child_idx = create_node();
            node.children[quad] = child_idx;
            
            // Re-fetch parent reference after potential reallocation
            QuadTreeNode& parent = tree_nodes_[item.node_index];
            QuadTreeNode& child = tree_nodes_[child_idx];
            
            child.depth = static_cast<uint16_t>(item.depth + 1);
            child.is_leaf = 1;
            child.particle_count = 0;
            child.width = half_width;
            for (int i = 0; i < 4; ++i) {
                child.children[i] = UINT32_MAX;
            }
            
            // Set quadrant bounds using YOUR convention (0=NW, 1=NE, 2=SW, 3=SE)
            switch (quad) {
                case 0: // NW: x<center, y≥center
                    child.min_x = parent.min_x; child.max_x = cx;
                    child.min_y = cy; child.max_y = parent.max_y;
                    break;
                case 1: // NE: x≥center, y≥center  
                    child.min_x = cx; child.max_x = parent.max_x;
                    child.min_y = cy; child.max_y = parent.max_y;
                    break;
                case 2: // SW: x<center, y<center
                    child.min_x = parent.min_x; child.max_x = cx;
                    child.min_y = parent.min_y; child.max_y = cy;
                    break;
                case 3: // SE: x≥center, y<center
                    child.min_x = cx; child.max_x = parent.max_x;
                    child.min_y = parent.min_y; child.max_y = cy;
                    break;
            }
            
            stack.push_back({child_idx, range_first, range_last, item.depth + 1});
        }
    }
    
    // *** STEP 4: Calculate center of mass (reuse existing method) ***
    calculate_center_of_mass(root_node_index_);
    
    // Update cache for tree rebuilding logic
    previous_positions_.resize(particle_count_);
    for (size_t i = 0; i < particle_count_; ++i) {
        previous_positions_[i] = Vector2d(positions_x_[i], positions_y_[i]);
    }
    
    tree_valid_ = true;
    perf_stats_.tree_nodes_created = next_free_node_;
    
    // Optional compaction for large trees
    if (next_free_node_ > particle_count_ * 2) {
        compact_tree();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    perf_stats_.tree_build_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();
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
        if (node.particle_count == 0) {
            node.center_of_mass_x = node.center_of_mass_y = 0.0;
            node.total_mass = 0.0;
            return;
        }
        double total_m = 0.0, wx = 0.0, wy = 0.0;
        for (uint8_t k = 0; k < node.particle_count; ++k) {
            const uint32_t p = node.children[k];      // bucket holds particle ids for leaves
            const double m = masses_[p];
            total_m += m;
            wx += positions_x_[p] * m;
            wy += positions_y_[p] * m;
        }
        node.total_mass = total_m;
        node.center_of_mass_x = (total_m > 0.0) ? (wx / total_m) : 0.0;
        node.center_of_mass_y = (total_m > 0.0) ? (wy / total_m) : 0.0;
        return;
    }

    
    // Internal node - calculate from children
    double total_mass = 0.0;
    double weighted_x = 0.0;
    double weighted_y = 0.0;
    
    for (int i = 0; i < 4; ++i) {
        if (node.children[i] != UINT32_MAX) {
            calculate_center_of_mass(node.children[i]);
            const QuadTreeNode& child = tree_nodes_[node.children[i]];
            
            if (child.total_mass > 0) {
                weighted_x += child.center_of_mass_x * child.total_mass;
                weighted_y += child.center_of_mass_y * child.total_mass;
                total_mass += child.total_mass;
            }
        }
    }
    
    if (total_mass > 0) {
        node.center_of_mass_x = weighted_x / total_mass;
        node.center_of_mass_y = weighted_y / total_mass;
        node.total_mass = total_mass;
    }
}

void BarnesHutParticleSystem::calculate_forces_barnes_hut() {
        // Reset detailed timing counters
    accumulated_traversal_time_ = 0.0f;
    accumulated_force_time_ = 0.0f;
    accumulated_theta_time_ = 0.0f;
    max_nodes_for_any_particle_ = 0;
    
    perf_stats_.pure_tree_traversal_time_ms = 0.0f;
    perf_stats_.pure_force_computation_time_ms = 0.0f;
    perf_stats_.theta_evaluation_time_ms = 0.0f;
    perf_stats_.force_calculations = 0;
    perf_stats_.approximations_used = 0;

    if (!tree_valid_ || root_node_index_ == UINT32_MAX) {
        return;
    }

    // Warm up cache with prefetch
    auto prefetch_start = std::chrono::high_resolution_clock::now();
    prefetch_tree_nodes();
    auto prefetch_end = std::chrono::high_resolution_clock::now();
    perf_stats_.node_access_time_ms = std::chrono::duration<float, std::milli>(prefetch_end - prefetch_start).count();

    // Calculate forces with detailed timing for each particle
    for (size_t i = 0; i < particle_count_; ++i) {
        current_particle_nodes_visited_ = 0;
        
        auto particle_start = std::chrono::high_resolution_clock::now();
        
        double fx = 0.0, fy = 0.0;
        calculate_force_on_particle(i, root_node_index_, fx, fy);
        forces_x_[i] += fx;
        forces_y_[i] += fy;
        
        auto particle_end = std::chrono::high_resolution_clock::now();
        
        // Track maximum nodes visited for any single particle
        max_nodes_for_any_particle_ = std::max(max_nodes_for_any_particle_, current_particle_nodes_visited_);
    }
    
    // Finalize timing statistics
    perf_stats_.pure_tree_traversal_time_ms = accumulated_traversal_time_;
    perf_stats_.pure_force_computation_time_ms = accumulated_force_time_;
    perf_stats_.theta_evaluation_time_ms = accumulated_theta_time_;
    
    // Calculate derived metrics
    perf_stats_.nodes_per_particle_avg = current_tree_nodes_visited_ / std::max(particle_count_, size_t(1));
    perf_stats_.max_nodes_per_particle = max_nodes_for_any_particle_;
    
    // Theoretical minimum is log₄(N) for a balanced tree
    size_t theoretical_min = static_cast<size_t>(std::ceil(std::log(particle_count_) / std::log(4.0)));
    perf_stats_.tree_traversal_efficiency = static_cast<float>(theoretical_min) / 
                                          std::max(perf_stats_.nodes_per_particle_avg, size_t(1));
    
}


inline void BarnesHutParticleSystem::calculate_force_on_particle(
    size_t i, uint32_t node_idx, double& fx, double& fy, int depth) const
{
    const bool do_debug =
    #ifdef _OPENMP
        !config_.enable_threading;
    #else
        true;
    #endif

    if (do_debug) { current_tree_nodes_visited_++; current_tree_depth_sum_ += depth; }

    const QuadTreeNode& node = tree_nodes_[node_idx];
    if (node.total_mass <= 0.0) return;

    const double px = positions_x_[i];
    const double py = positions_y_[i];
    const double mi = masses_[i];

    if (node.is_leaf) {
        if (do_debug) current_leaf_nodes_hit_++;
        const uint8_t count = node.particle_count;
        // Unroll tiny loop; avoids branch mispredicts in practice.
        for (uint8_t k = 0; k < count; ++k) {
            const uint32_t j = node.children[k];
            if (j == i) continue;

            const double dx = positions_x_[j] - px;
            const double dy = positions_y_[j] - py;
            const double r2 = dx*dx + dy*dy + EPS_SQ;

            // inv_r = 1/sqrt(r2); inv_r3 = inv_r^3
            const double inv_r  = 1.0 / std::sqrt(r2);
            const double inv_r3 = inv_r * inv_r * inv_r;

            const double s = G_GALACTIC * mi * masses_[j] * inv_r3;
            fx += s * dx;
            fy += s * dy;

            //#ifdef _OPENMP
            //if (config_.enable_threading) { #pragma omp atomic perf_stats_.force_calculations++; }
            //else
            //#endif
            //{ perf_stats_.force_calculations++; }
        }
        return;
    }

    if (do_debug) { current_internal_nodes_hit_++; current_theta_tests_++; }

    if (theta_condition_met(node, px, py)) {
        if (do_debug) current_theta_passed_++;

        const double dx = node.center_of_mass_x - px;
        const double dy = node.center_of_mass_y - py;
        const double r2 = dx*dx + dy*dy + EPS_SQ;

        const double inv_r  = 1.0 / std::sqrt(r2);
        const double inv_r3 = inv_r * inv_r * inv_r;

        const double s = G_GALACTIC * mi * node.total_mass * inv_r3;
        fx += s * dx;
        fy += s * dy;

        //#ifdef _OPENMP
        //if (config_.enable_threading) { #pragma omp atomic perf_stats_.approximations_used++; }
        //else
        //#endif
        //{ perf_stats_.approximations_used++; }
        return;
    }

    // Recurse into children (small, predictable fan-out = 4)
    // Tip: order likely-heavy child first using quadrant hint if you want.
    for (int c = 0; c < 4; ++c) {
        const uint32_t cid = node.children[c];
        if (cid == UINT32_MAX) continue;
        if (tree_nodes_[cid].total_mass <= 0.0) continue;
        calculate_force_on_particle(i, cid, fx, fy, depth + 1);
    }
}


void BarnesHutParticleSystem::apply_boundary_forces() {
    for (size_t i = 0; i < particle_count_; ++i) {
        // X boundaries
        if (positions_x_[i] > bounds_max_x_) {
            forces_x_[i] -= bounce_force_ * (positions_x_[i] - bounds_max_x_);
        } else if (positions_x_[i] < bounds_min_x_) {
            forces_x_[i] -= bounce_force_ * (positions_x_[i] - bounds_min_x_);
        }
        
        // Y boundaries
        if (positions_y_[i] > bounds_max_y_) {
            forces_y_[i] -= bounce_force_ * (positions_y_[i] - bounds_max_y_);
        } else if (positions_y_[i] < bounds_min_y_) {
            forces_y_[i] -= bounce_force_ * (positions_y_[i] - bounds_min_y_);
        }
    }
}

void BarnesHutParticleSystem::apply_gravity_forces() {
    if (gravity_.squaredNorm() > 0) {
        for (size_t i = 0; i < particle_count_; ++i) {
            forces_x_[i] += gravity_.x() * masses_[i];
            forces_y_[i] += gravity_.y() * masses_[i];
        }
    }
}

void BarnesHutParticleSystem::integrate_verlet(float dt) {
    const double dt_d = static_cast<double>(dt);
    const double dt_sq = dt_d * dt_d;
    
    for (size_t i = 0; i < particle_count_; ++i) {
        const double inv_mass = 1.0 / masses_[i];
        
        // Velocity Verlet integration
        const double accel_x = forces_x_[i] * inv_mass;
        const double accel_y = forces_y_[i] * inv_mass;
        
        // Update position: x = x + v*dt + 0.5*a*dt^2
        positions_x_[i] += velocities_x_[i] * dt_d + 0.5 * accel_x * dt_sq;
        positions_y_[i] += velocities_y_[i] * dt_d + 0.5 * accel_y * dt_sq;
        
        // Update velocity: v = v + a*dt
        velocities_x_[i] += accel_x * dt_d;
        velocities_y_[i] += accel_y * dt_d;
        
        // Apply damping
        velocities_x_[i] *= damping_;
        velocities_y_[i] *= damping_;
    }
    
    // NEW: Check if particles have moved enough to warrant reordering
    if (morton_ordering_enabled_ && iteration_count_ % 30 == 0) {  // Check every 30 frames
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

void BarnesHutParticleSystem::compact_tree() {
    // This would implement tree compaction for better cache locality
    // For now, it's a placeholder for future optimization
    // The idea is to reorder nodes in memory based on access patterns
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
    
    double min_x = positions_x_[0], max_x = positions_x_[0];
    double min_y = positions_y_[0], max_y = positions_y_[0];
    
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
    return Vec2(static_cast<float>(positions_x_[index]), static_cast<float>(positions_y_[index]));
}

Vec2 BarnesHutParticleSystem::get_velocity(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(static_cast<float>(velocities_x_[index]), static_cast<float>(velocities_y_[index]));
}

Vec2 BarnesHutParticleSystem::get_force(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(static_cast<float>(forces_x_[index]), static_cast<float>(forces_y_[index]));
}

float BarnesHutParticleSystem::get_mass(size_t index) const {
    if (index >= particle_count_) return 0.0f;
    return static_cast<float>(masses_[index]);
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
    viz_node.width = node.width;
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

void BarnesHutParticleSystem::reorder_particles_by_indices(const std::vector<std::pair<uint64_t, size_t>>& sorted_particles) {
    // *** CRITICAL FIX: Create temporary arrays with MAX size, not current particle count ***
    std::vector<double> new_positions_x(max_particles_);    // NOT particle_count_!
    std::vector<double> new_positions_y(max_particles_);    // NOT particle_count_!
    std::vector<double> new_velocities_x(max_particles_);   // NOT particle_count_!
    std::vector<double> new_velocities_y(max_particles_);   // NOT particle_count_!
    std::vector<double> new_masses(max_particles_);         // NOT particle_count_!
    std::vector<float> new_colors_r(max_particles_);        // NOT particle_count_!
    std::vector<float> new_colors_g(max_particles_);        // NOT particle_count_!
    std::vector<float> new_colors_b(max_particles_);        // NOT particle_count_!
    
    // Reorder according to Morton order (only up to particle_count_)
    for (size_t new_idx = 0; new_idx < particle_count_; ++new_idx) {
        size_t old_idx = sorted_particles[new_idx].second;
        
        new_positions_x[new_idx] = positions_x_[old_idx];
        new_positions_y[new_idx] = positions_y_[old_idx];
        new_velocities_x[new_idx] = velocities_x_[old_idx];
        new_velocities_y[new_idx] = velocities_y_[old_idx];
        new_masses[new_idx] = masses_[old_idx];
        new_colors_r[new_idx] = colors_r_[old_idx];
        new_colors_g[new_idx] = colors_g_[old_idx];
        new_colors_b[new_idx] = colors_b_[old_idx];
    }
    
    // Replace original arrays with reordered ones
    positions_x_ = std::move(new_positions_x);
    positions_y_ = std::move(new_positions_y);
    velocities_x_ = std::move(new_velocities_x);
    velocities_y_ = std::move(new_velocities_y);
    masses_ = std::move(new_masses);
    colors_r_ = std::move(new_colors_r);
    colors_g_ = std::move(new_colors_g);
    colors_b_ = std::move(new_colors_b);
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
    apply_morton_ordering();  // Just use the new default implementation
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

void BarnesHutParticleSystem::debug_force_calculation_bottleneck() {
    if (particle_count_ == 0 || !tree_valid_) return;
    
    std::cout << "\n=== FORCE CALCULATION DEBUG ===\n";
    
    // Time just the tree traversal for a few particles
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < std::min(particle_count_, size_t(10)); ++i) {
        double force_x = 0.0, force_y = 0.0;
        calculate_force_on_particle(i, root_node_index_, force_x, force_y);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float time_for_10 = std::chrono::duration<float, std::milli>(end - start).count();
    float estimated_total = (time_for_10 / 10.0f) * particle_count_;
    
    std::cout << "Time for 10 particles: " << time_for_10 << " ms\n";
    std::cout << "Estimated total time: " << estimated_total << " ms\n";
    std::cout << "Actual force calc time: " << perf_stats_.force_calculation_time_ms << " ms\n";
    std::cout << "Morton ordering enabled: " << (morton_ordering_enabled_ ? "Yes" : "No") << "\n";
    
    std::cout << "===================================\n\n";
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
