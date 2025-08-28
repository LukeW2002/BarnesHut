#include "BarnesHutParticleSystem.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <random>
#include <arm_neon.h>
#include <numeric>

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


BarnesHutParticleSystem::BarnesHutParticleSystem(size_t max_particles, EventBus& event_bus, const Config& config)
    : max_particles_(max_particles), particle_count_(0), event_bus_(event_bus),
      config_(config), iteration_count_(0), current_frame_(0), tree_valid_(false),
      root_node_index_(UINT32_MAX), next_free_node_(0),
      bounce_force_(1000.0f), damping_(0.999f), gravity_(0.0, 0.0),
      bounds_min_x_(-10.0), bounds_max_x_(10.0), bounds_min_y_(-10.0), bounds_max_y_(10.0),
      morton_ordering_enabled_(true), particles_need_reordering_(false), 
      indices_filled_(0) {  
    
    config_.theta_squared = config_.theta * config_.theta;
    
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
    
    // Morton and tree work buffers 
    morton_indices_.resize(max_particles_);         
    tmp_indices_.resize(max_particles_);           
    morton_keys_.resize(max_particles_);           
    current_accel_x_.resize(max_particles_);       
    current_accel_y_.resize(max_particles_);       
    node_stack_.reserve(4096);                     
    
    // Leaf arrays
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
    
    #ifdef _OPENMP
        if (config_.enable_threading) {
            std::cout << "OpenMP threading enabled with " << omp_get_max_threads() << " threads\n";
            omp_set_dynamic(0);
            omp_set_num_threads(8);
        }
    #endif
}

BarnesHutParticleSystem::~BarnesHutParticleSystem() = default;




//===========================================================================================
//==                                   MORTON CODE                                         ==
//===========================================================================================

class MortonCode {
public:
    static uint64_t encode_morton_2d(uint32_t x, uint32_t y) {
        return (expand_bits_2d(x) << 1) | expand_bits_2d(y);
    }
    

private:
    static uint64_t expand_bits_2d(uint32_t v) {
        uint64_t x = v;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
        x = (x | (x << 8))  & 0x00FF00FF00FF00FF;
        x = (x | (x << 4))  & 0x0F0F0F0F0F0F0F0F;
        x = (x | (x << 2))  & 0x3333333333333333;
        x = (x | (x << 1))  & 0x5555555555555555;
        return x;
    }
    
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
        /* Normalises [x,y] to [0,1] using current world bounds; clamps degenerate
           ranges to 1.0 to avoid division by zero
           Quantises to 21-bit integer coord.
           Interleaves bits MortonCode to produce a 42-bit Z-order key
           
            INPUTS:  x; y; min_x; max_y; min_y; max_y
            OUTPUS: uint64_t morton_key
        */
        double range_x = max_x - min_x;
        double range_y = max_y - min_y;
        
        if (range_x < 1e-10) range_x = 1.0;
        if (range_y < 1e-10) range_y = 1.0;
        
        double norm_x = std::clamp((x - min_x) / range_x, 0.0, 1.0);
        double norm_y = std::clamp((y - min_y) / range_y, 0.0, 1.0);
        
        const uint32_t max_coord = (1U << 21) - 1;
        uint32_t ix = static_cast<uint32_t>(norm_x * max_coord);
        uint32_t iy = static_cast<uint32_t>(norm_y * max_coord);
        
        return MortonCode::encode_morton_2d(ix, iy);
    }
};

inline void BarnesHutParticleSystem::radix_sort_indices() {
    /* 
     * 1. 4-pass LSD Radix sort, O(11N)≈ O(N) over a 11 bit per pass into (2048) buckets
     * Covers the 42 bit key. For N<2048 it falls back to std::sort 
     *
     * INPUTS: morton_keys_; morton_indices_
     * OUTPUT: morton_indices reordered so that morton_keys_[morton_indices_[i]] is
     *         non-decreasing
     * */
    const size_t N = particle_count_;
    if (N == 0) return;

    using IndexT = typename decltype(morton_indices_)::value_type;
    
    IndexT* __restrict out = tmp_indices_.data();       
    IndexT* __restrict in  = morton_indices_.data();     
    const uint64_t* __restrict keys = morton_keys_.data();

    constexpr int RADIX_BITS = 11;
    constexpr int KEY_BITS = 42;                    
    constexpr uint32_t RADIX = 1u << RADIX_BITS;   
    constexpr uint32_t RADIX_MASK = RADIX - 1;
    constexpr int NUM_PASSES = (KEY_BITS + RADIX_BITS - 1) / RADIX_BITS;  // = 4
    static_assert(NUM_PASSES == 4, "adjust passes for key width");

    uint32_t* count = radix_histogram_.data();

    if (N < 2048) {
        std::sort(in, in + N, [&](IndexT a, IndexT b){ return keys[a] < keys[b]; });
        return;
    }

    int shift = 0;
    for (int pass = 0; pass < NUM_PASSES; ++pass, shift += RADIX_BITS) {
        std::fill_n(count, RADIX, 0);

        for (size_t i = 0; i < N; ++i) {
            uint32_t bucket = (keys[in[i]] >> shift) & RADIX_MASK;
            ++count[bucket];
        }

        uint32_t sum = 0;
        for (uint32_t b = 0; b < RADIX; ++b) {
            uint32_t c = count[b];
            count[b] = sum;
            sum += c;
        }

        for (size_t i = 0; i < N; ++i) {
            IndexT idx = in[i];
            uint32_t bucket = (keys[idx] >> shift) & RADIX_MASK;
            out[count[bucket]++] = idx;
        }

        std::swap(in, out);
    }
}


inline void BarnesHutParticleSystem::ensure_indices_upto(size_t N) {
    /*
     * Maintain an idenity index array without rewriting the entire prefix every frame
     * Input: N
     * */
    if (indices_filled_ < N) {
        std::iota(morton_indices_.begin() + indices_filled_,
                  morton_indices_.begin() + N,
                  static_cast<size_t>(indices_filled_));
        indices_filled_ = N;
    }
    else { std::iota(morton_indices_.begin(), morton_indices_.begin()+N, 0); indices_filled_ = N; }
}

inline void BarnesHutParticleSystem::sort_by_morton_key() {
    // 1. Ensures Morton_keys_ has capacity pre-sized to max_particles_
    // 2. Waterline fills morton_indices_ with [0,(N-1)]  
    // 3. Computes One Key per Particle with the MortonEncoder::encode_position
    // 4. Calls Radix Sort
    //
    // INPUTS:  positions_x_; positions_y_; particle_count_
    // OUTPUTS: morton_keys_[0,N) filled; morton_indices_[0,N) sorted by 
    //
    const size_t N = particle_count_;
    if (morton_keys_.size() < N) morton_keys_.resize(max_particles_);
    ensure_indices_upto(N);

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
    /*
     * 1. Looks at the two bits for that level with the level_shift
     * 2. Uses mask to bin consecutive indices into 4 contiguous sub-ranges
     * 3. remaps Z-order nibble into Canonical child order so that 
     *    the trees children are [SW, SE, NW, NE]
     * 
     * INPUT:  first; last; depth
     * OUTPUT: 4 pairs, {range_first, range_last} into the parents [first, last]
     * */
    std::array<std::pair<size_t, size_t>, 4> ranges;
    for (auto& r : ranges) r = {SIZE_MAX, SIZE_MAX}; 
    
    if (first > last) return ranges;
    
    const int level_shift = MORTON_TOTAL_BITS - 2 * (depth + 1);
    const uint64_t mask = 3ULL << level_shift;
    static constexpr int z_to_child[4] = {0, 2, 1, 3};
    
    for (size_t i = first; i <= last; ) {
        const size_t gi = morton_indices_[i];            
        const uint64_t ki = morton_keys_[gi];
        const int z_quad = int((ki & mask) >> level_shift);
        const int child_slot = z_to_child[z_quad];
        
        size_t j = i + 1;
        while (j <= last) {
            const size_t gj = morton_indices_[j];         
            const int z2 = int((morton_keys_[gj] & mask) >> level_shift);
            if (z2 != z_quad) break;
            ++j;
        }
        ranges[child_slot] = {i, j - 1};
        i = j;
    }
    
    #ifndef NDEBUG
    size_t total = 0;
    for (auto [a, b] : ranges) {
        if (a != SIZE_MAX) total += (b - a + 1);
    }
    assert(total == (last - first + 1));
    #endif
    
    return ranges;
}

void BarnesHutParticleSystem::check_for_morton_reordering_need() {
    /*
     *    Checks if the morton code needs reordering based on how far the particles have moved
     * 1. first calculates threshold based on the roots bounding bod
     * 2. then checks a sampled amount of pixels on if its moved too far away calculated threshold via distance
     *
     * INPUT: particle_count_, bounds_min_x/y, bounds_max_min_x/y position_x/y_, previous_positions  
     * OUTPUT: Non
     * */
    if (particle_count_ < 100) return;  
    
    const size_t sample_size = std::min(particle_count_, size_t(50));
    const double movement_threshold = 0.1;
    
    double world_size = std::max(bounds_max_x_ - bounds_min_x_, bounds_max_y_ - bounds_min_y_);
    double threshold_distance = movement_threshold * world_size;
    double threshold_distance_sq = threshold_distance * threshold_distance;
    
    size_t moved_particles = 0;
    
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = (i * particle_count_) / sample_size;
        if (idx < previous_positions_.size()) {
            double dx = positions_x_[idx] - previous_positions_[idx].x();
            double dy = positions_y_[idx] - previous_positions_[idx].y();
            if (dx*dx + dy*dy > threshold_distance_sq) {
                moved_particles++;
            }
        }
    }
    
    if (moved_particles > sample_size * 0.3) {
        particles_need_reordering_ = true;
    }
}




//===========================================================================================
//==                                   TREE BUILDING                                       ==
//===========================================================================================

void BarnesHutParticleSystem::build_tree() {
    auto total_start = std::chrono::high_resolution_clock::now();

    tree_nodes_.clear();
    next_free_node_ = 0;

    const size_t N = particle_count_;
    if (N == 0) {
        tree_valid_ = false;
        return;
    }
    
    leaf_pos_x_.clear();   
    leaf_pos_y_.clear();   
    leaf_mass_.clear();    
    leaf_idx_.clear();     
    leaf_offset_.clear();  
    leaf_count_.clear();

    const size_t leaf_threshold = std::clamp<size_t>(config_.max_particles_per_leaf, 1, 64);
    const size_t est_leaves = std::max<size_t>(1, (N + leaf_threshold - 1) / leaf_threshold);
    leaf_offset_.reserve(est_leaves);
    leaf_count_.reserve(est_leaves);

    leaf_pos_x_.reserve(N);
    leaf_pos_y_.reserve(N);
    leaf_mass_.reserve(N);
    leaf_idx_.reserve(N);

    particle_leaf_slot_.resize(N);

    sort_by_morton_key();

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

    node_stack_.clear();  
    node_stack_.emplace_back(root_node_index_, 0, N - 1, 0);

    while (!node_stack_.empty()) {
        auto item = node_stack_.back();  
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

            const uint32_t off = static_cast<uint32_t>(leaf_pos_x_.size());
            leaf_offset_.push_back(off);
            leaf_count_.push_back(cnt);

            for (uint32_t t = 0; t < cnt; ++t) {
                const size_t global_idx = morton_indices_[item.first + t];
                
                leaf_pos_x_.push_back(positions_x_[global_idx]);
                leaf_pos_y_.push_back(positions_y_[global_idx]);
                leaf_mass_.push_back(masses_[global_idx]);
                leaf_idx_.push_back(static_cast<uint32_t>(global_idx));
                
                particle_leaf_slot_[global_idx] = off + t;  
            }

            node.leaf_first = off;
            node.leaf_last = off + cnt - 1;

            for (int i = 0; i < 4; ++i) {
                node.children[i] = UINT32_MAX;
            }

            continue;
        }

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

    calculate_center_of_mass(root_node_index_);

    if (previous_positions_.size() != particle_count_) {
        previous_positions_.resize(particle_count_);
    }
    for (size_t i = 0; i < particle_count_; ++i) {
        previous_positions_[i] = Vector2d(positions_x_[i], positions_y_[i]);
    }

    tree_valid_ = true;

}


bool BarnesHutParticleSystem::should_rebuild_tree() const {
    if (!config_.enable_tree_caching || previous_positions_.size() != particle_count_) {
        return true;
    }

    const double ratio_threshold = std::clamp<double>(config_.tree_rebuild_threshold, 0.0, 1.0);

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

void BarnesHutParticleSystem::prefetch_tree_nodes() const {
    if (!tree_valid_ || tree_nodes_.empty()) return;
    
    size_t prefetch_count = std::min(size_t(64), tree_nodes_.size());
    
    for (size_t i = 0; i < prefetch_count; ++i) {
        __builtin_prefetch(&tree_nodes_[i], 0, 3);  // GCC builtin for cache prefetch
    }
}



//===========================================================================================
//==                                  CALCULATIONS                                         ==
//===========================================================================================


void BarnesHutParticleSystem::calculate_forces_barnes_hut() {
    if (!tree_valid_ || root_node_index_ == UINT32_MAX) return;

    prefetch_tree_nodes();

    const float* const positions_x = positions_x_.data();
    const float* const positions_y = positions_y_.data();
    const float* const masses      = masses_.data();

    #ifdef _OPENMP
    if (config_.enable_threading) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)particle_count_; ++i) {
            float fx = 0.0, fy = 0.0;
            (void)calculate_force_on_particle_iterative((size_t)i, fx, fy, positions_x, positions_y, masses);
            forces_x_[(size_t)i] += fx;
            forces_y_[(size_t)i] += fy;
        }
        return; // done
    }
    #endif

    for (size_t i = 0; i < particle_count_; ++i) {
        float fx = 0.0, fy = 0.0;
        (void)calculate_force_on_particle_iterative(i, fx, fy, positions_x, positions_y, masses);
        forces_x_[i] += fx;
        forces_y_[i] += fy;
    }
}


// If you are reading this function and can understand it, I am sorry. 
    // For I am to dumb to understand, the eldritch horrors of my own creation.
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

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
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
    
    float total_mass = 0.0f;
    float weighted_x = 0.0f;
    float weighted_y = 0.0f;
    
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
    
    if (total_mass > 0.0f) {
        node.com_x = weighted_x / total_mass;
        node.com_y = weighted_y / total_mass;
        node.total_mass = total_mass;
    } else {
        node.com_x = node.com_y = 0.0f;
        node.total_mass = 0.0f;
    }
    
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
    calculate_bounds();
    
    const float world_span_x = float(bounds_max_x_ - bounds_min_x_);
    const float world_span_y = float(bounds_max_y_ - bounds_min_y_);
    const float world_scale = std::max(world_span_x, world_span_y);
    
    const float eps = config_.softening_rel * world_scale;
    frame_eps2_ = eps * eps;
    
    root_com_x_ = tree_nodes_[root_node_index_].com_x;
    root_com_y_ = tree_nodes_[root_node_index_].com_y;
}

void BarnesHutParticleSystem::integrate_verlet(float dt) {
    const float dt_half = 0.5f * dt;

    // 0: current accelerations from previous forces
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

    // Periodic Morton reorder check
    if (morton_ordering_enabled_ && (iteration_count_ % 30 == 0)) {
        check_for_morton_reordering_need();
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


//===========================================================================================
//==                                  RENDER DATA                                         ==
//===========================================================================================

void BarnesHutParticleSystem::prepare_render_data() {
    for (size_t i = 0; i < particle_count_; ++i) {
        render_positions_[i * 2 + 0] = static_cast<float>(positions_x_[i]);
        render_positions_[i * 2 + 1] = static_cast<float>(positions_y_[i]);
        
        render_colors_[i * 3 + 0] = colors_r_[i];
        render_colors_[i * 3 + 1] = colors_g_[i];
        render_colors_[i * 3 + 2] = colors_b_[i];
        
        render_positions_x_[i] = static_cast<float>(positions_x_[i]);
        render_positions_y_[i] = static_cast<float>(positions_y_[i]);
        render_velocities_x_[i] = static_cast<float>(velocities_x_[i]);
        render_velocities_y_[i] = static_cast<float>(velocities_y_[i]);
        render_masses_[i] = static_cast<float>(masses_[i]);
    }
}

std::vector<BarnesHutParticleSystem::QuadTreeBox> BarnesHutParticleSystem::get_quadtree_boxes() const {
    /*
     * Return a list of AABBs for all quadtree nodes, for visualisation/debug
     *
     * INPUT:  Visualize_quadtree, tree_valid, tree_nodes_ root_node_index_
     * OUTPUT: returns std::vector<QuadTreeBox> possibly empty
     * */
    std::vector<QuadTreeBox> boxes;
    
    if (!visualize_quadtree_ || !tree_valid_ || tree_nodes_.empty()) {
        return boxes;
    }

    if (root_node_index_ != UINT32_MAX && root_node_index_ < tree_nodes_.size()) {
        collect_quadtree_boxes(root_node_index_, boxes);
    }
    
    return boxes;
}

void BarnesHutParticleSystem::collect_quadtree_boxes(uint32_t node_index, std::vector<QuadTreeBox>& boxes) const {
    if (node_index >= tree_nodes_.size()) return;
    
    const QuadTreeNode& node = tree_nodes_[node_index];
    
    boxes.emplace_back(
        static_cast<float>(node.min_x),
        static_cast<float>(node.min_y),
        static_cast<float>(node.max_x),
        static_cast<float>(node.max_y),
        static_cast<int>(node.depth),
        static_cast<int>(node.particle_count),
        node.is_leaf != 0
    );
    
    if (!node.is_leaf) {
        for (int i = 0; i < 4; ++i) {
            if (node.children[i] != UINT32_MAX) {
                collect_quadtree_boxes(node.children[i], boxes);
            }
        }
    }
}

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




//===========================================================================================
//==                                    Particle stuff?                                   ==
//===========================================================================================

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
    tree_valid_ = false; 
    
    if (morton_ordering_enabled_ && particle_count_ >= 100) {
        particles_need_reordering_ = true;
    }
    
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
    
    if (morton_ordering_enabled_) {
        particles_need_reordering_ = true;
    }
}

void BarnesHutParticleSystem::set_boundary(float min_x, float max_x, float min_y, float max_y) {
    bounds_min_x_ = min_x;
    bounds_max_x_ = max_x;
    bounds_min_y_ = min_y;
    bounds_max_y_ = max_y;
    tree_valid_ = false;  
    
    if (morton_ordering_enabled_) {
        particles_need_reordering_ = true;
    }
}

void BarnesHutParticleSystem::set_config(const Config& config) {
    config_ = config;
    config_.theta_squared = config_.theta * config_.theta;
    tree_valid_ = false;  
}


// Accessor methods
Vec2 BarnesHutParticleSystem::get_position(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(positions_x_[index], positions_y_[index]);  
}

Vec2 BarnesHutParticleSystem::get_velocity(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(velocities_x_[index], velocities_y_[index]);  
}

Vec2 BarnesHutParticleSystem::get_force(size_t index) const {
    if (index >= particle_count_) return Vec2();
    return Vec2(forces_x_[index], forces_y_[index]);  
}

float BarnesHutParticleSystem::get_mass(size_t index) const {
    if (index >= particle_count_) return 0.0f;
    return masses_[index];  
}

Vec3 BarnesHutParticleSystem::get_color(size_t index) const {
    if (index >= particle_count_) return Vec3();
    return Vec3(colors_r_[index], colors_g_[index], colors_b_[index]);
}


//===========================================================================================
//==                                   UPDATE                                              ==
//===========================================================================================


void BarnesHutParticleSystem::update(float dt) {
    if (particle_count_ == 0) return;
    current_frame_++;
    sort_by_morton_key();
    bool need_initial_forces = (iteration_count_ == 0) || (std::all_of(forces_x_.begin(), forces_x_.begin() + particle_count_, [](float f) { return f == 0.0f; }));
    if (need_initial_forces) {
        std::fill(forces_x_.begin(), forces_x_.begin() + particle_count_, 0.0);
        std::fill(forces_y_.begin(), forces_y_.begin() + particle_count_, 0.0);
        calculate_bounds();
        calculate_forces_barnes_hut();
    }
    integrate_verlet(dt);
    prepare_render_data();
    iteration_count_++;
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


//===========================================================================================
//==                                   TESTING HOOK                                        ==
//===========================================================================================


#ifdef BH_TESTING
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
  
  static void leaf_neon(const BarnesHutParticleSystem& s,
                        const BarnesHutParticleSystem::QuadTreeNode& node,
                        int i_local, float px, float py, float gi,
                        float& fx, float& fy,
                        const float* leaf_x, const float* leaf_y, const float* leaf_m) {
    s.process_leaf_forces_neon(node, i_local, px, py, gi, fx, fy, leaf_x, leaf_y, leaf_m);
  }
};
#endif
