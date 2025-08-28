#pragma once
#include "Vec2.h"
#include "EventSystem.h"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>
#include <unordered_map> 
#include <cmath>          

//#define USE_HOME_QUADRANT_FIRST  // Comment out to use Morton order instead
#define USE_MORTON_ORDER      // Uncomment for Morton traversal


using Vector2d = Eigen::Vector2d;
using VectorXd = Eigen::VectorXd;
using ArrayXd = Eigen::ArrayXd;
// Length unit: L₀ = 1 kpc  
// Mass unit: M₀ = 10⁹ M☉
// From search results: G = 4.3009×10⁻³ pc·(km/s)²·M☉⁻¹
// Converting: G = 4.3009×10⁻⁶ kpc·(km/s)²·(10⁹M☉)⁻¹ = 4.3009×10⁻⁶ kpc·(km/s)²·(10⁻⁹·10⁹M☉)⁻¹ = 4.3009 kpc·(km/s)²·(10⁹M☉)⁻¹


static constexpr float G_GALACTIC = 4.3009;           // kpc·(km/s)²·(10⁹M☉)⁻¹  
static constexpr float VELOCITY_UNIT_KMS = 65.58;     // km/s per code velocity unit
static constexpr float TIME_UNIT_MYR = 14.91;         // Myr per code time unit
static constexpr float DEFAULT_SOFTENING_KPC = 0.05;  // 50 pc softening
static constexpr float EPS_SQ = DEFAULT_SOFTENING_KPC * DEFAULT_SOFTENING_KPC ;

class BarnesHutParticleSystem {
public:
    struct QuadTreeBox {
        float min_x, min_y, max_x, max_y;  // Bounding box coordinates
        int depth;                         // Tree depth (for coloring)
        int particle_count;                // Number of particles in this node
        bool is_leaf;                      // Whether this is a leaf node
        
        QuadTreeBox(float minX, float minY, float maxX, float maxY, int d, int count, bool leaf)
            : min_x(minX), min_y(minY), max_x(maxX), max_y(maxY)
            , depth(d), particle_count(count), is_leaf(leaf) {}
    };
    std::vector<QuadTreeBox> get_quadtree_boxes() const;
    
    // Enable/disable quadtree visualization
    void set_quadtree_visualization_enabled(bool enabled) { visualize_quadtree_ = enabled; }
    bool is_quadtree_visualization_enabled() const { return visualize_quadtree_; }


    struct Config {
        float theta;                            // Barnes-Hut approximation parameter (0.5-1.0)
        float theta_squared;                    // theta^2 for optimization
        bool enable_tree_caching;              // Cache tree between updates when possible
        float tree_rebuild_threshold;          // Rebuild tree when >10% particles move significantly
        size_t max_particles_per_leaf;         // Maximum particles in leaf node
        size_t tree_depth_limit;               // Maximum tree depth to prevent infinite recursion
        bool enable_vectorization;             // Use Eigen vectorized operations
        bool enable_threading;                 // Enable OpenMP threading (if available)
        bool enable_morton_ordering;
        float softening_rel;
        
        // Default constructor
        Config() 
            : theta(0.75f)
            , theta_squared(0.5625f)
            , enable_tree_caching(true)
            , tree_rebuild_threshold(0.1f)
            , max_particles_per_leaf(16)
            , tree_depth_limit(20)
            , enable_vectorization(true)
            , enable_threading(true)
            , enable_morton_ordering(true)
            , softening_rel(1e-4f)
        {}
    };

    const Config& get_config() const { return config_; }

    struct TreeNode {
        double center_x, center_y, width;
        bool is_leaf;
        size_t particle_count;
        std::vector<TreeNode> children;
        
        TreeNode() 
            : center_x(0.0), center_y(0.0), width(0.0)
            , is_leaf(true), particle_count(0) 
        {}
    };


    BarnesHutParticleSystem(size_t max_particles, EventBus& event_bus, const Config& config = Config{});
    ~BarnesHutParticleSystem();

    // Particle management
    bool add_particle(const Vec2& pos, const Vec2& vel, float mass, const Vec3& color);
    void clear_particles();
    void remove_particle(size_t index);

    void nuclear_particle_separation();

    // Physics simulation
    void update(float dt);
    void set_boundary(float min_x, float max_x, float min_y, float max_y);

    // Configuration
    void set_bounce_force(float force) { bounce_force_ = force; }
    void set_damping(float damping) { damping_ = damping; }
    void set_gravity(const Vec2& gravity) { gravity_ = Vector2d(gravity.x, gravity.y); }
    void set_config(const Config& config);
    
    // NEW: Morton ordering control
    void set_morton_ordering_enabled(bool enabled);
    bool is_morton_ordering_enabled() const { return morton_ordering_enabled_; }

    // Performance and debugging
    size_t get_particle_count() const { return particle_count_; }
    size_t get_max_particles() const { return max_particles_; }

    // Access for testing and rendering
    const std::vector<float>& get_positions_x() const { return render_positions_x_; }
    const std::vector<float>& get_positions_y() const { return render_positions_y_; }
    const std::vector<float>& get_velocities_x() const { return render_velocities_x_; }
    const std::vector<float>& get_velocities_y() const { return render_velocities_y_; }
    const std::vector<float>& get_masses() const { return render_masses_; }

    // Individual particle access
    Vec2 get_position(size_t index) const;
    Vec2 get_velocity(size_t index) const;
    Vec2 get_force(size_t index) const;
    float get_mass(size_t index) const;
    Vec3 get_color(size_t index) const;

    // Tree visualization
    TreeNode get_tree_visualization() const;

    // Debug methods (declared only - implemented in .cpp)
    void print_performance_analysis() const;
    void debug_force_calculation_bottleneck();
    void diagnose_30ms_issue();
    void test_theta_performance();
    void diagnose_tree_traversal_bottleneck() const;  // NEW: Diagnose traversal issues
    void fix_overlapping_particles();  // NEW: Basic particle overlap fix
    
    // Advanced optimization methods
    void optimize_particle_layout();                    // Morton Z-order sorting for cache locality
    void fix_overlapping_particles_advanced();          // Advanced particle separation with spatial hashing
    void adaptive_theta_optimization();                 // Find optimal theta value automatically
    void run_comprehensive_optimization();              // Run all optimizations together

private:
    #ifdef BH_TESTING
        friend struct BHTestHooks;
    #endif
    struct TraversalFrame {
        uint32_t node_idx;
        #ifdef ENABLE_DEPTH_STATS
        int depth;
        TraversalFrame() : node_idx(UINT32_MAX), depth(0) {}  // <-- ADD THIS
        TraversalFrame(uint32_t idx, int d) : node_idx(idx), depth(d) {}
        #else
        TraversalFrame() : node_idx(UINT32_MAX) {}  // <-- ADD THIS
        TraversalFrame(uint32_t idx, int d = 0) : node_idx(idx) {}
        #endif
    };

    struct StackItem {
        uint32_t node_index;
        size_t first, last;
        int depth;
        
        StackItem(uint32_t idx, size_t f, size_t l, int d) 
            : node_index(idx), first(f), last(l), depth(d) {}
    };



    std::array<uint32_t, 1u << 11> radix_histogram_{};
    std::vector<float> current_accel_x_, current_accel_y_;  // Move from local in integrate_verlet
    std::vector<StackItem> node_stack_;                     // Move from local in build_tree
    size_t indices_filled_;


    // Cache performance estimator

    std::vector<float> leaf_pos_x_, leaf_pos_y_, leaf_mass_;
    std::vector<uint32_t> leaf_idx_;        // original particle indices in leaf order
    std::vector<uint32_t> leaf_offset_;     // per-leaf offset into leaf_* arrays
    std::vector<uint32_t> leaf_count_;      // per-leaf counts
    
    // Map from global particle index -> global slot in compact leaf arrays
    // (so we can get the local index to skip self)
    std::vector<uint32_t> particle_leaf_slot_;





    std::vector<uint64_t> morton_keys_;           // Morton keys for each particle
    std::vector<size_t> morton_indices_;          // Sorted particle indices by Morton key
    std::vector<size_t> tmp_indices_;             // Temporary buffer for radix sort
    void build_tree_morton_iterative();          // O(N) Morton-based tree builder
    void apply_morton_permutation_to_arrays();
    void sort_by_morton_key();                   // Sort particles by Morton key
    void radix_sort_indices();                   // Fast radix sort for Morton keys
    void ensure_indices_upto(size_t N);
    std::array<std::pair<size_t, size_t>, 4> split_morton_range(size_t first, size_t last, int depth) const;

    bool visualize_quadtree_ = false;
    void collect_quadtree_boxes(uint32_t node_index, std::vector<QuadTreeBox>& boxes) const;

    struct alignas(64) QuadTreeNode {
        // Core data (16 bytes)
        float com_x, com_y, total_mass, bound_r;
        
        // Bounds (16 bytes)
        float min_x, min_y, max_x, max_y;   
        
        // Tree structure and leaf data (32 bytes)
        uint32_t children[4];          // 16 bytes
        uint32_t leaf_first;           //  4 bytes
        uint32_t leaf_last;            //  4 bytes
        uint16_t depth;                //  2 bytes
        uint8_t  is_leaf;              //  1 byte
        uint8_t  particle_count;       //  1 byte (optional; redundant with last-first+1)
        uint32_t last_used_frame;      //  4 bytes
        
        QuadTreeNode() {
            com_x = com_y = total_mass = bound_r = 0.0f;
            min_x = min_y = max_x = max_y = 0.0f;
            leaf_first = leaf_last = UINT32_MAX;
            depth = 0;
            is_leaf = 1;
            particle_count = 0;
            last_used_frame = 0;
            children[0] = children[1] = children[2] = children[3] = UINT32_MAX;
        }
        
        inline float width() const { 
            return std::max(max_x - min_x, max_y - min_y); 
        }
    };
     inline bool calculate_force_on_particle_iterative(
        size_t i, float& fx, float& fy, 
        const float* positions_x, const float* positions_y, const float* masses) const;

    void process_leaf_forces_neon_centered(
    const QuadTreeNode& node, int i_local, float px_c, float py_c, float gi,
    float& fx, float& fy, float ox, float oy,
    const float* __restrict leaf_x,
    const float* __restrict leaf_y,
    const float* __restrict leaf_m) const;

    // Tree management
    std::vector<QuadTreeNode> tree_nodes_;      // Cache-friendly node storage
    uint32_t root_node_index_;
    uint32_t next_free_node_;
    uint32_t current_frame_;
    bool tree_valid_;
    std::vector<Vector2d> previous_positions_;  // For cache invalidation detection

    // NEW: Morton ordering state
    bool morton_ordering_enabled_;              // Enable/disable Morton Z-order optimization
    bool particles_need_reordering_;           // Flag indicating particles should be reordered
    std::vector<size_t> inv_perm_;              // old->new permutation mapping (reused)
    std::vector<uint8_t> visited_;              // Fast scratch buffer (not std::vector<bool>!)
    static constexpr int MORTON_TOTAL_BITS = 42;

    // Particle data (SOA for cache efficiency) - using double for precision
    size_t max_particles_;
    size_t particle_count_;
    
    std::vector<float> positions_x_, positions_y_;
    std::vector<float> velocities_x_, velocities_y_;
    std::vector<float> forces_x_, forces_y_;
    std::vector<float> masses_;
    std::vector<float> colors_r_, colors_g_, colors_b_;

    // Legacy float arrays for rendering compatibility  
    std::vector<float> render_positions_;
    std::vector<float> render_colors_;
    std::vector<float> render_positions_x_, render_positions_y_;
    std::vector<float> render_velocities_x_, render_velocities_y_;
    std::vector<float> render_masses_;

    // Physics parameters
    float bounce_force_;
    float damping_;
    Vector2d gravity_;
    double bounds_min_x_, bounds_max_x_;
    double bounds_min_y_, bounds_max_y_;
    mutable float frame_eps2_;               // Cached adaptive softening squared  
    mutable float root_com_x_, root_com_y_;  // Cached root center-of-mass

    // Barnes-Hut configuration
    Config config_;
    
    // Performance tracking
    EventBus& event_bus_;
    size_t iteration_count_;

    // Profiling counters - FIXED: Made all mutable for const methods
    mutable size_t current_tree_nodes_visited_;
    mutable size_t current_leaf_nodes_hit_;
    mutable size_t current_internal_nodes_hit_;
    mutable size_t current_theta_tests_;
    mutable size_t current_theta_passed_;
    mutable float current_tree_depth_sum_;
    mutable float cached_eps2_;
    mutable bool eps2_cache_valid_;

    // Core Barnes-Hut methods
    void build_tree();
    uint32_t create_node();
    uint32_t insert_particle(uint32_t node_index, size_t particle_index, int depth = 0);
    void calculate_center_of_mass(uint32_t node_index);
    
    // Force calculation
    void calculate_forces_barnes_hut();
    void compute_frame_constants();
    
    // Tree optimization and caching
    bool should_rebuild_tree() const;
    
    // NEW: Morton ordering methods
    bool should_apply_morton_ordering() const;         // Determine when to apply Morton ordering
    void apply_morton_ordering();                      // Apply Morton Z-order reordering
    void check_for_morton_reordering_need();          // Check if particles have moved enough to warrant reordering
    
    void integrate_verlet(float dt);  
    
    // Utility methods
    void prepare_render_data();
    void calculate_bounds();
    void update_performance_stats();
    void reset_profiling_counters();
    void update_detailed_performance_stats(float total_frame_time);
    TreeNode build_tree_visualization(uint32_t node_index) const;
    bool detect_performance_issues() const;
    void print_detailed_mac_performance() const;
    
    // Optimization helper methods  
    void reorder_particles_by_indices(const std::vector<std::pair<uint64_t, size_t>>& sorted_particles);
    void prefetch_tree_nodes() const;                  // Cache prefetching
    //
    // Temp for attempting to fix the allocation issue
    std::vector<float> tmp_posx_, tmp_posy_, tmp_velx_, tmp_vely_, tmp_mass_;
    std::vector<float> tmp_colr_, tmp_colg_, tmp_colb_;
    std::vector<uint64_t> tmp_keys_;

    
    // Inline helper methods for performance
    inline double distance_squared(float x1, float y1, float x2, float y2) const {
        const float dx = x1 - x2;
        const float dy = y1 - y2;
        return dx * dx + dy * dy;
    }
    inline bool theta_condition_met(const QuadTreeNode& node, float px, float py) const {
        const float dx = node.com_x - px;
        const float dy = node.com_y - py;
        const float dist_sq = dx*dx + dy*dy + frame_eps2_;
        const float bound_r_sq = node.bound_r * node.bound_r;
        return bound_r_sq < config_.theta_squared * dist_sq;
    }

    static inline float rsqrt_fast(float x) {
    #if defined(__aarch64__)
        float32x4_t vx = vdupq_n_f32(x);
        float32x4_t r  = vrsqrteq_f32(vx);
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(vx, r), r));  // 1 Newton refinement
        return vgetq_lane_f32(r, 0);
    #else
        // Fallback
        return 1.0f / std::sqrt(x);
    #endif
    }

    static inline float32x4_t rsqrt_nr_f32x4(float32x4_t x) {
        float32x4_t r = vrsqrteq_f32(x);
        // ARM's recommended refinement: r = r * vrsqrts(x*r, r)  
        r = vmulq_f32(r, vrsqrtsq_f32(vmulq_f32(x, r), r));
        return r;
    }
   
    template <class T, class IndexT>
    inline void gather_by_index(std::vector<T>& a,
                                const std::vector<IndexT>& idx,
                                std::vector<T>& tmp,
                                size_t N)
    {
        if (tmp.size() < N) tmp.resize(N);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < N; ++i) {
            tmp[i] = a[idx[i]];         // read old position, write to new compact spot
        }
        a.swap(tmp);                     // O(1) swap to make gathered data the primary
    }



   template<typename T>
    void permute_in_place(std::vector<T>& data, 
                         const std::vector<size_t>& old_to_new_perm, // dest index for each old index
                         size_t count, 
                         std::vector<uint8_t>& visited) noexcept {
        
        // Safety checks
        assert(count <= data.size());
        assert(old_to_new_perm.size() >= count);
        
        #ifndef NDEBUG
        for (size_t i = 0; i < count; ++i) {
            assert(old_to_new_perm[i] < count);
        }
        #endif
        
        if (count == 0) return;
        
        // Resize scratch buffer once (uint8_t is faster than std::vector<bool>)
        if (visited.size() < count) visited.resize(count);
        std::fill_n(visited.data(), count, 0);
        
        for (size_t start = 0; start < count; ++start) {
            if (visited[start]) continue;
            
            // Swap-chain: follow cycle and place each element at its destination
            size_t current = start;
            T temp = data[start];
            
            while (true) {
                size_t next = old_to_new_perm[current];  // where element at 'current' must go
                std::swap(temp, data[next]);             // drop temp at destination; pick up what's there
                visited[current] = 1;
                current = next;
                if (current == start) break;             // cycle closed
            }
        }
    } 
    inline float get_adaptive_softening_squared() const {
        // Cache eps2 calculation - only recompute when bounds change
        if (!eps2_cache_valid_) {
            const float world_span_x = float(bounds_max_x_ - bounds_min_x_);
            const float world_span_y = float(bounds_max_y_ - bounds_min_y_);
            const float world_scale = std::max(world_span_x, world_span_y);
            
            // Softening that scales with the scene
            const float eps = config_.softening_rel * world_scale;
            cached_eps2_ = eps * eps;
            eps2_cache_valid_ = true;
        }
        return cached_eps2_;
    }
        

    
    inline int get_quadrant(double x, double y, double cx, double cy) const {
    const int east = (x >= cx) ? 1 : 0;
    const int north = (y >= cy) ? 1 : 0;
    return (north << 1) | east;  // SW(0,0)->0, SE(1,0)->1, NW(0,1)->2, NE(1,1)->3
}
};

// Backwards compatibility typedef
using ParticleSystem = BarnesHutParticleSystem;
