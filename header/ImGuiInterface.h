#pragma once

// Forward declarations
class BarnesHutParticleSystem;
class MetalRenderer;
class EventBus;
struct GLFWwindow;

#include "Vec2.h"
#include <functional>

// ImGui interface management class
class ImGuiInterface {
public:
    struct PhysicsParams {
        float gravity_x = 0.0f;
        float gravity_y = 0.0f;
        float damping = 1.0f;
        float bounce_force = 1000.0f;
        float boundary_min_x = -50.0f;
        float boundary_max_x = 50.0f;
        float boundary_min_y = -50.0f;
        float boundary_max_y = 50.0f;
        float time_scale = 1.0f;
    };
    
    struct RenderParams {
        float particle_size = 1.5f;
        float background_r = 0.02f;
        float background_g = 0.02f;
        float background_b = 0.05f;
        float background_a = 1.0f;
    };
    
    struct PerformanceStats {
        float frame_time_ms = 0.0f;
        float fps = 0.0f;
        size_t particle_count = 0;
        size_t physics_iterations = 0;
        float avg_frame_time_ms = 0.0f;
        std::vector<float> frame_time_history;
        size_t history_index = 0;
    };

    // Constructor
    ImGuiInterface(BarnesHutParticleSystem& particle_system,
                   MetalRenderer& renderer,
                   EventBus& event_bus);
    
    // Initialization and cleanup
    bool initialize(GLFWwindow* window, void* metal_device);
    void cleanup();
    
    // Main render function
    void render();
    
    // Update performance stats
    void update_performance_stats(float frame_time);
    
    // Accessors
    const PhysicsParams& get_physics_params() const { return physics_params_; }
    const RenderParams& get_render_params() const { return render_params_; }
    bool is_paused() const { return paused_; }
    bool should_show_quadtree() const { return show_quadtree_visualization_; }
    
    // Setters
    void set_paused(bool paused) { paused_ = paused; }
    void set_running(bool running) { running_ = running; }
    
    // Callbacks for galaxy creation
    std::function<void()> on_create_milky_way;
    std::function<void()> on_create_andromeda;
    std::function<void()> on_create_galaxy_merger;
    std::function<void()> on_create_local_group;
    std::function<void()> on_create_twin_spiral;
    std::function<void()> on_create_dwarf_accretion;
    std::function<void()> on_create_tidal_flyby;

    
    // Callbacks for optimizations
    std::function<void()> on_run_optimizations;
    std::function<void()> on_fix_overlaps;
    std::function<void()> on_optimize_layout;
    std::function<void()> on_find_optimal_theta;
    std::function<void()> on_diagnose_performance;
    std::function<void()> on_compact_cache;
    
    // Callbacks for classic presets
    std::function<void()> on_create_galaxy_spiral;
    std::function<void()> on_create_solar_system;
    std::function<void()> on_create_cluster;
    std::function<void(int)> on_add_random_particles;

private:
    // References to main systems
    BarnesHutParticleSystem& particle_system_;
    MetalRenderer& renderer_;
    EventBus& event_bus_;
    
    // UI state
    bool running_ = true;
    bool paused_ = false;
    bool show_demo_window_ = false;
    bool show_performance_window_ = true;
    bool show_particle_controls_ = true;
    bool show_barnes_hut_controls_ = true;
    bool show_optimization_controls_ = true;
    bool show_realistic_presets_ = true;
    bool show_quadtree_visualization_ = false;
    
    // Parameters
    PhysicsParams physics_params_;
    RenderParams render_params_;
    PerformanceStats performance_stats_;
    
    // Render functions
    void render_main_menu_bar();
    void render_performance_window();
    void render_particle_controls();
    void render_barnes_hut_controls();
    void render_optimization_controls();
    void render_realistic_presets_window();
};
