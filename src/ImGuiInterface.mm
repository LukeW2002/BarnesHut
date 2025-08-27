#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "ImGuiInterface.h"
#include "BarnesHutParticleSystem.h"
#include "MetalRenderer.h"
#include "EventSystem.h"

// IMGUI includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

#include <GLFW/glfw3.h>
#include <iostream>

ImGuiInterface::ImGuiInterface(BarnesHutParticleSystem& particle_system,
                               MetalRenderer& renderer,
                               EventBus& event_bus)
    : particle_system_(particle_system)
    , renderer_(renderer)
    , event_bus_(event_bus) {
    
    // Initialize default physics parameters
    physics_params_.gravity_x = 0.0f;
    physics_params_.gravity_y = 0.0f;
    physics_params_.damping = 1.0f;
    physics_params_.bounce_force = 1000.0f;
    physics_params_.boundary_min_x = -50.0f;
    physics_params_.boundary_max_x = 50.0f;
    physics_params_.boundary_min_y = -50.0f;
    physics_params_.boundary_max_y = 50.0f;
    physics_params_.time_scale = 1.0f;
    
    // Initialize default render parameters
    render_params_.particle_size = 1.5f;
    render_params_.background_r = 0.02f;
    render_params_.background_g = 0.02f;
    render_params_.background_b = 0.05f;
    render_params_.background_a = 1.0f;
}

bool ImGuiInterface::initialize(GLFWwindow* window, void* metal_device) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Setup style
    ImGui::StyleColorsDark();
    
    // Setup Platform/Renderer backends
    if (!ImGui_ImplGlfw_InitForOther(window, true)) {
        std::cerr << "Failed to initialize ImGui GLFW backend\n";
        return false;
    }
    if (!ImGui_ImplMetal_Init((__bridge id<MTLDevice>)metal_device)) {
        std::cerr << "Failed to initialize ImGui Metal backend\n";
        return false;
    }
    
    return true;
}

void ImGuiInterface::cleanup() {
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiInterface::render() {
    // Start ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Render all UI components
    render_main_menu_bar();
    
    if (show_performance_window_) {
        render_performance_window();
    }
    
    if (show_particle_controls_) {
        render_particle_controls();
    }
    
    if (show_barnes_hut_controls_) {
        render_barnes_hut_controls();
    }
    
    if (show_optimization_controls_) {
        render_optimization_controls();
    }
    
    if (show_realistic_presets_) {
        render_realistic_presets_window();
    }
    
    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }
    
    // Prepare for rendering
    ImGui::Render();
}

void ImGuiInterface::render_main_menu_bar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Reset Simulation")) {
                if (on_create_galaxy_spiral) on_create_galaxy_spiral(); // Default reset
            }
            if (ImGui::MenuItem("Exit")) {
                running_ = false;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Debug")) {
            if (ImGui::MenuItem("Show Quadtree", "Q", &show_quadtree_visualization_)) {
                particle_system_.set_quadtree_visualization_enabled(show_quadtree_visualization_);
                renderer_.set_quadtree_visualization_enabled(show_quadtree_visualization_);
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Performance", nullptr, &show_performance_window_);
            ImGui::MenuItem("Particle Controls", nullptr, &show_particle_controls_);
            ImGui::MenuItem("Barnes-Hut Settings", nullptr, &show_barnes_hut_controls_);
            ImGui::MenuItem("Optimizations", nullptr, &show_optimization_controls_);
            ImGui::MenuItem("Realistic Galaxies", nullptr, &show_realistic_presets_);
            ImGui::MenuItem("Demo Window", nullptr, &show_demo_window_);
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Optimizations")) {
            if (ImGui::MenuItem("Run All Optimizations")) {
                if (on_run_optimizations) on_run_optimizations();
            }
            if (ImGui::MenuItem("Fix Particle Overlaps")) {
                if (on_fix_overlaps) on_fix_overlaps();
            }
            if (ImGui::MenuItem("Optimize Spatial Layout")) {
                if (on_optimize_layout) on_optimize_layout();
            }
            if (ImGui::MenuItem("Find Optimal Theta")) {
                if (on_find_optimal_theta) on_find_optimal_theta();
            }
            if (ImGui::MenuItem("Compact Tree Cache")) {
                if (on_compact_cache) on_compact_cache();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Diagnose Performance")) {
                if (on_diagnose_performance) on_diagnose_performance();
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Realistic Galaxies")) {
            if (ImGui::MenuItem("🌌 Milky Way Galaxy")) {
                if (on_create_milky_way) on_create_milky_way();
            }
            if (ImGui::MenuItem("🌌 Andromeda Galaxy (M31)")) {
                if (on_create_andromeda) on_create_andromeda();
            }
            if (ImGui::MenuItem("💫 Two-Spiral Binary Merger (≈100k)")) {
                if (on_create_galaxy_merger) on_create_galaxy_merger();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Galaxy Merger")) {
                if (on_create_tidal_flyby) on_create_tidal_flyby();
            }
            if (ImGui::MenuItem("🌌 Local Group")) {
                if (on_create_local_group) on_create_local_group();
            }
            if (ImGui::MenuItem("🌌 Twin Galaxy")) {
                if (on_create_twin_spiral) on_create_twin_spiral();
            }
            if (ImGui::MenuItem("🔥 Dwarf Accretion")) {
                if (on_create_dwarf_accretion) on_create_dwarf_accretion();
            }
            ImGui::EndMenu();
        }
        
        // Pause/Play button
        if (ImGui::Button(paused_ ? "Play" : "Pause")) {
            paused_ = !paused_;
        }
        
        // FPS display
        ImGui::SameLine();
        ImGui::Text("FPS: %.1f", performance_stats_.fps);
        
        // Particle count
        ImGui::SameLine();
        ImGui::Text("Particles: %zu", performance_stats_.particle_count);
        
        // Force calc time with performance indicator
        const auto& bh_stats = particle_system_.get_performance_stats();
        ImGui::SameLine();
        if (bh_stats.force_calculation_time_ms > 10.0f) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "Force: %.1fms ⚠️", bh_stats.force_calculation_time_ms);
        } else if (bh_stats.force_calculation_time_ms > 5.0f) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Force: %.1fms", bh_stats.force_calculation_time_ms);
        } else {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Force: %.1fms ✅", bh_stats.force_calculation_time_ms);
        }
        
        ImGui::EndMainMenuBar();
    }
}

void ImGuiInterface::render_performance_window() {
    if (!ImGui::Begin("Performance", &show_performance_window_)) {
        ImGui::End();
        return;
    }
    
    ImGui::Text("Frame Time: %.3f ms", performance_stats_.frame_time_ms);
    ImGui::Text("FPS: %.1f", performance_stats_.fps);
    ImGui::Text("Particles: %zu", performance_stats_.particle_count);
    ImGui::Text("Physics Iterations: %zu", performance_stats_.physics_iterations);
    
    ImGui::Separator();
    ImGui::Text("Barnes-Hut Performance");
    const auto& bh_stats = particle_system_.get_performance_stats();
    ImGui::Text("Tree Build Time: %.3f ms", bh_stats.tree_build_time_ms);
    ImGui::Text("Force Calculation: %.3f ms", bh_stats.force_calculation_time_ms);
    ImGui::Text("Integration Time: %.3f ms", bh_stats.integration_time_ms);
    ImGui::Text("Tree Nodes: %zu", bh_stats.tree_nodes_created);
    ImGui::Text("Tree Depth: %zu", bh_stats.tree_depth);
    ImGui::Text("Direct Calculations: %zu", bh_stats.force_calculations);
    ImGui::Text("Approximations Used: %zu", bh_stats.approximations_used);
    ImGui::Text("Efficiency Ratio: %.1f%%", bh_stats.efficiency_ratio * 100.0f);
    ImGui::Text("Tree Rebuilt: %s", bh_stats.tree_was_rebuilt ? "Yes" : "No");
    
    // Performance warning indicators
    if (bh_stats.avg_tree_depth_per_particle > 100) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "⚠️ HIGH TREE DEPTH: %.0f", bh_stats.avg_tree_depth_per_particle);
        ImGui::Text("   ↳ Try optimizations to fix overlapping particles");
    }
    
    if (bh_stats.efficiency_ratio < 0.5f) {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "⚠️ LOW EFFICIENCY: %.1f%%", bh_stats.efficiency_ratio * 100.0f);
        ImGui::Text("   ↳ Try increasing theta value");
    }
    
    if (bh_stats.force_calculation_time_ms > 15.0f && particle_system_.get_particle_count() < 10000) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "⚠️ SLOW FORCE CALC: %.1fms", bh_stats.force_calculation_time_ms);
        ImGui::Text("   ↳ Run comprehensive optimizations (Press O)");
    }
    
    ImGui::Separator();
    ImGui::Text("Renderer Stats");
    const auto& render_stats = renderer_.get_stats();
    ImGui::Text("Particles Rendered: %zu", render_stats.particles_rendered);
    ImGui::Text("Vertices Processed: %zu", render_stats.vertices_processed);
    ImGui::Text("Draw Calls: %zu", render_stats.draw_calls);
    ImGui::Text("Render Time: %.3f ms", render_stats.last_frame_time_ms);
    
    // Frame time graph
    if (!performance_stats_.frame_time_history.empty()) {
        ImGui::PlotLines("Frame Time (ms)", 
            performance_stats_.frame_time_history.data(),
            static_cast<int>(performance_stats_.frame_time_history.size()),
            0, nullptr, 0.0f, 50.0f, ImVec2(0, 80));
    }
    
    ImGui::End();
}

void ImGuiInterface::render_particle_controls() {
    if (!ImGui::Begin("Particle Controls", &show_particle_controls_)) {
        ImGui::End();
        return;
    }
    
    // Physics parameters
    ImGui::Text("Physics");
    ImGui::SliderFloat("Time Scale", &physics_params_.time_scale, 0.0f, 5.0f);
    ImGui::SliderFloat("Gravity X", &physics_params_.gravity_x, -20.0f, 20.0f);
    ImGui::SliderFloat("Gravity Y", &physics_params_.gravity_y, -20.0f, 20.0f);
    ImGui::SliderFloat("Damping", &physics_params_.damping, 0.9f, 1.0f);
    ImGui::SliderFloat("Bounce Force", &physics_params_.bounce_force, 100.0f, 5000.0f);
    
    ImGui::Separator();
    ImGui::Text("Boundaries");
    ImGui::SliderFloat("Min X", &physics_params_.boundary_min_x, -100.0f, 0.0f);
    ImGui::SliderFloat("Max X", &physics_params_.boundary_max_x, 0.0f, 100.0f);
    ImGui::SliderFloat("Min Y", &physics_params_.boundary_min_y, -100.0f, 0.0f);
    ImGui::SliderFloat("Max Y", &physics_params_.boundary_max_y, 0.0f, 100.0f);
    
    ImGui::Separator();
    ImGui::Text("Rendering");
    if (ImGui::SliderFloat("Particle Size", &render_params_.particle_size, 0.5f, 10.0f)) {
        renderer_.set_particle_size(render_params_.particle_size);
    }
    
    if (ImGui::ColorEdit3("Background", &render_params_.background_r)) {
        renderer_.set_background_color(
            render_params_.background_r,
            render_params_.background_g, 
            render_params_.background_b,
            render_params_.background_a
        );
    }
    
    ImGui::Separator();
    
    // Particle management
    ImGui::Text("Particle Management");
    static int particles_to_add = 1000;
    ImGui::SliderInt("Count to Add", &particles_to_add, 100, 10000);
    
    if (ImGui::Button("Add Random Particles")) {
        if (on_add_random_particles) on_add_random_particles(particles_to_add);
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Clear All")) {
        particle_system_.clear_particles();
    }
    
    // Classic preset configurations
    ImGui::Separator();
    ImGui::Text("Classic Presets");
    if (ImGui::Button("Galaxy Spiral")) {
        if (on_create_galaxy_spiral) on_create_galaxy_spiral();
    }
    ImGui::SameLine();
    if (ImGui::Button("Solar System")) {
        if (on_create_solar_system) on_create_solar_system();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cluster")) {
        if (on_create_cluster) on_create_cluster();
    }
    
    ImGui::End();
}

void ImGuiInterface::render_barnes_hut_controls() {
    if (!ImGui::Begin("Barnes-Hut Settings", &show_barnes_hut_controls_)) {
        ImGui::End();
        return;
    }
    
    auto config = particle_system_.get_config();
    bool config_changed = false;
    
    ImGui::Text("Algorithm Parameters");
    config_changed |= ImGui::SliderFloat("Theta", &config.theta, 0.1f, 2.0f);
    ImGui::Text("Lower = more accurate, higher = faster");
    
    config_changed |= ImGui::Checkbox("Enable Tree Caching", &config.enable_tree_caching);
    ImGui::Text("Cache tree between frames when particles don't move much");
    
    if (config.enable_tree_caching) {
        config_changed |= ImGui::SliderFloat("Rebuild Threshold", &config.tree_rebuild_threshold, 0.05f, 0.5f);
        ImGui::Text("Rebuild tree when >%.0f%% of particles move significantly", 
                   config.tree_rebuild_threshold * 100.0f);
    }
    
    config_changed |= ImGui::SliderInt("Max Particles per Leaf", 
                                      reinterpret_cast<int*>(&config.max_particles_per_leaf), 1, 10);
    
    config_changed |= ImGui::SliderInt("Tree Depth Limit", 
                                      reinterpret_cast<int*>(&config.tree_depth_limit), 10, 30);
    
#ifdef _OPENMP
    config_changed |= ImGui::Checkbox("Enable Threading", &config.enable_threading);
    ImGui::Text("Use OpenMP for parallel force calculations");
#else
    ImGui::Text("OpenMP not available - single threaded only");
#endif
    
    if (config_changed) {
        config.theta_squared = config.theta * config.theta;
        particle_system_.set_config(config);
    }
    
    ImGui::Separator();
    ImGui::Text("Performance Tips");
    ImGui::BulletText("Theta 0.5-0.8: Good balance of speed and accuracy");
    ImGui::BulletText("Theta 1.0+: Very fast but less accurate");
    ImGui::BulletText("Enable caching for stable simulations");
    ImGui::BulletText("Disable caching for chaotic motion");
    ImGui::BulletText("Threading helps with >10k particles");
    
    ImGui::End();
}

void ImGuiInterface::render_optimization_controls() {
    if (!ImGui::Begin("Optimization Controls", &show_optimization_controls_)) {
        ImGui::End();
        return;
    }
    
    ImGui::Text("Performance Optimization Tools");
    ImGui::Text("Fix slowdowns and improve Barnes-Hut efficiency");
    
    ImGui::Separator();
    
    // Quick optimization buttons
    if (ImGui::Button("🚀 Run All Optimizations", ImVec2(-1, 0))) {
        if (on_run_optimizations) on_run_optimizations();
    }
    ImGui::Text("Comprehensive optimization: fixes overlaps, sorts particles,\nfinds optimal theta, compacts tree for cache efficiency");
    
    ImGui::Separator();
    ImGui::Text("Individual Optimizations:");
    
    if (ImGui::Button("🔧 Fix Particle Overlaps", ImVec2(-1, 0))) {
        if (on_fix_overlaps) on_fix_overlaps();
    }
    ImGui::Text("Remove overlapping particles that cause deep tree recursion");
    
    if (ImGui::Button("📊 Optimize Spatial Layout", ImVec2(-1, 0))) {
        if (on_optimize_layout) on_optimize_layout();
    }
    ImGui::Text("Reorder particles using Morton Z-order for cache locality");
    
    if (ImGui::Button("🎯 Find Optimal Theta", ImVec2(-1, 0))) {
        if (on_find_optimal_theta) on_find_optimal_theta();
    }
    ImGui::Text("Test different theta values and pick the fastest");
    
    if (ImGui::Button("📚✨ Compact Tree Cache", ImVec2(-1, 0))) {
        if (on_compact_cache) on_compact_cache();
    }
    ImGui::Text("Reorganize tree nodes for better memory access patterns");
    
    ImGui::Separator();
    ImGui::Text("Diagnostic Tools:");
    
    if (ImGui::Button("🔍 Diagnose Performance", ImVec2(-1, 0))) {
        if (on_diagnose_performance) on_diagnose_performance();
    }
    ImGui::Text("Analyze performance bottlenecks and get recommendations");
    
    // Performance indicator
    const auto& stats = particle_system_.get_performance_stats();
    ImGui::Text("Current Performance:");
    
    if (stats.force_calculation_time_ms > 15.0f) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "🔴 Force Calc: %.1fms (SLOW)", stats.force_calculation_time_ms);
        ImGui::Text("   ↳ Run optimizations to improve performance");
    } else if (stats.force_calculation_time_ms > 5.0f) {
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "⚠️ Force Calc: %.1fms (OK)", stats.force_calculation_time_ms);
    } else {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "✅ Force Calc: %.1fms (GOOD)", stats.force_calculation_time_ms);
    }
    
    ImGui::Text("Tree Depth: %zu (target: <20)", stats.tree_depth);
    ImGui::Text("Efficiency: %.1f%% approximations", stats.efficiency_ratio * 100.0f);
    ImGui::Text("Speedup: %.1fx vs brute force", stats.speedup_vs_brute_force);
    
    ImGui::End();
}

void ImGuiInterface::render_realistic_presets_window() {
    if (!ImGui::Begin("Realistic Galaxy Simulations", &show_realistic_presets_)) {
        ImGui::End();
        return;
    }
    
    ImGui::Text("🌌 Scientifically Accurate Galaxy Simulations");
    ImGui::Text("Based on observational data and astrophysical models");
    
    ImGui::Separator();
    ImGui::Text("Individual Galaxies:");
    
    if (ImGui::Button("🌌 Milky Way Galaxy", ImVec2(-1, 0))) {
        if (on_create_milky_way) on_create_milky_way();
    }
    ImGui::Text("Our galaxy with realistic mass distribution:\n"
               "• Central SMBH (Sgr A*): 4.3M solar masses\n"
               "• Stellar disk: ~55B solar masses, exponential profile\n"
               "• Central bulge: ~20B solar masses, Hernquist profile\n"
               "• Dark matter halo: NFW profile, concentration ~12\n"
               "• Spiral structure with logarithmic arms");
    
    ImGui::Separator();
    
    if (ImGui::Button("🌌 Andromeda Galaxy (M31)", ImVec2(-1, 0))) {
        if (on_create_andromeda) on_create_andromeda();
    }
    ImGui::Text("Larger spiral galaxy, closest major neighbor:\n"
               "• More massive than Milky Way (~1.8x)\n"
               "• Larger SMBH: ~140M solar masses\n"
               "• Extended stellar disk and prominent bulge\n"
               "• Higher rotation velocities (~260 km/s)");
    
    ImGui::Separator();
    ImGui::Text("Galaxy Interactions & Evolution:");
    
    if (ImGui::Button("💥 Galaxy Merger Event", ImVec2(-1, 0))) {
        if (on_create_tidal_flyby) on_create_tidal_flyby();
    }
    ImGui::Text("Two galaxies on collision course:\n"
               "• Realistic orbital parameters from simulations\n"
               "• Tidal stripping and stellar streams\n"
               "• Star formation bursts during close passes\n"
               "• Final coalescence into elliptical remnant");
    
    if (ImGui::Button("🌌 Local Group Simulation", ImVec2(-1, 0))) {
        if (on_create_local_group) on_create_local_group();
    }
    ImGui::Text("Our local galactic neighborhood:\n"
               "• Milky Way and Andromeda (approaching collision)\n"
               "• Large & Small Magellanic Clouds\n"
               "• Realistic masses and orbital motions\n"
               "• Future Milkomeda merger preview");
    
    if (ImGui::Button("🔥 Dwarf Galaxy Accretion", ImVec2(-1, 0))) {
        if (on_create_dwarf_accretion) on_create_dwarf_accretion();
    }
    ImGui::Text("Hierarchical galaxy formation in action:\n"
               "• Multiple dwarf galaxies falling into main halo\n"
               "• Tidal disruption and stellar stream formation\n"
               "• Mass assembly through minor mergers\n"
               "• Dark matter substructure evolution");
    
    ImGui::Separator();
    ImGui::Text("Simulation Features:");
    ImGui::BulletText("Realistic mass-to-light ratios");
    ImGui::BulletText("Proper velocity dispersions & rotation curves");
    ImGui::BulletText("NFW dark matter halo profiles");
    ImGui::BulletText("Exponential disk & Hernquist bulge components");
    ImGui::BulletText("Supermassive black holes at galaxy centers");
    ImGui::BulletText("Spiral arm structure in disk galaxies");
    
    ImGui::Separator();
    ImGui::Text("Physical Parameters (scaled for simulation):");
    ImGui::Text("• Masses: Solar mass units (÷10⁹ for display)");
    ImGui::Text("• Distances: kpc units (scaled to screen coordinates)");
    ImGui::Text("• Velocities: km/s (scaled for stable orbits)");
    ImGui::Text("• Time: Myr per simulation step");
    
    if (ImGui::CollapsingHeader("Scientific References")) {
        ImGui::Text("Based on observational data from:");
        ImGui::BulletText("Gaia spacecraft astrometry");
        ImGui::BulletText("Milky Way rotation curve measurements");
        ImGui::BulletText("Dark matter halo mass estimates");
        ImGui::BulletText("ΛCDM cosmological simulations");
        ImGui::BulletText("Navarro-Frenk-White (NFW) profiles");
        ImGui::BulletText("Local Group galaxy kinematics");
    }
    
    ImGui::End();
}

void ImGuiInterface::update_performance_stats(float frame_time) {
    performance_stats_.frame_time_ms = frame_time * 1000.0f;
    performance_stats_.fps = 1.0f / frame_time;
    
    // Update frame time history for graphing
    if (performance_stats_.frame_time_history.size() < 100) {
        performance_stats_.frame_time_history.push_back(performance_stats_.frame_time_ms);
    } else {
        performance_stats_.frame_time_history[performance_stats_.history_index] = performance_stats_.frame_time_ms;
        performance_stats_.history_index = (performance_stats_.history_index + 1) % 100;
    }
}
