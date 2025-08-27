// main.mm - Enhanced with realistic galaxy simulation parameters
#include "BarnesHutParticleSystem.h"
#include "MetalRenderer.h"
#include "EventSystem.h"
#include "Vec2.h"

// IMGUI includes
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"

// GLFW for window management
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

// Import Metal and related frameworks directly
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>

#include <iostream>
#include <memory>
#include <random>
#include <chrono>

//static constexpr double SOFTENING_LENGTH = 0.05;  // 50 pc softening
//static constexpr double G_GALACTIC = 4.3009;              // kpc·(km/s)²·(10⁹M☉)⁻¹  
//static constexpr double VELOCITY_UNIT_KMS = 65.58;        // km/s per code velocity unit
//static constexpr double TIME_UNIT_MYR = 14.91;            // Myr per code time unit

// *** FIXED: Proper galactic-scale softening ***
//static constexpr double DEFAULT_SOFTENING_KPC = 0.050;    // 50 pc = 0.05 kpc (not 0.001!)
//static constexpr double EPS_SQ = DEFAULT_SOFTENING_KPC * DEFAULT_SOFTENING_KPC;

// World-scale constants for galaxy simulations
static constexpr double MILKY_WAY_RADIUS_KPC = 25.0;      // 25 kpc galactic radius
static constexpr double GALACTIC_BOUNDARY_KPC = 40.0;     // 40 kpc simulation boundary
static constexpr double SOLAR_CIRCLE_RADIUS_KPC = 8.2;    // Sun's distance from galactic center
static constexpr double FLAT_ROTATION_VELOCITY_KMS = 220.0; // Flat rotation curve velocity

// Helpers (put near the top of main.mm or in a small header)
static inline double hernquist_Menc(double r, double Mb, double a) {
    // r, a in kpc; Mb in 1e9 Msun
    return Mb * (r*r) / ((r + a) * (r + a));
}
static inline double disk_exp_Menc(double r, double Md, double Rd) {
    // Thin exponential disk cumulative mass (2D): Md [1 - e^{-r/Rd}(1 + r/Rd)]
    const double x = r / Rd;
    return Md * (1.0 - std::exp(-x) * (1.0 + x));
}
static inline double nfw_Menc(double r, double Mh, double rvir, double c) {
    // Normalized NFW: Mh * f(x)/f(c), x=r/rs, rs=rvir/c
    const double rs = rvir / c;
    const auto f = [](double y){ return std::log(1.0 + y) - y/(1.0 + y); };
    const double x  = r / rs;
    const double fc = f(c);
    return (fc > 0.0) ? Mh * (f(x) / fc) : 0.0;
}
static inline double vcirc_code(double r_kpc, double Menc_1e9Msun) {
    // v_code = sqrt(G * M / (r + eps)) in km/s, then divide by VELOCITY_UNIT_KMS
    const double v_kms = std::sqrt(G_GALACTIC * std::max(0.0, Menc_1e9Msun)
                                   / (r_kpc + DEFAULT_SOFTENING_KPC));
    return v_kms / VELOCITY_UNIT_KMS;
}
static inline double clamp_min_r(double r, double rmin) { return (r < rmin) ? rmin : r; }


class BarnesHutApp {
public:
    BarnesHutApp() 
        : event_bus_(),
          particle_system_(1000000, event_bus_), // Use default config
          renderer_(event_bus_),
          window_(nullptr),
          metal_device_(nil),
          command_queue_(nil),
          metal_layer_(nil),
          running_(false),
          paused_(false),
          show_demo_window_(false),
          show_performance_window_(true),
          show_particle_controls_(true),
          show_barnes_hut_controls_(true),
          show_optimization_controls_(true),
          show_realistic_presets_(true) {  // NEW: Realistic galaxy presets
        
        // Physics parameters (configurable via IMGUI)
        physics_params_.gravity_x = 0.0f;
        physics_params_.gravity_y = 0.0f;
        physics_params_.damping = 1.0f;
        physics_params_.bounce_force = 1000.0f;
        physics_params_.boundary_min_x = -50.0f;  // Larger boundaries for galaxy scales
        physics_params_.boundary_max_x = 50.0f;
        physics_params_.boundary_min_y = -50.0f;
        physics_params_.boundary_max_y = 50.0f;
        physics_params_.time_scale = 1.0f;
        
        // Rendering parameters
        render_params_.particle_size = 1.5f;  // Smaller for galaxy simulations
        render_params_.background_r = 0.02f;
        render_params_.background_g = 0.02f;
        render_params_.background_b = 0.05f;
        render_params_.background_a = 1.0f;
        
        // Camera parameters
        camera_.zoom = 0.5f;  // Zoomed out for galaxy scales
        camera_.center_x = 0.0f;
        camera_.center_y = 0.0f;
        
        // Barnes-Hut configuration - initialize properly
        barnes_hut_config_ = BarnesHutParticleSystem::Config(); // Use default constructor
        barnes_hut_config_.theta = 0.75f;
        barnes_hut_config_.enable_tree_caching = true;
        barnes_hut_config_.tree_rebuild_threshold = 0.1f;
        barnes_hut_config_.max_particles_per_leaf = 8;
        barnes_hut_config_.enable_threading = true;
        
        subscribe_to_events();
    }
    
    ~BarnesHutApp() {
        cleanup();
    }
    bool show_quadtree_visualization_ = false;
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        auto* app = static_cast<BarnesHutApp*>(glfwGetWindowUserPointer(window));
        
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            if (action == GLFW_PRESS) {
                app->enhanced_camera_.middle_mouse_dragging = true;
                
                double xpos, ypos;
                glfwGetCursorPos(window, &xpos, &ypos);
                app->enhanced_camera_.last_mouse_x = xpos;
                app->enhanced_camera_.last_mouse_y = ypos;
                
                // Hide cursor during drag
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
                
                // Show world position where we started dragging
                Vec2 world_pos = app->enhanced_camera_.screen_to_world(xpos, ypos);
                std::cout << "Started pan at world (" << world_pos.x << ", " << world_pos.y << ")\n";
                
            } else if (action == GLFW_RELEASE) {
                app->enhanced_camera_.middle_mouse_dragging = false;
                
                // Restore cursor
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
                
                std::cout << "Ended pan at world center (" << app->enhanced_camera_.world_center_x 
                          << ", " << app->enhanced_camera_.world_center_y << ")\n";
            }
        }
    }
    // Add cursor position callback for middle mouse panning
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        auto* app = static_cast<BarnesHutApp*>(glfwGetWindowUserPointer(window));
        
        if (app->enhanced_camera_.middle_mouse_dragging) {
            // Calculate mouse movement in pixels
            double dx = xpos - app->enhanced_camera_.last_mouse_x;
            double dy = ypos - app->enhanced_camera_.last_mouse_y;
            
            // Pan the camera (note: y is flipped in screen coordinates)
            app->enhanced_camera_.pan_by_screen_pixels(dx, -dy);
            
            // Update last mouse position
            app->enhanced_camera_.last_mouse_x = xpos;
            app->enhanced_camera_.last_mouse_y = ypos;
            
            // Optional: Show current world position under cursor
            Vec2 world_pos = app->enhanced_camera_.screen_to_world(xpos, ypos);
            if ((int)xpos % 10 == 0) {  // Only print occasionally to avoid spam
                std::cout << "Panning: mouse at world (" << world_pos.x << ", " << world_pos.y << ")\n";
            }
        }
    }
    
    // Enhanced scroll callback with zoom-to-point
        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        auto* app = static_cast<BarnesHutApp*>(glfwGetWindowUserPointer(window));
        
        // Get mouse position for zoom-to-point
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        
        // More sensitive zoom factor for better control
        double zoom_factor = (yoffset > 0) ? 1.2 : (1.0 / 1.2);
        
        // Store old zoom for comparison
        double old_zoom = app->enhanced_camera_.world_zoom;
        
        // Zoom at mouse position
        app->enhanced_camera_.zoom_at_screen_point(mouse_x, mouse_y, zoom_factor);
        
        // Only print debug if zoom actually changed
        if (std::abs(app->enhanced_camera_.world_zoom - old_zoom) > 1e-10) {
            double left, right, bottom, top;
            app->enhanced_camera_.get_world_bounds(left, right, bottom, top);
            double world_width = right - left;
            double world_height = top - bottom;
            
            std::cout << "Zoom: " << app->enhanced_camera_.world_zoom << " px/unit, "
                      << "viewing " << world_width << " x " << world_height << " world units"
                      << " (center: " << app->enhanced_camera_.world_center_x 
                      << ", " << app->enhanced_camera_.world_center_y << ")\n";
        }
    }
 
    bool initialize(int width = 1200, int height = 800) {
        // Initialize GLFW
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW\n";
            return false;
        }
        
        // Configure GLFW for Metal
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        
        // Create window
        window_ = glfwCreateWindow(width, height, "Realistic Galaxy Simulations - Barnes-Hut N-Body", nullptr, nullptr);
        if (!window_) {
            std::cerr << "Failed to create GLFW window\n";
            glfwTerminate();
            return false;
        }
        
        // Set up window callbacks
        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
        glfwSetKeyCallback(window_, key_callback);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetCursorPosCallback(window_, cursor_position_callback);
        glfwSetScrollCallback(window_, scroll_callback);
        
        enhanced_camera_.viewport_width = width;
        enhanced_camera_.viewport_height = height;
        // Initialize Metal
        if (!initialize_metal()) {
            std::cerr << "Failed to initialize Metal\n";
            return false;
        }
        
        // Initialize IMGUI
        if (!initialize_imgui()) {
            std::cerr << "Failed to initialize IMGUI\n";
            return false;
        }
        
        // Initialize MetalRenderer
        if (!renderer_.initialize((__bridge void*)metal_device_, (__bridge void*)command_queue_)) {
            std::cerr << "Failed to initialize MetalRenderer\n";
            return false;
        }
        
        // Set up initial particle configuration
        setup_initial_particles();
        
        // Configure physics system
        update_physics_parameters();
        
        running_ = true;
        return true;
    }
    
    void run() {
        auto last_time = std::chrono::high_resolution_clock::now();
        float time_accumulator = 0.0f;
        const float physics_dt = 1.0f / 120.0f; // 120 FPS physics
        
        while (running_ && !glfwWindowShouldClose(window_)) {
            auto current_time = std::chrono::high_resolution_clock::now();
            float frame_time = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            // Cap frame time to prevent spiral of death
            frame_time = std::min(frame_time, 0.05f);
            
            glfwPollEvents();
            
            // Fixed timestep physics
            if (!paused_) {
                time_accumulator += frame_time * physics_params_.time_scale;
                
                while (time_accumulator >= physics_dt) {
                    particle_system_.update(physics_dt);
                    time_accumulator -= physics_dt;
                }
            }
            
            // Render frame
            render_frame();
            
            // Update frame statistics
            update_performance_stats(frame_time);
        }
    }
    
private:
        struct EnhancedCamera {
        // === WORLD COORDINATES (Physics space in kpc) ===
        double world_center_x = 0.0;      
        double world_center_y = 0.0;      
        double world_zoom = 1.0;           // pixels per world unit
        
        // === RENDER VIEWPORT ===
        int viewport_width = 1200;
        int viewport_height = 800;
        
        // === MOUSE INTERACTION STATE ===
        bool middle_mouse_dragging = false;
        double last_mouse_x = 0.0;
        double last_mouse_y = 0.0;
        
        // === ZOOM LIMITS (more reasonable for debugging) ===
        double min_zoom = 0.1;           // Can see 10x more area
        double max_zoom = 100.0;         // Can see 100x less area
        
        // === COORDINATE TRANSFORMATIONS ===
        Vec2 world_to_screen(Vec2 world_pos) const {
            double screen_x = (world_pos.x - world_center_x) * world_zoom + viewport_width * 0.5;
            double screen_y = (world_pos.y - world_center_y) * world_zoom + viewport_height * 0.5;
            return Vec2(static_cast<float>(screen_x), static_cast<float>(screen_y));
        }
        
        Vec2 screen_to_world(double screen_x, double screen_y) const {
            double world_x = (screen_x - viewport_width * 0.5) / world_zoom + world_center_x;
            double world_y = (screen_y - viewport_height * 0.5) / world_zoom + world_center_y;
            return Vec2(static_cast<float>(world_x), static_cast<float>(world_y));
        }
        
        void get_world_bounds(double& left, double& right, double& bottom, double& top) const {
            double half_width_world = (viewport_width * 0.5) / world_zoom;
            double half_height_world = (viewport_height * 0.5) / world_zoom;
            
            left = world_center_x - half_width_world;
            right = world_center_x + half_width_world;
            bottom = world_center_y - half_height_world;
            top = world_center_y + half_height_world;
        }
        
        void pan_by_screen_pixels(double dx_pixels, double dy_pixels) {
            double dx_world = dx_pixels / world_zoom;
            double dy_world = dy_pixels / world_zoom;
            
            world_center_x -= dx_world;
            world_center_y -= dy_world;
        }
        
        void zoom_at_screen_point(double screen_x, double screen_y, double zoom_factor) {
            // Get world position under mouse before zoom
            Vec2 world_pos_before = screen_to_world(screen_x, screen_y);
            
            // Apply zoom with limits
            double new_zoom = world_zoom * zoom_factor;
            new_zoom = std::max(min_zoom, std::min(max_zoom, new_zoom));
            
            // Only update if zoom actually changed
            if (std::abs(new_zoom - world_zoom) > 1e-10) {
                world_zoom = new_zoom;
                
                // Get world position under mouse after zoom
                Vec2 world_pos_after = screen_to_world(screen_x, screen_y);
                
                // Adjust center to keep the same world point under the mouse
                world_center_x += (world_pos_before.x - world_pos_after.x);
                world_center_y += (world_pos_before.y - world_pos_after.y);
            }
        }
        
        // FIXED: Better conversion to renderer parameters
        MetalRenderer::CameraParams to_render_params() const {
            MetalRenderer::CameraParams render_params;
            
            // The Metal renderer expects these parameters:
            // - zoom: world units visible per screen unit (inverse of our world_zoom)
            // - center: world coordinates of view center
            render_params.zoom = static_cast<float>(1.0 / world_zoom);
            render_params.center_x = static_cast<float>(world_center_x);
            render_params.center_y = static_cast<float>(world_center_y);
            
            return render_params;
        }
        
        // Debug info
        void print_debug_info() const {
            double left, right, bottom, top;
            get_world_bounds(left, right, bottom, top);
            std::cout << "Camera Debug: center=(" << world_center_x << ", " << world_center_y 
                      << "), zoom=" << world_zoom << " px/unit"
                      << ", viewing [" << left << "," << right << "] x [" << bottom << "," << top << "] world units\n";
        }
    };
    
    
    EnhancedCamera enhanced_camera_;
    struct PhysicsParams {
        float gravity_x, gravity_y;
        float damping;
        float bounce_force;
        float boundary_min_x, boundary_max_x;
        float boundary_min_y, boundary_max_y;
        float time_scale;
    };
    
    struct RenderParams {
        float particle_size;
        float background_r, background_g, background_b, background_a;
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
    
    // Optimization methods for BarnesHutApp
    void run_particle_optimizations() {
        std::cout << "\n🚀 RUNNING PARTICLE SYSTEM OPTIMIZATIONS\n";
        particle_system_.run_comprehensive_optimization();
    }
    
    void quick_fix_overlaps() {
        std::cout << "\n🔧 QUICK FIX: Removing particle overlaps\n";
        particle_system_.fix_overlapping_particles_advanced();
    }
    
    void optimize_spatial_layout() {
        std::cout << "\n📊 OPTIMIZING: Spatial layout with Morton ordering\n";
        particle_system_.optimize_particle_layout();
    }
    
    void find_optimal_theta() {
        std::cout << "\n🎯 OPTIMIZING: Finding optimal theta value\n";
        particle_system_.adaptive_theta_optimization();
    }
    
    void diagnose_performance_issues() {
        std::cout << "\n🔍 DIAGNOSING: Performance bottlenecks\n";
        particle_system_.diagnose_tree_traversal_bottleneck();
        particle_system_.print_performance_analysis();
    }
    
    void compact_tree_cache() {
        std::cout << "\n📁✨ COMPACTING: Tree for cache efficiency\n";
        particle_system_.compact_tree_for_cache_efficiency();
    }
    void create_realistic_milky_way() {
    std::cout << "\n🌌 Creating CANONICAL Milky Way - debugging missing chunk...\n";
    particle_system_.clear_particles();

    std::mt19937 gen(std::random_device{}());
    const double scale = 0.01;
    
    const double disk_scale_length = 3.5 / scale;      
    const double bulge_radius = 0.6 / scale;           
    const double halo_scale_radius = 200.0 / scale;    
    const double max_radius = 25.0 / scale;            
    
    const double smbh_mass = 0.0043 / scale;           
    const double bulge_mass = 20.0 / scale;            
    const double disk_mass = 55.0 / scale;             
    const double halo_mass = 1200.0 / scale;           
    
    std::uniform_real_distribution<double> angle_dist(0, 2 * M_PI);
    std::normal_distribution<double> velocity_noise(0.0, 0.02);
    
    // Debug: Track particle distribution by angle
    std::vector<int> angle_bins(36, 0);  // 10-degree bins
    auto track_angle = [&](double x, double y) {
        double angle = std::atan2(y, x);
        if (angle < 0) angle += 2 * M_PI;
        int bin = (int)(angle * 18.0 / M_PI);  // Convert to 0-35 range
        if (bin >= 0 && bin < 36) angle_bins[bin]++;
    };
    
    struct Spawned {
        float x, y, vx, vy;
        double m;
        Vec3 color;
    };
    std::vector<Spawned> spawned;
    
    double Px = 0, Py = 0, Mtot = 0;

    // Central SMBH
    spawned.push_back({0, 0, 0, 0, smbh_mass, {1.0f, 1.0f, 0.8f}});
    Mtot += smbh_mass;

    // BULGE - Check for issues in Hernquist sampling
    int bulge_particles = 3000;
    std::cout << "Creating bulge with Hernquist profile...\n";
    
    int failed_bulge = 0;
    for (int i = 0; i < bulge_particles; ++i) {
        double u = std::max(1e-12, std::uniform_real_distribution<double>(0.0, 1.0)(gen));
        
        // ISSUE 1: Check Hernquist inverse CDF
        double sqrt_u = std::sqrt(u);
        if (sqrt_u >= 1.0) {
            failed_bulge++;
            continue;  // Skip invalid particles
        }
        
        double r = bulge_radius * sqrt_u / (1.0 - sqrt_u);
        r = std::min(std::max(r, 2.0 * DEFAULT_SOFTENING_KPC), max_radius);
        
        double th = angle_dist(gen);
        double x = r * std::cos(th);
        double y = r * std::sin(th);
        
        track_angle(x, y);

        double Menc = smbh_mass
                    + hernquist_Menc(r, bulge_mass, bulge_radius)
                    + disk_exp_Menc(r, disk_mass, disk_scale_length)
                    + nfw_Menc(r, halo_mass, halo_scale_radius, 10.0);

        double v = vcirc_code(r, Menc);
        double vx = -v * std::sin(th) + velocity_noise(gen);
        double vy = v * std::cos(th) + velocity_noise(gen);

        double m = bulge_mass / double(bulge_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, m, {1.0f, 0.8f, 0.4f}});
    }
    
    if (failed_bulge > 0) {
        std::cout << "⚠️  " << failed_bulge << " bulge particles failed generation\n";
    }

    // DISK - More robust exponential disk generation  
    int disk_particles = 80000;
    std::cout << "Creating disk with exponential profile...\n";
    
    // ISSUE 2: Replace the problematic Gamma(2, Rd) with simpler exponential
    std::exponential_distribution<double> exp_dist(1.0 / disk_scale_length);
    
    int failed_disk = 0;
    for (int i = 0; i < disk_particles; ++i) {
        // FIXED: Use single exponential instead of sum of two
        double r = exp_dist(gen);
        
        // Apply reasonable limits
        r = std::min(r, 4.0 * disk_scale_length);  // Max ~14 kpc
        r = std::max(r, 2.0 * DEFAULT_SOFTENING_KPC);  // Min softening
        
        // ISSUE 3: Check if angle distribution is uniform
        double th = angle_dist(gen);
        
        // Verify angle is in valid range
        if (th < 0 || th >= 2 * M_PI) {
            failed_disk++;
            th = std::fmod(th, 2 * M_PI);
            if (th < 0) th += 2 * M_PI;
        }
        
        double x = r * std::cos(th);
        double y = r * std::sin(th);
        
        // ISSUE 4: Check for NaN/inf values
        if (!std::isfinite(x) || !std::isfinite(y)) {
            std::cout << "⚠️  Invalid disk particle: r=" << r << ", th=" << th 
                      << ", x=" << x << ", y=" << y << "\n";
            failed_disk++;
            continue;
        }
        
        track_angle(x, y);

        double Menc = smbh_mass
                    + hernquist_Menc(r, bulge_mass, bulge_radius)
                    + disk_exp_Menc(r, disk_mass, disk_scale_length)
                    + nfw_Menc(r, halo_mass, halo_scale_radius, 10.0);

        double v = vcirc_code(r, Menc);
        
        // Check velocity is reasonable
        if (!std::isfinite(v) || v > 1000.0) {  // Sanity check
            std::cout << "⚠️  Extreme velocity: v=" << v << " at r=" << r << "\n";
            v = std::min(v, 100.0);  // Cap velocity
        }
        
        double vx = -v * std::sin(th) + velocity_noise(gen);
        double vy = v * std::cos(th) + velocity_noise(gen);

        double m = disk_mass / double(disk_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, m, {0.7f, 0.6f, 1.0f}});
    }
    
    if (failed_disk > 0) {
        std::cout << "⚠️  " << failed_disk << " disk particles had issues\n";
    }

    // HALO - Simplified NFW generation
    int halo_particles = 40000;
    std::cout << "Creating halo with NFW profile...\n";
    
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double c_halo = 10.0;
    const double rs = halo_scale_radius / c_halo;
    
    int failed_halo = 0;
    for (int i = 0; i < halo_particles; ++i) {
        double r = 0.0;
        
        // ISSUE 5: Simplified rejection sampling for NFW
        bool accepted = false;
        for (int attempt = 0; attempt < 50; ++attempt) {  // Limit attempts
            r = uniform(gen) * max_radius;
            double x_nfw = r / rs;
            
            if (x_nfw <= 0) continue;  // Avoid division by zero
            
            double f_nfw = std::log(1.0 + x_nfw) - x_nfw / (1.0 + x_nfw);
            double f_max = std::log(1.0 + max_radius / rs);
            
            if (f_max <= 0) break;  // Avoid division by zero
            
            if (uniform(gen) <= f_nfw / f_max) {
                accepted = true;
                break;
            }
        }
        
        if (!accepted) {
            r = uniform(gen) * max_radius;  // Fallback to uniform
            failed_halo++;
        }
        
        r = std::max(r, 2.0 * DEFAULT_SOFTENING_KPC);
        double th = angle_dist(gen);
        
        // Add 3D projection for more realistic halo
        double phi = std::acos(2.0 * uniform(gen) - 1.0);  // Isotropic
        double x = r * std::sin(phi) * std::cos(th);
        double y = r * std::sin(phi) * std::sin(th);
        
        track_angle(x, y);

        double sigma_kms = 100.0 / std::sqrt(1.0 + r / rs);
        double sigma_code = sigma_kms / VELOCITY_UNIT_KMS;
        std::normal_distribution<double> vel_disp(0.0, sigma_code);
        
        double vx = vel_disp(gen);
        double vy = vel_disp(gen);

        double m = halo_mass / double(halo_particles);
        Px += m * vx; Py += m * vy; Mtot += m;
        spawned.push_back({(float)x, (float)y, (float)vx, (float)vy, m, {0.2f, 0.0f, 0.4f}});
    }
    
    if (failed_halo > 0) {
        std::cout << "⚠️  " << failed_halo << " halo particles used fallback sampling\n";
    }

    // DEBUG: Print angle distribution
    std::cout << "\n🔍 ANGLE DISTRIBUTION ANALYSIS:\n";
    for (int i = 0; i < 36; ++i) {
        double angle_deg = i * 10.0;
        std::cout << "  " << angle_deg << "°-" << (angle_deg + 10) << "°: " << angle_bins[i] << " particles";
        if (angle_bins[i] < 100) {  // Flag suspiciously low bins
            std::cout << " ⚠️  LOW";
        }
        std::cout << "\n";
    }
    
    // Find minimum and maximum bins
    auto min_bin = *std::min_element(angle_bins.begin(), angle_bins.end());
    auto max_bin = *std::max_element(angle_bins.begin(), angle_bins.end());
    std::cout << "  Range: " << min_bin << " to " << max_bin << " particles per 10° sector\n";
    
    if (max_bin > 2 * min_bin) {
        std::cout << "⚠️  UNEVEN DISTRIBUTION DETECTED! Some angles have significantly fewer particles.\n";
    }

    // Center of mass correction
    double vx_cm = Px / Mtot;
    double vy_cm = Py / Mtot;
    
    for (auto& p : spawned) {
        float adj_vx = p.vx - (float)vx_cm;
        float adj_vy = p.vy - (float)vy_cm;
        particle_system_.add_particle(
            Vec2{p.x, p.y}, Vec2{adj_vx, adj_vy}, p.m, p.color
        );
    }

    // Physics boundaries
    physics_params_.boundary_min_x = -50.0f;
    physics_params_.boundary_max_x = 50.0f;
    physics_params_.boundary_min_y = -50.0f;
    physics_params_.boundary_max_y = 50.0f;
    physics_params_.bounce_force = 0.0f;
    physics_params_.damping = 1.0f;
    physics_params_.time_scale = 0.5f;
    
    update_physics_parameters();

    // Camera setup
    enhanced_camera_.world_center_x = 0.0;
    enhanced_camera_.world_center_y = 0.0;
    double min_screen_size = std::min(enhanced_camera_.viewport_width, enhanced_camera_.viewport_height);
    enhanced_camera_.world_zoom = min_screen_size / 60.0;

    std::cout << "✅ MW created: " << spawned.size() << " particles\n";
    
    // ISSUE 6: Also check if it's a rendering/buffer issue
    size_t actual_count = particle_system_.get_particle_count();
    std::cout << "   Particle system reports: " << actual_count << " particles\n";
    
    if (actual_count != spawned.size()) {
        std::cout << "⚠️  PARTICLE COUNT MISMATCH! Generated " << spawned.size() 
                  << " but system has " << actual_count << "\n";
    }
}
    
    void create_andromeda_like_galaxy() {
        std::cout << "\n🌌 Creating Andromeda-like (M31) galaxy...\n";
        particle_system_.clear_particles();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // M31 is more massive than Milky Way
        const float total_mass = 1.8e12f;  // Solar masses (scaled)
        const float disk_mass = 8.0e10f;   // Larger stellar disk
        const float bulge_mass = 4.0e10f;  // Larger central bulge
        const float halo_mass = total_mass - disk_mass - bulge_mass;
        
        const float disk_scale_length = 5.0f;  // Larger than MW
        const float disk_scale_height = 0.4f;  
        const float bulge_radius = 1.2f;       // Larger bulge
        const float virial_radius = 30.0f;     // Larger halo
        
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        std::normal_distribution<float> velocity_noise(0.0f, 0.8f);
        
        // Central SMBH (larger than Milky Way's)
        particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 1.4e8f / 1e9f, Vec3(1.0f, 0.9f, 0.3f));
        
        // Larger, more massive bulge
        int bulge_particles = 3000;
        std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
        
        for (int i = 0; i < bulge_particles; ++i) {
            float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
            float theta = angle_dist(gen);
            float x = r * cos(theta);
            float y = r * sin(theta);
            
            float v_circ = sqrt(80.0f / (r + 0.1f));
            float vx = -v_circ * sin(theta) + velocity_noise(gen);
            float vy = v_circ * cos(theta) + velocity_noise(gen);
            
            Vec3 color(1.0f, 0.7f, 0.3f);  // More reddish bulge
            particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                        bulge_mass / bulge_particles / 1e9f, color);
        }
        
        // Extended stellar disk
        int disk_particles = 18000;
        std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
        std::normal_distribution<float> height_dist(0.0f, disk_scale_height);
        
        for (int i = 0; i < disk_particles; ++i) {
            float r = std::min(disk_radius_dist(gen), disk_scale_length * 6.0f);
            float theta = angle_dist(gen);
            
            float x = r * cos(theta);
            float y = r * sin(theta);
            
            // M31 rotation curve (higher velocities)
            float v_circ = 26.0f * sqrt(r) / sqrt(r + 1.5f) + 20.0f;
            if (r > 10.0f) v_circ = 26.0f;
            
            float vx = -v_circ * sin(theta) + velocity_noise(gen);
            float vy = v_circ * cos(theta) + velocity_noise(gen);
            
            Vec3 color(0.9f, 0.7f, 1.0f);  // Bluer stars
            particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                        disk_mass / disk_particles / 1e9f, color);
        }
        
        // Larger dark matter halo
        int halo_particles = 10000;
        const float concentration = 10.0f;
        const float scale_radius = virial_radius / concentration;
        
        std::uniform_real_distribution<float> uniform(0, 1);
        std::uniform_real_distribution<float> halo_radius_dist(0, virial_radius);
        
        for (int i = 0; i < halo_particles; ++i) {
            float r, nfw_prob;
            do {
                r = halo_radius_dist(gen);
                float x = r / scale_radius;
                nfw_prob = 1.0f / (x * (1.0f + x) * (1.0f + x));
            } while (uniform(gen) > nfw_prob * scale_radius / r);
            
            float theta = angle_dist(gen);
            float phi = acos(2 * uniform(gen) - 1);
            
            float x = r * sin(phi) * cos(theta);
            float y = r * sin(phi) * sin(theta);
            
            std::normal_distribution<float> vel_disp(0.0f, 10.0f * sqrt(1.0f / (1.0f + r / scale_radius)));
            float vx = vel_disp(gen);
            float vy = vel_disp(gen);
            
            Vec3 color(0.4f, 0.1f, 0.6f);
            particle_system_.add_particle(Vec2(x, y), Vec2(vx, vy), 
                                        halo_mass / halo_particles / 1e9f, color);
        }
        
        particle_system_.fix_overlapping_particles_advanced();
        std::cout << "✅ Andromeda-like galaxy created with " << particle_system_.get_particle_count() << " particles\n";
    }
    
    void create_galaxy_merger_scenario() {
        std::cout << "\n💥 Creating galaxy merger scenario...\n";
        particle_system_.clear_particles();
        
        // Create two galaxies that will merge
        create_milky_way_at_position(Vec2(-15.0f, 0.0f), Vec2(2.0f, 1.0f), 1.0f, "MW-like");
        create_smaller_galaxy_at_position(Vec2(50.0f, 5.0f), Vec2(-3.0f, -0.5f), 0.6f, "Satellite");
        
        std::cout << "✅ Galaxy merger scenario created\n";
        std::cout << "   Two galaxies on collision course\n";
        std::cout << "   Watch as tidal forces create streams and eventually merge!\n";
    }
    
    void create_local_group_simulation() {
        std::cout << "\n🌌 Creating Local Group simulation...\n";
        particle_system_.clear_particles();
        
        // Milky Way
        create_milky_way_at_position(Vec2(-20.0f, 0.0f), Vec2(0.5f, 0.0f), 1.0f, "Milky Way");
        
        // Andromeda (approaching for collision in ~4.5 Gyr)
        create_andromeda_at_position(Vec2(30.0f, -10.0f), Vec2(-1.2f, 0.8f), 1.5f, "Andromeda");
        
        // Large Magellanic Cloud
        create_smaller_galaxy_at_position(Vec2(-18.0f, -3.0f), Vec2(1.0f, 0.2f), 0.1f, "LMC");
        
        // Small Magellanic Cloud  
        create_smaller_galaxy_at_position(Vec2(-19.0f, -4.0f), Vec2(0.8f, 0.1f), 0.05f, "SMC");
        
        std::cout << "✅ Local Group simulation created\n";
        std::cout << "   Milky Way, Andromeda, and satellite galaxies\n";
        std::cout << "   Realistic masses and trajectories\n";
    }

    void create_two_armed_spiral_galaxy(
    const Vec2& center,
    const Vec2& bulk_vel,
    double mass_scale = 1.0,          // scales MW-like masses below
    double pitch_deg = 16.0,          // typical grand-design pitch ~10–25°
    int disk_particles = 60000,
    int bulge_particles = 3000,
    int halo_particles  = 25000,
    double arm_scatter_rad = 0.18,    // angular scatter around the arm (radians)
    double arm_phase0 = 0.0,          // rotate arms
    bool clockwise = false            // flip rotation sense
) {
    // --- MW-like numbers in 1e9 Msun / kpc, scaled like your MW function ---
    const double scale = 0.01; // same convention you use elsewhere
    const double Rd    = (3.5 / scale) * mass_scale;  // disk scale length (kpc)
    const double a_b   = (0.6 / scale) * mass_scale;  // Hernquist bulge scale (kpc)
    const double rvir  = (200.0 / scale) * mass_scale; // halo "virial" (kpc)
    const double rmax  = (25.0  / scale) * mass_scale; // spawn cutoff (kpc)

    const double M_bh   = (0.0043 / scale) * mass_scale;   // SMBH (1e9 Msun)
    const double M_bul  = (20.0   / scale) * mass_scale;
    const double M_disk = (55.0   / scale) * mass_scale;
    const double M_halo = (1200.0 / scale) * mass_scale;
    const double c_halo = 10.0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::exponential_distribution<double> Rexp(1.0 / Rd);   // P(r) ~ e^{-r/Rd}
    std::normal_distribution<double>  ThScatter(0.0, arm_scatter_rad);
    std::normal_distribution<double>  Vnoise(0.0, 0.02);    // small vel noise

    auto add_particle = [&](double x, double y, double vx, double vy, double m, const Vec3& col){
        particle_system_.add_particle(center + Vec2{(float)x,(float)y},
                                      bulk_vel + Vec2{(float)vx,(float)vy},
                                      (float)m, col);
    };

    // --- Central SMBH ---
    add_particle(0,0, 0,0, M_bh, Vec3(1.0f,1.0f,0.8f));

    // --- Bulge (Hernquist) ---
    for (int i=0;i<bulge_particles;i++){
        // Inverse CDF for Hernquist: r = a * sqrt(u) / (1 - sqrt(u))
        double u = std::max(1e-12, U01(rng));
        double s = std::sqrt(u);
        double r = std::min(std::max(a_b * s/(1.0 - s), 2.0*DEFAULT_SOFTENING_KPC), rmax);
        double th = 2*M_PI*U01(rng);
        double x = r * std::cos(th), y = r * std::sin(th);

        // Circular speed from total enclosed mass
        double Menc = M_bh
                    + hernquist_Menc(r, M_bul, a_b)
                    + disk_exp_Menc(r, M_disk, Rd)
                    + nfw_Menc(r, M_halo, rvir, c_halo);
        double v = vcirc_code(r, Menc);
        if (clockwise) v = -v;

        add_particle(x,y, -v*std::sin(th)+Vnoise(rng),  v*std::cos(th)+Vnoise(rng),
                     M_bul/double(bulge_particles), Vec3(1.0f,0.8f,0.4f));
    }

    // --- Disk with m=2 logarithmic spiral overdensities ---
    // Log-spiral: r = r0 * exp(b*theta), with b = tan(pitch)
    double pitch = pitch_deg * M_PI/180.0;
    double b = std::tan(pitch);
    double r0 = Rd; // reference radius sets arm phase origin

    for (int i=0;i<disk_particles;i++){
        // Sample radius from exponential disk, clamp
        double r = std::min(std::max(Rexp(rng), 2.0*DEFAULT_SOFTENING_KPC), 4.0*Rd);

        // Choose which arm (0 or 1), arms separated by pi
        int arm = (U01(rng) < 0.5) ? 0 : 1;

        // Ideal arm angle that yields this radius on a log-spiral
        // theta_arm = (ln(r/r0))/b + phase + arm*pi
        double th_arm = (std::log(r / r0))/b + arm_phase0 + arm*M_PI;

        // Scatter around the arm to make a *twin-armed* grand design
        double th = th_arm + ThScatter(rng);
        if (clockwise) th = -th;

        double x = r * std::cos(th), y = r * std::sin(th);

        // Circular speed from enclosed mass (BH + bulge + disk + halo)
        double Menc = M_bh
                    + hernquist_Menc(r, M_bul, a_b)
                    + disk_exp_Menc(r, M_disk, Rd)
                    + nfw_Menc(r, M_halo, rvir, c_halo);
        double v = vcirc_code(r, Menc);
        if (clockwise) v = -v;

        double vx = -v*std::sin(th) + Vnoise(rng);
        double vy =  v*std::cos(th) + Vnoise(rng);

        add_particle(x,y, vx,vy, M_disk/double(disk_particles), Vec3(0.8f,0.65f,1.0f));
    }

    // --- Simple quasi-isotropic halo sampling (projected to 2D sim plane) ---
    // (You can replace with your 3D->2D projection variant if desired.)
    for (int i=0;i<halo_particles;i++){
        // Sample radius with a crude rejection to roughly follow NFW M(r)
        double r, accepted = false;
        const double rs = rvir / c_halo;
        for (int tries=0; tries<40 && !accepted; ++tries){
            r = U01(rng) * rmax;
            double x = r/rs;
            double f = std::log(1.0+x) - x/(1.0+x);
            double fmax = std::log(1.0 + rmax/rs);
            accepted = (U01(rng) <= std::max(0.0, f)/std::max(1e-6, fmax));
        }
        r = std::max(r, 2.0*DEFAULT_SOFTENING_KPC);
        double th = 2*M_PI*U01(rng);
        if (clockwise) th = -th;

        // simple, small velocity dispersion that falls with r
        double sigma_code = (100.0 / VELOCITY_UNIT_KMS) / std::sqrt(1.0 + r/rs);
        std::normal_distribution<double> Vd(0.0, sigma_code);

        add_particle(r*std::cos(th), r*std::sin(th), Vd(rng), Vd(rng),
                     M_halo/double(halo_particles), Vec3(0.25f,0.12f,0.5f));
    }
}

// === Convenience: spawn a *pair* of twin-armed spirals on a near-circular orbit ===
void create_twin_spiral_pair() {
    std::cout << "\n🌌 Spawning twin-armed spiral *pair*\n";
    particle_system_.clear_particles();

    // Separation & masses consistent with the builder above
    const double scale = 0.01;
    const double D = 80.0 / scale;              // kpc separation
    const double Mgal = (0.0043 + 20.0 + 55.0 + 1200.0) / scale; // 1e9 Msun total (rough)
    const double G = G_GALACTIC;                // keep your code's G
    const double v_rel_kms = std::sqrt(G * (2.0*Mgal) / D); // two-body circular
    const double v_code    = v_rel_kms / VELOCITY_UNIT_KMS;

    // Place galaxies at +/- D/2 along x; give opposite y-velocities
    Vec2 c1(-float(0.5*D), 0.0f), c2(float(0.5*D), 0.0f);
    Vec2 v1(0.0f,  float( 0.5*v_code));
    Vec2 v2(0.0f,  float(-0.5*v_code));

    // Build each galaxy; flip handedness for visual variety
    create_two_armed_spiral_galaxy(c1, v1, 1.0, 16.0, 60000, 3000, 25000, 0.18, 0.0, false);
    create_two_armed_spiral_galaxy(c2, v2, 1.0, 16.0, 60000, 3000, 25000, 0.18, M_PI*0.35, true);

    // Housekeeping that keeps BH happy in dense starts
    particle_system_.fix_overlapping_particles_advanced();
    particle_system_.optimize_particle_layout();

    // Wider bounds, no wall bounces
    physics_params_.boundary_min_x = -150.0f;
    physics_params_.boundary_max_x =  150.0f;
    physics_params_.boundary_min_y = -150.0f;
    physics_params_.boundary_max_y =  150.0f;
    physics_params_.bounce_force   = 0.0f;
    update_physics_parameters();

    std::cout << "✅ Twin spirals created: " << particle_system_.get_particle_count() << " particles\n";
}

    
    void create_dwarf_galaxy_accretion() {
        std::cout << "\n🔄 Creating dwarf galaxy accretion event...\n";
        particle_system_.clear_particles();
        
        // Main spiral galaxy
        create_milky_way_at_position(Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f), 1.0f, "Host Galaxy");
        
        // Multiple dwarf galaxies being accreted
        std::vector<Vec2> positions = {
            Vec2(12.0f, 8.0f), Vec2(-15.0f, -6.0f), Vec2(18.0f, -12.0f),
            Vec2(-10.0f, 14.0f), Vec2(22.0f, 3.0f)
        };
        std::vector<Vec2> velocities = {
            Vec2(-1.5f, -1.0f), Vec2(1.8f, 0.5f), Vec2(-2.0f, 1.2f),
            Vec2(0.8f, -1.8f), Vec2(-1.8f, -0.3f)
        };
        
        for (size_t i = 0; i < positions.size(); ++i) {
            create_smaller_galaxy_at_position(positions[i], velocities[i], 0.08f, "Dwarf");
        }
        
        std::cout << "✅ Dwarf galaxy accretion scenario created\n";
        std::cout << "   Multiple small galaxies falling into main halo\n";
    }
    
    // Helper methods for creating galaxies at specific positions
    void create_milky_way_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        std::normal_distribution<float> velocity_noise(0.0f, 0.3f);
        
        const float disk_scale_length = 3.5f * mass_scale;
        const float bulge_radius = 0.5f * mass_scale;
        
        // Central SMBH
        particle_system_.add_particle(center, velocity, 4.3e6f / 1e9f * mass_scale, Vec3(1.0f, 1.0f, 0.5f));
        
        // Bulge
        int bulge_particles = static_cast<int>(1000 * mass_scale);
        std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
        
        for (int i = 0; i < bulge_particles; ++i) {
            float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
            float theta = angle_dist(gen);
            Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
            
            float v_circ = sqrt(50.0f * mass_scale / (r + 0.1f));
            Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                     v_circ * cos(theta) + velocity_noise(gen));
            
            particle_system_.add_particle(pos, vel, 2.0e10f / bulge_particles / 1e9f * mass_scale, 
                                        Vec3(1.0f, 0.8f, 0.4f));
        }
        
        // Disk
        int disk_particles = static_cast<int>(8000 * mass_scale);
        std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
        
        for (int i = 0; i < disk_particles; ++i) {
            float r = std::min(disk_radius_dist(gen), disk_scale_length * 5.0f);
            float theta = angle_dist(gen);
            Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
            
            float v_circ = 22.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 1.0f) + 15.0f;
            if (r > 8.0f) v_circ = 22.0f;
            
            Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                     v_circ * cos(theta) + velocity_noise(gen));
            
            particle_system_.add_particle(pos, vel, 5.5e10f / disk_particles / 1e9f * mass_scale,
                                        Vec3(0.8f, 0.6f, 1.0f));
        }
        
        std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
    }
    
    void create_andromeda_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        std::normal_distribution<float> velocity_noise(0.0f, 0.4f);
        
        const float disk_scale_length = 5.0f * mass_scale;
        const float bulge_radius = 1.2f * mass_scale;
        
        // Larger SMBH
        particle_system_.add_particle(center, velocity, 1.4e8f / 1e9f * mass_scale, Vec3(1.0f, 0.9f, 0.3f));
        
        // Larger bulge
        int bulge_particles = static_cast<int>(1500 * mass_scale);
        std::gamma_distribution<float> bulge_radius_dist(2.0f, bulge_radius / 2.0f);
        
        for (int i = 0; i < bulge_particles; ++i) {
            float r = std::min(bulge_radius_dist(gen), bulge_radius * 3.0f);
            float theta = angle_dist(gen);
            Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
            
            float v_circ = sqrt(80.0f * mass_scale / (r + 0.1f));
            Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                     v_circ * cos(theta) + velocity_noise(gen));
            
            particle_system_.add_particle(pos, vel, 4.0e10f / bulge_particles / 1e9f * mass_scale,
                                        Vec3(1.0f, 0.7f, 0.3f));
        }
        
        // Extended disk
        int disk_particles = static_cast<int>(12000 * mass_scale);
        std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
        
        for (int i = 0; i < disk_particles; ++i) {
            float r = std::min(disk_radius_dist(gen), disk_scale_length * 6.0f);
            float theta = angle_dist(gen);
            Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
            
            float v_circ = 26.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 1.5f) + 20.0f;
            if (r > 10.0f) v_circ = 26.0f;
            
            Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                     v_circ * cos(theta) + velocity_noise(gen));
            
            particle_system_.add_particle(pos, vel, 8.0e10f / disk_particles / 1e9f * mass_scale,
                                        Vec3(0.9f, 0.7f, 1.0f));
        }
        
        std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
    }
    
    void create_smaller_galaxy_at_position(Vec2 center, Vec2 velocity, float mass_scale, const std::string& name) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(0, 2 * M_PI);
        std::normal_distribution<float> velocity_noise(0.0f, 0.2f);
        
        const float disk_scale_length = 2.0f * mass_scale;
        
        // Small central mass
        particle_system_.add_particle(center, velocity, 1.0e6f / 1e9f * mass_scale, Vec3(0.8f, 0.8f, 1.0f));
        
        // Compact disk
        int disk_particles = static_cast<int>(1500 * mass_scale);
        std::exponential_distribution<float> disk_radius_dist(1.0f / disk_scale_length);
        
        for (int i = 0; i < disk_particles; ++i) {
            float r = std::min(disk_radius_dist(gen), disk_scale_length * 4.0f);
            float theta = angle_dist(gen);
            Vec2 pos = center + Vec2(r * cos(theta), r * sin(theta));
            
            float v_circ = 15.0f * sqrt(r * mass_scale) / sqrt(r * mass_scale + 0.5f) + 8.0f;
            
            Vec2 vel = velocity + Vec2(-v_circ * sin(theta) + velocity_noise(gen),
                                     v_circ * cos(theta) + velocity_noise(gen));
            
            Vec3 color = (name == "LMC") ? Vec3(0.6f, 0.8f, 1.0f) :
                        (name == "SMC") ? Vec3(0.8f, 0.6f, 1.0f) :
                                         Vec3(0.7f, 0.9f, 0.8f);
            
            particle_system_.add_particle(pos, vel, 1.0e10f / disk_particles / 1e9f * mass_scale, color);
        }
        
        std::cout << "  Created " << name << " at (" << center.x << ", " << center.y << ")\n";
    }
    
    void subscribe_to_events() {
        event_bus_.subscribe<PhysicsUpdateEvent>(Events::PHYSICS_UPDATE,
            [this](const PhysicsUpdateEvent& event) {
                performance_stats_.physics_iterations = event.iteration_count;
                performance_stats_.particle_count = event.particle_count;
            });
    }

    // Parabolic flyby: massive primary (A) + lighter satellite (B)
// A & B start far apart, B swings to pericentre rp and gets tidally stripped.
void create_tidally_disrupting_flyby() {
    std::cout << "\n🌀 Creating *tidal flyby* (A tears B apart)\n";
    particle_system_.clear_particles();

    // --- Choose galaxy models already in your code ---
    // Primary: Andromeda-like (heavier), Satellite: compact dwarf
    const float s1 = 1.5f;   // primary mass scale (Andromeda helper)
    const float s2 = 0.30f;  // satellite mass scale (dwarf helper)

    // These masses match what the helpers actually spawn (in 1e9 Msun units):
    // Andromeda helper ~ (BH 0.14 + bulge 40 + disk 80) * s1
    const double M1 = (0.14 + 40.0 + 80.0) * s1;  // -> ~180 for s1=1.5
    // Dwarf helper ~ (BH 0.001 + disk 10) * s2
    const double M2 = (0.001 + 10.0) * s2;        // -> ~3.0 for s2=0.30
    const double Mtot = M1 + M2;

    // --- Orbit design (parabolic, e=1) in physical units ---
    // Distances in kpc, velocities in km/s then converted to your code units.
    const double r0 = 120.0;  // initial separation (kpc)
    const double rp = 18.0;   // desired pericentre (kpc) — close enough to shred
    const double mu = G_GALACTIC * Mtot; // km^2/s^2·kpc per (1e9 Msun); you already use this in vcirc_code :contentReference[oaicite:2]{index=2}

    // Parabolic relations:
    // specific angular momentum h = sqrt(2 mu rp)
    // speed at radius r0: v = sqrt(2 mu / r0)
    // tangential component at r0: vt = h / r0
    // radial component (inward):   vr = -sqrt(max(0, v^2 - vt^2))
    const double h  = std::sqrt(2.0 * mu * rp);
    const double v  = std::sqrt(2.0 * mu / r0);
    const double vt = h / r0;
    const double vr = -std::sqrt(std::max(0.0, v*v - vt*vt));

    // Convert to your simulation velocity units (divide by VELOCITY_UNIT_KMS) :contentReference[oaicite:3]{index=3}
    const double vt_code = vt / VELOCITY_UNIT_KMS;
    const double vr_code = vr / VELOCITY_UNIT_KMS;

    // Place galaxies along x; give relative velocity (vr along -x, vt along +y)
    const float xA = (float)(-0.5 * r0), yA = 0.0f;
    const float xB = (float)( 0.5 * r0), yB = 0.0f;

    // Split relative velocity by masses so COM stays at rest
    const double f1 =  M2 / Mtot;
    const double f2 =  M1 / Mtot;
    const Vec2 vrel((float)vr_code, (float)vt_code);

    Vec2 vA(-f1 * vrel.x, -f1 * vrel.y);  // primary moves opposite to B
    Vec2 vB( f2 * vrel.x,  f2 * vrel.y);

    // Build galaxies using your helpers (adds internal rotation etc.)
    create_andromeda_at_position(Vec2(xA, yA), vA, s1, "Primary A");   // :contentReference[oaicite:4]{index=4}
    create_smaller_galaxy_at_position(Vec2(xB, yB), vB, s2, "Victim B"); // :contentReference[oaicite:5]{index=5}

    // Make sure nothing bounces off boundaries, and give plenty of room
    physics_params_.boundary_min_x = -200.0f;
    physics_params_.boundary_max_x =  200.0f;
    physics_params_.boundary_min_y = -200.0f;
    physics_params_.boundary_max_y =  200.0f;
    physics_params_.bounce_force   = 0.0f;
    physics_params_.time_scale     = 0.7f;  // a touch slower = clearer flyby
    update_physics_parameters();

    // Help the tree at t=0
    particle_system_.fix_overlapping_particles_advanced();
    particle_system_.optimize_particle_layout(); // Morton ordering :contentReference[oaicite:6]{index=6}

    std::cout << "✅ Tidal flyby ready: r0=" << r0 << " kpc, rp=" << rp << " kpc\n";
}

    
    bool initialize_metal() {
        // Get default Metal device
        metal_device_ = MTLCreateSystemDefaultDevice();
        if (!metal_device_) {
            std::cerr << "Metal is not supported on this device\n";
            return false;
        }
        
        // Create command queue
        command_queue_ = [metal_device_ newCommandQueue];
        if (!command_queue_) {
            std::cerr << "Failed to create Metal command queue\n";
            return false;
        }
        
        // Create Metal layer for the window
        NSWindow* nsWindow = glfwGetCocoaWindow(window_);
        metal_layer_ = [CAMetalLayer layer];
        metal_layer_.device = metal_device_;
        metal_layer_.pixelFormat = MTLPixelFormatBGRA8Unorm;
        metal_layer_.framebufferOnly = YES;
        
        nsWindow.contentView.layer = metal_layer_;
        nsWindow.contentView.wantsLayer = YES;
        
        std::cout << "Metal initialized successfully\n";
        
        return true;
    }
    
    bool initialize_imgui() {
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        
        // Setup style
        ImGui::StyleColorsDark();
        
        // Setup Platform/Renderer backends
        if (!ImGui_ImplGlfw_InitForOther(window_, true)) {
            std::cerr << "Failed to initialize ImGui GLFW backend\n";
            return false;
        }
        
        // Initialize ImGui Metal backend with proper device
        if (!ImGui_ImplMetal_Init(metal_device_)) {
            std::cerr << "Failed to initialize ImGui Metal backend\n";
            return false;
        }
        
        return true;
    }
    
    void setup_initial_particles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(-15.0f, 15.0f);
        std::uniform_real_distribution<float> vel_dist(-5.0f, 5.0f);
        std::uniform_real_distribution<float> color_dist(0.3f, 1.0f);
        std::uniform_real_distribution<float> mass_dist(0.8f, 1.2f);
        
        // Clear existing particles
        particle_system_.clear_particles();
        
        // Start with fewer particles for better initial performance
        // User can add more through the UI
        for (int i = 0; i < 1000; ++i) {
            Vec2 pos(pos_dist(gen), pos_dist(gen));
            Vec2 vel(vel_dist(gen), vel_dist(gen));
            Vec3 color(color_dist(gen), color_dist(gen), color_dist(gen));
            float mass = mass_dist(gen);
            
            particle_system_.add_particle(pos, vel, mass, color);
        }
        
        std::cout << "Added " << particle_system_.get_particle_count() << " initial particles\n";
    }
    
    void update_physics_parameters() {
        particle_system_.set_gravity(Vec2(physics_params_.gravity_x, physics_params_.gravity_y));
        particle_system_.set_damping(physics_params_.damping);
        particle_system_.set_bounce_force(physics_params_.bounce_force);
        particle_system_.set_boundary(
            physics_params_.boundary_min_x, physics_params_.boundary_max_x,
            physics_params_.boundary_min_y, physics_params_.boundary_max_y
        );
    }
    
    void render_frame() {
    @autoreleasepool {
        // Get window size
        int width, height;
        glfwGetFramebufferSize(window_, &width, &height);
        enhanced_camera_.viewport_width = width;
        enhanced_camera_.viewport_height = height;
        
        // Update Metal layer size
        metal_layer_.drawableSize = CGSizeMake(width, height);
        
        // Get next drawable
        id<CAMetalDrawable> drawable = [metal_layer_ nextDrawable];
        if (!drawable) {
            return;
        }
        
        // Create Metal command buffer
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        commandBuffer.label = @"BarnesHutRenderCommandBuffer";
        
        // Use the convenient constructor for render pass descriptor
        MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(
            render_params_.background_r,
            render_params_.background_g,
            render_params_.background_b,
            render_params_.background_a
        );
        
        // Create render command encoder
        id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        renderEncoder.label = @"BarnesHutRenderEncoder";
        MetalRenderer::CameraParams render_camera = enhanced_camera_.to_render_params();

        if (show_quadtree_visualization_) {
            auto quadtree_boxes = particle_system_.get_quadtree_boxes();
            
            renderer_.render_quadtree_lines(render_camera, (__bridge void*)renderEncoder, quadtree_boxes);
        }

        // Start ImGui frame
        ImGui_ImplMetal_NewFrame(renderPassDescriptor);
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Render UI
        render_ui();
        
        // Prepare ImGui render data
        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        
        // Render particles using our MetalRenderer
        renderer_.render(render_camera, (__bridge void*)renderEncoder);
        
        // Render ImGui
        ImGui_ImplMetal_RenderDrawData(draw_data, commandBuffer, renderEncoder);
        
        // End encoding and present
        [renderEncoder endEncoding];
        [commandBuffer presentDrawable:drawable];
        [commandBuffer commit];
    }
}
    
    void render_ui() {
        // Main menu bar
        if (ImGui::BeginMainMenuBar()) {
            if (ImGui::BeginMenu("File")) {
                if (ImGui::MenuItem("Reset Simulation")) {
                    setup_initial_particles();
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
                ImGui::MenuItem("Realistic Galaxies", nullptr, &show_realistic_presets_);  // NEW
                ImGui::MenuItem("Demo Window", nullptr, &show_demo_window_);
                ImGui::EndMenu();
            }
            
            if (ImGui::BeginMenu("Optimizations")) {
                if (ImGui::MenuItem("Run All Optimizations")) {
                    run_particle_optimizations();
                }
                if (ImGui::MenuItem("Fix Particle Overlaps")) {
                    quick_fix_overlaps();
                }
                if (ImGui::MenuItem("Optimize Spatial Layout")) {
                    optimize_spatial_layout();
                }
                if (ImGui::MenuItem("Find Optimal Theta")) {
                    find_optimal_theta();
                }
                if (ImGui::MenuItem("Compact Tree Cache")) {
                    compact_tree_cache();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Diagnose Performance")) {
                    diagnose_performance_issues();
                }
                ImGui::EndMenu();
            }
            
            // NEW: Realistic galaxy menu
            if (ImGui::BeginMenu("Realistic Galaxies")) {
                if (ImGui::MenuItem("🌌 Milky Way Galaxy")) {
                    create_realistic_milky_way();
                }
                if (ImGui::MenuItem("🌌 Andromeda Galaxy (M31)")) {
                    create_andromeda_like_galaxy();
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Galaxy Merger")) {
                    create_tidally_disrupting_flyby();
                }
                if (ImGui::MenuItem("🌌 Local Group")) {
                    create_local_group_simulation();
                }

                if (ImGui::MenuItem("🌌 Twin Galaxy")) {
                    create_twin_spiral_pair();
                }
                if (ImGui::MenuItem("🔄 Dwarf Accretion")) {
                    create_dwarf_galaxy_accretion();
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
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f), "Force: %.1fms ✓", bh_stats.force_calculation_time_ms);
            }
            
            ImGui::EndMainMenuBar();
        }
        
        // Performance window
        if (show_performance_window_) {
            render_performance_window();
        }
        
        // Particle controls
        if (show_particle_controls_) {
            render_particle_controls();
        }
        
        // Barnes-Hut specific controls
        if (show_barnes_hut_controls_) {
            render_barnes_hut_controls();
        }
        
        // Optimization controls window
        if (show_optimization_controls_) {
            render_optimization_controls();
        }
        
        // NEW: Realistic galaxy presets window
        if (show_realistic_presets_) {
            render_realistic_presets_window();
        }
        
        // Demo window
        if (show_demo_window_) {
            ImGui::ShowDemoWindow(&show_demo_window_);
        }
    }
    
    // NEW: Realistic galaxy presets window
    void render_realistic_presets_window() {
        if (!ImGui::Begin("Realistic Galaxy Simulations", &show_realistic_presets_)) {
            ImGui::End();
            return;
        }
        
        ImGui::Text("🌌 Scientifically Accurate Galaxy Simulations");
        ImGui::Text("Based on observational data and astrophysical models");
        
        ImGui::Separator();
        ImGui::Text("Individual Galaxies:");
        
        if (ImGui::Button("🌌 Milky Way Galaxy", ImVec2(-1, 0))) {
            create_realistic_milky_way();
        }
        ImGui::Text("Our galaxy with realistic mass distribution:\n"
                   "• Central SMBH (Sgr A*): 4.3M solar masses\n"
                   "• Stellar disk: ~55B solar masses, exponential profile\n"
                   "• Central bulge: ~20B solar masses, Hernquist profile\n"
                   "• Dark matter halo: NFW profile, concentration ~12\n"
                   "• Spiral structure with logarithmic arms");
        
        ImGui::Separator();
        
        if (ImGui::Button("🌌 Andromeda Galaxy (M31)", ImVec2(-1, 0))) {
            create_andromeda_like_galaxy();
        }
        ImGui::Text("Larger spiral galaxy, closest major neighbor:\n"
                   "• More massive than Milky Way (~1.8x)\n"
                   "• Larger SMBH: ~140M solar masses\n"
                   "• Extended stellar disk and prominent bulge\n"
                   "• Higher rotation velocities (~260 km/s)");
        
        ImGui::Separator();
        ImGui::Text("Galaxy Interactions & Evolution:");
        
        if (ImGui::Button("💥 Galaxy Merger Event", ImVec2(-1, 0))) {
            create_galaxy_merger_scenario();
        }
        ImGui::Text("Two galaxies on collision course:\n"
                   "• Realistic orbital parameters from simulations\n"
                   "• Tidal stripping and stellar streams\n"
                   "• Star formation bursts during close passes\n"
                   "• Final coalescence into elliptical remnant");
        
        if (ImGui::Button("🌌 Local Group Simulation", ImVec2(-1, 0))) {
            create_local_group_simulation();
        }
        ImGui::Text("Our local galactic neighborhood:\n"
                   "• Milky Way and Andromeda (approaching collision)\n"
                   "• Large & Small Magellanic Clouds\n"
                   "• Realistic masses and orbital motions\n"
                   "• Future Milkomeda merger preview");
        
        if (ImGui::Button("🔄 Dwarf Galaxy Accretion", ImVec2(-1, 0))) {
            create_dwarf_galaxy_accretion();
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
    
    void render_performance_window() {
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
            ImGui::Text("   → Try optimizations to fix overlapping particles");
        }
        
        if (bh_stats.efficiency_ratio < 0.5f) {
            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "⚠️ LOW EFFICIENCY: %.1f%%", bh_stats.efficiency_ratio * 100.0f);
            ImGui::Text("   → Try increasing theta value");
        }
        
        if (bh_stats.force_calculation_time_ms > 15.0f && particle_system_.get_particle_count() < 10000) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "⚠️ SLOW FORCE CALC: %.1fms", bh_stats.force_calculation_time_ms);
            ImGui::Text("   → Run comprehensive optimizations (Press O)");
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
        
        ImGui::Separator();
        
        // Camera controls
        ImGui::Text("Camera");
        ImGui::SliderFloat("Zoom", &camera_.zoom, 0.05f, 10.0f);
        ImGui::SliderFloat("Center X", &camera_.center_x, -100.0f, 100.0f);
        ImGui::SliderFloat("Center Y", &camera_.center_y, -100.0f, 100.0f);
        
        if (ImGui::Button("Reset Camera")) {
            camera_.zoom = 0.5f;
            camera_.center_x = 0.0f;
            camera_.center_y = 0.0f;
        }
        
        ImGui::End();
    }
    
    void render_particle_controls() {
        if (!ImGui::Begin("Particle Controls", &show_particle_controls_)) {
            ImGui::End();
            return;
        }
        
        bool params_changed = false;
        
        // Physics parameters
        ImGui::Text("Physics");
        params_changed |= ImGui::SliderFloat("Time Scale", &physics_params_.time_scale, 0.0f, 5.0f);
        params_changed |= ImGui::SliderFloat("Gravity X", &physics_params_.gravity_x, -20.0f, 20.0f);
        params_changed |= ImGui::SliderFloat("Gravity Y", &physics_params_.gravity_y, -20.0f, 20.0f);
        params_changed |= ImGui::SliderFloat("Damping", &physics_params_.damping, 0.9f, 1.0f);
        params_changed |= ImGui::SliderFloat("Bounce Force", &physics_params_.bounce_force, 100.0f, 5000.0f);
        
        ImGui::Separator();
        ImGui::Text("Boundaries");
        params_changed |= ImGui::SliderFloat("Min X", &physics_params_.boundary_min_x, -100.0f, 0.0f);
        params_changed |= ImGui::SliderFloat("Max X", &physics_params_.boundary_max_x, 0.0f, 100.0f);
        params_changed |= ImGui::SliderFloat("Min Y", &physics_params_.boundary_min_y, -100.0f, 0.0f);
        params_changed |= ImGui::SliderFloat("Max Y", &physics_params_.boundary_max_y, 0.0f, 100.0f);
        
        if (params_changed) {
            update_physics_parameters();
        }
        
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
            add_random_particles(particles_to_add);
        }
        
        ImGui::SameLine();
        if (ImGui::Button("Clear All")) {
            particle_system_.clear_particles();
        }
        
        // Original preset configurations
        ImGui::Separator();
        ImGui::Text("Classic Presets");
        if (ImGui::Button("Galaxy Spiral")) {
            create_galaxy_spiral();
        }
        ImGui::SameLine();
        if (ImGui::Button("Solar System")) {
            create_solar_system();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cluster")) {
            create_cluster();
        }
        
        ImGui::End();
    }
    
    void render_barnes_hut_controls() {
        if (!ImGui::Begin("Barnes-Hut Settings", &show_barnes_hut_controls_)) {
            ImGui::End();
            return;
        }
        
        bool config_changed = false;
        
        ImGui::Text("Algorithm Parameters");
        config_changed |= ImGui::SliderFloat("Theta", &barnes_hut_config_.theta, 0.1f, 2.0f);
        ImGui::Text("Lower = more accurate, higher = faster");
        
        config_changed |= ImGui::Checkbox("Enable Tree Caching", &barnes_hut_config_.enable_tree_caching);
        ImGui::Text("Cache tree between frames when particles don't move much");
        
        if (barnes_hut_config_.enable_tree_caching) {
            config_changed |= ImGui::SliderFloat("Rebuild Threshold", &barnes_hut_config_.tree_rebuild_threshold, 0.05f, 0.5f);
            ImGui::Text("Rebuild tree when >%.0f%% of particles move significantly", 
                       barnes_hut_config_.tree_rebuild_threshold * 100.0f);
        }
        
        config_changed |= ImGui::SliderInt("Max Particles per Leaf", 
                                          reinterpret_cast<int*>(&barnes_hut_config_.max_particles_per_leaf), 1, 10);
        
        config_changed |= ImGui::SliderInt("Tree Depth Limit", 
                                          reinterpret_cast<int*>(&barnes_hut_config_.tree_depth_limit), 10, 30);
        
#ifdef _OPENMP
        config_changed |= ImGui::Checkbox("Enable Threading", &barnes_hut_config_.enable_threading);
        ImGui::Text("Use OpenMP for parallel force calculations");
#else
        ImGui::Text("OpenMP not available - single threaded only");
#endif
        
        if (config_changed) {
            barnes_hut_config_.theta_squared = barnes_hut_config_.theta * barnes_hut_config_.theta;
            particle_system_.set_config(barnes_hut_config_);
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
    
    void render_optimization_controls() {
        if (!ImGui::Begin("Optimization Controls", &show_optimization_controls_)) {
            ImGui::End();
            return;
        }
        
        ImGui::Text("Performance Optimization Tools");
        ImGui::Text("Fix slowdowns and improve Barnes-Hut efficiency");
        
        ImGui::Separator();
        
        // Quick optimization buttons
        if (ImGui::Button("🚀 Run All Optimizations", ImVec2(-1, 0))) {
            run_particle_optimizations();
        }
        ImGui::Text("Comprehensive optimization: fixes overlaps, sorts particles,\nfinds optimal theta, compacts tree for cache efficiency");
        
        ImGui::Separator();
        ImGui::Text("Individual Optimizations:");
        
        if (ImGui::Button("🔧 Fix Particle Overlaps", ImVec2(-1, 0))) {
            quick_fix_overlaps();
        }
        ImGui::Text("Remove overlapping particles that cause deep tree recursion");
        
        if (ImGui::Button("📊 Optimize Spatial Layout", ImVec2(-1, 0))) {
            optimize_spatial_layout();
        }
        ImGui::Text("Reorder particles using Morton Z-order for cache locality");
        
        if (ImGui::Button("🎯 Find Optimal Theta", ImVec2(-1, 0))) {
            find_optimal_theta();
        }
        ImGui::Text("Test different theta values and pick the fastest");
        
        if (ImGui::Button("📁✨ Compact Tree Cache", ImVec2(-1, 0))) {
            compact_tree_cache();
        }
        ImGui::Text("Reorganize tree nodes for better memory access patterns");
        
        ImGui::Separator();
        ImGui::Text("Diagnostic Tools:");
        
        if (ImGui::Button("🔍 Diagnose Performance", ImVec2(-1, 0))) {
            diagnose_performance_issues();
        }
        ImGui::Text("Analyze performance bottlenecks and get recommendations");
        
        if (ImGui::Button("🧪 Test Theta Values", ImVec2(-1, 0))) {
            particle_system_.test_theta_performance();
        }
        ImGui::Text("Benchmark different theta parameters");
        
        ImGui::Separator();
        
        // Performance indicator
        const auto& stats = particle_system_.get_performance_stats();
        ImGui::Text("Current Performance:");
        
        if (stats.force_calculation_time_ms > 15.0f) {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f), "🔴 Force Calc: %.1fms (SLOW)", stats.force_calculation_time_ms);
            ImGui::Text("   → Run optimizations to improve performance");
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
    
    void add_random_particles(int count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(
            physics_params_.boundary_min_x * 0.8f, 
            physics_params_.boundary_max_x * 0.8f
        );
        std::uniform_real_distribution<float> vel_dist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> color_dist(0.2f, 1.0f);
        std::uniform_real_distribution<float> mass_dist(0.8f, 1.2f);
        
        for (int i = 0; i < count; ++i) {
            Vec2 pos(pos_dist(gen), pos_dist(gen));
            Vec2 vel(vel_dist(gen), vel_dist(gen));
            Vec3 color(color_dist(gen), color_dist(gen), color_dist(gen));
            float mass = mass_dist(gen);
            
            if (!particle_system_.add_particle(pos, vel, mass, color)) {
                std::cout << "Reached maximum particle limit (" << particle_system_.get_max_particles() << ")\n";
                break;
            }
        }
    }
    
    void create_galaxy_spiral() {
        particle_system_.clear_particles();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> noise(0.0f, 0.5f);
        
        for (int i = 0; i < 5000; ++i) {
            float angle = (i / 5000.0f) * 8 * M_PI;
            float radius = (i / 5000.0f) * 15.0f + 2.0f;
            
            float x = radius * cos(angle) + noise(gen);
            float y = radius * sin(angle) + noise(gen);
            
            // Orbital velocity
            float vel_magnitude = sqrt(10.0f / radius);  // Simplified orbital mechanics
            float vel_x = -vel_magnitude * sin(angle) + noise(gen) * 0.1f;
            float vel_y = vel_magnitude * cos(angle) + noise(gen) * 0.1f;
            
            Vec2 pos(x, y);
            Vec2 vel(vel_x, vel_y);
            Vec3 color(0.8f + noise(gen) * 0.2f, 0.6f + noise(gen) * 0.3f, 1.0f);
            
            particle_system_.add_particle(pos, vel, 1.0f, color);
        }
        
        // Add central massive object
        particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 100.0f, Vec3(1.0f, 1.0f, 0.5f));
        
        // Automatically optimize after creating galaxy (helps with clustered particles)
        std::cout << "Auto-optimizing galaxy simulation...\n";
        particle_system_.fix_overlapping_particles_advanced();
        particle_system_.optimize_particle_layout();
    }
    
    void create_solar_system() {
        particle_system_.clear_particles();
        
        // Central star
        particle_system_.add_particle(Vec2(0, 0), Vec2(0, 0), 50.0f, Vec3(1.0f, 1.0f, 0.3f));
        
        // Planets
        std::vector<float> planet_distances = {3, 5, 8, 12, 16};
        std::vector<Vec3> planet_colors = {
            Vec3(0.8f, 0.4f, 0.2f),  // Mercury-like
            Vec3(0.9f, 0.7f, 0.3f),  // Venus-like
            Vec3(0.3f, 0.5f, 1.0f),  // Earth-like
            Vec3(0.8f, 0.2f, 0.1f),  // Mars-like
            Vec3(0.9f, 0.8f, 0.6f)   // Jupiter-like
        };
        
        for (size_t i = 0; i < planet_distances.size(); ++i) {
            float distance = planet_distances[i];
            float velocity = sqrt(20.0f / distance);  // Orbital velocity
            
            particle_system_.add_particle(
                Vec2(distance, 0), 
                Vec2(0, velocity), 
                2.0f, 
                planet_colors[i]
            );
        }
    }
    
    void create_cluster() {
        particle_system_.clear_particles();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> pos_dist(0.0f, 5.0f);
        std::normal_distribution<float> vel_dist(0.0f, 1.0f);
        std::uniform_real_distribution<float> color_dist(0.5f, 1.0f);
        
        for (int i = 0; i < 3000; ++i) {
            Vec2 pos(pos_dist(gen), pos_dist(gen));
            Vec2 vel(vel_dist(gen), vel_dist(gen));
            Vec3 color(color_dist(gen), color_dist(gen), color_dist(gen));
            
            particle_system_.add_particle(pos, vel, 1.0f, color);
        }
        
        // Automatically fix overlaps for cluster (very important for dense clusters)
        std::cout << "Auto-fixing cluster overlaps...\n";
        particle_system_.fix_overlapping_particles_advanced();
    }
    
    void update_performance_stats(float frame_time) {
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
    
    void cleanup() {
        if (window_) {
            ImGui_ImplMetal_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();
            
            glfwDestroyWindow(window_);
            glfwTerminate();
        }
    }
    
    // GLFW callbacks
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        auto* app = static_cast<BarnesHutApp*>(glfwGetWindowUserPointer(window));
        app->renderer_.resize(width, height);
    }
   
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        auto* app = static_cast<BarnesHutApp*>(glfwGetWindowUserPointer(window));
        
        if (action == GLFW_PRESS) {
            switch (key) {
                case GLFW_KEY_ESCAPE:
                    app->running_ = false;
                    break;
                case GLFW_KEY_Q:  
                    app->show_quadtree_visualization_ = !app->show_quadtree_visualization_;
                    app->particle_system_.set_quadtree_visualization_enabled(app->show_quadtree_visualization_);
                    app->renderer_.set_quadtree_visualization_enabled(app->show_quadtree_visualization_);
                    std::cout << "Quadtree visualization: " << (app->show_quadtree_visualization_ ? "ON" : "OFF") << "\n";
                    break;
                case GLFW_KEY_SPACE:
                    app->paused_ = !app->paused_;
                    break;
                case GLFW_KEY_R:
                    app->setup_initial_particles();
                    break;
                case GLFW_KEY_G:
                    app->create_galaxy_spiral();
                    break;
                case GLFW_KEY_S:
                    app->create_solar_system();
                    break;
                case GLFW_KEY_C:
                    app->create_cluster();
                    break;
                case GLFW_KEY_D:  // Debug performance analysis
                    app->diagnose_performance_issues();
                    break;
                case GLFW_KEY_T:  // Test theta values
                    app->particle_system_.test_theta_performance();
                    break;
                case GLFW_KEY_O:  // Run comprehensive optimizations
                    app->run_particle_optimizations();
                    break;
                case GLFW_KEY_F:  // Quick fix overlaps
                    app->quick_fix_overlaps();
                    break;
                case GLFW_KEY_M:  // Morton ordering optimization
                    app->optimize_spatial_layout();
                    break;
                case GLFW_KEY_H:  // Find optimal theta
                    app->find_optimal_theta();
                    break;
                case GLFW_KEY_K:  // Compact tree cache
                    app->compact_tree_cache();
                    break;
                // NEW: Keyboard shortcuts for realistic galaxies
                case GLFW_KEY_1:  // Milky Way
                    app->create_realistic_milky_way();
                    break;
                case GLFW_KEY_2:  // Andromeda
                    app->create_andromeda_like_galaxy();
                    break;
                case GLFW_KEY_3:  // Galaxy merger
                    app->create_galaxy_merger_scenario();
                    break;
                case GLFW_KEY_4:  // Local Group
                    app->create_local_group_simulation();
                    break;
                case GLFW_KEY_5:  // Dwarf accretion
                    app->create_dwarf_galaxy_accretion();
                    break;
            }
        }
    }
    
    
    
    // Member variables
    EventBus event_bus_;
    BarnesHutParticleSystem particle_system_;
    MetalRenderer renderer_;
    
    GLFWwindow* window_;
    id<MTLDevice> metal_device_;
    id<MTLCommandQueue> command_queue_;
    CAMetalLayer* metal_layer_;
    
    bool running_;
    bool paused_;
    
    PhysicsParams physics_params_;
    RenderParams render_params_;
    MetalRenderer::CameraParams camera_;
    PerformanceStats performance_stats_;
    BarnesHutParticleSystem::Config barnes_hut_config_;
    
    // UI state
    bool show_demo_window_;
    bool show_performance_window_;
    bool show_particle_controls_;
    bool show_barnes_hut_controls_;
    bool show_optimization_controls_;
    bool show_realistic_presets_;  // NEW: Realistic galaxy presets window
};

int main() {
    std::cout << "Realistic Galaxy Simulations - Barnes-Hut N-Body\n";
    std::cout << "==============================================\n";
    std::cout << "High-performance astrophysical simulations with realistic initial conditions\n\n";
    
    BarnesHutApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize application\n";
        return -1;
    }
    
    std::cout << "Application initialized successfully\n";
    std::cout << "Controls:\n";
    std::cout << "  SPACE - Pause/Resume simulation\n";
    std::cout << "  R - Reset particles\n";
    std::cout << "  G - Create galaxy spiral\n";
    std::cout << "  S - Create solar system\n";
    std::cout << "  C - Create particle cluster\n";
    std::cout << "  D - Diagnose performance issues\n";
    std::cout << "  T - Test theta performance\n";
    std::cout << "  O - Run comprehensive optimizations ⚡\n";
    std::cout << "  F - Quick fix: Remove particle overlaps\n";
    std::cout << "  M - Optimize spatial layout (Morton ordering)\n";
    std::cout << "  H - Find optimal theta value\n";
    std::cout << "  K - Compact tree cache\n";
    std::cout << "  1 - Create realistic Milky Way galaxy 🌌\n";
    std::cout << "  2 - Create Andromeda galaxy (M31) 🌌\n";
    std::cout << "  3 - Galaxy merger scenario 💥\n";
    std::cout << "  4 - Local Group simulation 🌌\n";
    std::cout << "  5 - Dwarf galaxy accretion 🔄\n";
    std::cout << "  ESC - Exit\n";
    std::cout << "  Mouse Wheel - Zoom camera\n\n";
    std::cout << "Realistic Galaxy Features:\n";
    std::cout << "  - Scientifically accurate mass distributions\n";
    std::cout << "  - NFW dark matter halo profiles (c~12 for MW)\n";
    std::cout << "  - Exponential stellar disks with spiral structure\n";
    std::cout << "  - Hernquist bulge components\n";
    std::cout << "  - Central supermassive black holes\n";
    std::cout << "  - Realistic rotation curves (~220 km/s for MW)\n";
    std::cout << "  - Proper mass-to-light ratios\n";
    std::cout << "  - Galaxy merger dynamics\n";
    std::cout << "  - Tidal stripping and stellar streams\n";
    std::cout << "  - Hierarchical structure formation\n\n";
    std::cout << "Barnes-Hut Algorithm Features:\n";
    std::cout << "  - O(N log N) complexity vs O(N²) brute force\n";
    std::cout << "  - Tree caching for stable simulations\n";
    std::cout << "  - Cache-optimized memory layout\n";
    std::cout << "  - Morton Z-order spatial sorting\n";
    std::cout << "  - Advanced particle overlap detection\n";
    std::cout << "  - Adaptive theta optimization\n";
    std::cout << "  - Configurable theta parameter\n";
    std::cout << "  - Optional OpenMP threading\n";
    std::cout << "  - Real-time performance monitoring\n";
    std::cout << "  - Comprehensive performance diagnostics\n\n";
    std::cout << "⚡ Press 'O' to run optimizations if performance is slow!\n";
    std::cout << "🌌 Press '1' to create a realistic Milky Way galaxy!\n";
    std::cout << "💥 Press '3' to watch galaxies collide and merge!\n\n";
    
    app.run();
    
    std::cout << "Application shutdown\n";
    return 0;
}
