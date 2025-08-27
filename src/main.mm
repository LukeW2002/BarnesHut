// main.mm - Refactored with separated IMGUI and Galaxy components
#include "BarnesHutParticleSystem.h"
#include "MetalRenderer.h"
#include "EventSystem.h"
#include "ImGuiInterface.h"
#include "GalaxyFactory.h"
#include "Vec2.h"
#include "imgui.h"
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
#include <chrono>

class BarnesHutApp {
public:
    BarnesHutApp() 
        : event_bus_(),
          particle_system_(1000000, event_bus_), // Use default config
          renderer_(event_bus_),
          imgui_interface_(particle_system_, renderer_, event_bus_),
          galaxy_factory_(particle_system_),
          window_(nullptr),
          metal_device_(nil),
          command_queue_(nil),
          metal_layer_(nil),
          running_(false) {
        
        // Camera parameters
        enhanced_camera_.world_zoom = 0.5f;  // Zoomed out for galaxy scales
        enhanced_camera_.world_center_x = 0.0f;
        enhanced_camera_.world_center_y = 0.0f;
        
        setup_callbacks();
    }
    
    ~BarnesHutApp() {
        cleanup();
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
 
public:
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
        if (!imgui_interface_.initialize(window_, (__bridge void*)metal_device_)) {
            std::cerr << "Failed to initialize IMGUI\n";
            return false;
        }
        
        // Initialize MetalRenderer
        if (!renderer_.initialize((__bridge void*)metal_device_, (__bridge void*)command_queue_)) {
            std::cerr << "Failed to initialize MetalRenderer\n";
            return false;
        }
        
        // Set up initial particle configuration
        galaxy_factory_.create_galaxy_spiral(); // Default start
        
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
            if (!imgui_interface_.is_paused()) {
                const auto& physics_params = imgui_interface_.get_physics_params();
                time_accumulator += frame_time * physics_params.time_scale;
                
                while (time_accumulator >= physics_dt) {
                    particle_system_.update(physics_dt);
                    time_accumulator -= physics_dt;
                }
            }
            
            // Render frame
            render_frame();
            
            // Update frame statistics
            imgui_interface_.update_performance_stats(frame_time);
        }
    }

private:
    void setup_callbacks() {
        // Set up ImGui callbacks for galaxy creation
        imgui_interface_.on_create_milky_way = [this]() {
            galaxy_factory_.create_realistic_milky_way();
        };
        
        imgui_interface_.on_create_andromeda = [this]() {
            galaxy_factory_.create_andromeda_like_galaxy();
        };
        
        imgui_interface_.on_create_galaxy_merger = [this]() {
            galaxy_factory_.create_two_spiral_binary_merger_100k();
        };
        
        imgui_interface_.on_create_local_group = [this]() {
            galaxy_factory_.create_local_group_simulation();
        };
        
        imgui_interface_.on_create_twin_spiral = [this]() {
            galaxy_factory_.create_twin_spiral_pair();
        };
        
        imgui_interface_.on_create_dwarf_accretion = [this]() {
            galaxy_factory_.create_dwarf_galaxy_accretion();
        };
        
        imgui_interface_.on_create_tidal_flyby = [this]() {
            galaxy_factory_.create_tidally_disrupting_flyby();
        };
        
        // Classic presets
        imgui_interface_.on_create_galaxy_spiral = [this]() {
            galaxy_factory_.create_galaxy_spiral();
        };
        
        imgui_interface_.on_create_solar_system = [this]() {
            galaxy_factory_.create_solar_system();
        };
        
        imgui_interface_.on_create_cluster = [this]() {
            galaxy_factory_.create_cluster();
        };
        
        imgui_interface_.on_add_random_particles = [this](int count) {
            galaxy_factory_.add_random_particles(count);
        };
        
        // Optimization callbacks
        imgui_interface_.on_run_optimizations = [this]() {
            std::cout << "\n🚀 RUNNING PARTICLE SYSTEM OPTIMIZATIONS\n";
            particle_system_.run_comprehensive_optimization();
        };
        
        imgui_interface_.on_fix_overlaps = [this]() {
            std::cout << "\n🔧 QUICK FIX: Removing particle overlaps\n";
            particle_system_.fix_overlapping_particles_advanced();
        };
        
        imgui_interface_.on_optimize_layout = [this]() {
            std::cout << "\n📊 OPTIMIZING: Spatial layout with Morton ordering\n";
            particle_system_.optimize_particle_layout();
        };
        
        imgui_interface_.on_find_optimal_theta = [this]() {
            std::cout << "\n🎯 OPTIMIZING: Finding optimal theta value\n";
            particle_system_.adaptive_theta_optimization();
        };
        
        imgui_interface_.on_diagnose_performance = [this]() {
            std::cout << "\n🔍 DIAGNOSING: Performance bottlenecks\n";
            particle_system_.diagnose_tree_traversal_bottleneck();
            particle_system_.print_performance_analysis();
        };
        
        imgui_interface_.on_compact_cache = [this]() {
            std::cout << "\n📚✨ COMPACTING: Tree for cache efficiency\n";
            particle_system_.compact_tree_for_cache_efficiency();
        };
        
        // Set up galaxy factory callbacks for physics and camera updates
        galaxy_factory_.on_update_physics_params = [this](const GalaxyFactory::PhysicsConfig& config) {
            particle_system_.set_gravity(Vec2(0.0f, 0.0f)); // No external gravity for galaxies
            particle_system_.set_damping(config.damping);
            particle_system_.set_bounce_force(config.bounce_force);
            particle_system_.set_boundary(
                config.boundary_min_x, config.boundary_max_x,
                config.boundary_min_y, config.boundary_max_y
            );
        };
        
        galaxy_factory_.on_update_camera = [this](double center_x, double center_y, double zoom) {
            enhanced_camera_.world_center_x = center_x;
            enhanced_camera_.world_center_y = center_y;
            
            // Calculate zoom based on viewport size
            double min_screen_size = std::min(enhanced_camera_.viewport_width, enhanced_camera_.viewport_height);
            enhanced_camera_.world_zoom = min_screen_size / (zoom * 2.0); // Zoom factor to pixels per unit
        };
    }
    
    void update_physics_parameters() {
        const auto& params = imgui_interface_.get_physics_params();
        particle_system_.set_gravity(Vec2(params.gravity_x, params.gravity_y));
        particle_system_.set_damping(params.damping);
        particle_system_.set_bounce_force(params.bounce_force);
        particle_system_.set_boundary(
            params.boundary_min_x, params.boundary_max_x,
            params.boundary_min_y, params.boundary_max_y
        );
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

            ImGui_ImplMetal_NewFrame(renderPassDescriptor);
            
            const auto& render_params = imgui_interface_.get_render_params();
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(
                render_params.background_r,
                render_params.background_g,
                render_params.background_b,
                render_params.background_a
            );
            
            // Create render command encoder
            id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
            renderEncoder.label = @"BarnesHutRenderEncoder";
            MetalRenderer::CameraParams render_camera = enhanced_camera_.to_render_params();

            if (imgui_interface_.should_show_quadtree()) {
                auto quadtree_boxes = particle_system_.get_quadtree_boxes();
                renderer_.render_quadtree_lines(render_camera, (__bridge void*)renderEncoder, quadtree_boxes);
            }

            // Render ImGui interface
            imgui_interface_.render();
            
            // Render particles using our MetalRenderer
            renderer_.render(render_camera, (__bridge void*)renderEncoder);
            
            // Render ImGui
            imgui_interface_.render();
            renderer_.render(render_camera, (__bridge void*)renderEncoder);
            ImGui::Render();
            ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);

            
            // End encoding and present
            [renderEncoder endEncoding];
            [commandBuffer presentDrawable:drawable];
            [commandBuffer commit];
        }
    }
    
    void cleanup() {
        if (window_) {
            imgui_interface_.cleanup();
            
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
                    {
                        bool current_state = app->imgui_interface_.should_show_quadtree();
                        // Toggle through ImGui interface (it will handle the renderer updates)
                        app->particle_system_.set_quadtree_visualization_enabled(!current_state);
                        app->renderer_.set_quadtree_visualization_enabled(!current_state);
                        std::cout << "Quadtree visualization: " << (!current_state ? "ON" : "OFF") << "\n";
                    }
                    break;
                case GLFW_KEY_SPACE:
                    app->imgui_interface_.set_paused(!app->imgui_interface_.is_paused());
                    break;
                case GLFW_KEY_R:
                    app->galaxy_factory_.create_galaxy_spiral();
                    break;
                case GLFW_KEY_G:
                    app->galaxy_factory_.create_galaxy_spiral();
                    break;
                case GLFW_KEY_S:
                    app->galaxy_factory_.create_solar_system();
                    break;
                case GLFW_KEY_C:
                    app->galaxy_factory_.create_cluster();
                    break;
                case GLFW_KEY_D:  // Debug performance analysis
                    std::cout << "\n🔍 DIAGNOSING: Performance bottlenecks\n";
                    app->particle_system_.diagnose_tree_traversal_bottleneck();
                    app->particle_system_.print_performance_analysis();
                    break;
                case GLFW_KEY_T:  // Test theta values
                    app->particle_system_.test_theta_performance();
                    break;
                case GLFW_KEY_O:  // Run comprehensive optimizations
                    std::cout << "\n🚀 RUNNING PARTICLE SYSTEM OPTIMIZATIONS\n";
                    app->particle_system_.run_comprehensive_optimization();
                    break;
                case GLFW_KEY_F:  // Quick fix overlaps
                    std::cout << "\n🔧 QUICK FIX: Removing particle overlaps\n";
                    app->particle_system_.fix_overlapping_particles_advanced();
                    break;
                case GLFW_KEY_M:  // Morton ordering optimization
                    std::cout << "\n📊 OPTIMIZING: Spatial layout with Morton ordering\n";
                    app->particle_system_.optimize_particle_layout();
                    break;
                case GLFW_KEY_H:  // Find optimal theta
                    std::cout << "\n🎯 OPTIMIZING: Finding optimal theta value\n";
                    app->particle_system_.adaptive_theta_optimization();
                    break;
                case GLFW_KEY_K:  // Compact tree cache
                    std::cout << "\n📚✨ COMPACTING: Tree for cache efficiency\n";
                    app->particle_system_.compact_tree_for_cache_efficiency();
                    break;
                // Keyboard shortcuts for realistic galaxies
                case GLFW_KEY_1:  // Milky Way
                    app->galaxy_factory_.create_realistic_milky_way();
                    break;
                case GLFW_KEY_2:  // Andromeda
                    app->galaxy_factory_.create_andromeda_like_galaxy();
                    break;
                case GLFW_KEY_3:  // Galaxy merger
                    app->galaxy_factory_.create_tidally_disrupting_flyby();
                    break;
                case GLFW_KEY_4:  // Local Group
                    app->galaxy_factory_.create_local_group_simulation();
                    break;
                case GLFW_KEY_5:  // Dwarf accretion
                    app->galaxy_factory_.create_dwarf_galaxy_accretion();
                    break;
            }
        }
    }
    
    // Member variables
    EventBus event_bus_;
    BarnesHutParticleSystem particle_system_;
    MetalRenderer renderer_;
    ImGuiInterface imgui_interface_;
    GalaxyFactory galaxy_factory_;
    
    GLFWwindow* window_;
    id<MTLDevice> metal_device_;
    id<MTLCommandQueue> command_queue_;
    CAMetalLayer* metal_layer_;
    
    bool running_;
    
   EnhancedCamera enhanced_camera_;
};

int main() {
    std::cout << "Realistic Galaxy Simulations - Barnes-Hut N-Body (Refactored)\n";
    std::cout << "===========================================================\n";
    std::cout << "High-performance astrophysical simulations with realistic initial conditions\n";
    std::cout << "Now with clean separation of concerns:\n";
    std::cout << "  • ImGuiInterface.cpp - All UI management\n";
    std::cout << "  • GalaxyFactory.cpp - Galaxy creation and physics\n";
    std::cout << "  • main.mm - Application coordination\n\n";
    
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
    std::cout << "  5 - Dwarf galaxy accretion 🔥\n";
    std::cout << "  Q - Toggle quadtree visualization\n";
    std::cout << "  ESC - Exit\n";
    std::cout << "  Mouse Wheel - Zoom camera\n";
    std::cout << "  Middle Mouse Drag - Pan camera\n\n";
    
    std::cout << "Refactored Architecture Benefits:\n";
    std::cout << "  ✅ Cleaner code organization\n";
    std::cout << "  ✅ Easier to maintain and extend\n";
    std::cout << "  ✅ Separation of UI, physics, and rendering\n";
    std::cout << "  ✅ Reusable components\n";
    std::cout << "  ✅ Better testing capabilities\n\n";
    
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
