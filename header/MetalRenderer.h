#pragma once
#include "EventSystem.h"
#include <vector>

// Forward declarations to avoid including Metal headers in C++
#ifdef __OBJC__
    #import <Metal/Metal.h>
    #import <MetalKit/MetalKit.h>
    #import <simd/simd.h>
    typedef simd_float2 float2;
    typedef simd_float3 float3;
    typedef simd_float4x4 matrix_float4x4;
#else
    // C++ forward declarations
    struct float2 { float x, y; };
    struct float3 { float x, y, z; };
    struct matrix_float4x4 { float m[16]; };
    #ifdef __cplusplus
        extern "C" {
    #endif
        typedef struct objc_object* id;
    #ifdef __cplusplus
        }
    #endif
#endif

// Forward declaration
class BarnesHutParticleSystem;

// Define QuadTreeBox structure here to avoid incomplete type issues
struct QuadTreeBox {
    float min_x, min_y, max_x, max_y;
    int depth;
    bool is_leaf;
    int particle_count;
    
    QuadTreeBox() : min_x(0), min_y(0), max_x(0), max_y(0), depth(0), is_leaf(true), particle_count(0) {}
    QuadTreeBox(float minX, float minY, float maxX, float maxY, int d = 0, bool leaf = true, int count = 0)
        : min_x(minX), min_y(minY), max_x(maxX), max_y(maxY), depth(d), is_leaf(leaf), particle_count(count) {}
    
    // Conversion constructor template for compatibility with BarnesHutParticleSystem::QuadTreeBox
    template<typename T>
    QuadTreeBox(const T& other) 
        : min_x(other.min_x), min_y(other.min_y), max_x(other.max_x), max_y(other.max_y),
          depth(other.depth), is_leaf(other.is_leaf), particle_count(other.particle_count) {}
};

// Metal renderer for instanced particle rendering
class MetalRenderer {
public:
    struct RenderStats {
        size_t particles_rendered = 0;
        size_t frames_rendered = 0;
        float last_frame_time_ms = 0.0f;
        size_t vertices_processed = 0;
        size_t draw_calls = 0;
    };
    
    struct CameraParams {
        float zoom = 1.0f;
        float center_x = 0.0f;
        float center_y = 0.0f;
    };
    
    MetalRenderer(EventBus& event_bus);
    ~MetalRenderer();
    
    // Main rendering interface
    bool initialize(void* device, void* commandQueue);
    void render(const CameraParams& camera, void* renderEncoder);
    void resize(int width, int height);
    
    // Configuration
    void set_particle_size(float size) { particle_size_ = size; }
    void set_background_color(float r, float g, float b, float a);
    
    // Status and debugging
    const RenderStats& get_stats() const { return stats_; }
    bool is_initialized() const { return initialized_; }
    
    // For testing
    size_t get_current_particle_count() const { return current_particle_count_; }
    const std::vector<float>& get_positions() const { return positions_; }
    const std::vector<float>& get_colors() const { return colors_; }

    // Quadtree visualization - using our own QuadTreeBox struct
    void render_quadtree_lines(const CameraParams& camera, void* renderEncoder,
                              const std::vector<QuadTreeBox>& boxes);
    
    // Template method for automatic conversion from any compatible QuadTreeBox type
    template<typename BarnesHutQuadTreeBox>
    void render_quadtree_lines(const CameraParams& camera, void* renderEncoder,
                              const std::vector<BarnesHutQuadTreeBox>& barnes_hut_boxes) {
        std::vector<QuadTreeBox> converted_boxes;
        converted_boxes.reserve(barnes_hut_boxes.size());
        
        for (const auto& box : barnes_hut_boxes) {
            converted_boxes.emplace_back(box); // Uses template conversion constructor
        }
        
        render_quadtree_lines(camera, renderEncoder, converted_boxes);
    }
    
    // Toggle quadtree visualization
    void set_quadtree_visualization_enabled(bool enabled) { quadtree_visualization_enabled_ = enabled; }
    bool is_quadtree_visualization_enabled() const { return quadtree_visualization_enabled_; }
    
private:
    // Event handlers
    void on_render_update(const RenderUpdateEvent& event);
    
    EventBus& event_bus_;
    bool quadtree_visualization_enabled_; 
    
    // Particle data
    std::vector<float> positions_;    // [x0,y0,x1,y1,...]
    std::vector<float> colors_;       // [r0,g0,b0,r1,g1,b1,...]
    size_t current_particle_count_;
    
    // Render state
    bool initialized_;
    bool data_dirty_;
    float particle_size_;
    float background_r_, background_g_, background_b_, background_a_;
    int viewport_width_, viewport_height_;
    
    // Performance tracking
    RenderStats stats_;
    
    // Opaque pointer to implementation (PIMPL pattern)
    class Impl;
    Impl* impl_;
};
