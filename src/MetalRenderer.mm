// MetalRenderer.mm - Fixed structure
#include "MetalRenderer.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include "BarnesHutParticleSystem.h"

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

// Vertex structure for particle quad
struct Vertex {
    simd_float2 position;
    simd_float2 texCoord;
};

// Instance data structure (per particle)
struct InstanceData {
    simd_float2 position;
    simd_float3 color;
    float size;
};

// Uniforms for shaders
struct Uniforms {
    simd_float4x4 mvpMatrix;
    simd_float2 screenSize;
    float particleSize;
    float _padding;
};

// Line rendering structures
struct LineVertex {
    simd_float2 position;
    simd_float3 color;
};

struct LineUniforms {
    simd_float4x4 mvpMatrix;
    float line_width;
    float _padding[3];
};

// Metal shader source code embedded as strings
static const char* kParticleShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct InstanceData {
    float2 position;
    float3 color;
    float size;
};

struct Uniforms {
    float4x4 mvpMatrix;
    float2 screenSize;
    float particleSize;
    float _padding;
};

struct ParticleVertexOut {
    float4 position [[position]];
    float3 color;
    float2 texCoord;
    float pointSize [[point_size]];
};

vertex ParticleVertexOut vertex_main(const Vertex vertexIn [[stage_in]],
                                   const device InstanceData* instanceData [[buffer(1)]],
                                   const device Uniforms& uniforms [[buffer(2)]],
                                   uint instanceID [[instance_id]]) {
    ParticleVertexOut out;
    
    // Get instance data for this particle
    InstanceData instance = instanceData[instanceID];
    
    // Scale the quad vertex by particle size
    float2 scaledVertex = vertexIn.position * instance.size * uniforms.particleSize;
    
    // Translate to particle position
    float2 worldPosition = scaledVertex + instance.position;
    
    // Transform to clip space
    out.position = uniforms.mvpMatrix * float4(worldPosition, 0.0, 1.0);
    out.color = instance.color;
    out.texCoord = vertexIn.texCoord;
    
    return out;
}

fragment float4 fragment_main(ParticleVertexOut in [[stage_in]]) {
    // Create circular particle by computing distance from center
    float2 coord = in.texCoord * 2.0 - 1.0; // Convert to [-1, 1] range
    float dist = length(coord);
    
    // Smooth falloff for anti-aliasing
    float alpha = 1.0 - smoothstep(0.8, 1.0, dist);
    
    // Discard pixels outside the circle
    if (alpha <= 0.0) {
        discard_fragment();
    }
    
    // Apply some brightness variation based on distance from center
    float brightness = 1.0 - (dist * 0.3);
    float3 finalColor = in.color * brightness;
    
    return float4(finalColor, alpha);
}
)";

static const char* kLineShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct LineVertex {
    float2 position [[attribute(0)]];
    float3 color [[attribute(1)]];
};

struct LineUniforms {
    float4x4 mvpMatrix;
    float line_width;
};

struct LineVertexOut {
    float4 position [[position]];
    float3 color;
};

vertex LineVertexOut line_vertex_main(const LineVertex vertexIn [[stage_in]],
                                     const device LineUniforms& uniforms [[buffer(1)]]) {
    LineVertexOut out;
    out.position = uniforms.mvpMatrix * float4(vertexIn.position, 0.0, 1.0);
    out.color = vertexIn.color;
    return out;
}

fragment float4 line_fragment_main(LineVertexOut in [[stage_in]]) {
    return float4(in.color, 0.8); // Semi-transparent lines
}
)";

// PIMPL implementation class
class MetalRenderer::Impl {
public:
    // Metal objects
    id<MTLDevice> metal_device_;
    id<MTLCommandQueue> metal_command_queue_;
    id<MTLLibrary> shader_library_;
    id<MTLRenderPipelineState> render_pipeline_state_;
    
    // Line rendering pipeline and buffers
    id<MTLRenderPipelineState> line_pipeline_state_;
    id<MTLBuffer> line_vertex_buffer_;
    id<MTLBuffer> line_uniform_buffer_;
    
    // Buffers
    id<MTLBuffer> vertex_buffer_;        // Static quad vertices
    id<MTLBuffer> instance_buffer_;      // Dynamic instance data
    id<MTLBuffer> uniform_buffer_;       // Uniforms (MVP matrix, etc.)
    
    // Buffer management
    static constexpr size_t kMaxParticles = 1000000;
    static constexpr size_t kMaxLines = 5000000;  // Max quadtree edges
    static constexpr size_t kUniformBufferSize = sizeof(Uniforms);
    
    // Static quad vertices (will be instanced)
    static constexpr Vertex kQuadVertices[6] = {
        {{-0.5f, -0.5f}, {0.0f, 1.0f}},  // Bottom-left
        {{ 0.5f, -0.5f}, {1.0f, 1.0f}},  // Bottom-right
        {{ 0.5f,  0.5f}, {1.0f, 0.0f}},  // Top-right
        
        {{-0.5f, -0.5f}, {0.0f, 1.0f}},  // Bottom-left
        {{ 0.5f,  0.5f}, {1.0f, 0.0f}},  // Top-right
        {{-0.5f,  0.5f}, {0.0f, 0.0f}}   // Top-left
    };
    
    Impl() : metal_device_(nil), metal_command_queue_(nil), shader_library_(nil),
             render_pipeline_state_(nil), line_pipeline_state_(nil),
             line_vertex_buffer_(nil), line_uniform_buffer_(nil),
             vertex_buffer_(nil), instance_buffer_(nil), uniform_buffer_(nil) {}
    
    ~Impl() {
        // ARC will handle cleanup
    }
   
    bool initialize(id<MTLDevice> device, id<MTLCommandQueue> commandQueue) {
        metal_device_ = device;
        metal_command_queue_ = commandQueue;
        
        if (!metal_device_ || !metal_command_queue_) {
            std::cerr << "MetalRenderer: Invalid device or command queue\n";
            return false;
        }
        
        // Create shaders
        if (!create_shaders()) {
            std::cerr << "MetalRenderer: Failed to create shaders\n";
            return false;
        }
        
        // Create vertex buffers
        if (!create_vertex_buffers()) {
            std::cerr << "MetalRenderer: Failed to create vertex buffers\n";
            return false;
        }
        
        // Create render pipeline
        if (!create_render_pipeline()) {
            std::cerr << "MetalRenderer: Failed to create render pipeline\n";
            return false;
        }
        
        // Create line pipeline for quadtree visualization
        if (!create_line_pipeline()) {
            std::cerr << "MetalRenderer: Failed to create line pipeline\n";
            return false;
        }
        
        std::cout << "MetalRenderer: Initialized successfully\n";
        return true;
    }
    
    bool create_shaders() {
        NSError* error = nil;
        
        // Create particle shader library
        NSString* particleShaderSource = [NSString stringWithUTF8String:kParticleShaderSource];
        
        shader_library_ = [metal_device_ newLibraryWithSource:particleShaderSource
                                                      options:nil
                                                        error:&error];
        
        if (!shader_library_ || error) {
            if (error) {
                std::cerr << "Shader compilation error: " << [[error localizedDescription] UTF8String] << "\n";
            }
            return false;
        }
        
        return true;
    }
    
    bool create_vertex_buffers() {
        // Create static vertex buffer for the quad
        vertex_buffer_ = [metal_device_ newBufferWithBytes:kQuadVertices
                                                   length:sizeof(kQuadVertices)
                                                  options:MTLResourceStorageModeShared];
        
        if (!vertex_buffer_) {
            std::cerr << "Failed to create vertex buffer\n";
            return false;
        }
        
        // Create dynamic instance buffer
        size_t instanceBufferSize = kMaxParticles * sizeof(InstanceData);
        instance_buffer_ = [metal_device_ newBufferWithLength:instanceBufferSize
                                                      options:MTLResourceStorageModeShared];
        
        if (!instance_buffer_) {
            std::cerr << "Failed to create instance buffer\n";
            return false;
        }
        
        // Create uniform buffer
        uniform_buffer_ = [metal_device_ newBufferWithLength:kUniformBufferSize
                                                     options:MTLResourceStorageModeShared];
        
        if (!uniform_buffer_) {
            std::cerr << "Failed to create uniform buffer\n";
            return false;
        }
        
        std::cout << "Created buffers: vertex(" << sizeof(kQuadVertices) 
                  << " bytes), instance(" << instanceBufferSize 
                  << " bytes), uniform(" << kUniformBufferSize << " bytes)\n";
        
        return true;
    }
    
    bool create_render_pipeline() {
        // Get shader functions
        id<MTLFunction> vertexFunction = [shader_library_ newFunctionWithName:@"vertex_main"];
        id<MTLFunction> fragmentFunction = [shader_library_ newFunctionWithName:@"fragment_main"];
        
        if (!vertexFunction || !fragmentFunction) {
            std::cerr << "Failed to get shader functions\n";
            return false;
        }
        
        // Create vertex descriptor
        MTLVertexDescriptor* vertexDescriptor = [[MTLVertexDescriptor alloc] init];
        
        // Position attribute
        vertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
        vertexDescriptor.attributes[0].offset = offsetof(Vertex, position);
        vertexDescriptor.attributes[0].bufferIndex = 0;
        
        // Texture coordinate attribute
        vertexDescriptor.attributes[1].format = MTLVertexFormatFloat2;
        vertexDescriptor.attributes[1].offset = offsetof(Vertex, texCoord);
        vertexDescriptor.attributes[1].bufferIndex = 0;
        
        // Layout for vertex buffer
        vertexDescriptor.layouts[0].stride = sizeof(Vertex);
        vertexDescriptor.layouts[0].stepRate = 1;
        vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
        
        // Create pipeline descriptor
        MTLRenderPipelineDescriptor* pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        pipelineDescriptor.label = @"ParticleRenderPipeline";
        pipelineDescriptor.vertexFunction = vertexFunction;
        pipelineDescriptor.fragmentFunction = fragmentFunction;
        pipelineDescriptor.vertexDescriptor = vertexDescriptor;
        
        // Configure color attachment
        pipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        pipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
        
        // Alpha blending for particle transparency
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        
        NSError* error = nil;
        render_pipeline_state_ = [metal_device_ newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                                               error:&error];
        
        if (!render_pipeline_state_ || error) {
            if (error) {
                std::cerr << "Pipeline creation error: " << [[error localizedDescription] UTF8String] << "\n";
            }
            return false;
        }
        
        std::cout << "Created render pipeline successfully\n";
        return true;
    }

    bool create_line_pipeline() {
        NSError* error = nil;
        
        // Create line shader library
        NSString* lineShaderSource = [NSString stringWithUTF8String:kLineShaderSource];
        
        id<MTLLibrary> lineLibrary = [metal_device_ newLibraryWithSource:lineShaderSource
                                                                 options:nil
                                                                   error:&error];
        if (!lineLibrary || error) {
            if (error) {
                std::cerr << "Line shader compilation error: " << [[error localizedDescription] UTF8String] << "\n";
            }
            return false;
        }
        
        // Get shader functions
        id<MTLFunction> lineVertexFunction = [lineLibrary newFunctionWithName:@"line_vertex_main"];
        id<MTLFunction> lineFragmentFunction = [lineLibrary newFunctionWithName:@"line_fragment_main"];
        
        if (!lineVertexFunction || !lineFragmentFunction) {
            std::cerr << "Failed to get line shader functions\n";
            return false;
        }
        
        // Create vertex descriptor for lines
        MTLVertexDescriptor* lineVertexDescriptor = [[MTLVertexDescriptor alloc] init];
        
        // Position attribute
        lineVertexDescriptor.attributes[0].format = MTLVertexFormatFloat2;
        lineVertexDescriptor.attributes[0].offset = offsetof(LineVertex, position);
        lineVertexDescriptor.attributes[0].bufferIndex = 0;
        
        // Color attribute
        lineVertexDescriptor.attributes[1].format = MTLVertexFormatFloat3;
        lineVertexDescriptor.attributes[1].offset = offsetof(LineVertex, color);
        lineVertexDescriptor.attributes[1].bufferIndex = 0;
        
        // Layout
        lineVertexDescriptor.layouts[0].stride = sizeof(LineVertex);
        lineVertexDescriptor.layouts[0].stepRate = 1;
        lineVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
        
        // Create pipeline descriptor
        MTLRenderPipelineDescriptor* linePipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
        linePipelineDescriptor.label = @"QuadtreeLinePipeline";
        linePipelineDescriptor.vertexFunction = lineVertexFunction;
        linePipelineDescriptor.fragmentFunction = lineFragmentFunction;
        linePipelineDescriptor.vertexDescriptor = lineVertexDescriptor;
        
        // Configure for lines with alpha blending
        linePipelineDescriptor.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        linePipelineDescriptor.colorAttachments[0].blendingEnabled = YES;
        linePipelineDescriptor.colorAttachments[0].rgbBlendOperation = MTLBlendOperationAdd;
        linePipelineDescriptor.colorAttachments[0].alphaBlendOperation = MTLBlendOperationAdd;
        linePipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        linePipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        linePipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        linePipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        
        // Create pipeline state
        line_pipeline_state_ = [metal_device_ newRenderPipelineStateWithDescriptor:linePipelineDescriptor
                                                                             error:&error];
        if (!line_pipeline_state_ || error) {
            if (error) {
                std::cerr << "Line pipeline creation error: " << [[error localizedDescription] UTF8String] << "\n";
            }
            return false;
        }
        
        // Create line vertex buffer
        size_t lineBufferSize = kMaxLines * sizeof(LineVertex);
        line_vertex_buffer_ = [metal_device_ newBufferWithLength:lineBufferSize
                                                         options:MTLResourceStorageModeShared];
        
        // Create line uniform buffer
        line_uniform_buffer_ = [metal_device_ newBufferWithLength:sizeof(LineUniforms)
                                                          options:MTLResourceStorageModeShared];
        
        if (!line_vertex_buffer_ || !line_uniform_buffer_) {
            std::cerr << "Failed to create line buffers\n";
            return false;
        }
        
        std::cout << "Line rendering pipeline created successfully\n";
        return true;
    }
    
    void update_instance_data(const std::vector<float>& positions, 
                            const std::vector<float>& colors, 
                            size_t particle_count) {
        // *** CRITICAL: Add bounds checks to prevent buffer overruns ***
        if (positions.size() < particle_count * 2 || colors.size() < particle_count * 3) {
            std::cerr << "Warning: Insufficient data for particle count " << particle_count << "\n";
            return;
        }
        
        // Limit particles to buffer size
        const size_t n = std::min(particle_count, kMaxParticles);
        InstanceData* dst = static_cast<InstanceData*>([instance_buffer_ contents]);
        
        for (size_t i = 0; i < n; ++i) {
            dst[i].position = { positions[i*2+0], positions[i*2+1] };
            dst[i].color    = { colors[i*3+0], colors[i*3+1], colors[i*3+2] };
            dst[i].size     = 1.0f;
        }
    }
    
    simd_float4x4 create_mvp_matrix(const MetalRenderer::CameraParams& camera, int viewport_width, int viewport_height) {
        // The camera parameters now come from our enhanced camera system
        float zoom = camera.zoom;
        float world_center_x = camera.center_x;
        float world_center_y = camera.center_y;
        
        // Calculate the world space bounds visible on screen
        float world_half_width = (viewport_width * 0.5f)   * zoom;
        float world_half_height = (viewport_height * 0.5f) * zoom;
        
        float left = world_center_x - world_half_width;
        float right = world_center_x + world_half_width;
        float bottom = world_center_y - world_half_height;
        float top = world_center_y + world_half_height;
        
        // Near and far planes for orthographic projection
        float near_plane = -100.0f;
        float far_plane = 100.0f;
        
        // Create orthographic projection matrix
        simd_float4x4 ortho_matrix = {
            .columns[0] = simd_make_float4(2.0f / (right - left), 0, 0, 0),
            .columns[1] = simd_make_float4(0, 2.0f / (top - bottom), 0, 0),
            .columns[2] = simd_make_float4(0, 0, -2.0f / (far_plane - near_plane), 0),
            .columns[3] = simd_make_float4(
                -(right + left) / (right - left),
                -(top + bottom) / (top - bottom),
                -(far_plane + near_plane) / (far_plane - near_plane),
                1
            )
        };
        
        return ortho_matrix;
    }
    
    void render(const MetalRenderer::CameraParams& camera, 
                id<MTLRenderCommandEncoder> renderEncoder,
                const std::vector<float>& positions,
                const std::vector<float>& colors,
                size_t particle_count,
                float particle_size,
                int viewport_width,
                int viewport_height) {
        
        if (!renderEncoder || particle_count == 0) {
            return;
        }
        
        // Update instance data
        update_instance_data(positions, colors, particle_count);
        
        // Update uniforms with improved MVP matrix
        Uniforms* uniforms = static_cast<Uniforms*>([uniform_buffer_ contents]);
        uniforms->mvpMatrix = create_mvp_matrix(camera, viewport_width, viewport_height);
        uniforms->screenSize = simd_make_float2(viewport_width, viewport_height);
        uniforms->particleSize = particle_size;
        
        // Set render pipeline state
        [renderEncoder setRenderPipelineState:render_pipeline_state_];
        
        // Set vertex buffer (static quad)
        [renderEncoder setVertexBuffer:vertex_buffer_ offset:0 atIndex:0];
        
        // Set instance buffer (dynamic particle data)
        [renderEncoder setVertexBuffer:instance_buffer_ offset:0 atIndex:1];
        
        // Set uniform buffer
        [renderEncoder setVertexBuffer:uniform_buffer_ offset:0 atIndex:2];
        
        // Draw instanced primitives
        size_t particlesToRender = std::min(particle_count, kMaxParticles);
        [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle
                          vertexStart:0
                          vertexCount:6
                        instanceCount:particlesToRender];
    }

    void render_lines(const MetalRenderer::CameraParams& camera,
                     id<MTLRenderCommandEncoder> renderEncoder,
                     const std::vector<QuadTreeBox>& boxes,
                     int viewport_width, int viewport_height) {
        if (boxes.empty() || !renderEncoder) return;
            std::cout << "\n=== RENDER_LINES DEBUG ===\n";
    std::cout << "Viewport: " << viewport_width << "x" << viewport_height << "\n";
    std::cout << "Camera: center(" << camera.center_x << "," << camera.center_y << ") zoom=" << camera.zoom << "\n";
    
    for (size_t i = 0; i < std::min(boxes.size(), size_t(3)); ++i) {
        const auto& box = boxes[i];
        std::cout << "Box[" << i << "]: min(" << box.min_x << "," << box.min_y 
                  << ") max(" << box.max_x << "," << box.max_y << ") depth=" << box.depth << "\n";
    }
        
        // Build line vertices from quadtree boxes
        LineVertex* vertices = static_cast<LineVertex*>([line_vertex_buffer_ contents]);
        size_t vertex_count = 0;

            for (size_t box_idx = 0; box_idx < std::min(boxes.size(), size_t(2)); ++box_idx) {
        const auto& box = boxes[box_idx];
        if (vertex_count + 8 >= kMaxLines) break;
        
        std::cout << "Generating lines for box " << box_idx << ":\n";
        
        // Bottom line
        vertices[vertex_count++] = {simd_make_float2(box.min_x, box.min_y), simd_make_float3(1.0f, 0.0f, 0.0f)};
        vertices[vertex_count++] = {simd_make_float2(box.max_x, box.min_y), simd_make_float3(1.0f, 0.0f, 0.0f)};
        std::cout << "  Bottom: (" << box.min_x << "," << box.min_y << ") to (" << box.max_x << "," << box.min_y << ")\n";
        
        // Right line  
        vertices[vertex_count++] = {simd_make_float2(box.max_x, box.min_y), simd_make_float3(0.0f, 1.0f, 0.0f)};
        vertices[vertex_count++] = {simd_make_float2(box.max_x, box.max_y), simd_make_float3(0.0f, 1.0f, 0.0f)};
        std::cout << "  Right: (" << box.max_x << "," << box.min_y << ") to (" << box.max_x << "," << box.max_y << ")\n";
        
        // Top line
        vertices[vertex_count++] = {simd_make_float2(box.max_x, box.max_y), simd_make_float3(0.0f, 0.0f, 1.0f)};
        vertices[vertex_count++] = {simd_make_float2(box.min_x, box.max_y), simd_make_float3(0.0f, 0.0f, 1.0f)};
        std::cout << "  Top: (" << box.max_x << "," << box.max_y << ") to (" << box.min_x << "," << box.max_y << ")\n";
        
        // Left line
        vertices[vertex_count++] = {simd_make_float2(box.min_x, box.max_y), simd_make_float3(1.0f, 1.0f, 0.0f)};
        vertices[vertex_count++] = {simd_make_float2(box.min_x, box.min_y), simd_make_float3(1.0f, 1.0f, 0.0f)};
        std::cout << "  Left: (" << box.min_x << "," << box.max_y << ") to (" << box.min_x << "," << box.min_y << ")\n";
        
        if (box_idx < 2) break; // Only debug first 2 boxes
    }
        
        for (const auto& box : boxes) {
            if (vertex_count + 8 >= kMaxLines) break;  // Each box = 4 lines = 8 vertices
            
            // Color based on depth (deeper = more red, shallower = more blue)
            float depth_factor = std::min(box.depth / 10.0f, 1.0f);
            simd_float3 line_color;
            
            if (box.is_leaf) {
                // Leaf nodes: Green tint
                line_color = simd_make_float3(0.2f + depth_factor * 0.5f, 0.8f, 0.2f + depth_factor * 0.3f);
            } else {
                // Internal nodes: Blue to red gradient
                line_color = simd_make_float3(0.3f + depth_factor * 0.7f, 0.3f, 1.0f - depth_factor * 0.7f);
            }
            
            // Bottom line
            vertices[vertex_count++] = {simd_make_float2(box.min_x, box.min_y), line_color};
            vertices[vertex_count++] = {simd_make_float2(box.max_x, box.min_y), line_color};
            
            // Right line
            vertices[vertex_count++] = {simd_make_float2(box.max_x, box.min_y), line_color};
            vertices[vertex_count++] = {simd_make_float2(box.max_x, box.max_y), line_color};
            
            // Top line
            vertices[vertex_count++] = {simd_make_float2(box.max_x, box.max_y), line_color};
            vertices[vertex_count++] = {simd_make_float2(box.min_x, box.max_y), line_color};
            
            // Left line
            vertices[vertex_count++] = {simd_make_float2(box.min_x, box.max_y), line_color};
            vertices[vertex_count++] = {simd_make_float2(box.min_x, box.min_y), line_color};
        }
        
        if (vertex_count == 0) return;
        
        // Update uniforms
        LineUniforms* uniforms = static_cast<LineUniforms*>([line_uniform_buffer_ contents]);
        uniforms->mvpMatrix = create_mvp_matrix(camera, viewport_width, viewport_height);
        uniforms->line_width = 1.0f;
        
        // Set pipeline and draw
        [renderEncoder setRenderPipelineState:line_pipeline_state_];
        [renderEncoder setVertexBuffer:line_vertex_buffer_ offset:0 atIndex:0];
        [renderEncoder setVertexBuffer:line_uniform_buffer_ offset:0 atIndex:1];
        
        [renderEncoder drawPrimitives:MTLPrimitiveTypeLine
                          vertexStart:0
                          vertexCount:vertex_count];
    }
}; // END of Impl class

// MetalRenderer implementation (OUTSIDE the Impl class)
MetalRenderer::MetalRenderer(EventBus& event_bus) 
    : event_bus_(event_bus),
      quadtree_visualization_enabled_(false),
      current_particle_count_(0),
      initialized_(false),
      data_dirty_(false),
      particle_size_(2.0f),
      background_r_(0.02f), background_g_(0.02f), background_b_(0.05f), background_a_(1.0f),
      viewport_width_(800), viewport_height_(600),
      impl_(new Impl()) {
    
    // Subscribe to render updates
    event_bus_.subscribe<RenderUpdateEvent>(Events::RENDER_UPDATE,
        [this](const RenderUpdateEvent& event) {
            on_render_update(event);
        });
}

MetalRenderer::~MetalRenderer() {
    delete impl_;
}

bool MetalRenderer::initialize(void* device, void* commandQueue) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> mtlCommandQueue = (__bridge id<MTLCommandQueue>)commandQueue;
    
    initialized_ = impl_->initialize(mtlDevice, mtlCommandQueue);
    return initialized_;
}

void MetalRenderer::on_render_update(const RenderUpdateEvent& event) {
    current_particle_count_ = event.particle_count;
    
    if (event.particle_count > 0 && event.positions && event.colors) {
        // Copy position data (x,y interleaved)
        size_t pos_size = event.particle_count * 2;
        positions_.resize(pos_size);
        std::copy(event.positions, event.positions + pos_size, positions_.begin());
        
        // Copy color data (r,g,b interleaved)
        size_t color_size = event.particle_count * 3;
        colors_.resize(color_size);
        std::copy(event.colors, event.colors + color_size, colors_.begin());
        
        data_dirty_ = true;
    }
}

void MetalRenderer::render(const CameraParams& camera, void* renderEncoder) {
    if (!initialized_ || current_particle_count_ == 0 || !renderEncoder) {
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)renderEncoder;
    
    impl_->render(camera, encoder, positions_, colors_, current_particle_count_, 
                  particle_size_, viewport_width_, viewport_height_);
    
    // Update stats
    stats_.particles_rendered = current_particle_count_;
    stats_.frames_rendered++;
    stats_.vertices_processed = current_particle_count_ * 6; // 6 vertices per particle
    stats_.draw_calls = 1; // Single instanced draw call
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    stats_.last_frame_time_ms = duration.count() / 1000.0f;
    
    data_dirty_ = false;
}

void MetalRenderer::resize(int width, int height) {
    viewport_width_ = width;
    viewport_height_ = height;
    
    std::cout << "MetalRenderer: Resized to " << width << "x" << height << "\n";
}

void MetalRenderer::set_background_color(float r, float g, float b, float a) {
    background_r_ = r;
    background_g_ = g;
    background_b_ = b;
    background_a_ = a;
}

void MetalRenderer::render_quadtree_lines(const CameraParams& camera, void* renderEncoder,
                                         const std::vector<QuadTreeBox>& boxes) {
    if (!quadtree_visualization_enabled_ || boxes.empty() || !renderEncoder) {
        return;
    }
    
    id<MTLRenderCommandEncoder> encoder = (__bridge id<MTLRenderCommandEncoder>)renderEncoder;
    impl_->render_lines(camera, encoder, boxes, viewport_width_, viewport_height_);
}
