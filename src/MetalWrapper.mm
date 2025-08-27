// MetalWrapper.mm - C++ to Metal bridge functions
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <QuartzCore/CAMetalLayer.h>
#import <Cocoa/Cocoa.h>

// ADD GLFW headers for proper GLFWwindow type
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>

extern "C" {

// RENAMED function to avoid conflict with system MTLCreateSystemDefaultDevice
void* createMetalSystemDefaultDevice(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

// RENAMED function to avoid conflict with GLFW's glfwGetCocoaWindow
void* getCocoaWindowFromGLFW(void* window) {
    GLFWwindow* glfwWindow = (GLFWwindow*)window;
    NSWindow* nsWindow = glfwGetCocoaWindow(glfwWindow);  // Call actual GLFW function
    return (__bridge void*)nsWindow;
}

void* createMetalLayer(void* device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = mtlDevice;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    
    return (__bridge_retained void*)layer;
}

void setMetalLayerToWindow(void* window, void* layer) {
    NSWindow* nsWindow = (__bridge NSWindow*)window;
    CAMetalLayer* metalLayer = (__bridge CAMetalLayer*)layer;
    
    nsWindow.contentView.layer = metalLayer;
    nsWindow.contentView.wantsLayer = YES;
}

void* createCommandQueue(void* device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    return (__bridge_retained void*)queue;
}

void* getNextDrawable(void* layer) {
    CAMetalLayer* metalLayer = (__bridge CAMetalLayer*)layer;
    id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
    return (__bridge void*)drawable;
}

void setLayerDrawableSize(void* layer, double width, double height) {
    CAMetalLayer* metalLayer = (__bridge CAMetalLayer*)layer;
    metalLayer.drawableSize = CGSizeMake(width, height);
}

void* createCommandBuffer(void* commandQueue) {
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)commandQueue;
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = @"ParticleRenderCommandBuffer";
    // Use __bridge_retained to keep the object alive
    return (__bridge_retained void*)commandBuffer;
}

void* createRenderPassDescriptor(void* drawable, float r, float g, float b, float a) {
    id<CAMetalDrawable> mtlDrawable = (__bridge id<CAMetalDrawable>)drawable;
    
    MTLRenderPassDescriptor* renderPass = [MTLRenderPassDescriptor new];
    renderPass.colorAttachments[0].texture = mtlDrawable.texture;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(r, g, b, a);
    
    // Use __bridge_retained to keep the object alive
    return (__bridge_retained void*)renderPass;
}

void* createRenderCommandEncoder(void* commandBuffer, void* renderPass) {
    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
    MTLRenderPassDescriptor* passDescriptor = (__bridge MTLRenderPassDescriptor*)renderPass;
    
    id<MTLRenderCommandEncoder> encoder = [cmdBuffer renderCommandEncoderWithDescriptor:passDescriptor];
    encoder.label = @"ParticleRenderEncoder";
    
    // Use __bridge_retained to keep the object alive
    return (__bridge_retained void*)encoder;
}

void endRenderEncoding(void* encoder) {
    id<MTLRenderCommandEncoder> renderEncoder = (__bridge id<MTLRenderCommandEncoder>)encoder;
    [renderEncoder endEncoding];
}

void presentDrawable(void* commandBuffer, void* drawable) {
    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
    id<CAMetalDrawable> mtlDrawable = (__bridge id<CAMetalDrawable>)drawable;
    [cmdBuffer presentDrawable:mtlDrawable];
}

void commitCommandBuffer(void* commandBuffer) {
    id<MTLCommandBuffer> cmdBuffer = (__bridge id<MTLCommandBuffer>)commandBuffer;
    [cmdBuffer commit];
}

// Add cleanup functions to release retained objects
void releaseCommandBuffer(void* commandBuffer) {
    if (commandBuffer) {
        CFRelease(commandBuffer);
    }
}

void releaseRenderPassDescriptor(void* renderPass) {
    if (renderPass) {
        CFRelease(renderPass);
    }
}

void releaseRenderCommandEncoder(void* encoder) {
    if (encoder) {
        CFRelease(encoder);
    }
}

} // extern "C"
