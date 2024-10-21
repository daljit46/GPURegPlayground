#pragma once

#include <webgpu/webgpu_cpp.h>

class Image;

namespace gpu {

struct WGPUContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
};

struct WGPUImageBuffer {
    wgpu::Texture texture;
    wgpu::Extent3D size;
};

[[nodiscard]] WGPUImageBuffer createImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] WGPUImageBuffer createReadOnlyImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] Image createHostImageFromBuffer(const WGPUImageBuffer& buffer, WGPUContext& context);

[[nodiscard]] WGPUContext createWebGPUContext();

}
