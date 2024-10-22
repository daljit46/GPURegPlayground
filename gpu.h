#pragma once

#include <webgpu/webgpu_cpp.h>

class Image;

namespace gpu {

struct Context {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
};

struct ImageBuffer {
    wgpu::Texture texture;
    wgpu::Extent3D size;
};

[[nodiscard]] ImageBuffer createImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] ImageBuffer createReadOnlyImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] Image createHostImageFromBuffer(const ImageBuffer& buffer, Context& context);

void applyShaderTransform(const ImageBuffer& src, ImageBuffer& dst, const std::string& shaderCode, Context& context);

[[nodiscard]] Context createWebGPUContext();


[[nodiscard]] Context createWebGPUContext();

}
