#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class Image;

namespace gpu {

struct Context {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
};

struct ImageBuffer {
    wgpu::Texture texture;
    wgpu::Extent3D size;
};

struct Buffer {
    wgpu::Buffer buffer;
    wgpu::Extent3D size;
};

struct ShaderEntry {
    std::string name;
    std::string code;
    std::string entryPoint;
};

struct ComputeOperation {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
};

struct ComputeOperationData {
    ShaderEntry shader;
    std::vector<Buffer> inputBuffers;
    std::vector<ImageBuffer> inputImageBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<ImageBuffer> outputImageBuffers;
};

struct WorkgroupSize {
    uint32_t x = 16;
    uint32_t y = 16;
    uint32_t z = 0;
};

struct WorkgroupDimensions {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

[[nodiscard]] ImageBuffer createEmptyImageBuffer(const wgpu::Device& device);
[[nodiscard]] ImageBuffer createImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] ImageBuffer createReadOnlyImageBuffer(const Image& image, const wgpu::Device& device);
[[nodiscard]] Image createHostImageFromBuffer(const ImageBuffer& buffer, Context& context);

wgpu::ShaderModule createShaderModule(const std::string& name, const std::string& code, const Context& context);

void applyShaderTransform(const ImageBuffer& src, ImageBuffer& dst, const std::string& shaderCode, Context& context);

// Dispatches a computer shader with the given input and output buffers and images
ComputeOperation createComputeOperation(ComputeOperationData &operationData, Context &context);

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupDimensions workgroupDimensions,
                       Context& context);

[[nodiscard]] Context createWebGPUContext();

}
