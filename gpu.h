#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class Image;

namespace gpu {



struct TextureBuffer {
    wgpu::Texture texture;
    wgpu::Extent3D size;
};

enum class TextureUsage {
    ReadOnly,
    ReadWrite
};

enum class TextureFormat {
    R8Unorm,
    R32Float
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
    std::vector<TextureBuffer> inputImageBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<TextureBuffer> outputImageBuffers;
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

struct Context {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
};

TextureBuffer makeEmptyTextureBuffer(uint32_t width, uint32_t height,
                                       TextureUsage textureUsage,
                                       TextureFormat texureFormat,
                                       Context &context);
TextureBuffer makeTextureBufferFromHost(const Image& image, Context &context);
TextureBuffer makeReadOnlyTextureBuffer(const Image& image, Context &context);
Image makeHostImageFromBuffer(const TextureBuffer& buffer, Context& context);

wgpu::ShaderModule createShaderModule(const std::string& name, const std::string& code, const Context& context);

void applyShaderTransform(const TextureBuffer& src, TextureBuffer& dst, const std::string& shaderCode, Context& context);

// Dispatches a computer shader with the given input and output buffers and images
ComputeOperation createComputeOperation(ComputeOperationData &operationData, Context &context);

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupDimensions workgroupDimensions,
                       Context& context);

[[nodiscard]] Context createWebGPUContext();

}
