#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class Image;

namespace gpu {


enum class TextureUsage {
    ReadOnly,
    ReadWrite
};

enum class TextureFormat {
    R8Unorm,
    R32Float,
    RGBA8Unorm
};

struct TextureSpecification {
    uint32_t width = 0;
    uint32_t height = 0;
    TextureFormat format;
    TextureUsage usage;
};

struct TextureBuffer {
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
    std::vector<TextureBuffer> inputImageBuffers;
    std::vector<Buffer> outputBuffers;
    std::vector<TextureBuffer> outputImageBuffers;
};

struct WorkgroupSize {
    uint32_t x = 16;
    uint32_t y = 16;
    uint32_t z = 0;
};

struct WorkgroupGrid {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct Context {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
};

TextureBuffer makeEmptyTextureBuffer(const TextureSpecification& spec, Context &context);
TextureBuffer makeTextureBufferFromHost(const Image& image, Context &context);
TextureBuffer makeReadOnlyTextureBuffer(const Image& image, Context &context);
Image makeHostImageFromBuffer(const TextureBuffer& buffer, Context& context);

wgpu::ShaderModule createShaderModule(const std::string& name, const std::string& code, const Context& context);

void applyShaderTransform(const TextureBuffer& src, TextureBuffer& dst, const std::string& shaderCode, Context& context);

ComputeOperation createComputeOperation(ComputeOperationData &operationData, Context &context);

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupGrid workgroupDimensions,
                       Context& context);

[[nodiscard]] Context createWebGPUContext();

}
