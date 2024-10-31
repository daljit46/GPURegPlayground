#pragma once

#include <cstddef>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class Image;

namespace gpu {


enum class ResourceUsage {
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
    TextureFormat format = TextureFormat::R8Unorm;
    ResourceUsage usage = ResourceUsage::ReadOnly;
};

struct TextureBuffer {
    wgpu::Texture texture;
    wgpu::Extent3D size;
};

struct DataBuffer {
    wgpu::Buffer buffer;
    ResourceUsage usage = ResourceUsage::ReadOnly;
    size_t size = 0;
};

struct WorkgroupSize {
    uint32_t x = 16;
    uint32_t y = 16;
    uint32_t z = 1;
};

struct WorkgroupGrid {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;
};

struct ShaderEntry {
    std::string name;
    std::string entryPoint;
    std::string code;
    WorkgroupSize workgroupSize;
};

struct ComputeOperation {
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
};

struct ComputeOperationData {
    ShaderEntry shader;
    std::vector<DataBuffer> uniformBuffers;
    std::vector<DataBuffer> inputBuffers;
    std::vector<TextureBuffer> inputImageBuffers;
    std::vector<DataBuffer> outputBuffers;
    std::vector<TextureBuffer> outputImageBuffers;
    std::vector<wgpu::Sampler> samplers;
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

DataBuffer makeEmptyDataBuffer(size_t size, ResourceUsage usage, Context& context);
DataBuffer makeUniformBuffer(const uint8_t* data, size_t size, Context& context);
void readBufferFromGPU(void* data, const DataBuffer& buffer, Context& context);

wgpu::ShaderModule createShaderModule(const std::string& name, const std::string& code, const Context& context);
wgpu::Sampler createLinearSampler(const Context& context);

void applyShaderTransform(const TextureBuffer& src, TextureBuffer& dst, const std::string& shaderCode, Context& context);

ComputeOperation createComputeOperation(ComputeOperationData &operationData, Context &context);

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupGrid workgroupDimensions,
                       Context& context);

[[nodiscard]] Context createWebGPUContext();

}
