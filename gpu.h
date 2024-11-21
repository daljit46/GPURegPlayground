#pragma once

#include <cstddef>
#include <optional>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class CpuImage;

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

struct Texture {
    wgpu::Texture wgpuHandle;
    wgpu::Extent3D size;
};

enum class BufferType {
    Uniform,
    Storage
};

struct DataBuffer {
    wgpu::Buffer wgpuHandle;
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
    std::string name;
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
    DataBuffer timestampResolveBuffer;
};

struct ComputeOperationDescriptor {
    // Buffers in shader must be specified in the same order
    // as they are passed in this struct
    ShaderEntry shader;
    std::vector<DataBuffer> uniformBuffers;
    std::vector<DataBuffer> inputBuffers;
    std::vector<Texture> inputTextures;
    std::vector<DataBuffer> outputBuffers;
    std::vector<Texture> outputTextures;
    std::vector<wgpu::Sampler> samplers;
};


struct Context {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;

    [[nodiscard]] static Context newContext();

    Texture makeEmptyTexture(const TextureSpecification& spec);
    Texture makeTextureFromHost(const CpuImage& image);
    CpuImage downloadTexture(const Texture& buffer);

    DataBuffer makeEmptyBuffer(size_t size);
    DataBuffer makeUniformBuffer(const void* data, size_t size);
    void downloadBuffer(const DataBuffer& dataBuffer, void *data);
    void downloadBuffers(const std::vector<std::pair<DataBuffer*, void*>>& bufferMappingPairs);
    void writeToBuffer(const DataBuffer& dataBuffer, void* data);

    wgpu::ShaderModule makeShaderModule(const std::string& name, const std::string& code);
    wgpu::Sampler makeLinearSampler();

    ComputeOperation makeComputeOperation(const ComputeOperationDescriptor &operationDescriptor);
    void dispatchOperation(const ComputeOperation& operation, WorkgroupGrid workgroupDimensions);

    void updateUniformBuffer(const void *data, const DataBuffer &buffer, size_t size);

    using CompletionHandler = std::function<void()>;
    // Completion handler must be alive until the operation is completed
    void waitForAllQueueOperations();
};

}
