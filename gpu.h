#pragma once

#include "image.h"
#include <cstddef>
#include <utility>
#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <string>
#include <vector>

class PgmImage;
class NiftiImage;

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

struct TextureSize {
    uint32_t width = 0;
    uint32_t height = 1;
    uint32_t depth = 1;
};

struct TextureSpecification {
    TextureSize size = {};
    TextureFormat format = TextureFormat::R8Unorm;
    ResourceUsage usage = ResourceUsage::ReadOnly;
};

struct Texture {
    wgpu::Texture wgpuHandle;
    TextureSize size = {};
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

    // Assumes that each thread will process one work unit of the total work size per axis
    static WorkgroupGrid ForOneWorkUnitPerThread(uint32_t totalWorksizeX,
                                                 uint32_t totalWorksizeY,
                                                 uint32_t totalWorksizeZ,
                                                 const WorkgroupSize& workgroupSize);
};

struct ShaderEntry {
    std::string name;
    std::string entryPoint = "main";
    std::string code;
    WorkgroupSize workgroupSize;
};

struct Kernel {
    std::string name;
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
    DataBuffer timestampResolveBuffer;
};

struct KernelDescriptor {
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

    Texture makeEmptyTexture(const TextureSpecification& spec) const;
    Texture makeTextureFromHostPgm(const PgmImage& image);
    Texture makeTextureFromHostNifti(const NiftiImage& image);
    DataBuffer makeEmptyBuffer(size_t size);
    DataBuffer makeUniformBuffer(const void* data, size_t size);

    void downloadTexture(const Texture& buffer, void *data);
    void downloadBuffer(const DataBuffer& dataBuffer, void *data);
    void downloadBuffers(const std::vector<std::pair<DataBuffer*, void*>>& bufferMappingPairs);
    void writeToBuffer(const DataBuffer& dataBuffer, void* data) const;

    wgpu::ShaderModule makeShaderModule(const std::string& name, const std::string& code);
    wgpu::Sampler makeLinearSampler();

    Kernel makeKernel(const KernelDescriptor &kernelDescriptor);
    void dispatchKernel(const Kernel& kernel, WorkgroupGrid workgroupDimensions);

    void updateUniformBuffer(const void *data, const DataBuffer &buffer, size_t size);

    // Completion handler must be alive until the operation is completed
    void waitForAllQueueOperations();
};

}
