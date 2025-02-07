#pragma once

#include "image.h"
#include <cstddef>
#include <filesystem>
#include <map>
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

    uint32_t totalCount() const { return x * y * z; }
};

struct WorkgroupGrid {
    uint32_t x = 1;
    uint32_t y = 1;
    uint32_t z = 1;

    uint32_t totalCount() const { return x * y * z; }
    // Assumes that each thread will process one work unit of the total work size per axis
    static WorkgroupGrid ForOneWorkUnitPerThread(uint32_t totalWorksizeX,
                                                 uint32_t totalWorksizeY,
                                                 uint32_t totalWorksizeZ,
                                                 const WorkgroupSize& workgroupSize);
};

struct ShaderEntry {
    std::string name;
    std::string entryPoint = "main";
    std::filesystem::path filePath;
    WorkgroupSize workgroupSize;
    // When creating a kernel, placeholders in the shader code enclosed in {{}} will be
    // replaced with the valuesv from this map. By default the placeholder {{workgroup_size}}
    // will be replaced with the workgroup size in the shader code.
    std::map<std::string, std::string> placeHolders;
};

struct Kernel {
    std::string name;
    wgpu::ComputePipeline pipeline;
    wgpu::BindGroup bindGroup;
    DataBuffer timestampResolveBuffer;
    WorkgroupSize workgroupSize;
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

struct Limits {
    uint32_t maxStoragePerWorkgroup = 0; // in bytes
    uint32_t maxWorkgroupCountX = 0;
    uint32_t maxWorkgroupCountY = 0;
    uint32_t maxWorkgroupCountZ = 0;
};

struct Context {
    wgpu::Instance instance = nullptr;
    wgpu::Adapter adapter = nullptr;
    wgpu::Device device = nullptr;
    Limits limits;

    [[nodiscard]] static Context newContext();

    Texture makeEmptyTexture(const TextureSpecification& spec) const;
    Texture makeTextureFromHostPgm(const PgmImage& image) const;
    Texture makeTextureFromHostNifti(const NiftiImage& image) const;
    DataBuffer makeEmptyBuffer(size_t size) const;
    DataBuffer makeUniformBuffer(const void* data, size_t size) const;
    DataBuffer makeIndirectDispatchBuffer() const;

    void downloadTexture(const Texture& buffer, void *data) const;
    void downloadBuffer(const DataBuffer& dataBuffer, void *data) const;

    using BufferMappingPair = std::pair<const DataBuffer *, void *>;
    void downloadBuffers(const std::vector<BufferMappingPair> &bufferMappingPairs) const;
    void writeToBuffer(const DataBuffer& dataBuffer, const void *data) const;

    wgpu::Sampler makeLinearSampler() const;

    Kernel makeKernel(const KernelDescriptor &kernelDescriptor) const;
    void dispatchKernel(const Kernel& kernel, WorkgroupGrid workgroupDimensions) const;
    void dispatchKernelIndirect(const Kernel& kernel, const DataBuffer& indirectBuffer) const;
    void updateUniformBuffer(const void *data, const DataBuffer &buffer, size_t size) const;

    // Completion handler must be alive until the operation is completed
    void waitForAllQueueOperations() const;
};

}
