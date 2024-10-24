#include "gpu.h"
#include "image.h"
#include "utils.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_cpp.h>

#include <iostream>
#include <stdexcept>

namespace gpu {

namespace {
std::string parseAdapterType(wgpu::AdapterType type)
{
    switch(type) {
    case wgpu::AdapterType::DiscreteGPU: return "Discrete GPU";
    case wgpu::AdapterType::IntegratedGPU: return "Integrated GPU";
    case wgpu::AdapterType::CPU: return "CPU";
    case wgpu::AdapterType::Unknown: return "Unknown";
    default: return "Invalid";
    }
}

void printAdapterInfo(const wgpu::Adapter& adapter)
{
    std::cout << "--- Adapter Info ---\n";
    wgpu::AdapterInfo adapterInfo;
    adapter.GetInfo(&adapterInfo);

    std::cout << "Adapter vendor: " << std::string_view(adapterInfo.vendor) << '\n';
    std::cout << "Adapter type: " << parseAdapterType(adapterInfo.adapterType) << '\n';
    std::cout << "Adapter architecture: " << std::string_view(adapterInfo.architecture) << '\n';
    std::cout << "Adapter description: " << std::string_view(adapterInfo.description) << '\n';
    std::cout << "Adapter device: " << std::string_view(adapterInfo.device) << '\n';


    // Parse limits

    wgpu::SupportedLimits supportedLimits {};
    adapter.GetLimits(&supportedLimits);
    std::cout << "Adapter limits:\n";
    std::cout << "  maxTextureDimension1D: " << supportedLimits.limits.maxTextureDimension1D << '\n';
    std::cout << "  maxTextureDimension2D: " << supportedLimits.limits.maxTextureDimension2D << '\n';
    std::cout << "  maxTextureDimension3D: " << supportedLimits.limits.maxTextureDimension3D << '\n';
    std::cout << "  maxTextureArrayLayers: " << supportedLimits.limits.maxTextureArrayLayers << '\n';
    std::cout << "  maxBindGroups: " << supportedLimits.limits.maxBindGroups << '\n';
    std::cout << "  maxBindGroupsPlusVertexBuffers: " << supportedLimits.limits.maxBindGroupsPlusVertexBuffers << '\n';
    std::cout << "  maxBindingsPerBindGroup: " << supportedLimits.limits.maxBindingsPerBindGroup << '\n';
    std::cout << "  maxDynamicUniformBuffersPerPipelineLayout: " << supportedLimits.limits.maxDynamicUniformBuffersPerPipelineLayout << '\n';
    std::cout << "  maxDynamicStorageBuffersPerPipelineLayout: " << supportedLimits.limits.maxDynamicStorageBuffersPerPipelineLayout << '\n';
    std::cout << "  maxSampledTexturesPerShaderStage: " << supportedLimits.limits.maxSampledTexturesPerShaderStage << '\n';
    std::cout << "  maxSamplersPerShaderStage: " << supportedLimits.limits.maxSamplersPerShaderStage << '\n';
    std::cout << "  maxStorageBuffersPerShaderStage: " << supportedLimits.limits.maxStorageBuffersPerShaderStage << '\n';
    std::cout << "  maxStorageTexturesPerShaderStage: " << supportedLimits.limits.maxStorageTexturesPerShaderStage << '\n';
    std::cout << "  maxUniformBuffersPerShaderStage: " << supportedLimits.limits.maxUniformBuffersPerShaderStage << '\n';
    std::cout << "  maxUniformBufferBindingSize: " << supportedLimits.limits.maxUniformBufferBindingSize << '\n';
    std::cout << "  maxStorageBufferBindingSize: " << supportedLimits.limits.maxStorageBufferBindingSize << '\n';
    std::cout << "  minUniformBufferOffsetAlignment: " << supportedLimits.limits.minUniformBufferOffsetAlignment << '\n';
    std::cout << "  minStorageBufferOffsetAlignment: " << supportedLimits.limits.minStorageBufferOffsetAlignment << '\n';
    std::cout << "  maxVertexBuffers: " << supportedLimits.limits.maxVertexBuffers << '\n';
    std::cout << "  maxBufferSize: " << supportedLimits.limits.maxBufferSize << '\n';
    std::cout << "  maxVertexAttributes: " << supportedLimits.limits.maxVertexAttributes << '\n';
    std::cout << "  maxVertexBufferArrayStride: " << supportedLimits.limits.maxVertexBufferArrayStride << '\n';
    std::cout << "  maxInterStageShaderComponents: " << supportedLimits.limits.maxInterStageShaderComponents << '\n';
    std::cout << "  maxInterStageShaderVariables: " << supportedLimits.limits.maxInterStageShaderVariables << '\n';
    std::cout << "  maxColorAttachments: " << supportedLimits.limits.maxColorAttachments << '\n';
    std::cout << "  maxColorAttachmentBytesPerSample: " << supportedLimits.limits.maxColorAttachmentBytesPerSample << '\n';
    std::cout << "  maxComputeWorkgroupStorageSize: " << supportedLimits.limits.maxComputeWorkgroupStorageSize << '\n';
    std::cout << "  maxComputeInvocationsPerWorkgroup: " << supportedLimits.limits.maxComputeInvocationsPerWorkgroup << '\n';
    std::cout << "  maxComputeWorkgroupSizeX: " << supportedLimits.limits.maxComputeWorkgroupSizeX << '\n';
    std::cout << "  maxComputeWorkgroupSizeY: " << supportedLimits.limits.maxComputeWorkgroupSizeY << '\n';
    std::cout << "  maxComputeWorkgroupSizeZ: " << supportedLimits.limits.maxComputeWorkgroupSizeZ << '\n';
    std::cout << "  maxComputeWorkgroupsPerDimension: " << supportedLimits.limits.maxComputeWorkgroupsPerDimension << '\n';

    std::cout << "---------------------\n";
}

ImageBuffer createImageBufferFromHost(const Image &image,
                                      const wgpu::Device &device,
                                      const wgpu::TextureUsage& additionalFlags = {})
{
    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.format = wgpu::TextureFormat::R8Uint;
    descriptor.size = { image.width, image.height, 1};
    descriptor.sampleCount = 1;
    descriptor.viewFormatCount = 0;
    descriptor.viewFormats = nullptr;
    descriptor.usage = (
        wgpu::TextureUsage::TextureBinding | // to read texture in shaders
        wgpu::TextureUsage::CopyDst | // to upload input
        additionalFlags
        );

    auto gpuTexture = device.CreateTexture(&descriptor);
    auto queue = device.GetQueue();

    const wgpu::ImageCopyTexture copyTexture {
        .texture = gpuTexture,
        .mipLevel = 0,
        .origin = {0, 0, 0}
    };

    const wgpu::TextureDataLayout dataLayout {
        .nextInChain = nullptr,
        .offset = 0,
        .bytesPerRow = image.width,
        .rowsPerImage = image.height
    };

    queue.WriteTexture(&copyTexture,
                       image.data.data(),
                       image.data.size(),
                       &dataLayout,
                       &(descriptor.size));

    return ImageBuffer {
        .texture = gpuTexture,
        .size = descriptor.size
    };

}

}

Context createWebGPUContext()
{
    using namespace std::string_literals;

    Context context;
    wgpu::InstanceDescriptor instanceDescriptor {};
    instanceDescriptor.nextInChain = nullptr;
    // Required for using timed waits in async operations
    // e.g. for using wgpu::Instance::waitAny
    // https://webgpu-native.github.io/webgpu-headers/Asynchronous-Operations.html#Wait-Any
    instanceDescriptor.features.timedWaitAnyEnable = true;

    context.instance = wgpu::CreateInstance(&instanceDescriptor);

    wgpu::RequestAdapterOptions adapterOptions {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;

    struct RequestAdapterResult {
        WGPURequestAdapterStatus status = WGPURequestAdapterStatus_Error;
        wgpu::Adapter adapter = {};
        std::string message;
    };


    auto adapterCallback = [](WGPURequestAdapterStatus status,
                              WGPUAdapter adapter,
                              const char* message,
                              void * userdata) {
        if(status != WGPURequestAdapterStatus_Success) {
            throw std::runtime_error("Failed to create adapter: "s + message);
        }
        auto* result = static_cast<RequestAdapterResult*>(userdata);
        result->status = status;
        result->adapter = wgpu::Adapter::Acquire(adapter);
        result->message = message != nullptr ? message : "";
    };

    RequestAdapterResult adapterResult;
    context.instance.RequestAdapter(&adapterOptions, adapterCallback, &adapterResult);
    context.adapter = adapterResult.adapter;
    context.device = context.adapter.CreateDevice();


    auto onDeviceError = [](WGPUErrorType type, const char* message, void*) {
        std::cout << "Device error: " << type << "\n";
        if(message != nullptr) {
            std::cout << "Message: " << message << "\n";
        }
        std::cout << "\n";
    };

    auto onDeviceLost = [](WGPUDeviceLostReason reason, const char* message, void*) {
        std::cout << "Device lost: " << reason << "\n";
        if(message != nullptr) {
            std::cout << "Message: " << message << "\n";
        }
        std::cout << "\n";
    };

    context.device.SetUncapturedErrorCallback(onDeviceError, nullptr);
    context.device.SetDeviceLostCallback(onDeviceLost, nullptr);

    printAdapterInfo(context.adapter);

    return context;
}

ImageBuffer createEmptyImageBuffer(const wgpu::Device &device)
{

}

ImageBuffer createImageBuffer(const Image &image, const wgpu::Device &device)
{
    return createImageBufferFromHost(image, device, {});
}

ImageBuffer createReadOnlyImageBuffer(const Image &image, const wgpu::Device &device)
{
    return createImageBufferFromHost(image, device, wgpu::TextureUsage::CopySrc);
}

Image createHostImageFromBuffer(const ImageBuffer &buffer, Context &context)
{
    // WebGPU requires that the bytes per row is a multiple of 256
    auto paddedBytesPerRow = [](uint32_t width, uint32_t bytesPerPixel) {
        return (width * bytesPerPixel + 255) & ~255;
    };

    const auto stride = paddedBytesPerRow(buffer.size.width, 1);
    constexpr auto pixelSize = sizeof(uint8_t);

    wgpu::CommandEncoderDescriptor descriptor {};
    descriptor.label = "Image buffer to host encoder";

    auto encoder = context.device.CreateCommandEncoder(&descriptor);

    wgpu::BufferDescriptor outputBufferDescriptor {
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = stride * buffer.size.height * pixelSize
    };
    outputBufferDescriptor.mappedAtCreation = false;

    auto outputBuffer = context.device.CreateBuffer(&outputBufferDescriptor);

    const wgpu::ImageCopyTexture copyTexture {
        .texture = buffer.texture
    };
    const wgpu::ImageCopyBuffer copyBuffer {
        .layout = wgpu::TextureDataLayout{
            .offset = 0,
            .bytesPerRow = paddedBytesPerRow(buffer.size.width, pixelSize),
            .rowsPerImage = buffer.size.height
        },
        .buffer = outputBuffer
    };

    encoder.CopyTextureToBuffer(&copyTexture, &copyBuffer, &buffer.size);
    auto queue = context.device.GetQueue();
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    struct MapResult {
        bool ready = false;
        wgpu::Buffer buffer;
        const uint8_t* data = nullptr; // [stride * buffer.size.height
    };

    wgpu::BufferMapCallback onBufferMapped = [](WGPUBufferMapAsyncStatus status, void * userdata) {
        auto *mapResult = reinterpret_cast<MapResult*>(userdata);
        mapResult->ready = true;
        if(status == WGPUBufferMapAsyncStatus_Success) {
            const auto *const bufferData =  mapResult->buffer.GetConstMappedRange();
            mapResult->data = reinterpret_cast<const uint8_t*>(bufferData);
            mapResult->buffer.Unmap();
        }
        else {
            throw std::runtime_error("Failed to map buffer to host: " + std::to_string(status));
        }
    };

    MapResult mapResult {
        .buffer = outputBuffer
    };

    wgpu::BufferMapCallbackInfo mappingInfo {};
    mappingInfo.mode = wgpu::CallbackMode::WaitAnyOnly;
    mappingInfo.callback = onBufferMapped;
    mappingInfo.userdata = reinterpret_cast<void*>(&mapResult);

    auto bufferMapped = outputBuffer.MapAsync(wgpu::MapMode::Read,
                                              0,
                                              outputBuffer.GetSize(),
                                              mappingInfo
                                              );

    // Wait for mapping to finish
    context.instance.WaitAny(bufferMapped, std::numeric_limits<uint64_t>::max());

    // Copy the data to a new vector with the correct width
    std::vector<uint8_t> data;
    data.reserve(buffer.size.width * buffer.size.height);

    for(uint32_t y = 0; y < buffer.size.height; ++y) {
        const auto rowStart = mapResult.data + y * stride;
        data.insert(data.end(), rowStart, rowStart + buffer.size.width);
    }

    Image image;
    image.width = buffer.size.width;
    image.height = buffer.size.height;
    image.data = std::move(data);

    return image;
}

void applyShaderTransform(const ImageBuffer &src, ImageBuffer &dst, const std::string &shaderCode, Context &context)
{

}

wgpu::ShaderModule createShaderModule(const std::string &name, const std::string &code, const Context &context)
{
    wgpu::ShaderModuleWGSLDescriptor wgslDescriptor {};
    wgslDescriptor.code = code.c_str();
    wgpu::ShaderModuleDescriptor descriptor {};
    descriptor.nextInChain = &wgslDescriptor;
    descriptor.label = name.c_str();

    return context.device.CreateShaderModule(&descriptor);
}

ComputeOperation createComputeOperation(ComputeOperationData &data,
                                        Context &context)
{
    // Create BindGroupLayout with all input and output buffers
    std::vector<wgpu::BindGroupLayoutEntry> layoutEntries;
    layoutEntries.reserve(data.inputImageBuffers.size() +
                          data.inputBuffers.size() +
                          data.outputImageBuffers.size() +
                          data.outputBuffers.size()
                          );

    uint8_t bindingIndex = 0;
    for(const auto& imageBuffer : data.inputImageBuffers) {
        const wgpu::BindGroupLayoutEntry entry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout { .type = wgpu::BufferBindingType::ReadOnlyStorage} ,
            .storageTexture = {
                .access = wgpu::StorageTextureAccess::WriteOnly,
                .format = wgpu::TextureFormat::R8Uint,
                .viewDimension = wgpu::TextureViewDimension::e2D
            }
        };
        layoutEntries.push_back(entry);
    }

    for(const auto& buffer: data.inputBuffers) {
        const wgpu::BindGroupLayoutEntry entry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout { .type = wgpu::BufferBindingType::ReadOnlyStorage }
        };

        layoutEntries.push_back(entry);
    }

    for(const auto& imageBuffer: data.outputImageBuffers) {
        const wgpu::BindGroupLayoutEntry entry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout { .type = wgpu::BufferBindingType::Storage },
            .storageTexture = {
                .access = wgpu::StorageTextureAccess::WriteOnly,
                .format = wgpu::TextureFormat::R8Uint,
                .viewDimension = wgpu::TextureViewDimension::e2D
            }
        };
        layoutEntries.push_back(entry);
    }

    for(const auto& buffer: data.outputBuffers) {
        const wgpu::BindGroupLayoutEntry entry{
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout {.type = wgpu::BufferBindingType::Storage }
        };
        layoutEntries.push_back(entry);
    }

    const auto layoutDescLabel = data.shader.name + " layout descriptor";
    const wgpu::BindGroupLayoutDescriptor bindGroupLayoutDescriptor {
        .label = layoutDescLabel.c_str(),
        .entryCount = layoutEntries.size(),
        .entries = layoutEntries.data()
    };


    const wgpu::BindGroupLayout bindGroupLayout = context.device.CreateBindGroupLayout(&bindGroupLayoutDescriptor);

    // Create compute pipeline
    const wgpu::PipelineLayoutDescriptor pipelineLayoutDescriptor {
        .bindGroupLayoutCount = 1,
        .bindGroupLayouts = &bindGroupLayout
    };
    const wgpu::PipelineLayout pipelineLayout = context.device.CreatePipelineLayout(&pipelineLayoutDescriptor);

    const auto computePipelineLabel = data.shader.name + " compute pipeline";
    const wgpu::ComputePipelineDescriptor computePipelineDescriptor {
        .label = computePipelineLabel.c_str(),
        .layout = pipelineLayout,
        .compute = wgpu::ProgrammableStageDescriptor {
            .module = createShaderModule(data.shader.name, data.shader.code, context),
            .entryPoint = data.shader.entryPoint.c_str()
        },
    };

    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = bindGroupLayout,
            .entryCount = 0,
            .entries = nullptr
    };

    return ComputeOperation {
        .pipeline = context.device.CreateComputePipeline(&computePipelineDescriptor),
        .bindGroup = context.device.CreateBindGroup(&bindGroupDescriptor)
    };
}

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupDimensions workgroupDimensions,
                       Context& context)
{
    wgpu::CommandEncoder encoder = context.device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(operation.pipeline);
    pass.SetBindGroup(0, operation.bindGroup);
    pass.DispatchWorkgroups(workgroupDimensions.x, workgroupDimensions.y, workgroupDimensions.z);
    pass.End();

    auto commands = encoder.Finish();
    auto queue = context.device.GetQueue();
    queue.Submit(1, &commands);
}

}
