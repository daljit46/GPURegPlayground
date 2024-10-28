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

wgpu::TextureUsage convertUsageToWGPU(TextureUsage usage)
{
    switch(usage) {
    case TextureUsage::ReadOnly: {
        return wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    }
    case TextureUsage::ReadWrite: {
        return wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::CopyDst;
    }
    default: return wgpu::TextureUsage::None;
    }
}

wgpu::TextureFormat convertFormatToWGPU(TextureFormat format)
{
    switch(format) {
        case TextureFormat::R8Unorm: return wgpu::TextureFormat::R8Unorm;
        case TextureFormat::R32Float: return wgpu::TextureFormat::R32Float;
        case TextureFormat::RGBA8Unorm: return wgpu::TextureFormat::RGBA8Unorm;
        default: return wgpu::TextureFormat::Undefined;
    }
}

TextureBuffer createImageBufferFromHost(const Image &image,
                                        Context &context,
                                        const wgpu::TextureUsage& additionalFlags = {}
                                        )
{
    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.format = wgpu::TextureFormat::R8Unorm;
    descriptor.size = { image.width, image.height, 1};
    descriptor.sampleCount = 1;
    descriptor.viewFormatCount = 0;
    descriptor.viewFormats = nullptr;
    descriptor.usage = (
        wgpu::TextureUsage::TextureBinding | // to read texture in shaders
        wgpu::TextureUsage::CopyDst | // to be used as destination in copy operations
        additionalFlags
        );

    auto gpuTexture = context.device.CreateTexture(&descriptor);
    auto queue = context.device.GetQueue();

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

    return TextureBuffer {
        .texture = gpuTexture,
        .size = descriptor.size
    };

}

}

Context createWebGPUContext()
{
    using namespace std::string_literals;

    Context context;
    std::array<const char*, 1> enabledTogglesArray = { "allow_unsafe_apis" };

    wgpu::DawnTogglesDescriptor dawnToggles {};
    dawnToggles.enabledToggles = enabledTogglesArray.data();
    dawnToggles.enabledToggleCount = 1;

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
    std::vector<wgpu::FeatureName> requiredFeatures = {
        wgpu::FeatureName::R8UnormStorage
    };
    wgpu::DeviceDescriptor deviceDescriptor {};
    deviceDescriptor.nextInChain = &dawnToggles;
    deviceDescriptor.requiredFeatures = requiredFeatures.data();
    deviceDescriptor.requiredFeatureCount = requiredFeatures.size();

    context.device = context.adapter.CreateDevice(&deviceDescriptor);


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

TextureBuffer makeEmptyTextureBuffer(const TextureSpecification &spec, Context &context)
{
    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.format = convertFormatToWGPU(spec.format);
    descriptor.size = { spec.width, spec.height, 1};
    descriptor.sampleCount = 1;
    descriptor.viewFormatCount = 0;
    descriptor.viewFormats = nullptr;
    descriptor.usage = convertUsageToWGPU(spec.usage);

    return TextureBuffer {
        .texture = context.device.CreateTexture(&descriptor),
        .size = descriptor.size
    };
}

TextureBuffer makeTextureBufferFromHost(const Image &image, Context &context)
{
    return createImageBufferFromHost(image, context, {});
}

TextureBuffer makeReadOnlyTextureBuffer(const Image &image, Context &context)
{
    return createImageBufferFromHost(image, context, wgpu::TextureUsage::CopySrc);
}

Image makeHostImageFromBuffer(const TextureBuffer &buffer, Context &context)
{
    // WebGPU requires that the bytes per row is a multiple of 256
    auto paddedBytesPerRow = [](uint32_t width, uint32_t bytesPerPixel) {
        return (width * bytesPerPixel + 255) & ~255;
    };

    constexpr auto pixelSize = sizeof(uint8_t);
    const auto stride = paddedBytesPerRow(buffer.size.width, pixelSize);

    wgpu::CommandEncoderDescriptor descriptor {};
    descriptor.label = "Image buffer to host encoder";

    auto encoder = context.device.CreateCommandEncoder(&descriptor);

    wgpu::BufferDescriptor outputBufferDescriptor {
        .usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
        .size = uint64_t{stride * buffer.size.height}
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
            if(bufferData == nullptr) {
                throw std::runtime_error("Failed to get mapped range of buffer");
            }
            mapResult->data = reinterpret_cast<const uint8_t*>(bufferData);
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
        data.insert(data.end(), rowStart, rowStart + buffer.size.width * pixelSize);
    }

    Image image;
    image.width = buffer.size.width;
    image.height = buffer.size.height;
    image.data = std::move(data);


    mapResult.buffer.Unmap();

    return image;
}

void applyShaderTransform(const TextureBuffer &src, TextureBuffer &dst, const std::string &shaderCode, Context &context)
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
    std::vector<wgpu::BindGroupEntry> bindGroupEntries;
    layoutEntries.reserve(data.inputImageBuffers.size() +
                          data.inputBuffers.size() +
                          data.outputImageBuffers.size() +
                          data.outputBuffers.size()
                          );

    uint8_t bindingIndex = 0;
    for(const auto& imageBuffer : data.inputImageBuffers) {
        const wgpu::BindGroupLayoutEntry layoutEntry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .texture = wgpu::TextureBindingLayout {
                .sampleType = wgpu::TextureSampleType::Float,
                .viewDimension = wgpu::TextureViewDimension::e2D
            }
        };

        const wgpu::BindGroupEntry bindGroupEntry {
            .binding = layoutEntry.binding,
            .textureView = imageBuffer.texture.CreateView()
        };
        layoutEntries.push_back(layoutEntry);
        bindGroupEntries.push_back(bindGroupEntry);
    }

    for(const auto& buffer: data.inputBuffers) {
        const wgpu::BindGroupLayoutEntry entry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout { .type = wgpu::BufferBindingType::ReadOnlyStorage }
        };

        const wgpu::BindGroupEntry bindGroupEntry {
            .binding = entry.binding,
            .buffer = buffer.buffer
        };

        layoutEntries.push_back(entry);
        bindGroupEntries.push_back(bindGroupEntry);
    }

    for(const auto& imageBuffer: data.outputImageBuffers) {
        const wgpu::BindGroupLayoutEntry entry {
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .storageTexture = {
                .access = wgpu::StorageTextureAccess::WriteOnly,
                .format = imageBuffer.texture.GetFormat(),
                .viewDimension = wgpu::TextureViewDimension::e2D
            }
        };

        const wgpu::BindGroupEntry bindGroupEntry {
            .binding = entry.binding,
            .textureView = imageBuffer.texture.CreateView()
        };

        layoutEntries.push_back(entry);
        bindGroupEntries.push_back(bindGroupEntry);
    }

    for(const auto& buffer: data.outputBuffers) {
        const wgpu::BindGroupLayoutEntry entry{
            .binding = bindingIndex++,
            .visibility = wgpu::ShaderStage::Compute,
            .buffer = wgpu::BufferBindingLayout {.type = wgpu::BufferBindingType::Storage }
        };

        const wgpu::BindGroupEntry bindGroupEntry {
            .binding = entry.binding,
            .buffer = buffer.buffer
        };

        layoutEntries.push_back(entry);
        bindGroupEntries.push_back(bindGroupEntry);
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

    const std::string workgroupSizeStr = std::to_string(data.shader.workgroupSize.x) + ", " +
                                         std::to_string(data.shader.workgroupSize.y) + ", " +
                                         std::to_string(data.shader.workgroupSize.z);
    const auto shaderCode = Utils::replacePlaceholder(data.shader.code, "workgroup_size", workgroupSizeStr);
    const auto computePipelineLabel = data.shader.name + " compute pipeline";
    const wgpu::ComputePipelineDescriptor computePipelineDescriptor {
        .label = computePipelineLabel.c_str(),
        .layout = pipelineLayout,
        .compute = wgpu::ProgrammableStageDescriptor {
            .module = createShaderModule(data.shader.name, shaderCode, context),
            .entryPoint = data.shader.entryPoint.c_str()
        },
    };

    for(const auto& bindGroupLayoutEntry : layoutEntries) {
        const wgpu::BindGroupEntry entry {
            .binding = bindGroupLayoutEntry.binding,
        };
    }

    const wgpu::BindGroupDescriptor bindGroupDescriptor{
        .layout = bindGroupLayout,
        .entryCount = bindGroupEntries.size(),
        .entries = bindGroupEntries.data()
    };

    return ComputeOperation {
        .pipeline = context.device.CreateComputePipeline(&computePipelineDescriptor),
        .bindGroup = context.device.CreateBindGroup(&bindGroupDescriptor)
    };
}

void dispatchOperation(const ComputeOperation& operation,
                       WorkgroupGrid workgroupDimensions,
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
