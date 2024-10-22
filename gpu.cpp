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
    wgpu::AdapterInfo adapterInfo;
    adapter.GetInfo(&adapterInfo);

    std::cout << "Adapter vendor: " << std::string_view(adapterInfo.vendor) << '\n';
    std::cout << "Adapter type: " << parseAdapterType(adapterInfo.adapterType) << '\n';
    std::cout << "Adapter architecture: " << std::string_view(adapterInfo.architecture) << '\n';
    std::cout << "Adapter description: " << std::string_view(adapterInfo.description) << '\n';
    std::cout << "Adapter device: " << std::string_view(adapterInfo.device) << '\n';
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
    context.queue = context.device.GetQueue();


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
        const uint8_t* data; // [stride * buffer.size.height
    };

    wgpu::BufferMapCallback onBufferMapped = [](WGPUBufferMapAsyncStatus status, void * userdata) {
        auto *mapResult = reinterpret_cast<MapResult*>(userdata);
        mapResult->ready = true;
        if(status == WGPUBufferMapAsyncStatus_Success) {
            auto bufferData =  mapResult->buffer.GetConstMappedRange();
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

    // The data is now in mapResult.data
    // This is padded to 256 bytes per row

    // Copy the data to a new vector with the correct width
    std::vector<uint8_t> data;
    data.reserve(buffer.size.width * buffer.size.height);

    for(uint32_t y = 0; y < buffer.size.height; ++y) {
        auto rowStart = mapResult.data + y * stride;
        data.insert(data.end(), rowStart, rowStart + buffer.size.width);
    }

    Image image;
    image.width = buffer.size.width;
    image.height = buffer.size.height;
    image.data = std::move(data);

    return image;
}


}
