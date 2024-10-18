#include "wgpucontext.h"

#include <webgpu/webgpu_cpp.h>
#include <iostream>
#include <stdexcept>

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

    std::cout << "Adapter vendor: " << std::string_view(adapterInfo.vendor) << std::endl;
    std::cout << "Adapter type: " << parseAdapterType(adapterInfo.adapterType) << std::endl;
    std::cout << "Adapter architecture: " << std::string_view(adapterInfo.architecture) << std::endl;
    std::cout << "Adapter description: " << std::string_view(adapterInfo.description) << std::endl;
    std::cout << "Adapter device: " << std::string_view(adapterInfo.device) << std::endl;
}
}

WGPUContext createWebGPUContext()
{
    using namespace std::string_literals;

    WGPUContext context;
    context.instance = wgpu::CreateInstance();

    wgpu::RequestAdapterOptions adapterOptions {};
    adapterOptions.powerPreference = wgpu::PowerPreference::HighPerformance;

    struct RequestAdapterResult {
        WGPURequestAdapterStatus status;
        wgpu::Adapter adapter;
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
        result->message = message ? message : "";
    };

    RequestAdapterResult adapterResult;
    context.instance.RequestAdapter(&adapterOptions, adapterCallback, &adapterResult);
    context.adapter = adapterResult.adapter;
    context.device = context.adapter.CreateDevice();
    context.queue = context.device.GetQueue();


    auto onDeviceError = [](WGPUErrorType type, const char* message, void*) {
        std::cout << "Device error: " << type << std::endl;
        if(message) {
            std::cout << "Message: " << message << std::endl;
        }
        std::cout << std::endl;
    };

    auto onDeviceLost = [](WGPUDeviceLostReason reason, const char* message, void*) {
        std::cout << "Device lost: " << reason << std::endl;
        if(message) {
            std::cout << "Message: " << message << std::endl;
        }
        std::cout << std::endl;
    };

    context.device.SetUncapturedErrorCallback(onDeviceError, nullptr);
    context.device.SetDeviceLostCallback(onDeviceLost, nullptr);

    printAdapterInfo(context.adapter);

    return context;
}
