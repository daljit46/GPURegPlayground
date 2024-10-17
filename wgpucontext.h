#pragma once

#include <webgpu/webgpu_cpp.h>


struct WGPUContext {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
};

[[nodiscard]] WGPUContext createWebGPUContext();
