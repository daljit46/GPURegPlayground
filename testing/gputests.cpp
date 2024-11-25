#include <gtest/gtest.h>
#include "gpu.h"
#include "webgpu/webgpu_cpp.h"

class GpuTest : public ::testing::Test {
protected:
    void SetUp() override{
        wgpuContext = gpu::Context::newContext();
        EXPECT_NE(wgpuContext.instance, nullptr);
        EXPECT_NE(wgpuContext.adapter, nullptr);
        EXPECT_NE(wgpuContext.device, nullptr);
    }

    gpu::Context wgpuContext;
};


TEST_F(GpuTest, MakeEmptyBuffer)
{
    const gpu::DataBuffer buffer = wgpuContext.makeEmptyBuffer(1024);
    const wgpu::Buffer& wgpuHandle = buffer.wgpuHandle;
    EXPECT_NE(wgpuHandle.Get(), nullptr);
    EXPECT_EQ(wgpuHandle.GetSize(), 1024);
    EXPECT_EQ(wgpuHandle.GetMapState(), wgpu::BufferMapState::Unmapped);
}


TEST_F(GpuTest, WriteToBuffer)
{
    const gpu::DataBuffer buffer = wgpuContext.makeEmptyBuffer(1024);

    std::vector<uint8_t> data(1024);
    std::fill(data.begin(), data.end(), 0x42);

    wgpuContext.writeToBuffer(buffer, data.data());

    const wgpu::Buffer& wgpuHandle = buffer.wgpuHandle;

    std::vector<uint8_t> downloadedData(1024);
    wgpuContext.downloadBuffer(buffer, downloadedData.data());

    EXPECT_EQ(data, downloadedData);

    // Update the buffer again, this time with random data
    std::srand(unsigned(std::time(nullptr)));
    std::vector<int8_t> randomData(1024);
    std::generate(randomData.begin(), randomData.end(), std::rand);
    wgpuContext.writeToBuffer(buffer, randomData.data());

    std::vector<int8_t> downloadedRandomData(1024);
    wgpuContext.downloadBuffer(buffer, downloadedRandomData.data());

    EXPECT_EQ(randomData, downloadedRandomData);
}
