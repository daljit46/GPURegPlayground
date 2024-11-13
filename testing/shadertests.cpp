#include <gtest/gtest.h>
#include "gpu.h"
#include <filesystem>
#include "utils.h"
#include "image.h"

class ShaderTest : public ::testing::Test {
protected:
    void SetUp() override{
        wgpuContext = gpu::Context::newContext();
        EXPECT_NE(wgpuContext.instance, nullptr);
        EXPECT_NE(wgpuContext.adapter, nullptr);
        EXPECT_NE(wgpuContext.device, nullptr);
    }

    gpu::Context wgpuContext;
};


namespace {
uint8_t getPixel(int32_t x, int32_t y, const CpuImage& image) {
    if(x < 0 || y < 0 || x >= static_cast<int32_t>(image.width) || y >= static_cast<int32_t>(image.height)) {
        return 0;
    }
    const auto index = y * image.width + x;
    return image.data[index];
}

float getBilinearInterpolatedPixel(float x, float y, const CpuImage& img) {
    const auto x0 = static_cast<int32_t>(std::floor(x));
    const auto x1 = x0 + 1;
    const auto y0 = static_cast<int32_t>(std::floor(y));
    const auto y1 = y0 + 1;

    if (x0 < 0 || x1 >= static_cast<int32_t>(img.width) || y0 < 0 || y1 >= static_cast<int32_t>(img.height)) {
        return 0.0f;
    }

    const auto p00 = getPixel(x0, y0, img);
    const auto p01 = getPixel(x0, y1, img);
    const auto p10 = getPixel(x1, y0, img);
    const auto p11 = getPixel(x1, y1, img);

    const auto dx = x - x0;
    const auto dy = y - y0;

    const auto p0 = p00 * (1 - dx) + p10 * dx;
    const auto p1 = p01 * (1 - dx) + p11 * dx;

    return p0 * (1 - dy) + p1 * dy;
}

template <typename T>
float maxDifference(const std::vector<T>& a, const std::vector<T>& b) {
    float maxDiff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const float diff = std::abs(a[i] - b[i]);
        maxDiff = std::max(maxDiff, diff);
    }
    return maxDiff;
}

template<typename T>
float meanDifference(const std::vector<T>& a, const std::vector<T>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / a.size();
}
}

TEST_F(ShaderTest, GradientX)
{
    constexpr auto shaderPath = "shaders/gradientx.wgsl";
    const auto shaderSource = Utils::readFile(shaderPath);

    const auto cpuImage =  Utils::loadFromDisk("data/brain.pgm");
    const auto gpuImage = wgpuContext.makeTextureFromHost(cpuImage);
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(cpuImage.width * cpuImage.height * sizeof(float));

    const gpu::ComputeOperationDescriptor gradientComputeOpDesc {
        .shader = {
            .name = "gradientx",
            .entryPoint = "computeSobelX",
            .code = shaderSource,
            .workgroupSize = { 16, 16, 1 }
        },
        .inputTextures = { gpuImage },
        .outputBuffers = { outputBuffer }
    };

    auto gradientComputeOp = wgpuContext.makeComputeOperation(gradientComputeOpDesc);

    wgpuContext.dispatchOperation(gradientComputeOp,
                                  {
                                      cpuImage.width / 16,
                                      cpuImage.height / 16,
                                      1
                                  });

    std::vector<float> gpuOutput(cpuImage.width * cpuImage.height);
    wgpuContext.downloadBuffer(outputBuffer, gpuOutput.data());

    // Compute the gradient in the X direction on the CPU for comparison
    std::vector<float> cpuOutput(cpuImage.width * cpuImage.height);
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const uint32_t index = y * cpuImage.width + x;
            // Apply sobel operator in the X direction
            float sum = 0.0f;
            sum += getPixel(x - 1, y - 1, cpuImage) * -1.0F;
            sum += getPixel(x + 1, y - 1, cpuImage) * 1.0F;
            sum += getPixel(x - 1, y, cpuImage) * -2.0f;
            sum += getPixel(x + 1, y, cpuImage) * 2.0f;
            sum += getPixel(x - 1, y + 1, cpuImage) * -1.0F;
            sum += getPixel(x + 1, y + 1, cpuImage) * 1.0F;
            cpuOutput[index] = sum / 255.0f;
        }
    }

    float maxDifference = 0.0f;

    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const uint32_t index = y * cpuImage.width + x;
            const float cpuValue = cpuOutput[index];
            const float gpuValue = gpuOutput[index];
            const float difference = std::abs(cpuValue - gpuValue);
            maxDifference = std::max(maxDifference, difference);
        }
    }
    EXPECT_LT(maxDifference, 1e-4F);
}

TEST_F(ShaderTest, GradientY)
{
    constexpr auto shaderPath = "shaders/gradienty.wgsl";
    const auto shaderSource = Utils::readFile(shaderPath);

    const auto cpuImage =  Utils::loadFromDisk("data/brain.pgm");
    const auto gpuImage = wgpuContext.makeTextureFromHost(cpuImage);
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(cpuImage.width * cpuImage.height * sizeof(float));

    const gpu::ComputeOperationDescriptor gradientComputeOpDesc {
        .shader = {
            .name = "gradienty",
            .entryPoint = "computeSobelY",
            .code = shaderSource,
            .workgroupSize = { 16, 16, 1 }
        },
        .inputTextures = { gpuImage },
        .outputBuffers = { outputBuffer }
    };

    auto gradientComputeOp = wgpuContext.makeComputeOperation(gradientComputeOpDesc);

    wgpuContext.dispatchOperation(gradientComputeOp,
                                  {
                                      cpuImage.width / 16,
                                      cpuImage.height / 16,
                                      1
                                  });

    std::vector<float> gpuOutput(cpuImage.width * cpuImage.height);
    wgpuContext.downloadBuffer(outputBuffer, gpuOutput.data());

    std::vector<float> cpuOutput(cpuImage.width * cpuImage.height);
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const uint32_t index = y * cpuImage.width + x;
            // Apply sobel operator in the Y direction
            float sum = 0.0f;
            sum += getPixel(x - 1, y - 1, cpuImage) * -1.0F;
            sum += getPixel(x, y - 1, cpuImage) * -2.0f;
            sum += getPixel(x + 1, y - 1, cpuImage) * -1.0F;
            sum += getPixel(x - 1, y + 1, cpuImage) * 1.0F;
            sum += getPixel(x, y + 1, cpuImage) * 2.0f;
            sum += getPixel(x + 1, y + 1, cpuImage) * 1.0F;
            cpuOutput[index] = sum / 255.0f;
        }
    }

    float maxDifference = 0.0f;
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const uint32_t index = y * cpuImage.width + x;
            const float cpuValue = cpuOutput[index];
            const float gpuValue = gpuOutput[index];
            const float difference = std::abs(cpuValue - gpuValue);
            maxDifference = std::max(maxDifference, difference);
        }
    }
    EXPECT_LT(maxDifference, 1e-4F);
}

TEST_F(ShaderTest, TransformImage)
{
    constexpr auto shaderPath = "shaders/transformimage.wgsl";
    const auto shaderSource = Utils::readFile(shaderPath);

    const auto cpuImage =  Utils::loadFromDisk("data/brain.pgm");
    const auto gpuImage = wgpuContext.makeTextureFromHost(cpuImage);
    const auto outputImage = wgpuContext.makeEmptyTexture({
        .width = cpuImage.width,
        .height = cpuImage.height,
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    struct TransformParams {
        float angle = Utils::degreesToRadians(45.0);
        float tx = 200;
        float ty = 200;
        float _padding = 0.0F; // WGSL requires to align to 16 bytes
    } uniformParams;

    const gpu::ComputeOperationDescriptor transformComputeOpDesc {
        .shader = {
            .name = "transformimage",
            .entryPoint = "computeTransform",
            .code = shaderSource,
            .workgroupSize = { 16, 16, 1 }
        },
        .uniformBuffers = { wgpuContext.makeUniformBuffer(&uniformParams, sizeof(TransformParams)) },
        .inputTextures = { gpuImage },
        .outputTextures = { outputImage },
        .samplers = { wgpuContext.makeLinearSampler() }
    };

    auto transformComputeOp = wgpuContext.makeComputeOperation(transformComputeOpDesc);
    wgpuContext.dispatchOperation(transformComputeOp, { cpuImage.width / 16, cpuImage.height / 16, 1 });

    CpuImage gpuOutputImage = wgpuContext.downloadTexture(outputImage);
    Utils::saveToDisk(gpuOutputImage, "output_gpu.pgm");
    // Compute the transformed image on the CPU for comparison
    // using linear interpolation
    std::vector<uint8_t> cpuOutput(cpuImage.width * cpuImage.height);
    const auto cosTheta = std::cos(uniformParams.angle);
    const auto sinTheta = std::sin(uniformParams.angle);
    const auto centerX = cpuImage.width / 2.0F;
    const auto centerY = cpuImage.height / 2.0F;
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const float offsetX = x - centerX;
            const float offsetY = y - centerY;
            const float transformedX = offsetX * cosTheta - offsetY * sinTheta + centerX + uniformParams.tx;
            const float transformedY = offsetX * sinTheta + offsetY * cosTheta + centerY + uniformParams.ty;

            const auto value = getBilinearInterpolatedPixel(transformedX, transformedY, cpuImage);
            const auto index = y * cpuImage.width + x;
            cpuOutput[index] = static_cast<uint8_t>(value);
        }
    }

    CpuImage cpuOutputImage = {
        .width = cpuImage.width,
        .height = cpuImage.height,
        .data = cpuOutput
    };
    Utils::saveToDisk(cpuOutputImage, "output_cpu.pgm");

    EXPECT_LT(meanDifference(cpuOutput, gpuOutputImage.data) / 255.0, 1e-2F);
}
