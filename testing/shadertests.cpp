#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>

#include <iterator>
#include <limits>
#include <nifti1.h>
#include <nifti1_io.h>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <vector>
#include "gpu.h"
#include "spdlog/spdlog.h"
#include "transform.h"
#include "utils.h"
#include "image.h"
#include "reduce.h"

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
uint8_t getPixel(int32_t x, int32_t y, const PgmImage& image) {
    if(x < 0 || y < 0 || x >= static_cast<int32_t>(image.width) || y >= static_cast<int32_t>(image.height)) {
        return 0;
    }
    const auto index = y * image.width + x;
    return image.data[index];
}


float getBilinearInterpolatedPixel(float x, float y, const PgmImage& img) {
    const auto x0 = static_cast<int32_t>(std::floor(x));
    const auto x1 = x0 + 1;
    const auto y0 = static_cast<int32_t>(std::floor(y));
    const auto y1 = y0 + 1;

    if (x0 < 0 || x1 >= static_cast<int32_t>(img.width) || y0 < 0 || y1 >= static_cast<int32_t>(img.height)) {
        return 0.0F;
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

uint8_t getPixel3D(int32_t x, int32_t y, int32_t z, const NiftiImage& image) {
    if(x < 0 || y < 0 || z < 0 || x >= static_cast<int32_t>(image.width) || y >= static_cast<int32_t>(image.height) || z >= static_cast<int32_t>(image.depth)) {
        return 0;
    }
    const auto index = z * image.width * image.height + y * image.width + x;
    return image.at(static_cast<size_t>(index));
}
float getBilinearInterpolatedPixel3D(float x, float y, float z, const NiftiImage& img)
{
    const auto x0 = static_cast<int32_t>(std::floor(x));
    const auto x1 = x0 + 1;
    const auto y0 = static_cast<int32_t>(std::floor(y));
    const auto y1 = y0 + 1;
    const auto z0 = static_cast<int32_t>(std::floor(z));
    const auto z1 = z0 + 1;

    if (x0 < 0 || x1 >= static_cast<int32_t>(img.width) || y0 < 0 || y1 >= static_cast<int32_t>(img.height) || z0 < 0 || z1 >= static_cast<int32_t>(img.depth)) {
        return 0.0F;
    }

    const auto p000 = getPixel3D(x0, y0, z0, img);
    const auto p001 = getPixel3D(x0, y0, z1, img);
    const auto p010 = getPixel3D(x0, y1, z0, img);
    const auto p011 = getPixel3D(x0, y1, z1, img);
    const auto p100 = getPixel3D(x1, y0, z0, img);
    const auto p101 = getPixel3D(x1, y0, z1, img);
    const auto p110 = getPixel3D(x1, y1, z0, img);
    const auto p111 = getPixel3D(x1, y1, z1, img);

    const auto dx = x - x0;
    const auto dy = y - y0;
    const auto dz = z - z0;

    const auto p00 = p000 * (1 - dx) + p100 * dx;
    const auto p01 = p001 * (1 - dx) + p101 * dx;
    const auto p10 = p010 * (1 - dx) + p110 * dx;
    const auto p11 = p011 * (1 - dx) + p111 * dx;

    const auto p0 = p00 * (1 - dy) + p10 * dy;
    const auto p1 = p01 * (1 - dy) + p11 * dy;

    return p0 * (1 - dz) + p1 * dz;
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

template<typename T>
float ssd(const std::vector<T>& a, const std::vector<T>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}
}

TEST_F(ShaderTest, GradientX)
{
    const auto cpuImage =  Utils::loadFromDisk("data/brain.pgm");
    const auto gpuImage = wgpuContext.makeTextureFromHostPgm(cpuImage);
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(cpuImage.width * cpuImage.height * sizeof(float));

    const gpu::KernelDescriptor gradientKernelDesc {
        .shader = {
            .name = "gradientx",
            .entryPoint = "computeSobelX",
            .filePath = "shaders/gradientx.wgsl",
            .workgroupSize = { 16, 16, 1 }
        },
        .inputTextures = { gpuImage },
        .outputBuffers = { outputBuffer }
    };

    auto gradientKernel = wgpuContext.makeKernel(gradientKernelDesc);

    wgpuContext.dispatchKernel(gradientKernel,
                               {
                                   cpuImage.width / 16,
                                   cpuImage.height / 16,
                                   1
                               });

    std::vector<float> gpuOutput(static_cast<float>(cpuImage.width) * cpuImage.height);
    wgpuContext.downloadBuffer(outputBuffer, gpuOutput.data());

    // Compute the gradient in the X direction on the CPU for comparison
    std::vector<float> cpuOutput(cpuImage.width * cpuImage.height);
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const uint32_t index = y * cpuImage.width + x;
            // Apply sobel operator in the X direction
            float sum = 0.0f;
            sum += static_cast<float>(getPixel(x - 1, y - 1, cpuImage) * -1.0F);
            sum += static_cast<float>(getPixel(x + 1, y - 1, cpuImage) * 1.0F);
            sum += static_cast<float>(getPixel(x - 1, y, cpuImage) * -2.0F);
            sum += static_cast<float>(getPixel(x + 1, y, cpuImage) * 2.0F);
            sum += static_cast<float>(getPixel(x - 1, y + 1, cpuImage) * -1.0F);
            sum += static_cast<float>(getPixel(x + 1, y + 1, cpuImage) * 1.0F);
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
    const auto gpuImage = wgpuContext.makeTextureFromHostPgm(cpuImage);
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(cpuImage.width * cpuImage.height * sizeof(float));

    const gpu::KernelDescriptor gradientKernelDesc {
        .shader = {
            .name = "gradienty",
            .entryPoint = "computeSobelY",
            .filePath = shaderPath,
            .workgroupSize = { 16, 16, 1 }
        },
        .inputTextures = { gpuImage },
        .outputBuffers = { outputBuffer }
    };

    auto gradientKernel = wgpuContext.makeKernel(gradientKernelDesc);

    wgpuContext.dispatchKernel(gradientKernel,
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
            float sum = 0.0F;
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
    const auto gpuImage = wgpuContext.makeTextureFromHostPgm(cpuImage);
    const auto outputImage = wgpuContext.makeEmptyTexture({
        .size = { cpuImage.width, cpuImage.height, 1},
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    struct TransformParams {
        float angle = Utils::degreesToRadians(15.0);
        float tx = 100;
        float ty = 100;
        float _padding = 0.0F; // WGSL requires to align to 16 bytes
    } uniformParams;

    const gpu::KernelDescriptor transformKernelDesc {
        .shader = {
            .name = "transformimage",
            .entryPoint = "computeTransform",
            .filePath = shaderPath,
            .workgroupSize = { 16, 16, 1 }
        },
        .uniformBuffers = { wgpuContext.makeUniformBuffer(&uniformParams, sizeof(TransformParams)) },
        .inputTextures = { gpuImage },
        .outputTextures = { outputImage },
        .samplers = { wgpuContext.makeLinearSampler() }
    };

    auto transformKernel = wgpuContext.makeKernel(transformKernelDesc);
    wgpuContext.dispatchKernel(transformKernel, { cpuImage.width / 16, cpuImage.height / 16, 1 });

    PgmImage gpuOutputImage {
        .width = cpuImage.width,
        .height = cpuImage.height,
        .data = std::vector<uint8_t>(cpuImage.width * cpuImage.height)
    };
    wgpuContext.downloadTexture(outputImage, gpuOutputImage.data.data());
    Utils::saveToDisk(gpuOutputImage, "output_gpu.pgm");
    // Compute the transformed image on the CPU for comparison
    // using linear interpolation
    std::vector<uint8_t> cpuOutput(cpuImage.width * cpuImage.height);
    const auto cosTheta = std::cos(uniformParams.angle);
    const auto sinTheta = std::sin(uniformParams.angle);
    for (size_t y = 0; y < cpuImage.height; y++) {
        for (size_t x = 0; x < cpuImage.width; x++) {
            const float transformedX = x * cosTheta - y * sinTheta + uniformParams.tx;
            const float transformedY = x * sinTheta + y * cosTheta + uniformParams.ty;

            const auto value = getBilinearInterpolatedPixel(transformedX, transformedY, cpuImage);
            const auto index = y * cpuImage.width + x;
            cpuOutput[index] = static_cast<uint8_t>(value);
        }
    }

    PgmImage cpuOutputImage = {
        .width = cpuImage.width,
        .height = cpuImage.height,
        .data = cpuOutput
    };
    Utils::saveToDisk(cpuOutputImage, "output_gpu.pgm");
    Utils::saveToDisk(cpuOutputImage, "output_cpu.pgm");

    EXPECT_LT(meanDifference(cpuOutput, gpuOutputImage.data) / 255.0, 0.2F);
}

TEST_F(ShaderTest, TransformImage3D)
{
    constexpr auto shaderPath = "shaders/3d/transformimage_3d.wgsl";
    const auto shaderSource = Utils::readFile(shaderPath);

    const auto cpuImage =  Utils::loadNiftiFromDisk("data/test_file.nii");
    const auto gpuImage = wgpuContext.makeTextureFromHostNifti(cpuImage);

    const auto gpuOutputImage = wgpuContext.makeEmptyTexture({
        .size = { cpuImage.width, cpuImage.height, cpuImage.depth },
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    struct Uniforms {
        float alpha = Utils::degreesToRadians(0.0); // rotation about z-axis
        float beta = Utils::degreesToRadians(0.0); // rotation about y-axis
        float gamma = Utils::degreesToRadians(0.0); // rotation about x-axis
        float tx = 0;
        float ty = 0;
        float tz = -20;
        std::array<float, 2> _padding; // WGSL requires to align to 16 bytes
    } uniformParams;

    const gpu::KernelDescriptor transformKernelDesc {
        .shader = {
            .name = "transformimage3d",
            .filePath = shaderPath,
            .workgroupSize = { 4, 4, 4 }
        },
        .uniformBuffers = { wgpuContext.makeUniformBuffer(&uniformParams, sizeof(Uniforms)) },
        .inputTextures = { gpuImage },
        .outputTextures = { gpuOutputImage },
        .samplers = { wgpuContext.makeLinearSampler() }
    };

    auto transformKernel = wgpuContext.makeKernel(transformKernelDesc);
    wgpuContext.dispatchKernel(transformKernel, {
                                                    cpuImage.width + 3 / 4,
                                                    cpuImage.height + 3 / 4,
                                                    cpuImage.depth + 3/ 4
                                                });


    std::vector<uint8_t> gpuOutputData(cpuImage.width * cpuImage.height * cpuImage.depth);
    wgpuContext.downloadTexture(gpuOutputImage, gpuOutputData.data());

    // Compute the transformation on the CPU
    std::vector<uint8_t> cpuOutput(cpuImage.width * cpuImage.height * cpuImage.depth);
    const auto cosAlpha = std::cos(uniformParams.alpha);
    const auto sinAlpha = std::sin(uniformParams.alpha);
    const auto cosBeta = std::cos(uniformParams.beta);
    const auto sinBeta = std::sin(uniformParams.beta);
    const auto cosGamma = std::cos(uniformParams.gamma);
    const auto sinGamma = std::sin(uniformParams.gamma);

    for(size_t z = 0; z < cpuImage.depth; z++) {
        for(size_t y = 0; y < cpuImage.height; y++) {
            for(size_t x = 0; x < cpuImage.width; x++) {
                const float transformedX = cosAlpha * cosBeta * x + (cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma) * y + (cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma) * z + uniformParams.tx;
                const float transformedY = sinAlpha * cosBeta * x + (sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma) * y + (sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma) * z + uniformParams.ty;
                const float transformedZ = -sinBeta * x + cosBeta * sinGamma * y + cosBeta * cosGamma * z + uniformParams.tz;

                const auto value = getBilinearInterpolatedPixel3D(transformedX, transformedY, transformedZ, cpuImage);
                const auto index = z * cpuImage.width * cpuImage.height + y * cpuImage.width + x;
                cpuOutput[index] = static_cast<uint8_t>(value);
            }
        }
    }

    nifti_image* outputImage = nifti_copy_nim_info(cpuImage.handle());
    outputImage->data = reinterpret_cast<void*>(gpuOutputData.data());
    nifti_set_filenames(outputImage, "output_gpu.nii", 0, 0);
    auto status = nifti_image_write_status(outputImage);
    EXPECT_EQ(status, 0);
    outputImage->data = reinterpret_cast<void*>(cpuOutput.data());
    nifti_set_filenames(outputImage, "output_cpu.nii", 0, 0);
    status = nifti_image_write_status(outputImage);
    EXPECT_EQ(status, 0);

    const auto cpuGpuMeanDiff = meanDifference(cpuOutput, gpuOutputData);
    spdlog::info("Mean difference between CPU and GPU: {}", cpuGpuMeanDiff);
    EXPECT_LT(meanDifference(cpuOutput, gpuOutputData) / 255.0, 0.1F);

}


TEST_F(ShaderTest, Reduction)
{
    std::vector<int32_t> data(10000);
    // generate random data
    std::generate(data.begin(), data.end(), []() { return rand() % 10; });

    const auto inputBuffer = wgpuContext.makeEmptyBuffer(data.size() * sizeof(int32_t));
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(sizeof(int32_t));
    wgpuContext.writeToBuffer(inputBuffer, data.data());

    const auto shaderSource = Utils::readFile("shaders/reduction.wgsl");

    const gpu::KernelDescriptor reductionKernelDesc {
        .shader = {
            .name = "reduction",
            .entryPoint = "main",
            .filePath = "shaders/reduction.wgsl",
            .workgroupSize = { 256, 1, 1 }
        },
        .inputBuffers = { inputBuffer },
        .outputBuffers = { outputBuffer }
    };

    auto reductionKernel = wgpuContext.makeKernel(reductionKernelDesc);
    wgpuContext.dispatchKernel(reductionKernel, { static_cast<uint32_t>(data.size() + 255 / 256), 1, 1 });

    int32_t gpuResult;
    wgpuContext.downloadBuffer(outputBuffer, &gpuResult);

    // Compute the reduction on the CPU for comparison
    int32_t cpuResult = 0;
    for (size_t i = 0; i < data.size(); i++) {
        cpuResult += data[i];
    }

    EXPECT_EQ(cpuResult, gpuResult);
}


TEST_F(ShaderTest, ReductionFloat)
{
    std::vector<float> data(9999);
    // generate random data
    std::generate(data.begin(), data.end(), []() { return static_cast<float>((rand() % 100)/10.0); });

    const auto inputBuffer = wgpuContext.makeEmptyBuffer(data.size() * sizeof(float));
    wgpuContext.writeToBuffer(inputBuffer, data.data());

    const auto outputBuffer = wgpuContext.makeEmptyBuffer(sizeof(uint32_t));

    const gpu::KernelDescriptor reductionKernelDesc {
        .shader = {
            .name = "reduction_float",
            .entryPoint = "main",
            .filePath = "shaders/reduction_f32.wgsl",
            .workgroupSize = { 256, 1, 1 }
        },
        .inputBuffers = { inputBuffer },
        .outputBuffers = { outputBuffer }
    };

    auto reductionKernel = wgpuContext.makeKernel(reductionKernelDesc);
    wgpuContext.dispatchKernel(reductionKernel, { static_cast<uint32_t>(data.size() + 255 / 256), 1, 1 });

    float gpuResult;
    wgpuContext.downloadBuffer(outputBuffer, &gpuResult);

    // Compute the reduction on the CPU in double precision for comparison
    double cpuResult = 0;
    for (size_t i = 0; i < data.size(); i++) {
        cpuResult += static_cast<double>(data[i]);
    }

    EXPECT_NEAR(cpuResult, gpuResult, 1e-1);
}

TEST_F(ShaderTest, MultiStageReductionFloatNoPadding)
{
    auto gpuReduction = [&](const std::vector<float>& originalData){
        size_t originalSize = originalData.size();
        spdlog::info("Original data size: {}", originalSize);
        const gpu::WorkgroupSize wgSize = { 256, 1, 1};
        std::vector<float> data = originalData;
        const gpu::DataBuffer inputBuffer = wgpuContext.makeEmptyBuffer(data.size() * sizeof(float));
        spdlog::info("Input buffer size: {}", data.size());
        wgpuContext.writeToBuffer(inputBuffer, data.data());

        const gpu::ReductionDescriptor reductionDesc {
            .workgroupSize = 256,
            .groupSize = 1,
            .data = inputBuffer,
            .result = wgpuContext.makeEmptyBuffer(sizeof(float))
        };

        gpu::ReductionHelper reductionHelper(reductionDesc, wgpuContext);
        reductionHelper.dispatch(wgpuContext);

        float gpuResult = 0;
        wgpuContext.downloadBuffer(reductionDesc.result, &gpuResult);
        return gpuResult;
    };

    const std::array sizes = { 256, 512, 1 << 16, 2 << 19 };

    for(const auto size : sizes) {
        std::vector<float> data(size);
        std::generate(data.begin(), data.end(), []() { return static_cast<float>((rand() % 100)/10.0); });
        const auto gpuResult = gpuReduction(data);

        // Compute the reduction on the CPU in double precision for comparison
        const double cpuResult = std::accumulate(data.begin(), data.end(), 0.0);
        EXPECT_NEAR(cpuResult, gpuResult, 1e-1);
    }
}

TEST_F(ShaderTest, MultiStageReductionFloatNoPaddingWithNonSingularUnitSizes)
{
    const std::vector<std::pair<uint32_t, uint32_t>> unitSizesAndSizes = {
        {2, 512},
        {3, 3 << 8},
        {4, 4 << 9},
        {5, 5 << 10},
        {6, 6 << 11},
        {7, 7 << 12}
    };
    auto gpuReduction = [&](const std::vector<float>& originalData, uint32_t unitSize){
        size_t originalSize = originalData.size();
        spdlog::info("Original data size: {}", originalSize);
        const gpu::WorkgroupSize wgSize = { 256, 1, 1};
        std::vector<float> data = originalData;
        const gpu::DataBuffer inputBuffer = wgpuContext.makeEmptyBuffer(data.size() * sizeof(float));
        spdlog::info("Input buffer size: {}", data.size());
        wgpuContext.writeToBuffer(inputBuffer, data.data());

        const gpu::ReductionDescriptor reductionDesc {
            .workgroupSize = 256,
            .groupSize = unitSize,
            .data = inputBuffer,
            .result = wgpuContext.makeEmptyBuffer(sizeof(float) * unitSize)
        };

        gpu::ReductionHelper reductionHelper(reductionDesc, wgpuContext);
        reductionHelper.dispatch(wgpuContext);

        std::vector<float> gpuResult(unitSize);
        wgpuContext.downloadBuffer(reductionDesc.result, gpuResult.data());
        return gpuResult;
    };

    // Unit size means the elements in the input are treated as groups of unitSize elements
    for(const auto& [unitSize, dataSize] :  unitSizesAndSizes) {
        assert(dataSize % unitSize == 0);
        std::vector<float> data(dataSize);
        std::generate(data.begin(), data.end(), []() { return static_cast<float>((rand() % 100)/10.0); });

        const auto gpuResult = gpuReduction(data, unitSize);

        std::vector<float> cpuResult(unitSize, 0.0);
        for(size_t i = 0; i < data.size(); i += unitSize) {
            for(size_t j = 0; j < unitSize; j++) {
                cpuResult[j] += data[i + j];
            }
        }

        for(size_t i = 0; i < unitSize; i++) {
            EXPECT_NEAR(cpuResult[i], gpuResult[i], 1e-1);
        }
    }
}

TEST_F(ShaderTest, MultiStageReductionFloatWithPadding)
{
    auto gpuReduction = [&](const std::vector<float>& originalData){
        size_t originalSize = originalData.size();
        const gpu::WorkgroupSize wgSize = { 256, 1, 1};

        std::vector<float> data = originalData;
        if(originalSize < wgSize.x) {
            data.reserve(wgSize.x);
            std::fill_n(std::back_inserter(data), wgSize.x - originalSize, 0.0F);
        }
        // Check if data size is a multiple of the workgroup size
        // if not, pad the data with zeros
        else if(originalSize % wgSize.x != 0) {
            const size_t newSize = (originalSize / wgSize.x + 1) * wgSize.x;
            data.reserve(newSize);
            std::fill_n(std::back_inserter(data), newSize - originalSize, 0.0F);
        }

        const gpu::DataBuffer inputBuffer = wgpuContext.makeEmptyBuffer(data.size() * sizeof(float));
        wgpuContext.writeToBuffer(inputBuffer, data.data());

        const gpu::ReductionDescriptor reductionDesc {
            .workgroupSize = 256,
            .groupSize = 1,
            .data = inputBuffer,
            .result = wgpuContext.makeEmptyBuffer(sizeof(float))
        };

        gpu::ReductionHelper reductionHelper(reductionDesc, wgpuContext);
        reductionHelper.dispatch(wgpuContext);

        float gpuResult = 0;
        wgpuContext.downloadBuffer(reductionDesc.result, &gpuResult);
        return gpuResult;
    };

    const std::array sizes = { 10, 257, 10000, (1 << 19) + 1 };

    for(const auto size : sizes) {
        std::vector<float> data(size);
        std::generate(data.begin(), data.end(), []() { return static_cast<float>((rand() % 100)/10.0); });
        const auto gpuResult = gpuReduction(data);

        // Compute the reduction on the CPU in double precision for comparison
        const double cpuResult = std::accumulate(data.begin(), data.end(), 0.0);
        EXPECT_NEAR(cpuResult, gpuResult, 1e-1);
    }

}


TEST_F(ShaderTest, Downsample)
{
    auto brainImage = Utils::loadFromDisk("data/brain.pgm");

    const auto inputTexture = wgpuContext.makeTextureFromHostPgm(brainImage);
    const auto outputTexture = wgpuContext.makeEmptyTexture({
        .size = { brainImage.width / 2, brainImage.height / 2, 1 },
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    const gpu::KernelDescriptor downsampleOpDesc {
        .shader = {
            .name = "downsampling",
            .entryPoint = "main",
            .filePath = "shaders/downsample.wgsl",
            .workgroupSize = { 16, 16, 1 }
        },
        .inputTextures = { inputTexture },
        .outputTextures = { outputTexture }
    };

    auto downsampleOp = wgpuContext.makeKernel(downsampleOpDesc);
    wgpuContext.dispatchKernel(downsampleOp, { brainImage.width + 31 / 32, brainImage.height + 31 / 32, 1 });

    // Perform the downsampling on the CPU for comparison
    std::vector<uint8_t> cpuOutput(brainImage.width / 2 * brainImage.height / 2);
    for(size_t y = 0; y < brainImage.height / 2; ++y) {
        for(size_t x = 0; x < brainImage.width / 2; ++x) {
            uint8_t p00 = getPixel(x * 2, y * 2, brainImage);
            uint8_t p01 = getPixel(x * 2 + 1, y * 2, brainImage);
            uint8_t p10 = getPixel(x * 2, y * 2 + 1, brainImage);
            uint8_t p11 = getPixel(x * 2 + 1, y * 2 + 1, brainImage);

            cpuOutput[y * (brainImage.width / 2) + x] =
                static_cast<uint8_t>((static_cast<double>(p00) + p01 + p10 + p11) / 4.0);
        }
    }

    PgmImage gpuOutputImage = {
        .width = brainImage.width / 2,
        .height = brainImage.height / 2,
        .data = std::vector<uint8_t>(brainImage.width / 2 * brainImage.height / 2)
    };
    wgpuContext.downloadTexture(outputTexture, gpuOutputImage.data.data());

    // Save the output to disk for visual inspection
    PgmImage cpuOutputImage = {
        .width = brainImage.width / 2,
        .height = brainImage.height / 2,
        .data = cpuOutput
    };

    EXPECT_LT(meanDifference(cpuOutput, gpuOutputImage.data) / 255.0, 5e-3F);
}


TEST_F(ShaderTest, Downsample3D) {
    auto brainImage = Utils::loadNiftiFromDisk("data/test_file.nii");

    const auto inputTexture = wgpuContext.makeTextureFromHostNifti(brainImage);
    const auto outputTexture = wgpuContext.makeEmptyTexture({
        .size = { brainImage.width / 2, brainImage.height / 2, brainImage.depth / 2 },
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    const auto shaderSource = Utils::readFile("shaders/3d/downsample_3d.wgsl");
    const gpu::KernelDescriptor downsampleOpDesc {
        .shader = {
            .name = "downsampling3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/downsample_3d.wgsl",
            .workgroupSize = { 4, 4, 4 }
        },
        .inputTextures = { inputTexture },
        .outputTextures = { outputTexture }
    };

    auto downsampleOp = wgpuContext.makeKernel(downsampleOpDesc);

    wgpuContext.dispatchKernel(downsampleOp, { brainImage.width + 1 / 2, brainImage.height + 1 / 2, brainImage.depth + 1 / 2 });

    std::vector<uint8_t> cpuOutput(brainImage.width / 2 * brainImage.height / 2 * brainImage.depth / 2);
    for(size_t z = 0; z < brainImage.depth / 2; ++z) {
        for(size_t y = 0; y < brainImage.height / 2; ++y) {
            for(size_t x = 0; x < brainImage.width / 2; ++x) {
                uint8_t p000 = getPixel3D(x * 2, y * 2, z * 2, brainImage);
                uint8_t p001 = getPixel3D(x * 2, y * 2, z * 2 + 1, brainImage);
                uint8_t p010 = getPixel3D(x * 2, y * 2 + 1, z * 2, brainImage);
                uint8_t p011 = getPixel3D(x * 2, y * 2 + 1, z * 2 + 1, brainImage);
                uint8_t p100 = getPixel3D(x * 2 + 1, y * 2, z * 2, brainImage);
                uint8_t p101 = getPixel3D(x * 2 + 1, y * 2, z * 2 + 1, brainImage);
                uint8_t p110 = getPixel3D(x * 2 + 1, y * 2 + 1, z * 2, brainImage);
                uint8_t p111 = getPixel3D(x * 2 + 1, y * 2 + 1, z * 2 + 1, brainImage);

                cpuOutput[z * (brainImage.width / 2) * (brainImage.height / 2) + y * (brainImage.width / 2) + x] =
                    static_cast<uint8_t>((static_cast<double>(p000) + p001 + p010 + p011 + p100 + p101 + p110 + p111) / 8.0);
            }
        }
    }

    std::vector<uint8_t> gpuOutputData(brainImage.width / 2 * brainImage.height / 2 * brainImage.depth / 2);
    wgpuContext.downloadTexture(outputTexture, gpuOutputData.data());

    EXPECT_LT(meanDifference(cpuOutput, gpuOutputData) / 255.0, 5e-3F);
}

TEST_F(ShaderTest, ComputeSingleImageOperation3D) {
    auto brainImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const gpu::Texture inputTexture = wgpuContext.makeTextureFromHostNifti(brainImage);
    const double numberOfVoxels = brainImage.width * brainImage.height * brainImage.depth;

    // Output buffer of compute_mean shader is an intermediate array of floats
    // that needs to be reduce to a single float
    const gpu::WorkgroupSize workgroupSize { 8, 8, 4 };
    const auto workgroupGrid = gpu::WorkgroupGrid::ForOneWorkUnitPerThread(
        brainImage.width, brainImage.height, brainImage.depth, workgroupSize
    );
    const uint32_t intermediateBufferSize = workgroupGrid.totalCount();
    const gpu::DataBuffer intermediateBuffer = wgpuContext.makeEmptyBuffer(intermediateBufferSize * sizeof(float));
    const std::string shaderSource = Utils::readFile("shaders/3d/reduction_image_3d.wgsl");
    const gpu::KernelDescriptor computeMeanDesc {
        .shader = {
            .name = "reduction_image_3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/reduction_image_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                {"operations_size", "1u"},
                {"operations", "0u"}
            }
        },
        .inputTextures = { inputTexture },
        .outputBuffers = { intermediateBuffer }
    };

    const gpu::Kernel computeMeanKernel = wgpuContext.makeKernel(computeMeanDesc);
    wgpuContext.dispatchKernel(computeMeanKernel, workgroupGrid);

    // Perform the reduction on CPU for comparison
    std::vector<float> intermediateData(intermediateBufferSize);
    wgpuContext.downloadBuffer(intermediateBuffer, intermediateData.data());
    const float gpuMean = std::accumulate(intermediateData.begin(), intermediateData.end(), 0.0f) / numberOfVoxels;

    float cpuMean = 0.0f;
    for(size_t z = 0; z < brainImage.depth; z++) {
        for(size_t y = 0; y < brainImage.height; y++) {
            for(size_t x = 0; x < brainImage.width; x++) {
                cpuMean += getPixel3D(x, y, z, brainImage) / 256.0F;
            }
        }
    }
    cpuMean /= numberOfVoxels;

    EXPECT_NEAR(cpuMean, gpuMean, 1e-2);
}

TEST_F(ShaderTest, ComputeMultipleImageOperations3D)
{
    auto brainImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const gpu::Texture inputTexture = wgpuContext.makeTextureFromHostNifti(brainImage);
    const double numberOfVoxels = brainImage.width * brainImage.height * brainImage.depth;

    // Output buffer of compute_mean shader is an intermediate array of floats
    // that needs to be reduce to a single float
    const gpu::WorkgroupSize workgroupSize { 8, 8, 4 };
    const auto workgroupGrid = gpu::WorkgroupGrid::ForOneWorkUnitPerThread(
        brainImage.width, brainImage.height, brainImage.depth, workgroupSize
        );
    const uint32_t numberOfOperations = 3;
    const uint32_t intermediateBufferSize = workgroupGrid.totalCount() * numberOfOperations;
    const gpu::DataBuffer intermediateBuffer = wgpuContext.makeEmptyBuffer(intermediateBufferSize * sizeof(float));
    const std::string shaderSource = Utils::readFile("shaders/3d/reduction_image_3d.wgsl");
    const gpu::KernelDescriptor computeMeanDesc {
        .shader = {
            .name = "reduction_image_3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/reduction_image_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                {"operations_size", std::to_string(numberOfOperations)},
                {"operations", "0u, 1u, 2u"}
            }
        },
        .inputTextures = { inputTexture },
        .outputBuffers = { intermediateBuffer }
    };

    const gpu::Kernel computeMeanKernel = wgpuContext.makeKernel(computeMeanDesc);
    wgpuContext.dispatchKernel(computeMeanKernel, workgroupGrid);

    // Perform the reduction on CPU for comparison
    std::vector<float> intermediateData(intermediateBufferSize);
    wgpuContext.downloadBuffer(intermediateBuffer, intermediateData.data());

    float gpuSum = 0.0F;
    float gpuMin = std::numeric_limits<float>::max();
    float gpuMax = std::numeric_limits<float>::min();

    for(size_t i = 0; i < intermediateBufferSize; i += numberOfOperations) {
        gpuSum += intermediateData[i];
        gpuMin = std::min(gpuMin, intermediateData[i + 1]);
        gpuMax = std::max(gpuMax, intermediateData[i + 2]);
    }

    double cpuSum = 0.0F;
    double cpuMin = std::numeric_limits<double>::max();
    double cpuMax = std::numeric_limits<double>::min();
    for(size_t z = 0; z < brainImage.depth; z++) {
        for(size_t y = 0; y < brainImage.height; y++) {
            for(size_t x = 0; x < brainImage.width; x++) {
                const double pixel = static_cast<double>(getPixel3D(x, y, z, brainImage)) / 256.0;
                cpuSum += pixel;
                cpuMin = std::min(cpuMin, pixel);
                cpuMax = std::max(cpuMax, pixel);
            }
        }
    }

    spdlog::info("CPU sum: {}, GPU sum: {}", cpuSum, gpuSum);
    spdlog::info("CPU mean: {}, GPU mean: {}", cpuSum/numberOfVoxels, gpuSum/numberOfVoxels);
    EXPECT_NEAR(cpuSum/numberOfVoxels, gpuSum/numberOfVoxels, 1e-3);
    EXPECT_NEAR(cpuMin, gpuMin, 1e-3);
    EXPECT_NEAR(cpuMax, gpuMax, 1e-3);
}

TEST_F(ShaderTest, ComputeTransformedSingleImageOperation3D)
{
    NiftiTransformParams transformationParameters {
        .alpha = 0.2F, .beta = 0.3F, .gamma = 0.1F,
        .tx = 1.1F, .ty = -10.0F, .tz = -20.0F
    };
    auto brainImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const gpu::Texture inputTexture = wgpuContext.makeTextureFromHostNifti(brainImage);
    const double numberOfVoxels = brainImage.width * brainImage.height * brainImage.depth;

    // Output buffer of compute_mean shader is an intermediate array of floats
    // that needs to be reduce to a single float
    const gpu::WorkgroupSize workgroupSize { 8, 8, 4 };
    const auto workgroupGrid = gpu::WorkgroupGrid::ForOneWorkUnitPerThread(
        brainImage.width, brainImage.height, brainImage.depth, workgroupSize
        );
    const uint32_t intermediateBufferSize = workgroupGrid.totalCount();
    const gpu::DataBuffer intermediateBuffer = wgpuContext.makeEmptyBuffer(intermediateBufferSize * sizeof(float));
    const std::string shaderSource = Utils::readFile("shaders/3d/reduction_image_transformed_3d.wgsl");

    const gpu::DataBuffer transformationParametersBuffer =
        wgpuContext.makeEmptyBuffer(sizeof(NiftiTransformParams));
    wgpuContext.writeToBuffer(transformationParametersBuffer, &transformationParameters);

    const gpu::KernelDescriptor computeMeanDesc {
        .shader = {
            .name = "reduction_image_transformed_3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/reduction_image_transformed_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                {"operations_size", "1u"},
                {"operations", "0u"}
            }
        },
        .inputBuffers = { transformationParametersBuffer },
        .inputTextures = { inputTexture },
        .outputBuffers = { intermediateBuffer },
        .samplers = { wgpuContext.makeLinearSampler() },
    };

    const gpu::Kernel computeMeanKernel = wgpuContext.makeKernel(computeMeanDesc);
    wgpuContext.dispatchKernel(computeMeanKernel, workgroupGrid);

    // Perform the reduction on CPU for comparison
    std::vector<float> intermediateData(intermediateBufferSize);
    wgpuContext.downloadBuffer(intermediateBuffer, intermediateData.data());
    const float gpuMean = std::accumulate(intermediateData.begin(), intermediateData.end(), 0.0f) / numberOfVoxels;

    float cpuMean = 0.0f;
    auto transformedImage = transformNifti(brainImage, transformationParameters);
    for(size_t z = 0; z < brainImage.depth; z++) {
        for(size_t y = 0; y < brainImage.height; y++) {
            for(size_t x = 0; x < brainImage.width; x++) {
                cpuMean += getPixel3D(x, y, z, transformedImage) / 256.0F;
            }
        }
    }
    cpuMean /= numberOfVoxels;

    EXPECT_NEAR(cpuMean, gpuMean, 1e-2);
}

TEST_F(ShaderTest, Histogram3D)
{
    auto brainImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const gpu::Texture inputTexture = wgpuContext.makeTextureFromHostNifti(brainImage);
    const double numberOfVoxels = brainImage.width * brainImage.height * brainImage.depth;

    const gpu::WorkgroupSize workgroupSize { 8, 8, 4 };
    const auto workgroupGrid = gpu::WorkgroupGrid::ForOneWorkUnitPerThread(
        brainImage.width, brainImage.height, brainImage.depth, workgroupSize
    );

    const gpu::KernelDescriptor computeHistogramDesc {
        .shader = {
            .name = "histogram_3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/histogram_image_3d.wgsl",
            .workgroupSize = workgroupSize,
        },
        .inputTextures = { inputTexture },
        .outputBuffers = { wgpuContext.makeEmptyBuffer(256 * sizeof(uint32_t)) },
    };

    const gpu::Kernel computeHistogramKernel = wgpuContext.makeKernel(computeHistogramDesc);
    wgpuContext.dispatchKernel(computeHistogramKernel, workgroupGrid);

    std::vector<uint32_t> gpuHistogram(256);
    wgpuContext.downloadBuffer(computeHistogramDesc.outputBuffers[0], gpuHistogram.data());

    std::vector<uint32_t> cpuHistogram(256, 0);
    for(size_t z = 0; z < brainImage.depth; z++) {
        for(size_t y = 0; y < brainImage.height; y++) {
            for(size_t x = 0; x < brainImage.width; x++) {
                const uint8_t pixel = getPixel3D(x, y, z, brainImage);
                cpuHistogram[pixel]++;
            }
        }
    }

    for(size_t i = 0; i < 256; i++) {
        EXPECT_EQ(cpuHistogram[i], gpuHistogram[i]);
    }
}
