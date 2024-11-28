#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include "gpu.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include "image.h"
#include <nifti1.h>
#include <nifti1_io.h>
#include <cstddef>
#include <filesystem>
#include <vector>

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
    constexpr auto shaderPath = "shaders/gradientx.wgsl";
    const auto shaderSource = Utils::readFile(shaderPath);

    const auto cpuImage =  Utils::loadFromDisk("data/brain.pgm");
    const auto gpuImage = wgpuContext.makeTextureFromHostPgm(cpuImage);
    const auto outputBuffer = wgpuContext.makeEmptyBuffer(cpuImage.width * cpuImage.height * sizeof(float));

    const gpu::KernelDescriptor gradientKernelDesc {
        .shader = {
            .name = "gradientx",
            .entryPoint = "computeSobelX",
            .code = shaderSource,
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
            .code = shaderSource,
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
            .code = shaderSource,
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
        float alpha = Utils::degreesToRadians(0); // rotation about z-axis
        float beta = Utils::degreesToRadians(0); // rotation about y-axis
        float gamma = Utils::degreesToRadians(0); // rotation about x-axis
        float tx = 0;
        float ty = 0;
        float tz = -20;
        std::array<float, 2> _padding; // WGSL requires to align to 16 bytes
    } uniformParams;

    const gpu::KernelDescriptor transformKernelDesc {
        .shader = {
            .name = "transformimage3d",
            .code = shaderSource,
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
            .code = shaderSource,
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

    const auto shaderSource = Utils::readFile("shaders/reduction_f32.wgsl");

    const gpu::KernelDescriptor reductionKernelDesc {
        .shader = {
            .name = "reduction_float",
            .entryPoint = "main",
            .code = shaderSource,
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

TEST_F(ShaderTest, Downsample)
{
    auto brainImage = Utils::loadFromDisk("data/brain.pgm");

    const auto inputTexture = wgpuContext.makeTextureFromHostPgm(brainImage);
    const auto outputTexture = wgpuContext.makeEmptyTexture({
        .size = { brainImage.width / 2, brainImage.height / 2, 1 },
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    const auto shaderSource = Utils::readFile("shaders/downsample.wgsl");
    const gpu::KernelDescriptor downsampleOpDesc {
        .shader = {
            .name = "downsampling",
            .entryPoint = "main",
            .code = shaderSource,
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
