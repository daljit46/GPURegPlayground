#include "gpu.h"
#include "image.h"
#include "utils.h"
#include <iostream>
#include <chrono>


class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name(name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << name << ": " << duration << "ms" << std::endl;
    }

private:
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

int main()
{
    auto wgpuContext = gpu::Context::newContext();
    const CpuImage sourceImage = Utils::loadFromDisk("data/brain.pgm");
    const CpuImage targetImage = Utils::loadFromDisk("data/brain_translated.pgm");

    auto sourceTexture = wgpuContext.makeTextureFromHost(sourceImage);
    auto targetTexture = wgpuContext.makeTextureFromHost(targetImage);
    auto movingTexture = wgpuContext.makeEmptyTexture({
        .width = sourceImage.width,
        .height = sourceImage.height,
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });
    auto gradientXBuffer = wgpuContext.makeEmptyBuffer(sizeof(float) * sourceImage.width * sourceImage.height);
    auto gradientYBuffer = wgpuContext.makeEmptyBuffer(sizeof(float) * sourceImage.width * sourceImage.height);

    struct TransformationParameters {
        float angle = 0.0F;
        float translationX = 0.0F;
        float translationY = 0.0F;
        float padding = 0.0F;
    } transformationParams;

    auto paramsUniformBuffer = wgpuContext.makeUniformBuffer(&transformationParams, sizeof(TransformationParameters));

    const gpu::WorkgroupSize workgroupSize {16, 16, 1};
    const gpu::ComputeOperationDescriptor gradientXDesc {
        .shader = {
            .name="gradientx",
            .entryPoint = "computeSobelX",
            .code = Utils::readFile("shaders/gradientx.wgsl"),
            .workgroupSize = workgroupSize
        },
        .inputTextures = { movingTexture },
        .outputBuffers = { gradientXBuffer }
    };

    const gpu::ComputeOperationDescriptor gradientYDesc{
        .shader {
            .name="gradienty",
            .entryPoint = "computeSobelY",
            .code = Utils::readFile("shaders/gradienty.wgsl"),
            .workgroupSize = workgroupSize
        },
        .inputTextures = { movingTexture },
        .outputBuffers = { gradientYBuffer }
    };

    const gpu::ComputeOperationDescriptor transformDesc {
        .shader {
            .name = "transform",
            .entryPoint = "computeTransform",
            .code = Utils::readFile("shaders/transformimage.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = { paramsUniformBuffer },
        .inputTextures = { sourceTexture },
        .outputTextures = { movingTexture },
        .samplers = { wgpuContext.makeLinearSampler() }
    };

    // Output parameters are dssd_dtheta, dssd_dtx, dssd_dty, ssd
    std::array<uint32_t, 4> outputParameters = {0, 0, 0, 0};
    auto parametersOutputBuffer = wgpuContext.makeEmptyBuffer(sizeof(uint32_t) * outputParameters.size());
    wgpuContext.writeToBuffer(parametersOutputBuffer, outputParameters.data());

    const gpu::ComputeOperationDescriptor updateParamsDesc {
        .shader {
            .name = "updateparameters",
            .entryPoint = "updateParameters",
            .code = Utils::readFile("shaders/updateparameters.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = { paramsUniformBuffer },
        .inputBuffers = { gradientXBuffer, gradientYBuffer },
        .inputTextures = { targetTexture, movingTexture },
        .outputBuffers = { parametersOutputBuffer }
    };

    auto gradientXOp = wgpuContext.makeComputeOperation(gradientXDesc);
    auto gradientYOp = wgpuContext.makeComputeOperation(gradientYDesc);
    auto transformOp = wgpuContext.makeComputeOperation(transformDesc);
    auto updateParamsOp = wgpuContext.makeComputeOperation(updateParamsDesc);

    constexpr int maxIterations = 5;
    const auto angleLearningRate = 1e-6;
    const auto txLearningRate = 1e-4;
    const auto tyLearningRate = 1e-4;

    float minSSD = std::numeric_limits<float>::max();
    float minAngle = std::numeric_limits<float>::max();
    float minTx = std::numeric_limits<float>::max();
    float minTy = std::numeric_limits<float>::max();

    auto calcWorkgroupGrid = [](const CpuImage& image, const gpu::WorkgroupSize& workgroupSize) {
        return gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1};
    };

    for (int i = 0; i < maxIterations; i++) {
        wgpuContext.dispatchOperation(transformOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        wgpuContext.dispatchOperation(gradientXOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        wgpuContext.dispatchOperation(gradientYOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        wgpuContext.dispatchOperation(updateParamsOp, calcWorkgroupGrid(sourceImage, workgroupSize));

        wgpuContext.downloadBuffer(parametersOutputBuffer, outputParameters.data());

        const float dssd_dtheta = reinterpret_cast<float*>(outputParameters.data())[0];
        const float dssd_dtx = reinterpret_cast<float*>(outputParameters.data())[1];
        const float dssd_dty = reinterpret_cast<float*>(outputParameters.data())[2];
        const float ssd = reinterpret_cast<float*>(outputParameters.data())[3];

        if (ssd < minSSD) {
            minSSD = ssd;
            minAngle = outputParameters[0];
            minTx = outputParameters[1];
            minTy = outputParameters[2];
        }

        // transformationParams.angle -= angleLearningRate * dssd_dtheta;
        transformationParams.translationX -= txLearningRate * dssd_dtx;
        transformationParams.translationY -= tyLearningRate * dssd_dty;

        wgpuContext.writeToBuffer(paramsUniformBuffer, &transformationParams);

        // Reset output parameters to zero
        std::fill(outputParameters.begin(), outputParameters.end(), 0.0F);
        wgpuContext.writeToBuffer(parametersOutputBuffer, outputParameters.data());

        std::cout << "Iteration " << i << " SSD: " << ssd << std::endl;
        std::cout << "Angle: " << transformationParams.angle << std::endl;
        std::cout << "Tx: " << transformationParams.translationX << std::endl;
        std::cout << "Ty: " << transformationParams.translationY << std::endl;
        std::cout << "dssd_dtheta: " << dssd_dtheta << std::endl;
        std::cout << "dssd_dtx: " << dssd_dtx << std::endl;
        std::cout << "dssd_dty: " << dssd_dty << std::endl;
    }

    std::cout << "Minimum SSD: " << minSSD << std::endl;
    std::cout << "Minimum angle: " << minAngle << std::endl;
    std::cout << "Minimum tx: " << minTx << std::endl;
    std::cout << "Minimum ty: " << minTy << std::endl;

    // Save the final transformed image
    auto finalTransformedImage = wgpuContext.downloadTexture(movingTexture);
    const std::filesystem::path outputPath = "result.pgm";
    Utils::saveToDisk(finalTransformedImage, outputPath);
}
