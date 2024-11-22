#include "gpu.h"
#include "image.h"
#include "utils.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <matplot/matplot.h>
#include <spdlog/spdlog.h>
#include <string>
#include <vector>

template<typename T>
T sumVector(const std::vector<T>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0F);
}

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name) : name(name) {
        start = std::chrono::high_resolution_clock::now();
    }

    ~ScopedTimer() {
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        spdlog::info("{}: {}ms", name, duration);
    }

private:
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};



namespace CpuEngine {
double ssd(const CpuImage& source, const CpuImage& target) {
    assert(source.width == target.width && source.height == target.height);

    double sum = 0.0F;
    for (int i = 0; i < source.width * source.height; i++) {
        const double diff = (static_cast<double>(source.data[i]) - static_cast<double>(target.data[i]))/255.0;
        sum += diff * diff;
    }

    return sum;
}

CpuImage transform(const CpuImage& image, float angle, float tx, float ty) {
    CpuImage result {
        .width = image.width,
        .height = image.height,
    };

    result.data.resize(image.width * image.height);
    auto getPixel = [&image](int x, int y) -> uint8_t {
        if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
            return 0.0F;
        }

        const auto index = y * image.width + x;
        return image.data[index];
    };

    auto setPixel = [&result](int x, int y, uint8_t value) {
        const auto index = y * result.width + x;
        result.data[index] = value;
    };

    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            const float xNew = x * std::cos(angle) - y * std::sin(angle) + tx;
            const float yNew = x * std::sin(angle) + y * std::cos(angle) + ty;

            // Bilinear interpolation
            const int x0 = std::floor(xNew);
            const int y0 = std::floor(yNew);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float dx = xNew - x0;
            const float dy = yNew - y0;

            const float p00 = getPixel(x0, y0);
            const float p01 = getPixel(x0, y1);
            const float p10 = getPixel(x1, y0);
            const float p11 = getPixel(x1, y1);

            const float p0 = p00 * (1.0F - dx) + p10 * dx;
            const float p1 = p01 * (1.0F - dx) + p11 * dx;
            const float p = p0 * (1.0F - dy) + p1 * dy;

            setPixel(x, y, p);
        }
    }

    return result;
}

}

int main()
{
    spdlog::set_pattern("%^[%T] [%n:] %v%$");
    auto context = gpu::Context::newContext();
    const CpuImage sourceImage = Utils::loadFromDisk("data/brain.pgm");
    const CpuImage targetImage = CpuEngine::transform(sourceImage, 0.1, 100, 100);

    auto sourceTexture = context.makeTextureFromHost(sourceImage);
    auto targetTexture = context.makeTextureFromHost(targetImage);
    auto movingTexture = context.makeEmptyTexture({
        .width = sourceImage.width,
        .height = sourceImage.height,
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });
    auto gradientXBuffer = context.makeEmptyBuffer(sizeof(float) * sourceImage.width * sourceImage.height);
    auto gradientYBuffer = context.makeEmptyBuffer(sizeof(float) * sourceImage.width * sourceImage.height);

    struct TransformationParameters {
        float angle = 0.0F;
        float translationX = 0.0F;
        float translationY = 0.0F;
    } transformationParams;

    struct Uniforms {
        float cosAngle = 0.0;
        float sinAngle = 0.0;
        float translationX = 0.0;
        float translationY = 0.0;
    } uniforms;

    auto uniformsBuffer = context.makeUniformBuffer(&uniforms, sizeof(Uniforms));

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
        .uniformBuffers = { uniformsBuffer },
        .inputTextures = { sourceTexture },
        .outputTextures = { movingTexture },
        .samplers = { context.makeLinearSampler() }
    };

    // Output parameters are dssd_dtheta, dssd_dtx, dssd_dty, ssd
    std::array<uint32_t, 4> outputParameters = {0, 0, 0, 0};
    auto parametersOutputBuffer = context.makeEmptyBuffer(sizeof(uint32_t) * outputParameters.size());
    context.writeToBuffer(parametersOutputBuffer, outputParameters.data());

    const gpu::ComputeOperationDescriptor updateParamsDesc {
        .shader {
            .name = "updateparameters",
            .entryPoint = "updateParameters",
            .code = Utils::readFile("shaders/updateparameters.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = { uniformsBuffer },
        .inputBuffers = { gradientXBuffer, gradientYBuffer },
        .inputTextures = { targetTexture, movingTexture },
        .outputBuffers = { parametersOutputBuffer }
    };

    auto gradientXOp = context.makeComputeOperation(gradientXDesc);
    auto gradientYOp = context.makeComputeOperation(gradientYDesc);
    auto transformOp = context.makeComputeOperation(transformDesc);
    auto updateParamsOp = context.makeComputeOperation(updateParamsDesc);

    constexpr int maxIterations = 1000;
    const float txLearningRate = 5;
    const float tyLearningRate = 5;
    const float angleLearningRate = txLearningRate / 100;


    constexpr float epsilon = 1e-9;
    double angleAccumulator = 0.0;
    double txAccumulator = 0.0;
    double tyAccumulator = 0.0;

    float minSSD = std::numeric_limits<float>::max();
    float minAngle = std::numeric_limits<float>::max();
    float minTx = std::numeric_limits<float>::max();
    float minTy = std::numeric_limits<float>::max();

    auto calcWorkgroupGrid = [](const CpuImage& image, const gpu::WorkgroupSize& workgroupSize) {
        return gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1};
    };

    auto originalSSD = CpuEngine::ssd(sourceImage, targetImage);

    std::vector<float> ssdHistory;
    std::vector<float> transformationTimeHistory;
    std::vector<float> gradientXTimeHistory;
    std::vector<float> gradientYTimeHistory;
    std::vector<float> updateParamsTimeHistory;

    for (int i = 0; i < maxIterations; i++) {
        uniforms.cosAngle = std::cos(transformationParams.angle);
        uniforms.sinAngle = std::sin(transformationParams.angle);
        uniforms.translationX = transformationParams.translationX;
        uniforms.translationY = transformationParams.translationY;
        context.writeToBuffer(uniformsBuffer, &uniforms);
        // Reset output parameters to zero
        outputParameters = {};
        context.writeToBuffer(parametersOutputBuffer, outputParameters.data());

        context.dispatchOperation(transformOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        context.dispatchOperation(gradientXOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        context.dispatchOperation(gradientYOp, calcWorkgroupGrid(sourceImage, workgroupSize));
        context.dispatchOperation(updateParamsOp, calcWorkgroupGrid(sourceImage, workgroupSize));

        // Check how long the updateParamsOp takes
        struct TimeStamp {
            uint64_t start = 0;
            uint64_t end = 0;
        };

        TimeStamp updateParamsDuration;
        TimeStamp transformationDuration;
        TimeStamp gradientXDuration;
        TimeStamp gradientYDuration;

        const std::vector<std::pair<gpu::DataBuffer*, void*>> bufferMappingPairs = {
            { &parametersOutputBuffer, outputParameters.data()},
            { &updateParamsOp.timestampResolveBuffer, &updateParamsDuration},
            { &transformOp.timestampResolveBuffer, &transformationDuration},
            { &gradientXOp.timestampResolveBuffer, &gradientXDuration},
            { &gradientYOp.timestampResolveBuffer, &gradientYDuration}
        };

        context.downloadBuffers(bufferMappingPairs);

        updateParamsTimeHistory.push_back((updateParamsDuration.end - updateParamsDuration.start) / 1e6);
        transformationTimeHistory.push_back((transformationDuration.end - transformationDuration.start) / 1e6);
        gradientXTimeHistory.push_back((gradientXDuration.end - gradientXDuration.start) / 1e6);
        gradientXTimeHistory.push_back((gradientYDuration.end - gradientYDuration.start) / 1e6);

        // Data on the GPU is output as u32 because WebGPU does not support f32 atomics
        // So we need to bitcast the data to float
        const float dssd_dtheta = reinterpret_cast<float*>(outputParameters.data())[0];
        const float dssd_dtx = reinterpret_cast<float*>(outputParameters.data())[1];
        const float dssd_dty = reinterpret_cast<float*>(outputParameters.data())[2];
        const float ssd = reinterpret_cast<float*>(outputParameters.data())[3];

        if (ssd < minSSD) {
            minSSD = ssd;
            minAngle = transformationParams.angle;
            minTx = transformationParams.translationX;
            minTy = transformationParams.translationY;
        }

        // Update parameters using AdaGrad
        angleAccumulator += dssd_dtheta * dssd_dtheta;
        txAccumulator += dssd_dtx * dssd_dtx;
        tyAccumulator += dssd_dty * dssd_dty;

        transformationParams.angle -= angleLearningRate * dssd_dtheta / std::sqrt(angleAccumulator + epsilon);
        transformationParams.translationX -= txLearningRate * dssd_dtx / std::sqrt(txAccumulator + epsilon);
        transformationParams.translationY -= tyLearningRate * dssd_dty / std::sqrt(tyAccumulator + epsilon);

        spdlog::info("Iteration {} SSD: {}", i, ssd);
        spdlog::info("Angle: {}", transformationParams.angle);
        spdlog::info("Tx: {}", transformationParams.translationX);
        spdlog::info("Ty: {}", transformationParams.translationY);
        spdlog::info("dssd_dtheta: {}", dssd_dtheta);
        spdlog::info("dssd_dtx: {}", dssd_dtx);
        spdlog::info("dssd_dty: {}", dssd_dty);

        ssdHistory.push_back(ssd);

        if (i > 20) {
            auto averageSSD = std::accumulate(ssdHistory.begin() + i - 20, ssdHistory.begin() + i, 0.0F) / 20;
            if (std::abs(ssd - averageSSD) < 1e-3) {
                spdlog::info("No improvement in last 20 iterations. Stopping.");
                break;
            }
        }

        if(ssd < originalSSD * 1e-4) {
            spdlog::info("Converged");
            break;
        }
        spdlog::info("------------------------------------------------- \n");
    }

    spdlog::info("Transformation shader took: {}ms", sumVector(transformationTimeHistory));
    spdlog::info("GradientX shader took: {}ms", sumVector(gradientXTimeHistory));
    spdlog::info("GradientY shader took: {}ms", sumVector(gradientYTimeHistory));
    spdlog::info("UpdateParams shader took: {}ms", sumVector(updateParamsTimeHistory));

    matplot::plot(ssdHistory);
    matplot::title("SSD History");
    matplot::xlabel("Iterations");
    matplot::ylabel("SSD");
    matplot::show();

    spdlog::info("Minimum SSD: {}", minSSD);
    spdlog::info("Minimum angle: {}", minAngle);
    spdlog::info("Minimum tx: {}", minTx);
    spdlog::info("Minimum ty: {}", minTy);

    // Save the final transformed image
    auto finalTransformedImage = context.downloadTexture(movingTexture);
    const std::filesystem::path outputPath = "result.pgm";
    Utils::saveToDisk(finalTransformedImage, outputPath);
}
