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
