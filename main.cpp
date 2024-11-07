#include <iostream>
#include <limits>
#include <stdint.h>
#include <thread>
#include <vector>

#include "image.h"
#include "gpu.h"
#include "utils.h"

int main()
{
    try {

        auto wgpuContext = gpu::createWebGPUContext();
        auto image = Utils::loadFromDisk("data/brain.pgm");
        auto referenceImage = Utils::loadFromDisk("data/brain_transform.pgm");
        auto originalImageBuffer = gpu::makeReadOnlyTextureBuffer(image, wgpuContext);
        auto referenceImageBuffer = gpu::makeReadOnlyTextureBuffer(referenceImage, wgpuContext);

        gpu::TextureSpecification textureSpec = {
            .width = image.width,
            .height = image.height,
            .format = gpu::TextureFormat::R8Unorm,
            .usage = gpu::ResourceUsage::ReadWrite
        };
        auto gradientXBuffer = gpu::makeEmptyDataBuffer(image.width * image.height * sizeof(float), gpu::ResourceUsage::ReadWrite, wgpuContext);
        auto gradientYBuffer = gpu::makeEmptyDataBuffer(image.width * image.height * sizeof(float), gpu::ResourceUsage::ReadWrite, wgpuContext);

        const gpu::WorkgroupSize workgroupSize {16, 16, 1};

        auto calcWorkgroupGrid = [](const Image& image, const gpu::WorkgroupSize& workgroupSize) {
            return gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1};
        };

        struct TransformationParameters {
            float rotationAngle = 0.0F;
            float translationX = 0.0F;
            float translationY = 0.0F;
            float _padding = 0.0F; // WGSL requires to align to 16 bytes
        } parameters = {Utils::degreesToRadians(0.0F), 0, 0 };

        auto movingImageBuffer = gpu::makeEmptyTextureBuffer(gpu::TextureSpecification {
                                                                 .width = image.width,
                                                                 .height = image.height,
                                                                 .format = gpu::TextureFormat::R8Unorm,
                                                                 .usage = gpu::ResourceUsage::ReadWrite
                                                             },
                                                             wgpuContext);

        auto paramsBuffer = gpu::makeUniformBuffer(reinterpret_cast<uint8_t*>(&parameters),
                                                   sizeof(TransformationParameters),
                                                   wgpuContext);

        gpu::ComputeOperationData transformData {
            .shader {
                .name = "transform",
                .entryPoint = "computeTransform",
                .code = Utils::readFile("shaders/transformimage.wgsl"),
                .workgroupSize = workgroupSize
            },
            .uniformBuffers = { paramsBuffer },
            .inputImageBuffers = { originalImageBuffer },
            .outputImageBuffers = { movingImageBuffer },
            .samplers = { gpu::createLinearSampler(wgpuContext) }
        };

        auto transformOp = gpu::createComputeOperation(transformData, wgpuContext);
        gpu::dispatchOperation(transformOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);

        gpu::ComputeOperationData gradientXOpData {
                                                  .shader = {
                                                      .name="sobelx",
                                                      .entryPoint = "computeSobelX",
                                                      .code = Utils::readFile("shaders/gradientx.wgsl"),
                                                      .workgroupSize = workgroupSize
                                                  },
                                                  .inputImageBuffers = { movingImageBuffer },
                                                  .outputBuffers = { gradientXBuffer },
                                                  };

        gpu::ComputeOperationData gradientYOpData {
                                                  .shader = {
                                                      .name="sobely",
                                                      .entryPoint = "computeSobelY",
                                                      .code = Utils::readFile("shaders/gradienty.wgsl"),
                                                      .workgroupSize = workgroupSize
                                                  },
                                                  .inputImageBuffers = { movingImageBuffer },
                                                  .outputBuffers = { gradientYBuffer },
                                                  };

        auto gradientXOp = gpu::createComputeOperation(gradientXOpData, wgpuContext);
        auto gradientYOp = gpu::createComputeOperation(gradientYOpData, wgpuContext);

        auto gradientsOutputBuffer = gpu::makeEmptyDataBuffer(sizeof(int32_t) * 4, gpu::ResourceUsage::ReadWrite, wgpuContext);
        std::array<int32_t, 4> gradientOutputInitialData = {0, 0, 0, 0};
        gpu::updateDataBuffer(gradientOutputInitialData.data(), gradientsOutputBuffer, wgpuContext);

        gpu::ComputeOperationData updateGradientsOpData {
            .shader {
                .name = "updategradients",
                .entryPoint = "updateGradients",
                .code = Utils::readFile("shaders/updategradients.wgsl"),
                .workgroupSize = workgroupSize
            },
            .uniformBuffers = { paramsBuffer },
            .inputBuffers = { gradientXBuffer, gradientYBuffer },
            .inputImageBuffers = { referenceImageBuffer, movingImageBuffer },
            .outputBuffers = {gradientsOutputBuffer}
        };
        auto updateGradientsOp = gpu::createComputeOperation(updateGradientsOpData, wgpuContext);

        constexpr float threshold = 0.01F;
        constexpr int maxIterations = 500;
        int i = 0;

        const auto angleLearningRate = 1e-5;
        const auto txLearningRate = 1e-4;
        const auto tyLearningRate = 1e-4;
        float minSSD = std::numeric_limits<float>::max();
        float minAngle = 0.0F;
        float minTx = 0.0F;
        float minTy = 0.0F;

        while(i < maxIterations) {
            gpu::dispatchOperation(transformOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);
            gpu::dispatchOperation(gradientXOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);
            gpu::dispatchOperation(gradientYOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);
            gpu::dispatchOperation(updateGradientsOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);

            // Read the output buffer and check if the SSD is below a threshold
            std::array<int32_t, 4> updatedGradients{};
            gpu::readBufferFromGPU(updatedGradients.data(), gradientsOutputBuffer, wgpuContext);

            const auto ssd = updatedGradients[3]/1000.0F;
            if(ssd < minSSD) {
                minSSD = ssd;
                minAngle = parameters.rotationAngle;
                minTx = parameters.translationX;
                minTy = parameters.translationY;
            }


            // Update the parameters
            parameters.rotationAngle -= angleLearningRate * updatedGradients[0]/1000.0;
            // clamp the angle between -pi and pi
            parameters.rotationAngle = std::fmod(parameters.rotationAngle + M_PI, 2*M_PI) - M_PI;
            parameters.translationX -= txLearningRate * updatedGradients[1]/1000.0;
            parameters.translationY -= tyLearningRate * updatedGradients[2]/1000.0;
            gpu::updateUniformBuffer(paramsBuffer, reinterpret_cast<void*>(&parameters), sizeof(TransformationParameters), wgpuContext);

            // print parameters and gradients
            std::cout << "Iteration: " << i << std::endl;
            std::cout << "SSD: " << ssd << std::endl;
            std::cout << "Angle: " << Utils::radiansToDegrees(parameters.rotationAngle) << std::endl;
            std::cout << "Tx: " << parameters.translationX << std::endl;
            std::cout << "Ty: " << parameters.translationY << std::endl;

            // Reset the gradients before next iteration
            updatedGradients = {0,0,0,0};
            gpu::updateDataBuffer(updatedGradients.data(), gradientsOutputBuffer, wgpuContext);
            ++i;
            std::cout << std::endl;
        }

        std::cout << "Min SSD: " << minSSD << std::endl;
        std::cout << "Min Angle: " << Utils::radiansToDegrees(minAngle) << std::endl;
        std::cout << "Min Tx: " << minTx << std::endl;
        std::cout << "Min Ty: " << minTy << std::endl;

        // Save the final transformed image
        parameters.rotationAngle = minAngle;
        parameters.translationX = minTx;
        parameters.translationY = minTy;
        gpu::updateUniformBuffer(paramsBuffer, reinterpret_cast<void*>(&parameters), sizeof(TransformationParameters), wgpuContext);
        gpu::dispatchOperation(transformOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);
        auto transformedImage = gpu::makeHostImageFromBuffer(movingImageBuffer, wgpuContext);
        Utils::saveToDisk(transformedImage, "data/transformed_image.pgm");

    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
