#include <iostream>
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
        auto brainImageBuffer = gpu::makeReadOnlyTextureBuffer(image, wgpuContext);
        gpu::TextureSpecification textureSpec = {
            .width = image.width,
            .height = image.height,
            .format = gpu::TextureFormat::R8Unorm,
            .usage = gpu::ResourceUsage::ReadWrite
        };
        auto gradientXBuffer = gpu::makeEmptyTextureBuffer(textureSpec, wgpuContext);
        auto gradientYBuffer = gpu::makeEmptyTextureBuffer(textureSpec, wgpuContext);

        const gpu::WorkgroupSize workgroupSize {16, 16, 1};

        auto calcWorkgroupGrid = [](const Image& image, const gpu::WorkgroupSize& workgroupSize) {
            return gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1};
        };

        gpu::ComputeOperationData gradientXOpData {
                                                  .shader = {
                                                      .name="sobelx",
                                                      .entryPoint = "computeSobelX",
                                                      .code = Utils::readFile("shaders/gradientx.wgsl"),
                                                      .workgroupSize = workgroupSize
                                                  },
                                                  .inputImageBuffers = { brainImageBuffer },
                                                  .outputImageBuffers = { gradientXBuffer },
                                                  };

        gpu::ComputeOperationData gradientYOpData {
                                                  .shader = {
                                                      .name="sobely",
                                                      .entryPoint = "computeSobelY",
                                                      .code = Utils::readFile("shaders/gradienty.wgsl"),
                                                      .workgroupSize = workgroupSize
                                                  },
                                                  .inputImageBuffers = { brainImageBuffer },
                                                  .outputImageBuffers = { gradientYBuffer },
                                                  };

        auto gradientXOp = gpu::createComputeOperation(gradientXOpData, wgpuContext);
        auto gradientYOp = gpu::createComputeOperation(gradientYOpData, wgpuContext);
        gpu::dispatchOperation(gradientXOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);
        gpu::dispatchOperation(gradientYOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);

        struct TransformationParameters {
            float rotationAngle = 0.0F;
            float translationX = 0.0F;
            float translationY = 0.0F;
            float _padding = 0.0F; // WGSL requires to align to 16 bytes
        } parameters = {Utils::degreesToRadians(45.0F), 30, 20 };

        auto outputBuffer2 = gpu::makeEmptyTextureBuffer(gpu::TextureSpecification {
                                                             .width = image.width,
                                                             .height = image.height,
                                                             .format = gpu::TextureFormat::R8Unorm,
                                                             .usage = gpu::ResourceUsage::ReadWrite
                                                         },
                                                         wgpuContext);

        auto paramsBuffer = gpu::makeUniformBuffer(reinterpret_cast<uint8_t*>(&parameters),
                                                   sizeof(TransformationParameters),
                                                   wgpuContext);

        gpu::ComputeOperationData data2 {
            .shader {
                .name = "transform",
                .entryPoint = "computeTransform",
                .code = Utils::readFile("shaders/transformimage.wgsl"),
                .workgroupSize = workgroupSize
            },
            .uniformBuffers = { paramsBuffer },
            .inputImageBuffers = { gradientXBuffer },
            .outputImageBuffers = { outputBuffer2 },
            .samplers = { gpu::createLinearSampler(wgpuContext) }
        };

        auto transformOp = gpu::createComputeOperation(data2, wgpuContext);
        gpu::dispatchOperation(transformOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);

        auto gradientsOutputBuffer = gpu::makeEmptyDataBuffer(sizeof(int32_t) * 4, gpu::ResourceUsage::ReadWrite, wgpuContext);

        gpu::ComputeOperationData updateGradientsOpData {
            .shader {
                .name = "updategradients",
                .entryPoint = "updateGradients",
                .code = Utils::readFile("shaders/updategradients.wgsl"),
                .workgroupSize = workgroupSize
            },
            .uniformBuffers = { paramsBuffer },
            .inputImageBuffers = { gradientXBuffer, gradientYBuffer, brainImageBuffer, outputBuffer2 },
            .outputBuffers = {gradientsOutputBuffer}
        };
        auto updateGradientsOp = gpu::createComputeOperation(updateGradientsOpData, wgpuContext);

        constexpr float threshold = 0.01F;
        constexpr int maxIterations = 100;
        int i = 0;

        const auto angleLearningRate = 0.0001;
        const auto txLearningRate = 0.1;
        const auto tyLearningRate = 0.1;

        while(i < maxIterations) {
            gpu::dispatchOperation(updateGradientsOp, calcWorkgroupGrid(image, workgroupSize), wgpuContext);

            // Read the output buffer and check if the SSD is below a threshold
            std::array<int32_t, 4> data;
            gpu::readBufferFromGPU(data.data(), gradientsOutputBuffer, wgpuContext);

            // Update the parameters
            parameters.rotationAngle -= angleLearningRate * data[0]/1000.0;
            parameters.translationX -= txLearningRate * data[1]/1000.0;
            parameters.translationY -= tyLearningRate * data[2]/1000.0;

            gpu::updateUniformBuffer(paramsBuffer, reinterpret_cast<uint8_t*>(&parameters), sizeof(TransformationParameters), wgpuContext);
            // print data
            std::cout << "data: ";
            for(int j = 0; j < 4; ++j) {
                std::cout << data[j] << " ";
            }
            std::cout << std::endl;
            // if(data[3] < threshold) {
            //     break;
            // }

            data = {0,0,0,0};
            gpu::updateDataBuffer(data.data(), gradientsOutputBuffer, wgpuContext);

            ++i;
        }

        // auto outputImage = gpu::makeHostImageFromBuffer(outputBuffer, wgpuContext);
        // auto outputImage2 = gpu::makeHostImageFromBuffer(outputBuffer2, wgpuContext);
        // Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
        // Utils::saveToDisk(outputImage2, "data/brain_transform.pgm");
        auto gradientYImage = gpu::makeHostImageFromBuffer(gradientYBuffer, wgpuContext);
        Utils::saveToDisk(gradientYImage, "data/brain_sobely.pgm");
    }
    catch(const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
