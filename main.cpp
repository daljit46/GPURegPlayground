#include <stdint.h>
#include <thread>
#include <vector>

#include "image.h"
#include "gpu.h"
#include "utils.h"

int main()
{
    auto wgpuContext = gpu::createWebGPUContext();
    auto image = Utils::loadFromDisk("data/brain.pgm");
    auto wgpuImageBuffer = gpu::makeReadOnlyTextureBuffer(image, wgpuContext);
    auto outputBuffer = gpu::makeEmptyTextureBuffer(gpu::TextureSpecification {
                                                        .width = image.width,
                                                        .height = image.height,
                                                        .format = gpu::TextureFormat::R8Unorm,
                                                        .usage = gpu::ResourceUsage::ReadWrite
                                                    },
                                                    wgpuContext);

    const gpu::WorkgroupSize workgroupSize {16, 16, 1};
    gpu::ComputeOperationData data {
                                   .shader = {
                                       .name="sobelx",
                                       .entryPoint = "computeSobelX",
                                       .code = Utils::readFile("shaders/gradientx.wgsl"),
                                       .workgroupSize = workgroupSize
                                   },
                                   .inputImageBuffers = { wgpuImageBuffer },
                                   .outputImageBuffers = { outputBuffer },
                                   };

    auto computeOp = gpu::createComputeOperation(data, wgpuContext);
    gpu::dispatchOperation(computeOp, gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1}, wgpuContext);

    struct TransformationParameters {
        float rotationAngle = 0.0f;
        float translationX = 0.0f;
        float translationY = 0.0f;
        float _padding = 0.0f;
    } parameters = {50, 30, 20 };

    auto outputBuffer2 = gpu::makeEmptyTextureBuffer(gpu::TextureSpecification {
                                                         .width = image.width,
                                                         .height = image.height,
                                                         .format = gpu::TextureFormat::R8Unorm,
                                                         .usage = gpu::ResourceUsage::ReadWrite
                                                     },
                                                     wgpuContext);

    gpu::ComputeOperationData data2 {
                                    .shader {
                                        .name = "transform",
                                        .entryPoint = "computeTransform",
                                        .code = Utils::readFile("shaders/transformimage.wgsl"),
                                        .workgroupSize = workgroupSize
                                    },
                                    .uniformBuffers = { gpu::makeUniformBuffer(reinterpret_cast<uint8_t*>(&parameters),
                                                                              sizeof(TransformationParameters),
                                                                              wgpuContext) },
                                    .inputImageBuffers = { wgpuImageBuffer },
                                    .outputImageBuffers = { outputBuffer2 },
                                    };

    auto computeOp2 = gpu::createComputeOperation(data2, wgpuContext);
    gpu::dispatchOperation(computeOp2, gpu::WorkgroupGrid {image.width / workgroupSize.x, image.height / workgroupSize.y, 1}, wgpuContext);

    auto outputImage = gpu::makeHostImageFromBuffer(outputBuffer, wgpuContext);
    auto outputImage2 = gpu::makeHostImageFromBuffer(outputBuffer2, wgpuContext);
    Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
    Utils::saveToDisk(outputImage2, "data/brain_transform.pgm");
}
