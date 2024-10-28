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
                                                        .usage = gpu::TextureUsage::ReadWrite
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

    auto outputImage = gpu::makeHostImageFromBuffer(outputBuffer, wgpuContext);
    Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
}
