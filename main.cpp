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

    auto outputBufferList = std::vector<gpu::TextureBuffer>{outputBuffer};
    gpu::ComputeOperationData data {
                                   .shader = {
                                       .name="sobelx",
                                       .code=Utils::readFile("shaders/transformimage.wgsl"),
                                       .entryPoint = "computeSobelX",
                                   },
                                   .inputImageBuffers = {wgpuImageBuffer},
                                   .outputImageBuffers = outputBufferList,
                                   };

    auto computeOp = gpu::createComputeOperation(data, wgpuContext);
    gpu::dispatchOperation(computeOp, gpu::WorkgroupGrid {image.width / 8, image.height / 8, 1}, wgpuContext);

    auto outputImage = gpu::makeHostImageFromBuffer(outputBuffer, wgpuContext);
    Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
}
