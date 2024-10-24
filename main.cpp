#include <stdint.h>
#include <vector>

#include "image.h"
#include "gpu.h"
#include "utils.h"

int main()
{
    auto wgpuContext = gpu::createWebGPUContext();
    auto image = Utils::loadFromDisk("data/brain.pgm");
    auto wgpuImageBuffer = gpu::createReadOnlyImageBuffer(image, wgpuContext.device);
    auto outputBuffer = gpu::createEmptyImageBuffer(wgpuContext.device);
    auto outputBufferList = std::vector<gpu::ImageBuffer>{outputBuffer};
    std::vector<gpu::Buffer> outputRawBufferList = {};
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
    gpu::dispatchOperation(computeOp, {8, 8, 1}, wgpuContext);

    auto outputImage = gpu::createHostImageFromBuffer(outputBuffer, wgpuContext);
    Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
}
