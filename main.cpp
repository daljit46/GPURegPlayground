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
    auto outputBuffer = gpu::makeEmptyTextureBuffer(image.width, image.height,
                                                    gpu::TextureUsage::ReadWrite,
                                                    gpu::TextureFormat::R32Float,
                                                    wgpuContext
                                                    );
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
    gpu::dispatchOperation(computeOp, {image.width * image.height / 64, 1, 1}, wgpuContext);

    // Let's wait a few milliseconds for the operation to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    auto outputImage = gpu::makeHostImageFromBuffer(outputBuffer, wgpuContext);
    Utils::saveToDisk(outputImage, "data/brain_sobelx.pgm");
}
