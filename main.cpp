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
    auto cpuImage = gpu::createHostImageFromBuffer(wgpuImageBuffer, wgpuContext);
}
