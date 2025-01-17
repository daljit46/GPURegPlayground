#include "gpu.h"
#include <cstdint>

namespace gpu {

struct GpuReductionDescriptor {
    // Workgroup size must be a multiple of 2
    uint32_t workgroupSize = 256;
    // if unitSize > 1, the operation will be performed as if unitSize separate values
    // e.g. {2, 2, 1, 1} with unitSize = 2 will be reduced to {4, 2}
    uint32_t unitSize = 1;
    gpu::DataBuffer data;
    gpu::DataBuffer result;
};

// Performs a reduction on the GPU with multiple kernel dispatches
void floatReduction(const GpuReductionDescriptor& dataDesc, const gpu::Context& gpuContext);

}
