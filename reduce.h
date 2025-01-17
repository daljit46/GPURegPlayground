#include "gpu.h"
#include <cstdint>
#include <vector>

namespace gpu {

struct ReductionDescriptor {
    // Workgroup size must be a multiple of 2
    uint32_t workgroupSize = 256;
    // if unitSize > 1, the operation will be performed as if unitSize separate values
    // e.g. {2, 2, 1, 1} with unitSize = 2 will be reduced to {4, 2}
    uint32_t unitSize = 1;
    // Size of number of elements in the input buffer must be a multiple of workgroupSize
    gpu::DataBuffer data;
    gpu::DataBuffer result;
};

struct ReductionHelper {
    explicit ReductionHelper(const gpu::ReductionDescriptor& dataDesc, const gpu::Context& gpuContext);

    void dispatch(const gpu::Context& gpuContext);
    void dispatchIndirect(const gpu::Context& gpuContext, const gpu::DataBuffer& indirectBuffer);

private:
    std::vector<gpu::DataBuffer> m_partialSums;
    std::vector<gpu::Kernel> m_kernels;
    size_t m_numberOfElements = 0;
};

}
