#include "gpu.h"
#include <cstdint>
#include <vector>

namespace gpu {


enum class ReductionOperation {
    Sum,
    Min,
    Max
};

struct ReductionDescriptor {
    // Workgroup size must be a multiple of 2
    uint32_t workgroupSize = 256;
    // if groupSize > 1, the operation will be performed as if the input buffer was split
    // into groups of groupSize elements. For each element in the group, the corresponding
    // operation in the operations list will be performed.
    // Example: groupSize = 4, operations = [Sum, Min, Max, Sum]
    // The first element of the group will be the sum of every 4 elements in the input buffer,
    // the second element will be the minimum of every 4 elements in the input buffer, etc.
    uint32_t groupSize = 1;
    // Size of number of elements in the input buffer must be a multiple of workgroupSize
    gpu::DataBuffer data;
    gpu::DataBuffer result;
    // List of operations for each item in the group
    std::vector<ReductionOperation> operations;
};

struct ReductionHelper {
    explicit ReductionHelper(const gpu::ReductionDescriptor& dataDesc, const gpu::Context& gpuContext);

    void dispatch(const gpu::Context& gpuContext) const;
private:
    std::vector<gpu::DataBuffer> m_partialSums;
    std::vector<gpu::Kernel> m_kernels;
    size_t m_totalNumberOfUnits = 0;
    uint32_t m_unitSize = 1;
};

}
