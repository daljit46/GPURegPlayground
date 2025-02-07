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
    // [[1, 2, 3, 4], [5, 6, 7, 8]] -> [1 + 5, min(2, 6), max(3, 7), 4 + 8] = [6, 2, 7, 12]
    uint32_t groupSize = 1;
    // Size of number of elements in the input buffer must be a multiple of workgroupSize
    gpu::DataBuffer data;
    gpu::DataBuffer result;
    // List of operations for each item in the group
    std::vector<ReductionOperation> operations = {ReductionOperation::Sum};
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
