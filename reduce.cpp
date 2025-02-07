#include "reduce.h"
#include "gpu.h"
#include "spdlog/spdlog.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

gpu::ReductionHelper::ReductionHelper(const ReductionDescriptor &dataDesc, const gpu::Context &gpuContext)
{
    assert(dataDesc.data.size > 0 && "Input buffer size must be greater than 0");
    assert(dataDesc.data.size % sizeof(float) == 0 && "Input buffer size must be a multiple of 4 bytes");
    assert((dataDesc.data.size / sizeof(float)) / dataDesc.groupSize % dataDesc.workgroupSize == 0 &&
           "Number of units in the input buffer must be a multiple of workgroup size");
    assert(dataDesc.workgroupSize % 2 == 0 && "Workgroup size must be a multiple of 2");
    assert(dataDesc.data.size >= dataDesc.groupSize * sizeof(float) && "Input buffer size must be greater than or equal to unitSize * 4 bytes");
    assert(dataDesc.operations.size() == dataDesc.groupSize && "Number of operations must match unit size");

    const gpu::WorkgroupSize wgSize = { dataDesc.workgroupSize, 1, 1 };
    const size_t totalNumberOfUnits = dataDesc.data.size / sizeof(float) / dataDesc.groupSize;
    const gpu::DataBuffer &inputBuffer = dataDesc.data;
    const size_t numStages = static_cast<size_t>(std::ceil(std::log2(totalNumberOfUnits) / std::log2(wgSize.x)));

    for(size_t i = 0; i < numStages; ++i) {
        const bool isLastStage = i == numStages - 1;
        const size_t partialSumUnits = std::ceil(1.0 * totalNumberOfUnits / std::pow(wgSize.x, i + 1));
        if(!isLastStage) {
            m_partialSums.emplace_back(gpuContext.makeEmptyBuffer(partialSumUnits * sizeof(float) * dataDesc.groupSize));
        }
        else {
            m_partialSums.emplace_back(dataDesc.result);
        }

        const std::string operationString = [&]() {
            std::string s;
            for(const ReductionOperation operation : dataDesc.operations) {
                s += s.empty() ? "" : ",";
                switch (operation) {
                    case ReductionOperation::Sum: s += "0u"; break;
                    case ReductionOperation::Min: s += "1u"; break;
                    case ReductionOperation::Max: s += "2u"; break;
                }
            }
            return s;
        }();


        const gpu::KernelDescriptor reductionKernelDesc {
            .shader = {
                .name = "reduction_float_multi_stage",
                .entryPoint = "main",
                .filePath = "shaders/reduction_f32_multi_stage.wgsl",
                .workgroupSize = wgSize,
                .placeHolders = {
                    { "operations", operationString },
                    { "group_size", std::to_string(dataDesc.groupSize) }
                }
            },
            .inputBuffers = { i == 0 ? inputBuffer : m_partialSums[i - 1] },
            .outputBuffers = { m_partialSums[i] }
        };

        m_kernels.push_back(gpuContext.makeKernel(reductionKernelDesc));
    }

    m_totalNumberOfUnits = totalNumberOfUnits;
    m_unitSize = dataDesc.groupSize;
    assert(m_kernels.size() == numStages && "Number of kernels does not match number of stages");
}

void gpu::ReductionHelper::dispatch(const gpu::Context& gpuContext) const
{
    int i = 0;
    for(const gpu::Kernel& reductionKernel : m_kernels) {
        const auto wgSize = reductionKernel.workgroupSize;
        const uint32_t groupsX = m_partialSums[i].size / (sizeof(float) * m_unitSize);
        assert(groupsX > 0 && "Number of workgroups must be greater than 0");

        const gpu::WorkgroupGrid grid { groupsX, 1, 1 };
        gpuContext.dispatchKernel(reductionKernel, grid);

        ++i;
    }
}

