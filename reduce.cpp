#include "reduce.h"
#include "gpu.h"
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
    assert(dataDesc.data.size / sizeof(float) % dataDesc.workgroupSize == 0 &&
           "Number of elements in the input buffer must be a multiple of workgroup size");
    assert(dataDesc.workgroupSize % 2 == 0 && "Workgroup size must be a multiple of 2");
    const gpu::WorkgroupSize wgSize = { dataDesc.workgroupSize, 1, 1 };

    const size_t N = dataDesc.data.size / sizeof(float);
    const gpu::DataBuffer &inputBuffer = dataDesc.data;
    const size_t numStages = static_cast<size_t>(std::ceil(std::log2(N) / std::log2(wgSize.x)));

    auto shaderSource = Utils::readFile("shaders/reduction_f32_multi_stage.wgsl");
    shaderSource = Utils::replacePlaceholder(shaderSource, "unit_size", std::to_string(dataDesc.unitSize));

    for(size_t i = 0; i < numStages; ++i) {
        const bool isLastStage = i == numStages - 1;
        const size_t partialSumSize = static_cast<size_t>(std::ceil(1.0 * N / std::pow(wgSize.x * dataDesc.unitSize, i + 1)));

        if(!isLastStage) {
            m_partialSums.emplace_back(gpuContext.makeEmptyBuffer(partialSumSize * sizeof(float)));
        }
        const gpu::KernelDescriptor reductionKernelDesc {
            .shader = {
                .name = "reduction_float_multi_stage",
                .entryPoint = "main",
                .code = shaderSource,
                .workgroupSize = wgSize
            },
            .inputBuffers = { i == 0 ? inputBuffer : m_partialSums[i - 1] },
            .outputBuffers = { isLastStage ? dataDesc.result : m_partialSums[i] }
        };

        m_kernels.push_back(gpuContext.makeKernel(reductionKernelDesc));
    }

    m_numberOfElements = N;
    assert(m_kernels.size() == numStages && "Number of kernels must match number of stages");
}

void gpu::ReductionHelper::dispatch(const gpu::Context& gpuContext)
{
    int i = 0;
    for(const gpu::Kernel& reductionKernel : m_kernels) {
        const auto wgSize = reductionKernel.workgroupSize;
        auto numElementsThisStage = std::max(1.0, m_numberOfElements / std::pow(wgSize.x, i));
        auto groupsX = static_cast<uint32_t>(std::ceil(numElementsThisStage / wgSize.x));

        gpu::WorkgroupGrid const grid { groupsX, 1, 1 };
        gpuContext.dispatchKernel(reductionKernel, grid);
        ++i;
    }
}

void gpu::ReductionHelper::dispatchIndirect(const Context &gpuContext, const DataBuffer &indirectBuffer)
{
    int i = 0;
    for(const gpu::Kernel& reductionKernel : m_kernels) {
        const auto wgSize = reductionKernel.workgroupSize;
        auto numElementsThisStage = std::max(1.0, m_numberOfElements / std::pow(wgSize.x, i));
        auto groupsX = static_cast<uint32_t>(std::ceil(numElementsThisStage / wgSize.x));

        gpu::WorkgroupGrid const grid { groupsX, 1, 1 };
        gpuContext.dispatchKernelIndirect(reductionKernel, indirectBuffer);
        ++i;
    }
}
