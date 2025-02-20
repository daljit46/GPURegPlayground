#include "adamoptimiser.h"
#include "adabeliefoptimiser.h"
#include "image.h"
#include "gpu.h"
#include "reduce.h"
#include "spdlog/spdlog.h"
#include "scopedtimer.h"
#include "transform.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <matplot/matplot.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <future>



struct TransformationParameters {
    float alpha = Utils::degreesToRadians(0.0F);
    float beta = 0.0F;
    float gamma = 0.0F;
    float tx = 0.0F;
    float ty = 0.0F;
    float tz = 0.0F;
};

struct SSDGradients {
    float ssd          = 0;
    float dssd_dalpha  = 0;
    float dssd_dbeta   = 0;
    float dssd_dgamma  = 0;
    float dssd_dtx     = 0;
    float dssd_dty     = 0;
    float dssd_dtz     = 0;

    SSDGradients operator+(const SSDGradients &other) const {
        return {
            .ssd = ssd + other.ssd,
            .dssd_dalpha = dssd_dalpha + other.dssd_dalpha,
            .dssd_dbeta = dssd_dbeta + other.dssd_dbeta,
            .dssd_dgamma = dssd_dgamma + other.dssd_dgamma,
            .dssd_dtx = dssd_dtx + other.dssd_dtx,
            .dssd_dty = dssd_dty + other.dssd_dty,
            .dssd_dtz = dssd_dtz + other.dssd_dtz
        };
    }
};


// Let I=I(x,y,z) be the target image and J = J(T(x, y, z)) be the moving image
// Let I' = I - mean(I) and J' = J - mean(J)
// NOTE: J = J(T(x,y,z)) where T is the transformation function
// NCC = A / sqrt(B * C) where
// A = sum(I' * J')
// B = sum(I' * I')
// C = sum(J' * J')
// The sum is over all voxels in the images.
// dNCC/dp_k = 1/[sqrt(B) * C^3/2] * [dA/dp_k * C - 0.5 A * dC/dp_k] where
// p_k is the k-th transformation parameter
// where dA/dp_k = sum[I' * (gradJ) dotted dT/dp_k - d/dp_k(mean(J))]
// where dC/dp_k = 2 * sum[J' * (gradJ dotted dT/dp_k - d/dp_k(mean(J))]
// For simplicity, we will assume that d/dp_k(mean(J)) = 0 even though this is not strictly true

struct NCCPartialSums {
    float sumA = 0;
    float sumB = 0;
    float sumC = 0;

    float dA_dalpha = 0;
    float dA_dbeta  = 0;
    float dA_dgamma = 0;
    float dA_dtx    = 0;
    float dA_dty    = 0;
    float dA_dtz    = 0;

    float dC_dalpha = 0;
    float dC_dbeta  = 0;
    float dC_dgamma = 0;
    float dC_dtx    = 0;
    float dC_dty    = 0;
    float dC_dtz    = 0;

    NCCPartialSums operator+(const NCCPartialSums& other) {
        return {
            .sumA = sumA + other.sumA,
            .sumB = sumB + other.sumB,
            .sumC = sumC + other.sumC,
            .dA_dalpha = dA_dalpha + other.dA_dalpha,
            .dA_dbeta  = dA_dbeta  + other.dA_dbeta,
            .dA_dgamma = dA_dgamma + other.dA_dgamma,
            .dA_dtx    = dA_dtx    + other.dA_dtx,
            .dA_dty    = dA_dty    + other.dA_dty,
            .dA_dtz    = dA_dtz    + other.dA_dtz,
        };
    }
};

struct MIGradients {
    float grad_alpha = 0;
    float grad_beta  = 0;
    float grad_gamma = 0;
    float grad_tx    = 0;
    float grad_ty    = 0;
    float grad_tz    = 0;

    MIGradients operator+(const MIGradients &other) {
        return {
            .grad_alpha = grad_alpha + other.grad_alpha,
            .grad_beta  = grad_beta  + other.grad_beta,
            .grad_gamma = grad_gamma + other.grad_gamma,
            .grad_tx    = grad_tx    + other.grad_tx,
            .grad_ty    = grad_ty    + other.grad_ty,
            .grad_tz    = grad_tz    + other.grad_tz
        };
    }
};


gpu::Texture downsample3DTexture(
    gpu::Context &context,
    const gpu::Texture &inputTexture,
    const gpu::WorkgroupSize &workgroupSize)
{
    gpu::Texture outputTexture = context.makeEmptyTexture({
        .size = { inputTexture.size.width / 2,
                 inputTexture.size.height / 2,
                 inputTexture.size.depth / 2 },
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    gpu::KernelDescriptor downsampleDesc {
        .shader = {
            .name = "downsample_3d",
            .entryPoint = "main",
            .filePath = "shaders/3d/downsample_3d.wgsl",
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = {},
        .inputTextures = { inputTexture },
        .outputTextures = { outputTexture },
        .samplers = { context.makeLinearSampler() }
    };

    auto downsampleKernel = context.makeKernel(downsampleDesc);

    // Dispatch threads for the new (half-sized) volume
    const gpu::WorkgroupGrid grid {
        .x = (outputTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (outputTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (outputTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    context.dispatchKernel(downsampleKernel, grid);
    return outputTexture;
}

struct SingleLevelResult {
    float finalSSD = 0.0F;
    std::vector<float> ssdHistory;
    // Final transform parameters for chaining to the next level
    float alpha = 0.0F; float beta  = 0.0F; float gamma = 0.0F;
    float tx    = 0.0F; float ty    = 0.0F; float tz    = 0.0F;
};


SingleLevelResult registerAtSingleResolutionGPUOnly(
    gpu::Context &context,
    const gpu::Texture &sourceTexture,
    const gpu::Texture &targetTexture,
    TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    float rotationLearningRate,
    float translationLearningRate,
    int maxIterations
)
{
    const gpu::WorkgroupGrid workgrid {
        .x = (sourceTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (sourceTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (sourceTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    gpu::DataBuffer transformationBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    context.writeToBuffer(transformationBuffer, &transformationParams);

    // Size of array of SSD gradients is the number of workgroups in the grid times size of SSDGradients struct
    const uint32_t workgroupCount = workgrid.x * workgrid.y * workgrid.z;
    spdlog::info("Workgroup Count: {}", workgroupCount);

    constexpr uint32_t reductionWorkgroupSize = 256;
    // The SSD Gradient buffer must have a size that is a multiple of the reduction workgroup size
    // as that's required by the ReductionHelper
    size_t ssdGradientsCount = workgroupCount + (reductionWorkgroupSize - workgroupCount % reductionWorkgroupSize);
    spdlog::info("SSD Gradients Count: {}", ssdGradientsCount);
    gpu::DataBuffer ssdGradientsBuffer = context.makeEmptyBuffer(ssdGradientsCount * sizeof(SSDGradients));
    gpu::DataBuffer reductionResultBuffer = context.makeEmptyBuffer(sizeof(SSDGradients));

    const gpu::KernelDescriptor gradientDescentDesc {
        .shader = {
            .name = "gradientdescent",
            .entryPoint = "main",
            .filePath = "shaders/3d/ssd/gradientdescent_3d.wgsl",
            .workgroupSize = workgroupSize
        },
        .inputBuffers = { transformationBuffer },
        .inputTextures  = { targetTexture, sourceTexture },
        .outputBuffers  = { ssdGradientsBuffer },
        .samplers       = { context.makeLinearSampler() }
    };

    const gpu::ReductionDescriptor reductionDesc{
        .workgroupSize = reductionWorkgroupSize,
        .groupSize = sizeof(SSDGradients) / sizeof(float),
        .data = ssdGradientsBuffer,
        .result = reductionResultBuffer
    };

    gpu::ReductionHelper reductionHelper(reductionDesc, context);

    // AdaBelief optimiser needs to keep track of state across iterations
    // 6 learning rates, 6 first moments and 6 second moments
    std::array<float, 18> initialAdamState = {
        // Learning rates for 3 angles and 3 translations
        rotationLearningRate, rotationLearningRate, rotationLearningRate,
        translationLearningRate,  translationLearningRate,  translationLearningRate,
        // First moments
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        // Second moments
        0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F
    };

    const gpu::DataBuffer adamStateBuffer = context.makeEmptyBuffer(sizeof(float) * initialAdamState.size());
    context.writeToBuffer(adamStateBuffer, initialAdamState.data());

    const gpu::DataBuffer minSSDBuffer = context.makeEmptyBuffer(sizeof(float));
    const gpu::DataBuffer minTransformParametersBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    const gpu::DataBuffer ssdHistoryBuffer = context.makeEmptyBuffer(maxIterations * sizeof(SSDGradients));
    const gpu::DataBuffer currentIterationBuffer = context.makeEmptyBuffer(sizeof(uint32_t));
    const gpu::DataBuffer stopIterationBuffer = context.makeEmptyBuffer(sizeof(uint32_t));
    uint32_t stopIteration = 0u;
    context.writeToBuffer(stopIterationBuffer, &stopIteration);

    const gpu::KernelDescriptor optimiserStepDesc {
        .shader = {
            .name = "optimiser",
            .entryPoint = "main",
            .filePath = "shaders/3d/ssd/optimiser_3d.wgsl",
            .workgroupSize = {1, 1, 1}
        },
        .outputBuffers = {
            reductionResultBuffer, // TODO: this is actually an input to the shader
            adamStateBuffer,
            transformationBuffer,
            minSSDBuffer,
            minTransformParametersBuffer,
            ssdHistoryBuffer,
            currentIterationBuffer,
            stopIterationBuffer
        },
    };


    auto gradientDescentKernel  = context.makeKernel(gradientDescentDesc);
    auto adamStepKernel = context.makeKernel(optimiserStepDesc);

    float minSSD = std::numeric_limits<float>::max();


    for (int i = 0; i < maxIterations; ++i) {
        context.dispatchKernel(gradientDescentKernel, workgrid);
        reductionHelper.dispatch(context);
        context.dispatchKernel(adamStepKernel, {1, 1, 1});

        // Every 5 iterations, check if we should stop
        if (i % 5 == 0) {
            context.downloadBuffer(stopIterationBuffer, &stopIteration);
            if (stopIteration) {
                break;
            }
        }
    }

    // Download the SSD history from GPU
    std::vector<SSDGradients> ssdGradientsHistory(maxIterations);

    std::vector<gpu::Context::BufferMappingPair> buffersToDownload = {
        {&ssdHistoryBuffer, ssdGradientsHistory.data()},
        {&minSSDBuffer, &minSSD},
        {&minTransformParametersBuffer, &transformationParams}
    };

    context.downloadBuffers(buffersToDownload);

    spdlog::info("SSD History:");
    for (int i = 0; i < maxIterations; ++i) {
        SSDGradients ssdGradient = ssdGradientsHistory[i];
        SSDGradients zeroGradient = {};
        if(ssdGradient.ssd == zeroGradient.ssd && i > 0) {
            break;
        }
        spdlog::info("{} SSD: {}, dssd_dalpha: {}, dssd_dbeta: {}, dssd_dgamma: {}, dssd_dtx: {}, dssd_dty: {}, dssd_dtz: {}",
                     i, ssdGradient.ssd, ssdGradient.dssd_dalpha, ssdGradient.dssd_dbeta, ssdGradient.dssd_dgamma,
                     ssdGradient.dssd_dtx, ssdGradient.dssd_dty, ssdGradient.dssd_dtz);
    }


    spdlog::info("Final SSD: {}", minSSD);
    spdlog::info("Final Transform: Alpha: {}, Beta: {}, Gamma: {}, Tx: {}, Ty: {}, Tz: {}",
                 transformationParams.alpha, transformationParams.beta, transformationParams.gamma,
                 transformationParams.tx, transformationParams.ty, transformationParams.tz);


    std::vector<float> ssdHistory;
    ssdHistory.reserve(maxIterations);
    for (int i = 0; i < maxIterations; ++i) {
        ssdHistory.push_back(ssdGradientsHistory[i].ssd);
    }
    // Return final result for chaining
    SingleLevelResult result;
    result.finalSSD = minSSD;
    result.ssdHistory = ssdHistory;
    result.alpha = transformationParams.alpha;
    result.beta  = transformationParams.beta;
    result.gamma = transformationParams.gamma;
    result.tx    = transformationParams.tx;
    result.ty    = transformationParams.ty;
    result.tz    = transformationParams.tz;
    return result;
}



SingleLevelResult registerAtSingleResolution(
    gpu::Context &context,
    const gpu::Texture &sourceTexture,
    const gpu::Texture &targetTexture,
    const TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    float rotationLearningRate,
    float translationLearningRate,
    int maxIterations)
{
    TransformationParameters newTransformationParams = transformationParams;
    // Setup Adam with 6 parameters
    std::vector<AdaBeliefOptimiser::Parameter> parameters = {
        {.value = newTransformationParams.alpha, .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.beta,  .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.gamma, .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.tx,    .learning_rate = translationLearningRate },
        {.value = newTransformationParams.ty,    .learning_rate = translationLearningRate },
        {.value = newTransformationParams.tz,    .learning_rate = translationLearningRate }
    };
    AdaBeliefOptimiser optimizer(parameters);

    const gpu::WorkgroupGrid workgrid {
        .x = (sourceTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (sourceTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (sourceTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    auto transformationParamsBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);

    SSDGradients ssdGradients;

    // Size of array of SSD gradients is the number of workgroups in the grid times size of SSDGradients struct
    const uint32_t workgroupCount = workgrid.x * workgrid.y * workgrid.z;
    spdlog::info("Workgroup Count: {}", workgroupCount);
    const size_t ssdGradientsSize = sizeof(SSDGradients) * workgroupCount;
    gpu::DataBuffer ssdGradientsBuffer = context.makeEmptyBuffer(ssdGradientsSize);

    const gpu::KernelDescriptor gradientDescentDesc {
        .shader = {
            .name = "gradientdescent",
            .entryPoint = "main",
            .filePath = "shaders/3d/ssd/gradientdescent_3d.wgsl",
            .workgroupSize = workgroupSize
        },
        .inputBuffers = { transformationParamsBuffer },
        .inputTextures  = { targetTexture, sourceTexture },
        .outputBuffers  = { ssdGradientsBuffer },
        .samplers       = { context.makeLinearSampler() }
    };

    auto gradientDescentKernel  = context.makeKernel(gradientDescentDesc);

    float minSSD = std::numeric_limits<float>::max();

    std::vector<float> ssdHistory;
    ssdHistory.reserve(maxIterations);

    for (int i = 0; i < maxIterations; ++i) {
        context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);

        ssdGradients = {};
        context.dispatchKernel(gradientDescentKernel, workgrid);

        std::vector<SSDGradients> ssdGradientsVec(workgroupCount);
        context.downloadBuffer(ssdGradientsBuffer, ssdGradientsVec.data());
        ssdGradients = std::reduce(ssdGradientsVec.begin(), ssdGradientsVec.end(), SSDGradients{});

        const float ssd         = ssdGradients.ssd;
        const float dssd_dalpha = ssdGradients.dssd_dalpha;
        const float dssd_dbeta  = ssdGradients.dssd_dbeta;
        const float dssd_dgamma = ssdGradients.dssd_dgamma;
        const float dssd_dtx    = ssdGradients.dssd_dtx;
        const float dssd_dty    = ssdGradients.dssd_dty;
        const float dssd_dtz    = ssdGradients.dssd_dtz;

        spdlog::info(
            "SSD: {} dAlpha: {} dBeta: {} dGamma: {} dTx: {} dTy: {} dTz: {}",
            ssd, dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz
            );

        if (ssd < minSSD) {
            minSSD = ssd;
        }

        auto newParams = optimizer.step({
            dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz
        });

        newTransformationParams.alpha = newParams[0].value;
        newTransformationParams.beta  = newParams[1].value;
        newTransformationParams.gamma = newParams[2].value;
        newTransformationParams.tx    = newParams[3].value;
        newTransformationParams.ty    = newParams[4].value;
        newTransformationParams.tz    = newParams[5].value;

        ssdHistory.push_back(ssd);

        spdlog::info(
            "Iteration: {} | SSD: {} | Alpha: {} Beta: {} Gamma: {} Tx: {} Ty: {} Tz: {}",
            i,
            ssd,
            newTransformationParams.alpha,
            newTransformationParams.beta,
            newTransformationParams.gamma,
            newTransformationParams.tx,
            newTransformationParams.ty,
            newTransformationParams.tz
            );

        // if we see no improvement in the last 20 iterations, break
        if(i > 10) {
            auto mean = std::accumulate(ssdHistory.end()-10, ssdHistory.end(), 0.0F) / 10;
            if (std::abs(ssd - mean) < 0.01) {
                break;
            }
        }
    }

    // Return final result for chaining
    SingleLevelResult result;
    result.finalSSD = minSSD;
    result.ssdHistory = ssdHistory;
    result.alpha = newTransformationParams.alpha;
    result.beta  = newTransformationParams.beta;
    result.gamma = newTransformationParams.gamma;
    result.tx    = newTransformationParams.tx;
    result.ty    = newTransformationParams.ty;
    result.tz    = newTransformationParams.tz;
    return result;
}


SingleLevelResult registerAtSingleResolutionNCC(
    gpu::Context &context,
    const gpu::Texture &sourceTexture,
    const gpu::Texture &targetTexture,
    const TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    float rotationLearningRate,
    float translationLearningRate,
    int maxIterations
)
{
    // Plan:
    // 1. Compute the mean of the target image
    // In the iteration loop:
    // - Compute the mean of the moving image
    // - Dispatch the kernel to compute the NCC partial sums
    // - Download the partial sums on CPU and compute the gradients
    // - Update the transformation parameters
    // - Repeat


    TransformationParameters newTransformationParams = transformationParams;
    // Setup Adam with 6 parameters
    const std::vector<AdaBeliefOptimiser::Parameter> parameters = {
        {.value = newTransformationParams.alpha, .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.beta,  .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.gamma, .learning_rate = rotationLearningRate },
        {.value = newTransformationParams.tx,    .learning_rate = translationLearningRate },
        {.value = newTransformationParams.ty,    .learning_rate = translationLearningRate },
        {.value = newTransformationParams.tz,    .learning_rate = translationLearningRate }
    };
    AdaBeliefOptimiser optimizer(parameters);

    const gpu::WorkgroupGrid workgrid {
        .x = (sourceTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (sourceTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (sourceTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    NCCPartialSums nccPartialSums;
    const size_t nccPartialSumsSize = sizeof(NCCPartialSums) * workgrid.totalCount();
    gpu::DataBuffer nccPartialSumsBuffer = context.makeEmptyBuffer(nccPartialSumsSize);
    gpu::DataBuffer targetMeanBuffer = context.makeEmptyBuffer(sizeof(float));
    gpu::DataBuffer movingMeanBuffer = context.makeEmptyBuffer(sizeof(float));
    gpu::DataBuffer transformationParamsBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);

    // meanIntermediateBufferSize needs to be a multiple of workgroupSize
    const size_t reductionWorkgroupSize = 256;
    const size_t meanIntermediateBufferSize = sizeof(float) * Utils::nextMultipleOf(workgrid.totalCount(), reductionWorkgroupSize);
    gpu::DataBuffer targetMeanIntermediateBuffer = context.makeEmptyBuffer(meanIntermediateBufferSize);
    const gpu::KernelDescriptor targetMeanKernelDesc {
        .shader = {
            .filePath = "shaders/3d/reduction_image_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = { {"operation", "0u"} }
        },
        .inputTextures = { targetTexture    },
        .outputBuffers = { targetMeanIntermediateBuffer },
    };
    gpu::Kernel targetMeanKernel = context.makeKernel(targetMeanKernelDesc);
    context.dispatchKernel(targetMeanKernel, workgrid);

    const gpu::ReductionDescriptor targetMeanReductionDesc {
        .workgroupSize = reductionWorkgroupSize,
        .groupSize = 1,
        .data = targetMeanIntermediateBuffer,
        .result = targetMeanBuffer
    };
    gpu::ReductionHelper targetMeanReductionHelper(targetMeanReductionDesc, context);
    targetMeanReductionHelper.dispatch(context);

    gpu::DataBuffer sourceMeanIntermediateBuffer = context.makeEmptyBuffer(meanIntermediateBufferSize);

    const gpu::KernelDescriptor movingMeanKernelDesc {
        .shader = {
            .filePath = "shaders/3d/reduction_image_transformed_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = { {"operation", "0u"} }
        },
        .inputBuffers  = { transformationParamsBuffer },
        .inputTextures = { sourceTexture },
        .outputBuffers = { sourceMeanIntermediateBuffer },
        .samplers      = { context.makeLinearSampler() }
    };
    gpu::Kernel movingMeanKernel = context.makeKernel(movingMeanKernelDesc);

    const gpu::ReductionDescriptor movingMeanReductionDesc {
        .workgroupSize = 256,
        .groupSize = 1,
        .data = sourceMeanIntermediateBuffer,
        .result = movingMeanBuffer
    };
    gpu::ReductionHelper movingMeanReductionHelper(movingMeanReductionDesc, context);

    const gpu::KernelDescriptor updateGradientsDesc {
        .shader = {
            .name = "Update Gradients NCC",
            .filePath = "shaders/3d/ncc/updategradients_ncc_3d.wgsl",
            .workgroupSize = workgroupSize
        },
        .inputBuffers   = { transformationParamsBuffer, targetMeanBuffer, movingMeanBuffer },
        .inputTextures  = { targetTexture, sourceTexture },
        .outputBuffers  = { nccPartialSumsBuffer },
        .samplers       = { context.makeLinearSampler() }
    };

    const gpu::Kernel updateGradientsKernel = context.makeKernel(updateGradientsDesc);
    float maxNCC = std::numeric_limits<float>::lowest();
    std::vector<float> nccHistory;
    nccHistory.reserve(maxIterations);

    for(int i = 0; i < maxIterations; ++i) {
        context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);

        nccPartialSums = {};
        context.dispatchKernel(movingMeanKernel, workgrid);
        movingMeanReductionHelper.dispatch(context);

        context.dispatchKernel(updateGradientsKernel, workgrid);
        std::vector<NCCPartialSums> nccPartialSumsVec(workgrid.totalCount());
        context.downloadBuffer(nccPartialSumsBuffer, nccPartialSumsVec.data());
        nccPartialSums = std::reduce(nccPartialSumsVec.begin(), nccPartialSumsVec.end(), NCCPartialSums{});

        const float ncc = nccPartialSums.sumA / std::sqrt(nccPartialSums.sumB * nccPartialSums.sumC);
        nccHistory.push_back(ncc);
        if(ncc > maxNCC) {
            maxNCC = ncc;
        }

        // dNCC/dp_k = 1/[sqrt(B) * C^3/2] * [dA/dp_k * C - 0.5 A * C * dC/dp_k] where

        const float sumA = nccPartialSums.sumA;
        const float sumB = nccPartialSums.sumB;
        const float sumC = nccPartialSums.sumC;
        const float dA_dalpha = nccPartialSums.dA_dalpha;
        const float dA_dbeta  = nccPartialSums.dA_dbeta;
        const float dA_dgamma = nccPartialSums.dA_dgamma;
        const float dA_dtx    = nccPartialSums.dA_dtx;
        const float dA_dty    = nccPartialSums.dA_dty;
        const float dA_dtz    = nccPartialSums.dA_dtz;
        const float dC_dalpha = nccPartialSums.dC_dalpha;
        const float dC_dbeta  = nccPartialSums.dC_dbeta;
        const float dC_dgamma = nccPartialSums.dC_dgamma;
        const float dC_dtx    = nccPartialSums.dC_dtx;
        const float dC_dty    = nccPartialSums.dC_dty;
        const float dC_dtz    = nccPartialSums.dC_dtz;

        const float denominmator = std::sqrt(sumB) * std::pow(sumC, 1.5F);
        const float dNCC_dalpha = ( sumC*dA_dalpha - 0.5F*sumA*dC_dalpha ) / denominmator;
        const float dNCC_dbeta  = ( sumC*dA_dbeta  - 0.5F*sumA*dC_dbeta  ) / denominmator;
        const float dNCC_dgamma = ( sumC*dA_dgamma - 0.5F*sumA*dC_dgamma ) / denominmator;
        const float dNCC_dtx    = ( sumC*dA_dtx    - 0.5F*sumA*dC_dtx    ) / denominmator;
        const float dNCC_dty    = ( sumC*dA_dty    - 0.5F*sumA*dC_dty    ) / denominmator;
        const float dNCC_dtz    = ( sumC*dA_dtz    - 0.5F*sumA*dC_dtz    ) / denominmator;

        spdlog::info(
            "Iteration: {} | NCC: {} | Alpha: {} Beta: {} Gamma: {} Tx: {} Ty: {} Tz: {}",
            i,
            ncc,
            newTransformationParams.alpha,
            newTransformationParams.beta,
            newTransformationParams.gamma,
            newTransformationParams.tx,
            newTransformationParams.ty,
            newTransformationParams.tz
            );

        auto newParams = optimizer.step({
            -dNCC_dalpha, -dNCC_dbeta, -dNCC_dgamma, -dNCC_dtx, -dNCC_dty, -dNCC_dtz
        });

        newTransformationParams.alpha = newParams[0].value;
        newTransformationParams.beta  = newParams[1].value;
        newTransformationParams.gamma = newParams[2].value;
        newTransformationParams.tx    = newParams[3].value;
        newTransformationParams.ty    = newParams[4].value;
        newTransformationParams.tz    = newParams[5].value;

        if(i > 10) {
            auto mean = std::accumulate(nccHistory.end()-10, nccHistory.end(), 0.0F) / 10;
            if (std::abs(ncc - mean) < 1e-3) {
                break;
            }
        }
    }

    SingleLevelResult result;
    result.finalSSD = maxNCC;
    result.ssdHistory = nccHistory;
    result.alpha = newTransformationParams.alpha;
    result.beta  = newTransformationParams.beta;
    result.gamma = newTransformationParams.gamma;
    result.tx    = newTransformationParams.tx;
    result.ty    = newTransformationParams.ty;
    result.tz    = newTransformationParams.tz;

    return result;
}


SingleLevelResult registerAtSingleResolutionMI(
    gpu::Context &context,
    const gpu::Texture &sourceTexture,
    const gpu::Texture &targetTexture,
    const TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    float rotationLearningRate,
    float translationLearningRate,
    int maxIterations,
    int numBins
    )
{
    TransformationParameters newTransformationParams = transformationParams;
    // Plan for MI:
    // 1. (Pre‐step) Ensure that the intensity bounds (min/max) for both target and moving images
    //    are available in buffers (minMaxTarget and minMaxMoving). These are used in the MI shaders.
    // 2. In each iteration:
    //    a. Dispatch the MI Joint Histogram kernel (Pass 1) to accumulate a soft joint histogram
    //       of target intensities and the transformed moving image.
    //    b. Dispatch the MI Lookup kernel (Pass 2) to normalize the histogram and compute
    //       the log–ratio lookup table L_ij along with the MI value.
    //    c. Dispatch the MI Gradient kernel (Pass 3) to compute per–voxel contributions to the
    //       gradient of MI (i.e. partial sums for each transformation parameter).
    //    d. Download and reduce the MI partial sums from the GPU.
    //    e. Update the transformation parameters.
    //    f. Repeat until convergence or maxIterations.

    const std::vector<AdaBeliefOptimiser::Parameter> parameters = {
        { .value = newTransformationParams.alpha, .learning_rate = rotationLearningRate },
        { .value = newTransformationParams.beta,  .learning_rate = rotationLearningRate },
        { .value = newTransformationParams.gamma, .learning_rate = rotationLearningRate },
        { .value = newTransformationParams.tx,    .learning_rate = translationLearningRate },
        { .value = newTransformationParams.ty,    .learning_rate = translationLearningRate },
        { .value = newTransformationParams.tz,    .learning_rate = translationLearningRate }
    };
    AdaBeliefOptimiser optimizer(parameters, 0.9);

    // Determine workgroup grid over the image domain.
    const gpu::WorkgroupGrid workgrid {
        .x = (targetTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (targetTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (targetTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    gpu::DataBuffer transformationParamsBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);

    gpu::DataBuffer jointHistogramBuffer = context.makeEmptyBuffer(sizeof(uint32_t) * numBins * numBins);
    gpu::DataBuffer miLookupBuffer       = context.makeEmptyBuffer(sizeof(float) * numBins * numBins);
    gpu::DataBuffer miResultBuffer       = context.makeEmptyBuffer(sizeof(float)); // holds final MI value

    const size_t miPartialSumsSize = sizeof(MIGradients) * workgrid.totalCount();
    gpu::DataBuffer miPartialSumsBuffer = context.makeEmptyBuffer(miPartialSumsSize);


    // Compute min max values for target and moving images.
    gpu::DataBuffer minMaxTargetBuffer = context.makeEmptyBuffer(sizeof(float) * 2);
    gpu::DataBuffer minMaxMovingBuffer = context.makeEmptyBuffer(sizeof(float) * 2);
    const size_t minMaxIntermediateBufferSize = Utils::nextMultipleOf(workgrid.totalCount(), workgroupSize.totalCount()) * 2;
    spdlog::info("Workgrid Total Count: {}", workgrid.totalCount());
    spdlog::info("Min Max Intermediate Buffer Size: {}", minMaxIntermediateBufferSize);
    spdlog::info("Workgroup Total Count: {}", workgroupSize.totalCount());
    gpu::DataBuffer minMaxIntermediateSourceBuffer = context.makeEmptyBuffer(sizeof(float) * minMaxIntermediateBufferSize);
    gpu::DataBuffer minMaxIntermediateTargetBuffer = context.makeEmptyBuffer(sizeof(float) * minMaxIntermediateBufferSize);

    const gpu::KernelDescriptor minMaxTargetKernelDesc {
        .shader = {
            .filePath = "shaders/3d/reduction_image_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                { "operations_size" , "2u" },
                { "operations" , "1u, 2u" } // min, max
            }
        },
        .inputTextures = { targetTexture },
        .outputBuffers = { minMaxIntermediateTargetBuffer }
    };
    const gpu::Kernel minMaxTargetKernel = context.makeKernel(minMaxTargetKernelDesc);
    context.dispatchKernel(minMaxTargetKernel, workgrid);


    const gpu::KernelDescriptor minMaxMovingTargetKernelDesc {
        .shader = {
            .name = "Min Max Moving",
            .filePath = "shaders/3d/reduction_image_transformed_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                { "operations_size" , "2u" },
                { "operations" , "1u, 2u" } // min, max
            }
        },
        .inputBuffers  = { transformationParamsBuffer },
        .inputTextures = { sourceTexture },
        .outputBuffers = { minMaxIntermediateSourceBuffer },
        .samplers      = { context.makeLinearSampler() }
    };
    const gpu::Kernel minMaxMovingKernel = context.makeKernel(minMaxMovingTargetKernelDesc);
    context.dispatchKernel(minMaxMovingKernel, workgrid);

    gpu::ReductionHelper minMaxMovingReductionHelper(
        {
         .workgroupSize = workgroupSize.totalCount(),
         .groupSize = 2,
         .data = minMaxIntermediateSourceBuffer,
         .result = minMaxMovingBuffer,
         .operations = { gpu::ReductionOperation::Min, gpu::ReductionOperation::Max }
        },
        context
    );
    gpu::ReductionHelper minMaxTargetReductionHelper(
        {
         .workgroupSize = workgroupSize.totalCount(),
         .groupSize = 2,
         .data = minMaxIntermediateTargetBuffer,
         .result = minMaxTargetBuffer,
         .operations = { gpu::ReductionOperation::Min, gpu::ReductionOperation::Max }
        },
        context
    );
    minMaxTargetReductionHelper.dispatch(context);
    minMaxMovingReductionHelper.dispatch(context);

    auto linearSampler = context.makeLinearSampler();

    gpu::KernelDescriptor miJointHistDesc {
        .shader = {
            .filePath = "shaders/3d/mi/joint_histogram_3d.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                { "numBins", std::to_string(numBins) }
            }
        },
        .inputBuffers  = { transformationParamsBuffer, minMaxTargetBuffer, minMaxMovingBuffer },
        .inputTextures = { targetTexture, sourceTexture },
        .outputBuffers = { jointHistogramBuffer },
        .samplers      = { linearSampler }
    };
    gpu::Kernel miJointHistogramKernel = context.makeKernel(miJointHistDesc);

    const gpu::DataBuffer totalMassBuffer = context.makeEmptyBuffer(sizeof(float));
    gpu::KernelDescriptor miLookupDesc {
        .shader = {
            .filePath = "shaders/3d/mi/compute_probabilities_lookup.wgsl",
            .workgroupSize = {1,1,1},
            .placeHolders = {
                { "numBins", std::to_string(numBins) }
            }
        },
        .inputBuffers  = { jointHistogramBuffer },
        .outputBuffers = { miLookupBuffer, miResultBuffer, totalMassBuffer },
    };
    gpu::Kernel miLookupKernel = context.makeKernel(miLookupDesc);

    gpu::KernelDescriptor miGradientDesc {
        .shader = {
            .filePath = "shaders/3d/mi/update_gradients_mi.wgsl",
            .workgroupSize = workgroupSize,
            .placeHolders = {
                { "numBins", std::to_string(numBins) }
            }
        },
        .inputBuffers  = { transformationParamsBuffer, minMaxTargetBuffer,
                           minMaxMovingBuffer, miLookupBuffer, totalMassBuffer
                         },
        .inputTextures = { targetTexture, sourceTexture },
        .outputBuffers = { miPartialSumsBuffer },
        .samplers      = { linearSampler }
    };

    gpu::Kernel miGradientKernel = context.makeKernel(miGradientDesc);

    std::vector<float> miHistory;
    miHistory.reserve(maxIterations);
    float maxMI = -std::numeric_limits<float>::infinity();

    std::vector<float> zeros(numBins * numBins, 0.0f);

    for (int i = 0; i < maxIterations; ++i) {
        context.writeToBuffer(transformationParamsBuffer, &newTransformationParams);
        context.dispatchKernel(minMaxMovingKernel, workgrid);

        minMaxMovingReductionHelper.dispatch(context);

        context.writeToBuffer(jointHistogramBuffer, zeros.data());
        context.dispatchKernel(miJointHistogramKernel, workgrid);

        context.dispatchKernel(miLookupKernel, {1, 1, 1});
        float miValue;
        context.downloadBuffer(miResultBuffer, &miValue);

        context.dispatchKernel(miGradientKernel, workgrid);

        std::vector<MIGradients> miGradientsList(workgrid.totalCount());
        context.downloadBuffer(miPartialSumsBuffer, miGradientsList.data());
        MIGradients miGradients = std::reduce(
            miGradientsList.begin(), miGradientsList.end(), MIGradients{0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

        miHistory.push_back(miValue);
        if (miValue > maxMI)
            maxMI = miValue;

        spdlog::info(
            "Iteration: {} | MI: {} | Alpha: {} Beta: {} Gamma: {} Tx: {} Ty: {} Tz: {}",
            i,
            miValue,
            newTransformationParams.alpha,
            newTransformationParams.beta,
            newTransformationParams.gamma,
            newTransformationParams.tx,
            newTransformationParams.ty,
            newTransformationParams.tz
            );

        const float dMI_dalpha = miGradients.grad_alpha;
        const float dMI_dbeta  = miGradients.grad_beta;
        const float dMI_dgamma = miGradients.grad_gamma;
        const float dMI_dtx    = miGradients.grad_tx;
        const float dMI_dty    = miGradients.grad_ty;
        const float dMI_dtz    = miGradients.grad_tz;

        // spdlog::info(
        //     "dMI_dalpha: {} dMI_dbeta: {} dMI_dgamma: {} dMI_dtx: {} dMI_dty: {} dMI_dtz: {}",
        //     dMI_dalpha, dMI_dbeta, dMI_dgamma, dMI_dtx, dMI_dty, dMI_dtz
        // );

        // Since MI is maximized during registration, we use the negative gradient
        auto newParams = optimizer.step({
            -dMI_dalpha, -dMI_dbeta, -dMI_dgamma, -dMI_dtx, -dMI_dty, -dMI_dtz
        });

        newTransformationParams.alpha = newParams[0].value;
        newTransformationParams.beta  = newParams[1].value;
        newTransformationParams.gamma = newParams[2].value;
        newTransformationParams.tx    = newParams[3].value;
        newTransformationParams.ty    = newParams[4].value;
        newTransformationParams.tz    = newParams[5].value;

        if (i > 10) {
            float recentMean = std::accumulate(miHistory.end()-10, miHistory.end(), 0.0F) / 10;
            if (std::abs(miValue - recentMean) < 1e-5)
                break;
        }
    }

    SingleLevelResult result;
    result.finalSSD = maxMI;  // Here, MI is maximized.
    result.ssdHistory = miHistory;
    result.alpha = newTransformationParams.alpha;
    result.beta  = newTransformationParams.beta;
    result.gamma = newTransformationParams.gamma;
    result.tx    = newTransformationParams.tx;
    result.ty    = newTransformationParams.ty;
    result.tz    = newTransformationParams.tz;

    return result;
}


int main(int argc, char **argv)
{
    enum class Metric { SSD, NCC, MI };

    std::vector<std::string> appArgs(argv, argv + argc);
    bool gpuOnlyVersion = std::find(appArgs.begin(), appArgs.end(), "--gpuonly") != appArgs.end();
    bool graphResults = std::find(appArgs.begin(), appArgs.end(), "--graph") != appArgs.end();
    bool randomInitialisation = std::find(appArgs.begin(), appArgs.end(), "--random") != appArgs.end();

    if (std::find(appArgs.begin(), appArgs.end(), "--trace") != appArgs.end()) {
        spdlog::set_level(spdlog::level::trace);
    }

    Metric metric = Metric::SSD;
    if(std::find(appArgs.begin(), appArgs.end(), "--ncc") != appArgs.end()) {
        spdlog::info("Using NCC as the metric");
        metric = Metric::NCC;
    }
    else if(std::find(appArgs.begin(), appArgs.end(), "--mi") != appArgs.end()) {
        spdlog::info("Using MI as the metric");
        metric = Metric::MI;
    }
    else {
        spdlog::info("Using SSD as the metric");
    }

    ScopedTimer timer ("main");
    auto context = gpu::Context::newContext();

    // Target transform
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> angleDist(-0.5, 0.5);
    std::uniform_real_distribution<float> translationDist(-30.0, 30.0);

    const float targetAlpha = randomInitialisation ? angleDist(gen) : 0.1F;
    const float targetBeta  = randomInitialisation ? angleDist(gen) : 0.4F;
    const float targetGamma = randomInitialisation ? angleDist(gen) : -0.3F;
    const float targetTx    = randomInitialisation ? translationDist(gen) : 10.0F;
    const float targetTy    = randomInitialisation ? translationDist(gen) : 29.0F;
    const float targetTz    = randomInitialisation ? translationDist(gen) : -23.0F;

    spdlog::info("Target Alpha: {}", targetAlpha);
    spdlog::info("Target Beta: {}", targetBeta);
    spdlog::info("Target Gamma: {}", targetGamma);
    spdlog::info("Target Tx: {}", targetTx);
    spdlog::info("Target Ty: {}", targetTy);
    spdlog::info("Target Tz: {}", targetTz);

    const NiftiImage sourceImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const NiftiImage targetImage = transformNifti(sourceImage,
                                                  { targetAlpha, targetBeta, targetGamma, targetTx, targetTy, targetTz});

    // Save the artificially transformed target
    // Utils::saveToDisk(targetImage, "target.nii");

    auto sourceTextureFull = context.makeTextureFromHostNifti(sourceImage);
    auto targetTextureFull = context.makeTextureFromHostNifti(targetImage);

    // ---------------------------------------------------
    // Create a 4-level pyramid for source & target
    // Level 3: full resolution
    // Level 2: half resolution
    // Level 1: quarter resolution
    // Level 0: eighth resolution
    // ---------------------------------------------------
    const gpu::WorkgroupSize downsampleWG = {4, 4, 4};
    auto sourceTextureHalf   = downsample3DTexture(context, sourceTextureFull, downsampleWG);
    auto targetTextureHalf   = downsample3DTexture(context, targetTextureFull, downsampleWG);
    auto sourceTextureQuarter = downsample3DTexture(context, sourceTextureHalf,  downsampleWG);
    auto targetTextureQuarter = downsample3DTexture(context, targetTextureHalf,  downsampleWG);
    auto sourceTextureEighth = downsample3DTexture(context, sourceTextureQuarter, downsampleWG);
    auto targetTextureEighth = downsample3DTexture(context, targetTextureQuarter, downsampleWG);


    const std::vector<gpu::Texture> sourcePyramid { sourceTextureEighth, sourceTextureQuarter, sourceTextureHalf, sourceTextureFull };
    const std::vector<gpu::Texture> targetPyramid { targetTextureEighth, targetTextureQuarter, targetTextureHalf, targetTextureFull };

    TransformationParameters transformationParams; // all zero by default

    // We'll run the same number of iterations at each level
    constexpr int maxIterations = 500;

    // For gradient-based approach, set learning rates for angles & translations.
    // Using max dimension from the *full* resolution
    const float maxImageDim = static_cast<float>(std::max(sourceImage.width, std::max(sourceImage.height, sourceImage.depth)));
    const float translationLearningRate = 2.0;
    const float angleLearningRate       = translationLearningRate / maxImageDim;

    // We'll store SSD history for final plotting
    std::vector<float> globalSSDHistory;

    const gpu::WorkgroupSize workgroupSize = {8,8,4};


    for (int level = 0; level < 4; ++level)
    {
        SingleLevelResult result;

        constexpr int numBinsMI = 32;
        // For the lowest level (level 0) and MI metric, try 5 random initializations.
        if(level == 0 && metric == Metric::MI)
        {
            spdlog::info("Performing 5 random initializations at the lowest pyramid level (MI).");
            float bestMI = -std::numeric_limits<float>::infinity();
            SingleLevelResult bestResult;
            constexpr int restartCount = 5;
            for (int trial = 0; trial < restartCount; ++trial)
            {
                // Since at level 0, the image is 1/8th the size of the full resolution,
                // we scale the expected translation by 8.
                const auto translationScalingFactor = 8.0;
                TransformationParameters trialParams;
                // For the first trial, use the initial parameters.
                if(trial > 0) {
                    trialParams.alpha = angleDist(gen);
                    trialParams.beta  = angleDist(gen);
                    trialParams.gamma = angleDist(gen);
                    trialParams.tx    = translationDist(gen) / translationScalingFactor;
                    trialParams.ty    = translationDist(gen) / translationScalingFactor;
                    trialParams.tz    = translationDist(gen) / translationScalingFactor;
                }
                spdlog::info("Level 0, Trial {}: Initial parameters: Alpha: {}, Beta: {}, Gamma: {}, Tx: {}, Ty: {}, Tz: {}",
                             trial, trialParams.alpha, trialParams.beta, trialParams.gamma,
                             trialParams.tx, trialParams.ty, trialParams.tz);

                SingleLevelResult trialResult = registerAtSingleResolutionMI(
                    context,
                    sourcePyramid[level],
                    targetPyramid[level],
                    trialParams,
                    workgroupSize,
                    angleLearningRate / std::pow(2, level + 1),
                    translationLearningRate / std::pow(2, level + 1),
                    maxIterations,
                    numBinsMI >> (4 - level)
                    );
                spdlog::info("Level 0, Trial {}: Final MI = {}", trial, trialResult.finalSSD);

                // Keep the trial with the highest MI.
                if(trialResult.finalSSD > bestMI)
                {
                    bestMI = trialResult.finalSSD;
                    bestResult = trialResult;
                }
            }
            result = bestResult;
            // Use the best trial's parameters as the starting point for the next levels.
            transformationParams.alpha = result.alpha;
            transformationParams.beta  = result.beta;
            transformationParams.gamma = result.gamma;
            transformationParams.tx    = result.tx;
            transformationParams.ty    = result.ty;
            transformationParams.tz    = result.tz;
            globalSSDHistory.insert(globalSSDHistory.end(), result.ssdHistory.begin(), result.ssdHistory.end());
        }
        else {
            // For all other levels (or metrics), perform registration as before.
            result = [&]() {
                if(metric == Metric::SSD) {
                    return gpuOnlyVersion ?
                               registerAtSingleResolutionGPUOnly(
                                   context,
                                   sourcePyramid[level],
                                   targetPyramid[level],
                                   transformationParams, // updated in place
                                   workgroupSize,
                                   angleLearningRate / std::pow(2, level + 1),
                                   translationLearningRate / std::pow(2, level + 1),
                                   maxIterations
                                   ) :
                               registerAtSingleResolution(
                                   context,
                                   sourcePyramid[level],
                                   targetPyramid[level],
                                   transformationParams, // updated in place
                                   workgroupSize,
                                   angleLearningRate / std::pow(2, level + 1),
                                   translationLearningRate / std::pow(2, level + 1),
                                   maxIterations
                                   );
                }
                else if(metric == Metric::NCC) {
                    return registerAtSingleResolutionNCC(
                        context,
                        sourcePyramid[level],
                        targetPyramid[level],
                        transformationParams,
                        workgroupSize,
                        angleLearningRate / std::pow(2, level + 1),
                        translationLearningRate / std::pow(2, level + 1),
                        maxIterations
                        );
                }
                else { // MI
                    return registerAtSingleResolutionMI(
                        context,
                        sourcePyramid[level],
                        targetPyramid[level],
                        transformationParams,
                        workgroupSize,
                        angleLearningRate / std::pow(2, level + 1),
                        translationLearningRate / std::pow(2, level + 1),
                        maxIterations,
                        numBinsMI >> (4 - level)
                        );
                }
            }();

            // Update transformation parameters from this level's registration.
            transformationParams.alpha = result.alpha;
            transformationParams.beta  = result.beta;
            transformationParams.gamma = result.gamma;
            transformationParams.tx    = result.tx;
            transformationParams.ty    = result.ty;
            transformationParams.tz    = result.tz;
            globalSSDHistory.insert(globalSSDHistory.end(), result.ssdHistory.begin(), result.ssdHistory.end());

        }

        // If not yet at the finest level, rescale translation parameters
        // to the next finer resolution. Rotations remain the same.
        if (level < 3)
        {
            // Because next level is double the dimension of the current,
            // the translation in voxel-space effectively doubles as well.
            transformationParams.tx *= 2.0F;
            transformationParams.ty *= 2.0F;
            transformationParams.tz *= 2.0F;
            spdlog::info("Upscaled translation for next level: Tx={}, Ty={}, Tz={}",
                         transformationParams.tx, transformationParams.ty, transformationParams.tz
                         );
        }
    }

    spdlog::info("Multi-resolution registration done.");
    spdlog::info("Initial parameters:");
    spdlog::info("  Alpha: {}", targetAlpha);
    spdlog::info("  Beta:  {}", targetBeta);
    spdlog::info("  Gamma: {}", targetGamma);
    spdlog::info("  Tx:    {}", targetTx);
    spdlog::info("  Ty:    {}", targetTy);
    spdlog::info("  Tz:    {}", targetTz);

    spdlog::info("Final parameters:");
    spdlog::info("  Alpha: {}", transformationParams.alpha);
    spdlog::info("  Beta:  {}", transformationParams.beta);
    spdlog::info("  Gamma: {}", transformationParams.gamma);
    // The transformation of the image is assumed to be wrt the center of the
    // voxels of the image, so we need to subtract half the voxel size from the
    // final translation parameters.
    const auto finalTx = transformationParams.tx - 0.5F;
    const auto finalTy = transformationParams.ty - 0.5F;
    const auto finalTz = transformationParams.tz - 0.5F;
    spdlog::info("  Tx:    {}", finalTx);
    spdlog::info("  Ty:    {}", finalTy);
    spdlog::info("  Tz:    {}", finalTz);

    spdlog::info("Difference between final transform and target:");
    spdlog::info("  Alpha: {}", transformationParams.alpha - targetAlpha);
    spdlog::info("  Beta:  {}", transformationParams.beta - targetBeta);
    spdlog::info("  Gamma: {}", transformationParams.gamma - targetGamma);
    spdlog::info("  Tx:    {}", finalTx - targetTx);
    spdlog::info("  Ty:    {}", finalTy - targetTy);
    spdlog::info("  Tz:    {}", finalTz - targetTz);

    if(graphResults) {
        // Plot the global SSD history
        matplot::plot(globalSSDHistory);
        matplot::title("SSD History (Multi-Level)");
        matplot::xlabel("Iteration");
        matplot::ylabel("SSD");
        matplot::show();
    }

    return 0;
}
