#include "adamoptimiser.h"
#include "image.h"
#include "gpu.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "scopedtimer.h"
#include "transform.h"

#include "reduce.h"
#include <matplot/matplot.h>
#include <vector>



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
            .code = Utils::readFile("shaders/3d/downsample_3d.wgsl"),
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
            .code = Utils::readFile("shaders/3d/gradientdescent_3d.wgsl"),
            .workgroupSize = workgroupSize
        },
        .inputBuffers = { transformationBuffer },
        .inputTextures  = { targetTexture, sourceTexture },
        .outputBuffers  = { ssdGradientsBuffer },
        .samplers       = { context.makeLinearSampler() }
    };

    const gpu::ReductionDescriptor reductionDesc{
        .workgroupSize = reductionWorkgroupSize,
        .unitSize = sizeof(SSDGradients) / sizeof(float),
        .data = ssdGradientsBuffer,
        .result = reductionResultBuffer
    };

    gpu::ReductionHelper reductionHelper(reductionDesc, context);

    // Adam optimiser needs to keep track of state across iterations
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

    const gpu::KernelDescriptor adamStepDesc {
        .shader = {
            .name = "adamstep",
            .entryPoint = "main",
            .code = Utils::readFile("shaders/3d/adamstep_3d.wgsl"),
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
    auto adamStepKernel = context.makeKernel(adamStepDesc);

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
        spdlog::info("SSD: {}, dssd_dalpha: {}, dssd_dbeta: {}, dssd_dgamma: {}, dssd_dtx: {}, dssd_dty: {}, dssd_dtz: {}",
                     ssdGradientsHistory[i].ssd, ssdGradientsHistory[i].dssd_dalpha,
                     ssdGradientsHistory[i].dssd_dbeta, ssdGradientsHistory[i].dssd_dgamma,
                     ssdGradientsHistory[i].dssd_dtx, ssdGradientsHistory[i].dssd_dty,
                     ssdGradientsHistory[i].dssd_dtz);
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
    TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    float rotationLearningRate,
    float translationLearningRate,
    int maxIterations)
{
    // Setup Adam with 6 parameters
    std::vector<AdamOptimizer::Parameter> parameters = {
        {.value = transformationParams.alpha, .learning_rate = rotationLearningRate },
        {.value = transformationParams.beta,  .learning_rate = rotationLearningRate },
        {.value = transformationParams.gamma, .learning_rate = rotationLearningRate },
        {.value = transformationParams.tx,    .learning_rate = translationLearningRate },
        {.value = transformationParams.ty,    .learning_rate = translationLearningRate },
        {.value = transformationParams.tz,    .learning_rate = translationLearningRate }
    };
    AdamOptimizer optimizer(parameters);

    const gpu::WorkgroupGrid workgrid {
        .x = (sourceTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (sourceTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (sourceTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    auto transformationParamsBuffer = context.makeEmptyBuffer(sizeof(TransformationParameters));
    context.writeToBuffer(transformationParamsBuffer, &transformationParams);

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
            .code = Utils::readFile("shaders/3d/gradientdescent_3d.wgsl"),
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
        context.writeToBuffer(transformationParamsBuffer, &transformationParams);

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

        transformationParams.alpha = newParams[0].value;
        transformationParams.beta  = newParams[1].value;
        transformationParams.gamma = newParams[2].value;
        transformationParams.tx    = newParams[3].value;
        transformationParams.ty    = newParams[4].value;
        transformationParams.tz    = newParams[5].value;

        ssdHistory.push_back(ssd);

        spdlog::info(
            "Iteration: {} | SSD: {} | Alpha: {} Beta: {} Gamma: {} Tx: {} Ty: {} Tz: {}",
            i,
            ssd,
            transformationParams.alpha,
            transformationParams.beta,
            transformationParams.gamma,
            transformationParams.tx,
            transformationParams.ty,
            transformationParams.tz
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
    result.alpha = transformationParams.alpha;
    result.beta  = transformationParams.beta;
    result.gamma = transformationParams.gamma;
    result.tx    = transformationParams.tx;
    result.ty    = transformationParams.ty;
    result.tz    = transformationParams.tz;
    return result;
}


int main(int argc, char **argv)
{
    std::vector<std::string> appArgs(argv, argv + argc);
    bool gpuOnlyVersion = false;
    if (std::find(appArgs.begin(), appArgs.end(), "--gpuonly") != appArgs.end()) {
        gpuOnlyVersion = true;
    }
    if (std::find(appArgs.begin(), appArgs.end(), "--trace") != appArgs.end()) {
        spdlog::set_level(spdlog::level::trace);
    }

    ScopedTimer timer ("main");
    auto context = gpu::Context::newContext();

    // Target transform
    const float targetAlpha = Utils::degreesToRadians(10.0F);
    const float targetBeta  = 0.4F;
    const float targetGamma = -0.3F;
    const float targetTx    = 10.0F;
    const float targetTy    = 29.0F;
    const float targetTz    = -23.0F;

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


    std::vector<gpu::Texture> sourcePyramid { sourceTextureEighth, sourceTextureQuarter, sourceTextureHalf, sourceTextureFull };
    std::vector<gpu::Texture> targetPyramid { targetTextureEighth, targetTextureQuarter, targetTextureHalf, targetTextureFull };

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
        spdlog::info("\n\n=== Registering at pyramid level {} (0=coarse, 3=full) ===", level);
        auto result = [&]() {
            if(gpuOnlyVersion) {
                return registerAtSingleResolutionGPUOnly(
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
            else return registerAtSingleResolution(
                context,
                sourcePyramid[level],
                targetPyramid[level],
                transformationParams, // updated in place
                workgroupSize,
                angleLearningRate / std::pow(2, level + 1),
                translationLearningRate / std::pow(2, level + 1),
                maxIterations
            );
        }();
        globalSSDHistory.insert( globalSSDHistory.end(),
            result.ssdHistory.begin(),
            result.ssdHistory.end()
        );

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

    // Plot the global SSD history
    // matplot::plot(globalSSDHistory);
    // matplot::title("SSD History (Multi-Level)");
    // matplot::xlabel("Iteration");
    // matplot::ylabel("SSD");
    // matplot::show();

    return 0;
}
