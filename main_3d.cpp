#include "adamoptimiser.h"
#include "image.h"
#include "gpu.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "scopedtimer.h"
#include "transform.h"
#include <matplot/matplot.h>



struct TransformationParameters {
    float alpha = Utils::degreesToRadians(0.0F);
    float beta = 0.0F;
    float gamma = 0.0F;
    float tx = 0.0F;
    float ty = 0.0F;
    float tz = 0.0F;
    std::array<float, 2> _padding; // WebGPU requires 16 byte alignment
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
    float finalSSD;
    std::vector<float> ssdHistory;
    // Final transform parameters for chaining to the next level
    float alpha, beta, gamma, tx, ty, tz;
};

SingleLevelResult registerAtSingleResolution(
    gpu::Context &context,
    const gpu::Texture &sourceTexture,
    const gpu::Texture &targetTexture,
    TransformationParameters &transformationParams,
    const gpu::WorkgroupSize &workgroupSize,
    int maxIterations,
    AdamOptimizer &optimizer)
{
    const gpu::WorkgroupGrid workgrid {
        .x = (sourceTexture.size.width  + workgroupSize.x - 1) / workgroupSize.x,
        .y = (sourceTexture.size.height + workgroupSize.y - 1) / workgroupSize.y,
        .z = (sourceTexture.size.depth  + workgroupSize.z - 1) / workgroupSize.z
    };

    // Create an empty "moving" texture matching sourceTexture dimension
    gpu::Texture movingTexture = context.makeEmptyTexture({
        .size = sourceTexture.size,
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    // Uniform buffer holding the transform parameters
    auto uniformsBuffer = context.makeUniformBuffer(
        &transformationParams,
        sizeof(TransformationParameters)
        );

    const gpu::KernelDescriptor transformDesc {
        .shader = {
            .name = "transform",
            .entryPoint = "main",
            .code = Utils::readFile("shaders/3d/transformimage_3d.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = {uniformsBuffer},
        .inputTextures  = {sourceTexture},
        .outputTextures = {movingTexture},
        .samplers       = { context.makeLinearSampler() }
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

    } ssdGradients;

    // Size of array of SSD gradients is the number of workgroups in the grid times size of SSDGradients struct
    const uint32_t workgroupCount = workgrid.x * workgrid.y * workgrid.z;
    spdlog::info("Workgroup Count: {}", workgroupCount);
    const size_t ssdGradientsSize = sizeof(SSDGradients) * workgroupCount;
    gpu::DataBuffer ssdGradientsBuffer = context.makeEmptyBuffer(ssdGradientsSize);

    const gpu::KernelDescriptor updateParamsDesc {
        .shader = {
            .name = "updateparameters",
            .entryPoint = "main",
            .code = Utils::readFile("shaders/3d/updateparameters_3d.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = { uniformsBuffer },
        .inputTextures  = { targetTexture, movingTexture },
        .outputBuffers  = { ssdGradientsBuffer },
        .samplers       = { context.makeLinearSampler() }
    };

    auto transformKernel = context.makeKernel(transformDesc);
    auto updateParamsOP  = context.makeKernel(updateParamsDesc);

    float minSSD = std::numeric_limits<float>::max();

    std::vector<float> ssdHistory;
    ssdHistory.reserve(maxIterations);

    for (int i = 0; i < maxIterations; ++i) {
        context.writeToBuffer(uniformsBuffer, &transformationParams);

        ssdGradients = {};
        context.dispatchKernel(transformKernel, workgrid);
        context.dispatchKernel(updateParamsOP, workgrid);

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

        // Logging
        spdlog::info(
            "SSD1: {} dAlpha: {} dBeta: {} dGamma: {} dTx: {} dTy: {} dTz: {}",
            ssd, dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz
            );

        // Keep track of minimum
        if (ssd < minSSD) {
            minSSD = ssd;
        }

        // Update parameters with Adam
        auto newParams = optimizer.step({
            dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz
        });

        // Set new transformation parameters
        transformationParams.alpha = newParams[0].value;
        transformationParams.beta  = newParams[1].value;
        transformationParams.gamma = newParams[2].value;
        transformationParams.tx    = newParams[3].value;
        transformationParams.ty    = newParams[4].value;
        transformationParams.tz    = newParams[5].value;

        ssdHistory.push_back(ssd);

        // Print iteration info
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


// ---------------------------------------------------
// Main
// ---------------------------------------------------
int main()
{
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

    // Load the original images (full resolution)
    const NiftiImage sourceImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const NiftiImage targetImage = transformNifti(sourceImage,
                                                  { targetAlpha, targetBeta, targetGamma, targetTx, targetTy, targetTz});

    // Save the artificially transformed target
    // Utils::saveToDisk(targetImage, "target.nii");

    // Convert to GPU textures (full resolution)
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

    // ---------------------------------------------------
    // Prepare the transformation parameters (start at identity)
    // ---------------------------------------------------
    TransformationParameters transformationParams; // all zero by default

    // We'll run the same number of iterations at each level
    constexpr int maxIterations = 500;

    // For gradient-based approach, set learning rates for angles & translations.
    // Using max dimension from the *full* resolution
    const float maxImageDim = std::max(
        {float(sourceImage.width), float(sourceImage.height), float(sourceImage.depth)}
        );
    const float translationLearningRate = 1.0F;
    const float angleLearningRate       = translationLearningRate / maxImageDim;

    // Setup Adam with 6 parameters
    std::vector<AdamOptimizer::Parameter> parameters = {
        {.value = transformationParams.alpha, .learning_rate = angleLearningRate },
        {.value = transformationParams.beta,  .learning_rate = angleLearningRate },
        {.value = transformationParams.gamma, .learning_rate = angleLearningRate },
        {.value = transformationParams.tx,    .learning_rate = translationLearningRate },
        {.value = transformationParams.ty,    .learning_rate = translationLearningRate },
        {.value = transformationParams.tz,    .learning_rate = translationLearningRate }
    };
    AdamOptimizer optimizer(parameters);

    // We'll store SSD history for final plotting
    std::vector<float> globalSSDHistory;

    const gpu::WorkgroupSize workgroupSize = {8,8,4};


    for (int level = 0; level < 4; ++level)
    {
        spdlog::info("\n\n=== Registering at pyramid level {} (0=coarse, 3=full) ===", level);

        // We might want to adjust the learning rates for coarser levels:
        // e.g., bigger learning rate for coarse, smaller for fine, etc.
        // For simplicity, we keep them the same in this example.

        // Update transformation parameters in case we up-scaled from previous step
        parameters[0].value = transformationParams.alpha;
        parameters[1].value = transformationParams.beta;
        parameters[2].value = transformationParams.gamma;
        parameters[3].value = transformationParams.tx;
        parameters[4].value = transformationParams.ty;
        parameters[5].value = transformationParams.tz;
        // Decrease learning rate as we go to up-scaled levels
        parameters[0].learning_rate = angleLearningRate / std::pow(2, level + 1);
        parameters[1].learning_rate = angleLearningRate / std::pow(2, level + 1);
        parameters[2].learning_rate = angleLearningRate / std::pow(2, level + 1);
        parameters[3].learning_rate = translationLearningRate / std::pow(2, level + 1);
        parameters[4].learning_rate = translationLearningRate / std::pow(2, level + 1);
        parameters[5].learning_rate = translationLearningRate / std::pow(2, level + 1);

        optimizer = AdamOptimizer(parameters);

        auto result = registerAtSingleResolution(
            context,
            sourcePyramid[level],
            targetPyramid[level],
            transformationParams, // updated in place
            workgroupSize,
            maxIterations,
            optimizer
            );

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
            transformationParams.tx *= 2.0f;
            transformationParams.ty *= 2.0f;
            transformationParams.tz *= 2.0f;
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
    spdlog::info("  Tx:    {}", transformationParams.tx);
    spdlog::info("  Ty:    {}", transformationParams.ty);
    spdlog::info("  Tz:    {}", transformationParams.tz);

    spdlog::info("Difference between final transform and target:");
    spdlog::info("  Alpha: {}", transformationParams.alpha - targetAlpha);
    spdlog::info("  Beta:  {}", transformationParams.beta - targetBeta);
    spdlog::info("  Gamma: {}", transformationParams.gamma - targetGamma);
    spdlog::info("  Tx:    {}", transformationParams.tx - targetTx);
    spdlog::info("  Ty:    {}", transformationParams.ty - targetTy);
    spdlog::info("  Tz:    {}", transformationParams.tz - targetTz);

    // Plot the global SSD history
    // matplot::plot(globalSSDHistory);
    // matplot::title("SSD History (Multi-Level)");
    // matplot::xlabel("Iteration");
    // matplot::ylabel("SSD");
    // matplot::show();

    return 0;
}
