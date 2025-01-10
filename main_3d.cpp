#include "adamoptimiser.h"
#include "image.h"
#include "gpu.h"
#include "utils.h"
#include "spdlog/spdlog.h"
#include "transform.h"
#include <matplot/matplot.h>


int main()
{
    auto context = gpu::Context::newContext();
    const float targetAlpha = Utils::degreesToRadians(10.0F);
    const float targetBeta = 0.1F;
    const float targetGamma = 0.3F;
    const float targetTx = 10.0F;
    const float targetTy = 13.0F;
    const float targetTz = 4.0F;

    // Print target values
    spdlog::info("Target Alpha: {}", targetAlpha);
    spdlog::info("Target Beta: {}", targetBeta);
    spdlog::info("Target Gamma: {}", targetGamma);
    spdlog::info("Target Tx: {}", targetTx);
    spdlog::info("Target Ty: {}", targetTy);
    spdlog::info("Target Tz: {}", targetTz);

    const NiftiImage sourceImage = Utils::loadNiftiFromDisk("data/test_file.nii");
    const NiftiImage targetImage = transformNifti(sourceImage,{
       targetAlpha, targetBeta, targetGamma, targetTx, targetTy, targetTz
    });

    Utils::saveToDisk(targetImage, "target.nii");

    const gpu::Texture sourceTexture = context.makeTextureFromHostNifti(sourceImage);;
    const gpu::Texture targetTexture = context.makeTextureFromHostNifti(targetImage);
    const gpu::Texture movingTexture = context.makeEmptyTexture ({
        .size = {sourceImage.width, sourceImage.height, sourceImage.depth},
        .format = gpu::TextureFormat::R8Unorm,
        .usage = gpu::ResourceUsage::ReadWrite
    });

    int64_t nonZeroSource = 0;
    int64_t nonZeroTarget = 0;
    for(size_t z = 0; z < sourceImage.depth; z++) {
        for(size_t y = 0; y < sourceImage.height; y++) {
            for(size_t x = 0; x < sourceImage.width; x++) {
                if(sourceImage.at(x, y, z) != 0) {
                    nonZeroSource++;
                }
                if(targetImage.at(x, y, z) != 0) {
                    nonZeroTarget++;
                }
            }
        }
    }
    spdlog::info("Non-zero voxels in source image: {}", nonZeroSource);
    spdlog::info("Non-zero voxels in target image: {}", nonZeroTarget);


    struct TransformationParameters {
        float alpha = Utils::degreesToRadians(0.0F);
        float beta = 0.0F;
        float gamma = 0.0F;
        float tx = 0.0F;
        float ty = 0.0F;
        float tz = 0.0F;
        std::array<float, 2> _padding; // WebGPU requires 16 byte alignment
    } transformationParams;

    auto uniformsBuffer = context.makeUniformBuffer(&transformationParams, sizeof(TransformationParameters));

    const gpu::WorkgroupSize workgroupSize = {4, 4, 4};

    const gpu::KernelDescriptor transformDesc {
        .shader = {
            .name = "transform",
            .entryPoint = "main",
            .code = Utils::readFile("shaders/3d/transformimage_3d.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = {uniformsBuffer},
        .inputTextures = {sourceTexture},
        .outputTextures = {movingTexture},
        .samplers = { context.makeLinearSampler() }
    };

    struct OutputParameters {
        uint32_t ssd = 0;
        uint32_t dssd_dalpha = 0;
        uint32_t dssd_dbeta = 0;
        uint32_t dssd_dgamma = 0;
        uint32_t dssd_dtx = 0;
        uint32_t dssd_dty = 0;
        uint32_t dssd_dtz = 0;
        uint32_t _padding;
    } outputParams;

    auto paramsBuffer = context.makeEmptyBuffer(sizeof(OutputParameters));
    context.writeToBuffer(paramsBuffer, &outputParams);

    const gpu::KernelDescriptor updateParamsDesc {
        .shader = {
            .name = "updateparameters",
            .entryPoint = "main",
            .code = Utils::readFile("shaders/3d/updateparameters_3d.wgsl"),
            .workgroupSize = workgroupSize
        },
        .uniformBuffers = {uniformsBuffer},
        .inputTextures = { targetTexture, movingTexture },
        .outputBuffers = {paramsBuffer},
        .samplers = { context.makeLinearSampler() }
    };

    auto transformKernel = context.makeKernel(transformDesc);
    auto updateParamsOP = context.makeKernel(updateParamsDesc);

    constexpr int maxIterations = 500;
    const float alphaLearningRate = 1e-4;
    const float betaLearningRate = 1e-4;
    const float gammaLearningRate = 1e-4;
    const float txLearningRate = 1e-1;
    const float tyLearningRate = 1e-1;
    const float tzLearningRate = 1e-1;

    std::vector<AdamOptimizer::Parameter> parameters =  {
        {.value = transformationParams.alpha, .learning_rate = alphaLearningRate },
        {.value = transformationParams.beta, .learning_rate = betaLearningRate },
        {.value = transformationParams.gamma, .learning_rate = gammaLearningRate },
        {.value = transformationParams.tx, .learning_rate = txLearningRate },
        {.value = transformationParams.ty, .learning_rate = tyLearningRate },
        {.value = transformationParams.tz, .learning_rate = tzLearningRate }
    };

    AdamOptimizer optimizer(parameters);

    float minSSD = std::numeric_limits<float>::max();
    float minAlpha = std::numeric_limits<float>::max();
    float minBeta = std::numeric_limits<float>::max();
    float minGamma = std::numeric_limits<float>::max();
    float minTx = std::numeric_limits<float>::max();
    float minTy = std::numeric_limits<float>::max();
    float minTz = std::numeric_limits<float>::max();

    const gpu::WorkgroupGrid workgrid {
        .x = sourceImage.width + workgroupSize.x - 1 / workgroupSize.x,
        .y = sourceImage.height + workgroupSize.y - 1 / workgroupSize.y,
        .z = sourceImage.depth + workgroupSize.y - 1 / workgroupSize.z
    };

    // context.writeToBuffer(uniformsBuffer, &transformationParams);
    // context.dispatchKernel(transformKernel, workgrid);

    // // Download the transformed image
    // std::vector<uint8_t> transformedData(sourceImage.width * sourceImage.height * sourceImage.depth);
    // context.downloadTexture(movingTexture, transformedData.data());

    // nifti_image *transformedNifti = nifti_copy_nim_info(sourceImage.handle());
    // transformedNifti->data = transformedData.data();
    // NiftiImage transformedImage(transformedNifti, false);
    // Utils::saveToDisk(transformedImage, "transformed_gpu.nii");

    // return 0;

    std::vector<float> ssdHistory;

    for(int i = 0; i < maxIterations; ++i) {
        context.writeToBuffer(uniformsBuffer, &transformationParams);
        outputParams = {};
        context.writeToBuffer(paramsBuffer, &outputParams);

        context.dispatchKernel(transformKernel, workgrid);
        context.dispatchKernel(updateParamsOP, workgrid);

        context.downloadBuffer(paramsBuffer, &outputParams);

        const float ssd = reinterpret_cast<float*>(&outputParams.ssd)[0];
        const float dssd_dalpha = reinterpret_cast<float*>(&outputParams.dssd_dalpha)[0];
        const float dssd_dbeta = reinterpret_cast<float*>(&outputParams.dssd_dbeta)[0];
        const float dssd_dgamma = reinterpret_cast<float*>(&outputParams.dssd_dgamma)[0];
        const float dssd_dtx = reinterpret_cast<float*>(&outputParams.dssd_dtx)[0];
        const float dssd_dty = reinterpret_cast<float*>(&outputParams.dssd_dty)[0];
        const float dssd_dtz = reinterpret_cast<float*>(&outputParams.dssd_dtz)[0];

        spdlog::info("SSD: {} dSSD/dAlpha: {} dSSD/dBeta: {} dSSD/dGamma: {} dSSD/dTx: {} dSSD/dTy: {} dSSD/dTz: {}",
                     ssd, dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz);

        if(ssd < minSSD) {
            minSSD = ssd;
            minAlpha = transformationParams.alpha;
            minBeta = transformationParams.beta;
            minGamma = transformationParams.gamma;
            minTx = transformationParams.tx;
            minTy = transformationParams.ty;
            minTz = transformationParams.tz;
        }

        auto newParams = optimizer.step({dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz});
        transformationParams.alpha = newParams[0].value;
        transformationParams.beta = newParams[1].value;
        transformationParams.gamma = newParams[2].value;
        transformationParams.tx = newParams[3].value;
        transformationParams.ty = newParams[4].value;
        transformationParams.tz = newParams[5].value;

        ssdHistory.push_back(ssd);

        spdlog::info("Iteration: {} SSD: {} Alpha: {} Beta: {} Gamma: {} Tx: {} Ty: {} Tz: {}", i, ssd, transformationParams.alpha, transformationParams.beta, transformationParams.gamma, transformationParams.tx, transformationParams.ty, transformationParams.tz);

    }

    matplot::plot(ssdHistory);
    matplot::title("SSD History");
    matplot::xlabel("Iteration");
    matplot::ylabel("SSD");
    matplot::show();
}
