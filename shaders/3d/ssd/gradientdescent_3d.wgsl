// NOTE: the total workgroup size of this shader must be 32 <= 2^N <= 1024
// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

#include "../rigidtransformation.wgsl"


struct SSDGradients {
    ssd: f32,
    dssd_dalpha: f32,
    dssd_dbeta: f32,
    dssd_dgamma: f32,
    dssd_dtx: f32,
    dssd_dty: f32,
    dssd_dtz: f32,
};

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var targetImage: texture_3d<f32>;
@group(0) @binding(2) var movingImage: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> ssdGrads: array<SSDGradients>;
@group(0) @binding(4) var linearSampler: sampler;

const workgroupSize = vec3<u32>({{workgroup_size}});
const workgroupInvocations = workgroupSize.x * workgroupSize.y * workgroupSize.z;

var<workgroup> local_gradients : array<SSDGradients, workgroupInvocations>;

fn reduceLocalGradients(index: u32, offset: u32) {
    if(index < offset) {
        let data1 = local_gradients[index];
        let data2 = local_gradients[index + offset];
        local_gradients[index] = SSDGradients(
            data1.ssd + data2.ssd,
            data1.dssd_dalpha + data2.dssd_dalpha,
            data1.dssd_dbeta + data2.dssd_dbeta,
            data1.dssd_dgamma + data2.dssd_dgamma,
            data1.dssd_dtx + data2.dssd_dtx,
            data1.dssd_dty + data2.dssd_dty,
            data1.dssd_dtz + data2.dssd_dtz
        );
    }
}


fn finiteDiff(image: texture_3d<f32>, id: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        textureLoad(image, vec3<u32>(id.x + 1u, id.y, id.z), 0).r - textureLoad(image, vec3<u32>(id.x - 1u, id.y, id.z), 0).r,
        textureLoad(image, vec3<u32>(id.x, id.y + 1u, id.z), 0).r - textureLoad(image, vec3<u32>(id.x, id.y - 1u, id.z), 0).r,
        textureLoad(image, vec3<u32>(id.x, id.y, id.z + 1u), 0).r - textureLoad(image, vec3<u32>(id.x, id.y, id.z - 1u), 0).r
    );
}


@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>,
    @builtin(num_workgroups) numWorkgroups: vec3<u32>
)
{
    let index = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let dim = vec3<f32>(textureDimensions(targetImage, 0));

    if(f32(id.x) >= dim.x || f32(id.y) >= dim.y || f32(id.z) >= dim.z) {
        local_gradients[index] = SSDGradients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    else {
        let mat = rotationMatrix(params);
        let dmatDalpha = dmatDalpha(params);
        let dmatDbeta = dmatDbeta(params);
        let dmatDgamma = dmatDgamma(params);

        let voxelCenter = vec3<f32>(id.xyz) + vec3<f32>(0.5, 0.5, 0.5);
        let transformed = mat * voxelCenter + vec3<f32>(params.tx, params.ty, params.tz);
        let movingValue = textureSampleLevel(movingImage, linearSampler, transformed / dim, 0).r;
        let offset = vec3<f32>(1.0, 0.0, 0.0);
        let gradMoving = vec3<f32>(
            textureSampleLevel(movingImage, linearSampler, (transformed + offset) / dim, 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset) / dim, 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yxy) / dim, 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yxy) / dim, 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yyx) / dim, 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yyx) / dim, 0).r
        )/2.0;

        let error = movingValue - textureLoad(targetImage, id, 0).r;
        let gradXYZalpha = dmatDalpha * voxelCenter;
        let gradXYZbeta = dmatDbeta * voxelCenter;
        let gradXYZgamma = dmatDgamma * voxelCenter;

        let gradAlpha = 2 * error * dot(gradMoving, gradXYZalpha);
        let gradBeta = 2 * error * dot(gradMoving, gradXYZbeta);
        let gradGamma = 2 * error * dot(gradMoving, gradXYZgamma);
        let gradTx = 2 * error * gradMoving.x;
        let gradTy = 2 * error * gradMoving.y;
        let gradTz = 2 * error * gradMoving.z;

        local_gradients[index] = SSDGradients(error * error, gradAlpha, gradBeta, gradGamma, gradTx, gradTy, gradTz);
    }


    workgroupBarrier();

    // Perform tree based reduction (unrolled for performance)
    // Probably could be made faster when subgroups are no longer experimental in WebGPU
    if(workgroupInvocations >= 1024) { reduceLocalGradients(index, 512); } workgroupBarrier();
    if(workgroupInvocations >= 512)  { reduceLocalGradients(index, 256); } workgroupBarrier();
    if(workgroupInvocations >= 256)  { reduceLocalGradients(index, 128); } workgroupBarrier();
    if(workgroupInvocations >= 128)  { reduceLocalGradients(index, 64);  } workgroupBarrier();
    // We no longer need to use barries when the index <= 32 because
    // instructions are SIMD within a wavefront/warp (technically on AMD this limit should be 64)
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    if(workgroupInvocations >= 64) { reduceLocalGradients(index, 32); }
    if(workgroupInvocations >= 32) { reduceLocalGradients(index, 16); }
    if(workgroupInvocations >= 16) { reduceLocalGradients(index, 8); }
    if(workgroupInvocations >= 8)  { reduceLocalGradients(index, 4); }
    if(workgroupInvocations >= 4)  { reduceLocalGradients(index, 2); }
    if(workgroupInvocations >= 2)  { reduceLocalGradients(index, 1); }

    if(index == 0u) {
        let wgIndex = workgroupId.x + workgroupId.y * numWorkgroups.x + workgroupId.z * numWorkgroups.x * numWorkgroups.y;
        ssdGrads[wgIndex] = local_gradients[0];
    }
}
