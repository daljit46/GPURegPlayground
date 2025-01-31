// NOTE: the total workgroup size of this shader must be 32 <= 2^N <= 1024
// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct TransformationParameters {
    alpha: f32, // rotation around z-axis
    beta: f32,  // rotation around y-axis
    gamma: f32, // rotation around x-axis
    tx: f32,
    ty: f32,
    tz: f32
};

// Let I=I(x,y,z) be the target image and J = J(T(x, y, z)) be the moving image
// Let I' = I - mean(I) and J' = J - mean(J)
// NOTE: J = J(T(x,y,z)) where T is the transformation function
// NCC = A / sqrt(B * C) where
// A = sum(I' * J')
// B = sum(I' * I')
// C = sum(J' * J')
// The sum is over all voxels in the images.
// dNCC/dp_k = 1/[sqrt(B) * C^3/2] * [dA/dp_k * C - 0.5 A * C * dC/dp_k] where
// p_k is the k-th transformation parameter
// where dA/dp_k = sum[I' * (gradJ) dotted dT/dp_k - d/dp_k(mean(J))]
// where dC/dp_k = 2 * sum[J' * (gradJ dotted dT/dp_k - d/dp_k(mean(J))]
// For simplicity, we will assume that d/dp_k(mean(J)) = 0 even though this is not strictly true

// 60 bytes
struct NCCPartialSums {
    sumA: f32,
    sumB: f32,
    sumC: f32,

    dA_dalpha: f32,
    dA_dbeta: f32,
    dA_dgamma: f32,
    dA_dtx: f32,
    dA_dty: f32,
    dA_dtz: f32,

    dC_dalpha: f32,
    dC_dbeta: f32,
    dC_dgamma: f32,
    dC_dtx: f32,
    dC_dty: f32,
    dC_dtz: f32
};

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var<storage, read> targetMean: f32;
@group(0) @binding(2) var<storage, read> movingMean: f32;
@group(0) @binding(3) var targetImage: texture_3d<f32>;
@group(0) @binding(4) var movingImage: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> nccPartialSums: array<NCCPartialSums>;
@group(0) @binding(6) var linearSampler: sampler;

const workgroupSize = vec3<u32>({{workgroup_size}});
const numThreadsPerWorkgroup = workgroupSize.x * workgroupSize.y * workgroupSize.z;

var<workgroup> localPartialSums : array<NCCPartialSums, numThreadsPerWorkgroup>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>,
    @builtin(num_workgroups) numWorkgroups: vec3<u32>
)
{
    let index = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let dim = textureDimensions(targetImage, 0);

    var partialSums = NCCPartialSums(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    );

    if(id.x < dim.x && id.y < dim.y && id.z < dim.z) {
        let sinAlpha = sin(params.alpha);
        let cosAlpha = cos(params.alpha);
        let sinBeta  = sin(params.beta);
        let cosBeta  = cos(params.beta);
        let sinGamma = sin(params.gamma);
        let cosGamma = cos(params.gamma);

        // WebGPU uses column-major matrices
        let mat = mat3x3<f32>(
            cosAlpha * cosBeta, sinAlpha * cosBeta, -sinBeta,
            cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma, cosBeta * sinGamma,
            cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma
        );
        // x' = column 1 dotted with (x, y, z)
        // we need dx'/dalpha, dx'/dbeta, dx'/dgamma, dy'/dalpha, dy'/dbeta, dy'/dgamma, dz'/dalpha, dz'/dbeta, dz'/dgamma
        let dmatDalpha = mat3x3<f32>(
            -sinAlpha * cosBeta, cosAlpha * cosBeta, 0.0,
            -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, 0.0,
            -sinAlpha * sinBeta * cosGamma + cosAlpha * sinGamma, cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, 0.0
        );
        let dmatDbeta = mat3x3<f32>(
            -cosAlpha * sinBeta, -sinAlpha * sinBeta, -cosBeta,
            cosAlpha * cosBeta * sinGamma, sinAlpha * cosBeta * sinGamma, -sinBeta * sinGamma,
            cosAlpha * cosBeta * cosGamma, sinAlpha * cosBeta * cosGamma, -sinBeta * cosGamma
        );
        let dmatDgamma = mat3x3<f32>(
            0.0, 0.0, 0.0,
            cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma,
            -cosAlpha * sinBeta * sinGamma + sinAlpha * cosGamma, -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, -cosBeta * sinGamma
        );

        let targetIntensity = textureLoad(targetImage, id, 0).r;
        let movingIntensity = textureSampleLevel(movingImage, linearSampler,
                                (mat * vec3<f32>(id.xyz) + vec3<f32>(params.tx, params.ty, params.tz)) /
                                vec3<f32>(dim), 0).r;
        let IPrime = targetIntensity - targetMean / f32(dim.x * dim.y * dim.z);
        let JPrime = movingIntensity - movingMean / f32(dim.x * dim.y * dim.z);

        let offset = vec3<f32>(1.0, 0.0, 0.0);
        let voxelCentre = vec3<f32>(id.xyz) + vec3<f32>(0.5, 0.5, 0.5);
        let transformed = mat * voxelCentre + vec3<f32>(params.tx, params.ty, params.tz);
        let gradMoving = vec3<f32>(
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.xyy) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.xyy) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yxy) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yxy) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yyx) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yyx) / vec3<f32>(dim), 0).r
        ) * 0.5;

        let dTDalpha = dmatDalpha * vec3<f32>(id.xyz);
        let dTDbeta = dmatDbeta * vec3<f32>(id.xyz);
        let dTDgamma = dmatDgamma * vec3<f32>(id.xyz);

        // gradJ dotted with dT/dp_k
        let dJDalpha = dot(gradMoving, dTDalpha);
        let dJDbeta = dot(gradMoving, dTDbeta);
        let dJDgamma = dot(gradMoving, dTDgamma);

        partialSums.sumA = IPrime * JPrime;
        partialSums.sumB = IPrime * IPrime;
        partialSums.sumC = JPrime * JPrime;

        // dA/dp_k = sum[I' * (gradJ) dotted dT/dp_k]
        partialSums.dA_dalpha = IPrime * dJDalpha;
        partialSums.dA_dbeta  = IPrime * dJDbeta;
        partialSums.dA_dgamma = IPrime * dJDgamma;
        // For translations dT/dtx = (1, 0, 0), dT/dty = (0, 1, 0), dT/dtz = (0, 0, 1)
        // so e.g. dA/dtx = I' * gradJ dotted (1, 0, 0)
        partialSums.dA_dtx    = IPrime * gradMoving.x;
        partialSums.dA_dty    = IPrime * gradMoving.y;
        partialSums.dA_dtz    = IPrime * gradMoving.z;

        // dC/dp_k = sum[J' * (gradJ dotted dT/dp_k)]
        partialSums.dC_dalpha = 2 * JPrime * dJDalpha;
        partialSums.dC_dbeta  = 2 * JPrime * dJDbeta;
        partialSums.dC_dgamma = 2 * JPrime * dJDgamma;
        partialSums.dC_dtx    = 2 * JPrime * gradMoving.x;
        partialSums.dC_dty    = 2 * JPrime * gradMoving.y;
        partialSums.dC_dtz    = 2 * JPrime * gradMoving.z;

        localPartialSums[index] = partialSums;
    }

    workgroupBarrier();

    // Perform tree-based parallel reduction
    var pairOffset = numThreadsPerWorkgroup / 2u;
    while (pairOffset > 0u) {
        if (index < pairOffset) {
            let data1 = localPartialSums[index];
            let data2 = localPartialSums[index + pairOffset];
            localPartialSums[index] = NCCPartialSums(
                data1.sumA + data2.sumA,
                data1.sumB + data2.sumB,
                data1.sumC + data2.sumC,
                data1.dA_dalpha + data2.dA_dalpha,
                data1.dA_dbeta + data2.dA_dbeta,
                data1.dA_dgamma + data2.dA_dgamma,
                data1.dA_dtx + data2.dA_dtx,
                data1.dA_dty + data2.dA_dty,
                data1.dA_dtz + data2.dA_dtz,
                data1.dC_dalpha + data2.dC_dalpha,
                data1.dC_dbeta + data2.dC_dbeta,
                data1.dC_dgamma + data2.dC_dgamma,
                data1.dC_dtx + data2.dC_dtx,
                data1.dC_dty + data2.dC_dty,
                data1.dC_dtz + data2.dC_dtz
            );
        }
        workgroupBarrier();
        pairOffset = pairOffset / 2u;
    }

    if (index == 0u) {
        let data = localPartialSums[0];
        let wgIndex = workgroupId.x + workgroupId.y * numWorkgroups.x + workgroupId.z * numWorkgroups.x * numWorkgroups.y;
        nccPartialSums[wgIndex] = NCCPartialSums(
            data.sumA,
            data.sumB,
            data.sumC,
            data.dA_dalpha,
            data.dA_dbeta,
            data.dA_dgamma,
            data.dA_dtx,
            data.dA_dty,
            data.dA_dtz,
            data.dC_dalpha,
            data.dC_dbeta,
            data.dC_dgamma,
            data.dC_dtx,
            data.dC_dty,
            data.dC_dtz
        );
    }
}
