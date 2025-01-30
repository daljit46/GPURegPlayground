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
// NOTE: J = J(T(x,y,z)) where T is the transformation function
// NCC = A / sqrt(B * C) where
// A = sum((I - mean(I)) * (J - mean(J)))
// B = sum((I - mean(I))^2)
// C = sum((J - mean(J))^2)
// The sum is over all voxels in the images.
// dNCC/dp_k = 1/[sqrt(B) * C^3/2] * [dA/dp_k * C - 0.5 A * C * dC/dp_k] where
// p_k is the k-th transformation parameter
// where dA/dp_k = sum[I * (gradJ) dotted dT/dp_k - d/dp_k(mean(J))]
// where dC/dp_k = 2 * sum[J' * (gradJ /dotted dT/dp_k - d/dp_k(mean(J))]
// For simplicity, we will assume that d/dp_k(mean(J)) = 0 even though this is not strictly true

// 60 bytes
struct NCCPartialSums {
    sumA: f32,      // sum((I - mean(I)) * (J - mean(J)))
    sumB: f32,      // sum((I - mean(I))^2), technically this could be precomputed in a separate shader
    sumC: f32,      // sum((J - mean(J))^2)

    dA_dalpha: f32, // sum[I * (gradJ) dotted dT/dalpha]
    dA_dbeta: f32,  // sum[I * (gradJ) dotted dT/dbeta]
    dA_dgamma: f32, // sum[I * (gradJ) dotted dT/dgamma]
    dA_dtx: f32,    // sum[I * (gradJ) dotted dT/dtx]
    dA_dty: f32,    // sum[I * (gradJ) dotted dT/dty]
    dA_dtz: f32,    // sum[I * (gradJ) dotted dT/dtz]

    dC_dalpha: f32, // sum[J' * (gradJ dotted dT/dalpha)]
    dC_dbeta: f32,  // sum[J' * (gradJ dotted dT/dbeta)]
    dC_dgamma: f32, // sum[J' * (gradJ dotted dT/dgamma)]
    dC_dtx: f32,    // sum[J' * (gradJ dotted dT/dtx)]
    dC_dty: f32,    // sum[J' * (gradJ dotted dT/dty)]
    dC_dtz: f32     // sum[J' * (gradJ dotted dT/dtz)]
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
        0.0, 0.0, 0.0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
        let IPrime = targetIntensity - targetMean;
        let JPrime = movingIntensity - movingMean;

        let offset = vec3<f32>(1.0, 0.0, 0.0);
        let gradMoving = vec3<f32>(
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz + offset) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz - offset) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz + offset.yxy) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz - offset.yxy) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz + offset.yyx) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (mat * vec3<f32>(id.xyz - offset.yyx) + vec3<f32>(params.tx, params.ty, params.tz)) / vec3<f32>(dim), 0).r
        ) * 0.5;

        let dTDalpha = dmatDalpha * vec3<f32>(id.xyz);
        let dTDbeta = dmatDbeta * vec3<f32>(id.xyz);
        let dTDgamma = dmatDgamma * vec3<f32>(id.xyz);

        // dJ/dp_k = sum[J' * (gradJ dotted dT/dp_k)]
        let dJDalpha = dot(gradMoving, dTDalpha);
        let dJDbeta = dot(gradMoving, dTDbeta);
        let dJDgamma = dot(gradMoving, dTDgamma);

        partialSums.sumA = IPrime * JPrime;
        partialSums.sumB = IPrime * IPrime;
        partialSums.sumC = JPrime * JPrime;

        // dA/dp_k = sum[I * (gradJ) dotted dT/dp_k]
        partialSums.dA_dalpha = IPrime * dJDalpha;
        partialSums.dA_dbeta  = IPrime * dJDbeta;
        partialSums.dA_dgamma = IPrime * dJDgamma;
        partialSums.dA_dtx    = IPrime;
        partialSums.dA_dty    = IPrime;
        partialSums.dA_dtz    = IPrime;

        partialSums.dC_dalpha = JPrime * dJDalpha;
        partialSums.dC_dbeta  = JPrime * dJDbeta;
        partialSums.dC_dgamma = JPrime * dJDgamma;
        partialSums.dC_dtx    = JPrime;
        partialSums.dC_dty    = JPrime;
        partialSums.dC_dtz    = JPrime;
    }
}
