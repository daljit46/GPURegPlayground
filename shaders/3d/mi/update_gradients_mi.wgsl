// Enable chromium_internal_graphite if needed.
enable chromium_internal_graphite;

#include "../rigidtransformation.wgsl"

const numBins: u32 = {{numBins}};
const workgroupSize: vec3<u32> = vec3<u32>({{workgroup_size}});

fn cubicBSpline(x: f32) -> f32 {
    let ax = abs(x);
    if (ax <= 1.0) {
        return (2.0 / 3.0) - (ax * ax) + (0.5 * ax * ax * ax);
    } else if (ax < 2.0) {
        return (1.0 / 6.0) * pow(2.0 - ax, 3.0);
    } else {
        return 0.0;
    }
}

fn cubicBSplineDerivative(x: f32) -> f32 {
    let ax = abs(x);
    if (ax <= 1.0) {
        if (x >= 0.0) {
            return -2.0 * x + 1.5 * x * x;
        } else {
            return -2.0 * x - 1.5 * x * x;
        }
    } else if (ax < 2.0) {
        // derivative of (1/6)*(2-ax)^3 is -0.5*(2-ax)^2*sign(x)
        return -0.5 * (2.0 - ax) * (2.0 - ax) * (select(-1.0, 1.0, x >= 0.0));
    } else {
        return 0.0;
    }
}

struct MIPartialSums {
    grad_alpha: f32,
    grad_beta: f32,
    grad_gamma: f32,
    grad_tx: f32,
    grad_ty: f32,
    grad_tz: f32,
};

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var<storage, read> minMaxTarget: vec2<f32>;
@group(0) @binding(2) var<storage, read> minMaxMoving: vec2<f32>;
@group(0) @binding(3) var<storage, read> miLookup: array<f32>;
@group(0) @binding(4) var<storage, read> totalMass: f32;
@group(0) @binding(5) var targetImage: texture_3d<f32>;
@group(0) @binding(6) var movingImage: texture_3d<f32>;
@group(0) @binding(7) var<storage, read_write> miPartialSums: array<MIPartialSums>;
@group(0) @binding(8) var linearSampler: sampler;

var<workgroup> localPartialSums: array<MIPartialSums, workgroupSize.x * workgroupSize.y * workgroupSize.z>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>,
    @builtin(num_workgroups) numWorkgroups: vec3<u32>
) {
    let index: u32 = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let dim: vec3<u32> = textureDimensions(targetImage, 0);

    let N: f32 = f32(dim.x * dim.y * dim.z);

    let mat: mat3x3<f32> = rotationMatrix(params);
    let dmatDalpha: mat3x3<f32> = dmatDalpha(params);
    let dmatDbeta: mat3x3<f32> = dmatDbeta(params);
    let dmatDgamma: mat3x3<f32> = dmatDgamma(params);

    var partial: MIPartialSums = MIPartialSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if (globalId.x < dim.x && globalId.y < dim.y && globalId.z < dim.z) {
        let voxelCentre: vec3<f32> = vec3<f32>(globalId) + vec3<f32>(0.5, 0.5, 0.5);
        let transformed: vec3<f32> = mat * voxelCentre + vec3<f32>(params.tx, params.ty, params.tz);
        let targetIntensity: f32 = textureLoad(targetImage, globalId, 0).r;
        let movingIntensity: f32 = textureSampleLevel(movingImage, linearSampler, transformed / vec3<f32>(dim), 0).r;

        let offset: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);
        let gradMoving: vec3<f32> = vec3<f32>(
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.xyy) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.xyy) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yxy) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yxy) / vec3<f32>(dim), 0).r,
            textureSampleLevel(movingImage, linearSampler, (transformed + offset.yyx) / vec3<f32>(dim), 0).r -
            textureSampleLevel(movingImage, linearSampler, (transformed - offset.yyx) / vec3<f32>(dim), 0).r
        ) * 0.5;

        let dTDalpha: vec3<f32> = dmatDalpha * voxelCentre;
        let dTDbeta: vec3<f32>  = dmatDbeta  * voxelCentre;
        let dTDgamma: vec3<f32> = dmatDgamma * voxelCentre;

        let rangeTarget: f32 = minMaxTarget.y - minMaxTarget.x;
        let rangeMoving: f32 = minMaxMoving.y - minMaxMoving.x;
        let binTarget: f32 = ((targetIntensity - minMaxTarget.x) / rangeTarget) * f32(numBins - 1u);
        let binMoving: f32 = ((movingIntensity - minMaxMoving.x) / rangeMoving) * f32(numBins - 1u);
        let binCenterTarget: f32 = floor(binTarget);
        let binCentreMoving: f32 = floor(binMoving);

        let i_min = u32(max(0.0, binCenterTarget - 1.0));
        let i_max = u32(min(f32(numBins - 1u), binCenterTarget + 2.0));
        let j_min = u32(max(0.0, binCentreMoving - 1.0));
        let j_max = u32(min(f32(numBins - 1u), binCentreMoving + 2.0));

        // The chain–rule factor for differentiating the moving intensity through the bin mapping.
        // b_j = J - min J / rangeMoving * (numBins - 1)
        // db_j/dq = (numBins - 1) / (rangeMoving) * dJ/dq
        let chainFactor: f32 = f32(numBins - 1u) / rangeMoving;

        // For each bin pair in the local support, accumulate the derivative.
        for (var i: u32 = i_min; i <= i_max; i = i + 1u) {
            let weightTarget: f32 = cubicBSpline(binTarget - f32(i));
            for (var j: u32 = j_min; j <= j_max; j = j + 1u) {
                let idx: u32 = i * numBins + j;
                let L: f32 = miLookup[idx];

                // Compute derivative of the moving side weight.
                let derivMoving: f32 = cubicBSplineDerivative(binMoving - f32(j));
                // The contribution factor from this bin.
                let contribution: f32 = weightTarget * derivMoving * chainFactor * L;

                let dJ: f32 = dot(gradMoving, dTDalpha);
                let dJ_alpha: f32 = dot(gradMoving, dTDalpha);
                let dJ_beta:  f32 = dot(gradMoving, dTDbeta);
                let dJ_gamma: f32 = dot(gradMoving, dTDgamma);
                // For translations, note that dT/dtx = (1,0,0), etc.
                let dJ_tx: f32 = gradMoving.x;
                let dJ_ty: f32 = gradMoving.y;
                let dJ_tz: f32 = gradMoving.z;

                // The voxel’s contribution to the MI derivative for each parameter
                // normalized by the number of voxels.
                partial.grad_alpha = partial.grad_alpha + (dJ_alpha * contribution) / totalMass;
                partial.grad_beta  = partial.grad_beta  + (dJ_beta  * contribution) / totalMass;
                partial.grad_gamma = partial.grad_gamma + (dJ_gamma * contribution) / totalMass;
                partial.grad_tx    = partial.grad_tx    + (dJ_tx    * contribution) / totalMass;
                partial.grad_ty    = partial.grad_ty    + (dJ_ty    * contribution) / totalMass;
                partial.grad_tz    = partial.grad_tz    + (dJ_tz    * contribution) / totalMass;
            }
        }
    }
    localPartialSums[index] = partial;
    workgroupBarrier();

    // Parallel tree–based reduction in workgroup local memory.
    var pairOffset: u32 = workgroupSize.x * workgroupSize.y * workgroupSize.z / 2u;
    while (pairOffset > 0u) {
        if (index < pairOffset) {
            let a = localPartialSums[index];
            let b = localPartialSums[index + pairOffset];
            localPartialSums[index] = MIPartialSums(
                a.grad_alpha + b.grad_alpha,
                a.grad_beta  + b.grad_beta,
                a.grad_gamma + b.grad_gamma,
                a.grad_tx    + b.grad_tx,
                a.grad_ty    + b.grad_ty,
                a.grad_tz    + b.grad_tz
            );
        }
        workgroupBarrier();
        pairOffset = pairOffset / 2u;
    }

    if (index == 0u) {
        let wgIndex: u32 = workgroupId.x + workgroupId.y * numWorkgroups.x + workgroupId.z * numWorkgroups.x * numWorkgroups.y;
        miPartialSums[wgIndex] = localPartialSums[0];
    }
}
