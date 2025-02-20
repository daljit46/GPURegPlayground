// Enable the chromium_internal_graphite feature so that r8unorm can be used.
enable chromium_internal_graphite;

#include "../../atomic_utils.wgsl"
#include "../rigidtransformation.wgsl"

const numBins: u32 = {{numBins}};
const workgroupSize: vec3<u32> = vec3<u32>({{workgroup_size}});
const workgroupInvocations = workgroupSize.x * workgroupSize.y * workgroupSize.z;
// We use u32 atomics in workgroup memory because bitcasting + CAS can be quite slow.
// In workgroup memory, we can store values as u32 by multiplying by a scaling factor
// and create a "scaled" local histogram. The global histogram will be updated by a
// leader thread that will divide the scaled values by the scaling factor.
// Here we can choose a large scaling factor because our workgroup size is small,
// so there is no risk of overflow.
const scalingFactor = 1000000.0;
// Cubic B–spline kernel with compact support = 2
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

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var<storage, read> minMaxTarget: vec2<f32>;
@group(0) @binding(2) var<storage, read> minMaxMoving: vec2<f32>;
@group(0) @binding(3) var targetImage: texture_3d<f32>;
@group(0) @binding(4) var movingImage: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> globalHistogram: array<atomic<u32>>; // size = numBins*numBins
@group(0) @binding(6) var linearSampler: sampler;

var<workgroup> localHistogram: array<atomic<u32>, numBins * numBins>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
  @builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(local_invocation_id) localId: vec3<u32>
) {
    let dim: vec3<u32> = textureDimensions(targetImage, 0);
    let localIndex: u32 = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let totalBins: u32 = numBins * numBins;
    let binsPerThread: u32 = (totalBins + workgroupInvocations - 1u) / workgroupInvocations;
    for (var i: u32 = 0u; i < binsPerThread; i = i + 1u) {
        let bin: u32 = localIndex + i * workgroupInvocations;
        if (bin < totalBins) {
            atomicStore(&localHistogram[bin], 0u);
        }
    }
    workgroupBarrier();

    if (globalId.x < dim.x && globalId.y < dim.y && globalId.z < dim.z) {
        let mat: mat3x3<f32> = rotationMatrix(params);
        let voxelCentre: vec3<f32> = vec3<f32>(globalId) + vec3<f32>(0.5, 0.5, 0.5);
        let transformed: vec3<f32> = mat * voxelCentre + vec3<f32>(params.tx, params.ty, params.tz);

        let targetIntensity: f32 = textureLoad(targetImage, globalId, 0).r;
        let movingIntensity: f32 = textureSampleLevel(movingImage, linearSampler, transformed / vec3<f32>(dim), 0).r;

        // Map intensities into bin–space.
        let rangeTarget: f32 = minMaxTarget.y - minMaxTarget.x;
        let rangeMoving: f32 = minMaxMoving.y - minMaxMoving.x;
        let binTarget: f32 = ((targetIntensity - minMaxTarget.x) / rangeTarget) * f32(numBins - 1u);
        let binMoving: f32 = ((movingIntensity - minMaxMoving.x) / rangeMoving) * f32(numBins - 1u);

        let centerTarget: f32 = floor(binTarget);
        let centerMoving: f32 = floor(binMoving);

        let i_min: u32 = u32(max(0.0, centerTarget - 1.0));
        let i_max: u32 = u32(min(f32(numBins - 1u), centerTarget + 2.0));
        let j_min: u32 = u32(max(0.0, centerMoving - 1.0));
        let j_max: u32 = u32(min(f32(numBins - 1u), centerMoving + 2.0));

        // Loop over the local support.
        for (var i: u32 = i_min; i <= i_max; i = i + 1u) {
            let weightTarget: f32 = cubicBSpline(binTarget - f32(i));
            for (var j: u32 = j_min; j <= j_max; j = j + 1u) {
                let weightMoving: f32 = cubicBSpline(binMoving - f32(j));
                let jointWeight: f32 = weightTarget * weightMoving;
                if (jointWeight > 0.0) {
                    let binIndex: u32 = i * numBins + j;
                    let value = u32(round(jointWeight * scalingFactor));
                    atomicAdd(&localHistogram[binIndex], value);
                }
            }
        }
    }
    workgroupBarrier();

    // Merge local histogram into the global histogram.
    for (var i: u32 = localIndex; i < totalBins; i = i + workgroupInvocations) {
        let localCount = f32(atomicLoad(&localHistogram[i])) / scalingFactor;
        if (localCount > 0.0) {
            atomicAddF32InGlobalMemory(&globalHistogram[i], localCount);
        }
    }
}
