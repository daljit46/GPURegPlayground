enable chromium_internal_graphite;

struct TransformationParameters {
    alpha: f32, // rotation around z-axis
    beta: f32,  // rotation around y-axis
    gamma: f32, // rotation around x-axis
    tx: f32,
    ty: f32,
    tz: f32
};

// Computes the mean of a 3D texture transformed by the given parameters.
// The output needs to be an intermediate array of size >= number of dispatched workgroups.
// To compute the final mean, another reduction step is required.
// Workgroup size needs to be a power of 2.
const workgroupSize = vec3<u32>({{workgroup_size}});
// 0: sum, 1: min, 2: max
const operation = {{operation}};

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var inputTexture: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> outputArray: array<f32>;
@group(0) @binding(3) var linearSampler: sampler;

var<workgroup> localIntensities : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;


fn reductionOperation(a: f32, b: f32, operation: u32) -> f32 {
    switch (operation) {
        case 0: { return a + b; }
        case 1: { return min(a, b); }
        case 2: { return max(a, b); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroupId: vec3<u32>,
        @builtin(num_workgroups) numWorkgroups: vec3<u32>
)
{
    let coords : vec3<f32> = vec3<f32>(id.xyz);
    let dim = vec3<f32>(textureDimensions(inputTexture, 0));
    let index = local_id.x + local_id.y * workgroupSize.x + local_id.z * workgroupSize.x * workgroupSize.y;

    var intensity : f32 = 0.0;

    let cosAlpha = cos(params.alpha);
    let sinAlpha = sin(params.alpha);
    let cosBeta = cos(params.beta);
    let sinBeta = sin(params.beta);
    let cosGamma = cos(params.gamma);
    let sinGamma = sin(params.gamma);
    // WebGPU uses column-major matrices
    let mat = mat3x3<f32>(
        cosAlpha * cosBeta, sinAlpha * cosBeta, -sinBeta,
        cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma, cosBeta * sinGamma,
        cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma
    );
    let voxelCenter = coords + vec3<f32>(0.5, 0.5, 0.5);
    let transformed = mat * voxelCenter + vec3<f32>(params.tx, params.ty, params.tz);

    if (transformed.x > 0.0 && transformed.y > 0.0 && transformed.z > 0.0
        && transformed.x < dim.x && transformed.y < dim.y && transformed.z < dim.z
        && coords.x < dim.x && coords.y < dim.y && coords.z < dim.z) {
        intensity = textureSampleLevel(inputTexture, linearSampler, transformed / dim, 0).r;
    }

    localIntensities[index] = intensity;
    workgroupBarrier();

    var offset = workgroupSize.x * workgroupSize.y * workgroupSize.z / 2;
    while (offset > 0) {
        if (index < offset) {
            let data1 = localIntensities[index];
            let data2 = localIntensities[index + offset];
            localIntensities[index] = reductionOperation(data1, data2, operation);
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    if(index == 0) {
        let wgIndex = workgroupId.x + workgroupId.y * numWorkgroups.x + workgroupId.z * numWorkgroups.x * numWorkgroups.y;
        outputArray[wgIndex] = localIntensities[0];
    }
}
