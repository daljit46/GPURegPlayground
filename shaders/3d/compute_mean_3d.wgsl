enable chromium_internal_graphite;

// Computes the mean of a 3D texture. The output needs to be an intermediate array
// of size >= number of dispatched workgroups.
// To compute the final mean, another reduction step is required.
// Workgroup size needs to be a power of 2.
const workgroupSize = vec3<u32>({{workgroup_size}});

@group(0) @binding(0) var inputTexture: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> outputArray: array<f32>;

var<workgroup> localIntensities : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroupId: vec3<u32>)
{
    let coords : vec3<f32> = vec3<f32>(id.xyz);
    let dim = vec3<f32>(textureDimensions(inputTexture, 0));
    let index = local_id.x + local_id.y * workgroupSize.x + local_id.z * workgroupSize.x * workgroupSize.y;

    var intensity : f32 = 0.0;
    if (coords.x < dim.x && coords.y < dim.y && coords.z < dim.z) {
        intensity = textureLoad(inputTexture, id.xyz, 0).r;
    }
    localIntensities[index] = intensity;
    workgroupBarrier();

    var offset = workgroupSize.x * workgroupSize.y * workgroupSize.z / 2;
    while (offset > 0) {
        if (index < offset) {
            let data1 = localIntensities[index];
            let data2 = localIntensities[index + offset];
            localIntensities[index] = data1 + data2;
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    if(index == 0) {
        let wgIndex = workgroupId.x + workgroupId.y * workgroupSize.x + workgroupId.z * workgroupSize.x * workgroupSize.y;
        outputArray[wgIndex] = localIntensities[0];
    }
}
