// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

@group(0) @binding(0) var inputTexture: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> histogramBuffer: array<atomic<u32>>;

const numBins = 256u;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec3<f32>(textureDimensions(inputTexture, 0));
    let coords = vec3<f32>(id.xyz);

    if (coords.x >= dim.x || coords.y >= dim.y || coords.z >= dim.z) {
        return;
    }

    let color = textureLoad(inputTexture, id.xyz, 0).r;

    let bin = u32(color * f32(numBins - 1u));
    atomicAdd(&histogramBuffer[bin], 1u);
}
