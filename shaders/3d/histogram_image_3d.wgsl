// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

const numBins = {{numBins}};

// Compute the histogram for an image
@group(0) @binding(0) var<storage, read> minMax: vec2<f32>;
@group(0) @binding(1) var inputTexture: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec3<f32>(textureDimensions(inputTexture, 0));
    let coords : vec3<f32> = vec3<f32>(id.xyz);
    let range = minMax.y - minMax.x;

    if (range == 0.0) {
        return;
    }

    if (coords.x >= dim.x || coords.y >= dim.y || coords.z >= dim.z) {
        return;
    }

    let intensity = textureLoad(inputTexture, id.xyz, 0).r;
    // bin = (intensity - min) / range * numBins
    let bin = u32(round((intensity - minMax.x) / range * f32(numBins - 1)));
    atomicAdd(&histogram[bin], 1u);
}
