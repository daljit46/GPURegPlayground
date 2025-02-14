// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

// Number of bins is the same for both images
const numBins = {{numBins}};

// Compute the histogram for an image
@group(0) @binding(0) var<storage, read> minMax1: vec2<f32>;
@group(0) @binding(1) var<storage, read> minMax2: vec2<f32>;
@group(0) @binding(2) var inputTexture1: texture_3d<f32>;
@group(0) @binding(3) var inputTexture2: texture_3d<f32>;
// 2D histogram flattened to 1D
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Assume that the dimensions of the two images are the same
    let dim = vec3<f32>(textureDimensions(inputTexture1, 0));
    let coords : vec3<f32> = vec3<f32>(id.xyz);
    let range1 = minMax1.y - minMax1.x;
    let range2 = minMax2.y - minMax2.x;

    if (range1 == 0.0 || range2 == 0.0) {
        return;
    }

    if (coords.x >= dim.x || coords.y >= dim.y || coords.z >= dim.z) {
        return;
    }

    let intensity1 = textureLoad(inputTexture1, id.xyz, 0).r;
    let intensity2 = textureLoad(inputTexture2, id.xyz, 0).r;
    // bin = (intensity - min) / range * numBins
    let bin1 = u32(round((intensity1 - minMax1.x) / range1 * f32(numBins - 1)));
    let bin2 = u32(round((intensity2 - minMax2.x) / range2 * f32(numBins - 1)));

    // Flatten 2D histogram to 1D using row-major order
    let bin = bin1 * numBins + bin2;
    atomicAdd(&histogram[bin], 1u);
}
