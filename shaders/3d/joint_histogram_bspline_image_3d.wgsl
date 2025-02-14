// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

// Number of bins is the same for both images
const numBins = {{numBins}};


// Cubic B-spline kernel with compact support = 2
fn cubicBSpline(x: f32) -> f32 {
    let ax = abs(x);
    if (ax <= 1.0) {
        return (2.0/3.0) - (ax * ax) + (0.5 * ax * ax * ax);
    } else if (ax < 2.0) {
        return (1.0/6.0) * pow(2.0 - ax, 3.0);
    } else {
        return 0.0;
    }
}

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
    let bin1 = (intensity1 - minMax1.x) / range1 * f32(numBins - 1);
    let bin2 = (intensity2 - minMax2.x) / range2 * f32(numBins - 1);
    let bin1_center = floor(bin1);
    let bin2_center = floor(bin2);

    // Compute the weights for the four bins
    let i_min = u32(max(0, bin1_center - 1));
    let i_max = u32(min(f32(numBins - 1), bin1_center + 2));
    let j_min = u32(max(0, bin2_center - 1));
    let j_max = u32(min(f32(numBins - 1), bin2_center + 2));

    for(var i = i_min; i <= i_max; i = i+1u) {
        let w1 = cubicBSpline(bin1 - f32(i));
        for(var j = j_min; j <= j_max; j += 1u) {
            let w2 = cubicBSpline(bin2 - f32(j));
            let jointWeight = w1 * w2;
            let bin = i * numBins + j;
            atomicAdd(&histogram[bin], u32(round(jointWeight * 100.0)));
        }
    }
}
