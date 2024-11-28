// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

// A struct for 3D rigid transform parameters
struct Parameters {
    alpha: f32, // rotation around z-axis
    beta: f32,  // rotation around y-axis
    gamma: f32, // rotation around x-axis
    tx: f32,
    ty: f32,
    tz: f32
};


@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var inputTexture: texture_3d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_3d<r8unorm, write>;
@group(0) @binding(3) var linearSampler: sampler;


@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec3<f32>(textureDimensions(inputTexture, 0));
    let coords : vec3<f32> = vec3<f32>(id.xyz);

    if (coords.x >= dim.x || coords.y >= dim.y || coords.z >= dim.z) {
        return;
    }

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

    let transformed = mat * coords + vec3<f32>(params.tx, params.ty, params.tz);
    if(transformed.x < 0.0 || transformed.y < 0.0 || transformed.z < 0.0) {
        textureStore(outputTexture, id.xyz, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
    else {
        // let color = textureSampleLevel(inputTexture, linearSampler, transformed / dim, 0);
        let color = textureLoad(inputTexture, vec3<i32>(transformed), 0);
        textureStore(outputTexture, id.xyz, vec4<f32>(color.r, 0.0, 0.0, 0.0));
    }
}
