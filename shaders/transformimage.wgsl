// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct Parameters {
    cosAngle: f32,
    sinAngle: f32,
    tx: f32,
    ty: f32,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<r8unorm, write>;
@group(0) @binding(3) var linearSampler: sampler;

@compute @workgroup_size({{workgroup_size}})
fn computeTransform(@builtin(global_invocation_id) id: vec3<u32>) {
    let dim = vec2<f32>(textureDimensions(inputTexture, 0));
    let coords : vec2<f32> = vec2<f32>(id.xy);

    if (coords.x >= dim.x || coords.y >= dim.y) {
        return;
    }

    // WebGPU uses column-major matrices
    let mat = mat2x2<f32>(
        params.cosAngle, params.sinAngle,
        -params.sinAngle, params.cosAngle
    );

    let transformed = mat * coords + vec2<f32>(params.tx, params.ty);

    let color = textureSampleLevel(inputTexture, linearSampler, transformed / dim, 0);
    textureStore(outputTexture, id.xy, vec4<f32>(color.r, 1.0, 0.0, 1.0));
}
