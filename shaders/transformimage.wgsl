// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct Parameters {
    angle: f32,
    tx: f32,
    ty: f32,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var inputTexture: texture_2d<f32>;
@group(0) @binding(2) var outputTexture: texture_storage_2d<r8unorm, write>;

@compute @workgroup_size({{workgroup_size}})
fn computeTransform(@builtin(global_invocation_id) id: vec3<u32>) {
    // We need to rotate the image about its center
    let dim : vec2<u32> = textureDimensions(inputTexture, 0);
    let center = vec2<f32>(dim) / 2.0;
    let uv = vec2<f32>(id.xy);
    let uvFromCenter = uv - center;
    let rotated = vec2<f32>(
        uvFromCenter.x * cos(params.angle) - uvFromCenter.y * sin(params.angle),
        uvFromCenter.x * sin(params.angle) + uvFromCenter.y * cos(params.angle)
    );
    let translated = rotated + center + vec2<f32>(params.tx, params.ty);

    // Read the input image and write it to the output
    // using the transformed coordinates
    let color = textureLoad(inputTexture, vec2<i32>(id.xy), 0);
    textureStore(outputTexture, vec2<i32>(translated), color);
}
