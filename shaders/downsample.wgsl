// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_2d<r8unorm, write>;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= textureDimensions(outputTexture).x || id.y >= textureDimensions(outputTexture).y) {
        return;
    }
    let offset = vec2<u32>(0, 1);
    let result = (
        textureLoad(inputTexture, 2 * id.xy + offset.xx, 0) +
        textureLoad(inputTexture, 2 * id.xy + offset.yx, 0) +
        textureLoad(inputTexture, 2 * id.xy + offset.xy, 0) +
        textureLoad(inputTexture, 2 * id.xy + offset.yy, 0)
    ) * 0.25;

    textureStore(outputTexture, id.xy, result);
}
