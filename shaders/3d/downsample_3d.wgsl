// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

@group(0) @binding(0) var inputTexture: texture_3d<f32>;
@group(0) @binding(1) var outputTexture: texture_storage_3d<r8unorm, write>;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x >= textureDimensions(outputTexture).x || id.y >= textureDimensions(outputTexture).y
        || id.z >= textureDimensions(outputTexture).z) {
        return;
    }

    let offset = vec3<u32>(0, 1, 1);

    let result = (
        textureLoad(inputTexture, 2 * id.xyz + offset.xxx, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.xyx, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.yxx, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.yyx, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.xxy, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.yxy, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.xyy, 0) +
        textureLoad(inputTexture, 2 * id.xyz + offset.yyy, 0)
    ) * 0.125;

    textureStore(outputTexture, id.xyz, result);
}
