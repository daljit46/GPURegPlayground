// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

@group(0) @binding(0) var movingImage: texture_3d<f32>;
@group(0) @binding(1) var outputGradientImage: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size({{workgroup_size}})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let imageSize = textureDimensions(movingImage);
    // Compute the derivative of the image in x, y, and z directions using finite differences
    // Clamp the values to the image boundaries
    let x1 = max(0u, id.x - 1);
    let x2 = min(imageSize.x - 1u, id.x + 1);
    let y1 = max(0u, id.y - 1);
    let y2 = min(imageSize.y - 1u, id.y + 1);
    let z1 = max(0u, id.z - 1);
    let z2 = min(imageSize.z - 1u, id.z + 1);

    let dx = textureLoad(movingImage, vec3<u32>(x2, id.y, id.z), 0).r -
             textureLoad(movingImage, vec3<u32>(x1, id.y, id.z), 0).r;
    let dy = textureLoad(movingImage, vec3<u32>(id.x, y2, id.z), 0).r -
             textureLoad(movingImage, vec3<u32>(id.x, y1, id.z), 0).r;
    let dz = textureLoad(movingImage, vec3<u32>(id.x, id.y, z2), 0).r -
             textureLoad(movingImage, vec3<u32>(id.x, id.y, z1), 0).r;

    let gradient = vec4<f32>(dx, dy, dz, 1.0);
    textureStore(outputGradientImage, id, gradient);
}
