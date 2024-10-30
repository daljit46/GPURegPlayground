// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

// Compute the difference in intensity between two r8unorm textures
@group(0) @binding(0) var texture1: texture_2d<u32>;
@group(0) @binding(1) var texture2: texture_2d<u32>;
@group(0) @binding(2) var<storage> outputBuffer: array<i32>

@compute @workgroup_size({{workgroup_size}})
fn computeDifference(@builtin(global_invocation_id) id: vec3<u32>) {
    let intensity1 = textureLoad(texture1, id.xy, 0).r;
    let intensity2 = textureLoad(texture2, id.xy, 0).r;
    let difference = i32(intensity1) - i32(intensity2);

    // assume outbuffer stores row-major matrix
    let index = id.x + id.y * u32(textureDimensions(texture1, 0).x);
    outputBuffer[index] = difference;
}