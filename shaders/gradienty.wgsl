// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

@group(0) @binding(0) var inputTexture: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> outputArray: array<f32>;

@compute @workgroup_size({{workgroup_size}})
fn computeSobelY(@builtin(global_invocation_id) id: vec3<u32>) {
    // Apply the Sobel Y operator to the input texture
    var sum = 0.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x - 1, id.y - 1), 0).r * -1.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x, id.y - 1), 0).r * -2.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x + 1, id.y - 1), 0).r * -1.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x - 1, id.y + 1), 0).r * 1.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x, id.y + 1), 0).r * 2.0;
    sum += textureLoad(inputTexture, vec2<u32>(id.x + 1, id.y + 1), 0).r * 1.0;

    // Write the result to the output array
    let dimensions = textureDimensions(inputTexture);
    let index = id.y * dimensions.x + id.x;
    outputArray[index] = sum;
}
