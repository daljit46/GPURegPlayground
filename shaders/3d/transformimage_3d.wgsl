// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

// A struct for 3D rigid transform parameters
struct Parameters {
    theta: f32,
    phi: f32,
    psi: f32,
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

    // WebGPU uses column-major matrices
    let cosTheta = cos(params.theta);
    let sinTheta = sin(params.theta);
    let cosPhi = cos(params.phi);
    let sinPhi = sin(params.phi);
    let cosPsi = cos(params.psi);
    let sinPsi = sin(params.psi);

    let mat = mat3x3<f32>(
        cosTheta * cosPhi, cosTheta * sinPhi * sinPsi - sinTheta * cosPsi, cosTheta * sinPhi * cosPsi + sinTheta * sinPsi,
        sinTheta * cosPhi, sinTheta * sinPhi * sinPsi + cosTheta * cosPsi, sinTheta * sinPhi * cosPsi - cosTheta * sinPsi,
        -sinPhi, cosPhi * sinPsi, cosPhi * cosPsi
    );

    let transformed = mat * coords + vec3<f32>(params.tx, params.ty, params.tz);

    let color = textureSampleLevel(inputTexture, linearSampler, transformed / dim, 0);
    textureStore(outputTexture, id.xyz, vec4<f32>(color.r, 1.0, 0.0, 1.0));   
}
