// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct Parameters {
    angle: f32,
    tx: f32,
    ty: f32,
};

struct Output {
    dssd_dtheta: atomic<i32>,
    dssd_dtx: atomic<i32>,
    dssd_dty: atomic<i32>,
    ssd: atomic<i32>,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var gradientTextureX: texture_2d<f32>;
@group(0) @binding(2) var gradientTextureY: texture_2d<f32>;
@group(0) @binding(3) var originalImage: texture_2d<f32>;
@group(0) @binding(4) var transformedImage: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> output: Output;

const scaling_float_to_int = 1000.0;

@compute @workgroup_size({{workgroup_size}})
fn updateGradients(@builtin(global_invocation_id) id: vec3<u32>) {
    // dssd_dtheta = -2 * sum(discrepancy * (dI/dx * dx/dtheta + dI/dy * dy/dtheta) )
    // where I is the moving image
    // and discrepancy = (Reference Image - I)
    // and dx/dtheta = -y, dy/dtheta = x

    let dim = vec2<f32>(textureDimensions(gradientTextureX, 0));
    let coords : vec2<f32> = vec2<f32>(id.xy);

    if (coords.x >= dim.x || coords.y >= dim.y) {
        return;
    }

    let center : vec2<f32> = vec2<f32>(dim.xy / 2.0);
    let offset : vec2<f32> = coords - center;

    let cosTheta = cos(params.angle);
    let sinTheta = sin(params.angle);

    let discrepancy = textureLoad(originalImage, id.xy, 0).r - textureLoad(transformedImage, id.xy, 0).r;
    let dIdX = textureLoad(gradientTextureX, id.xy, 0).r;
    let dIdY = textureLoad(gradientTextureY, id.xy, 0).r;
    
    // x = cos(theta) * x - sin(theta) * y
    let dx_dtheta = -sinTheta * f32(id.x) - cosTheta * f32(id.y);
    // y = sin(theta) * x + cos(theta) * y
    let dy_dtheta = cosTheta * f32(id.x) - sinTheta * f32(id.y);
    let dssd_dtheta = -2.0 * discrepancy * (dIdX * dx_dtheta + dIdY * dy_dtheta);

    let dssd_dtx = -2.0 * discrepancy * dIdX;
    let dssd_dty = -2.0 * discrepancy * dIdY;
    atomicAdd(&output.dssd_dtheta, i32(dssd_dtheta * scaling_float_to_int));
    atomicAdd(&output.dssd_dtx, i32(dssd_dtx * scaling_float_to_int));
    atomicAdd(&output.dssd_dty, i32(dssd_dty * scaling_float_to_int));
    atomicAdd(&output.ssd, i32(discrepancy * discrepancy * scaling_float_to_int));
}
