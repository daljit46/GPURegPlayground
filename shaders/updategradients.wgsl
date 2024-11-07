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
@group(0) @binding(1) var<storage, read> gradientBufferX: array<f32>;
@group(0) @binding(2) var<storage, read> gradientBufferY: array<f32>;
@group(0) @binding(3) var referenceImage: texture_2d<f32>;
@group(0) @binding(4) var movingImage: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> output: Output;

const scaling_float_to_int = 1000.0;

@compute @workgroup_size({{workgroup_size}})
fn updateGradients(@builtin(global_invocation_id) id: vec3<u32>) {
    // dssd_dtheta = -2 * sum(error * (dI/dx * dx/dtheta + dI/dy * dy/dtheta) )
    // where I is the moving image
    // and error = (M - R)
    // and dx/dtheta = -y, dy/dtheta = x
    let dim = vec2<f32>(textureDimensions(referenceImage, 0));

    if (f32(id.x) >= dim.x || f32(id.y) >= dim.y) {
        return;
    }

    let center = dim / 2.0;
    let offset = vec2<f32>(id.xy) - center;

    let cosTheta = cos(params.angle);
    let sinTheta = sin(params.angle);

    let dIdX = gradientBufferX[id.y * u32(dim.x) + id.x];
    let dIdX = gradientBufferY[id.y * u32(dim.x) + id.x];
    let valueOriginal : f32 = textureLoad(movingImage, id.xy, 0).r;
    let valueWarped : f32 = textureLoad(referenceImage, id.xy, 0).r;
    let error : f32 = (valueWarped - valueOriginal);

    // x = cos(theta) * x - sin(theta) * y
    // y = sin(theta) * x + cos(theta) * y
    let dx_dtheta = -sinTheta * f32(offset.x) - cosTheta * f32(offset.y);
    let dy_dtheta = cosTheta * f32(offset.x) - sinTheta * f32(offset.y);
    let dssd_dtheta = -2.0 * error * (dIdX * dx_dtheta + dIdY * dy_dtheta);
    let dssd_dtx = -2.0 * error * dIdX;
    let dssd_dty = -2.0 * error * dIdY;

    atomicAdd(&output.dssd_dtheta, i32(dssd_dtheta * scaling_float_to_int));
    atomicAdd(&output.dssd_dtx, i32(dssd_dtx * scaling_float_to_int));
    atomicAdd(&output.dssd_dty, i32(dssd_dty * scaling_float_to_int));
    atomicAdd(&output.ssd, i32(error * error));
}
