// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct Parameters {
    cosAngle: f32,
    sinAngle: f32,
    tx: f32,
    ty: f32,
};

struct Output {
    dssd_dtheta: atomic<u32>,
    dssd_dtx: atomic<u32>,
    dssd_dty: atomic<u32>,
    ssd: atomic<u32>,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> gradientBufferX: array<f32>;
@group(0) @binding(2) var<storage, read> gradientBufferY: array<f32>;
@group(0) @binding(3) var targetImage: texture_2d<f32>;
@group(0) @binding(4) var movingImage: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> output: Output;


const workgroupSize = vec3<u32>({{workgroup_size}});

var<workgroup> local_ssd: array<f32, workgroupSize.x * workgroupSize.y>;
var<workgroup> local_dssd_dtheta: array<f32, workgroupSize.x * workgroupSize.y>;
var<workgroup> local_dssd_dtx: array<f32, workgroupSize.x * workgroupSize.y>;
var<workgroup> local_dssd_dty: array<f32, workgroupSize.x * workgroupSize.y>;

// WGSL doesn't support atomicAdd for f32, so we use bitcasting to u32
fn atomicAddF32(sum: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {
    var old = 0u;
    loop {
      let new_value = value + bitcast<f32>(old);
      let exchange_result = atomicCompareExchangeWeak(sum, old, bitcast<u32>(new_value));
      if exchange_result.exchanged {
         return new_value;
      }
      old = exchange_result.old_value;
    }
}

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, 1)
fn updateParameters(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
)
{
    let index = localId.y * workgroupSize.x + localId.x;
    let dim = vec2<f32>(textureDimensions(targetImage, 0));

    if (f32(id.x) >= dim.x || f32(id.y) >= dim.y) {
        local_ssd[index] = 0;
        local_dssd_dtheta[index] = 0;
        local_dssd_dtx[index] = 0;
        local_dssd_dty[index] = 0;
    }
    else {
        let gradMovingX = gradientBufferX[id.y * u32(dim.x) + id.x];
        let gradMovingY = gradientBufferY[id.y * u32(dim.x) + id.x];
        let targetValue : f32 = textureLoad(targetImage, id.xy, 0).r;
        let movingValue : f32 = textureLoad(movingImage, id.xy, 0).r;
        let error : f32 = (movingValue - targetValue);

        let gradXTheta = -params.sinAngle * f32(id.x) - params.cosAngle * f32(id.y);
        let gradYTheta = params.cosAngle * f32(id.x) - params.sinAngle * f32(id.y);
        let dssd_dtheta = 2.0 * error * (gradMovingX * gradXTheta + gradMovingY * gradYTheta);
        let dssd_dtx = 2.0 * error * gradMovingX;
        let dssd_dty = 2.0 * error * gradMovingY;

        local_ssd[index] = error * error;
        local_dssd_dtheta[index] = dssd_dtheta;
        local_dssd_dtx[index] = dssd_dtx;
        local_dssd_dty[index] = dssd_dty;
    }

    workgroupBarrier();

    // Perform tree based reduction
    var pairOffset = workgroupSize.x * workgroupSize.y / 2u;
    while(pairOffset > 0u) {
        if(index < pairOffset) {
            local_ssd[index] += local_ssd[index + pairOffset];
            local_dssd_dtheta[index] += local_dssd_dtheta[index + pairOffset];
            local_dssd_dtx[index] += local_dssd_dtx[index + pairOffset];
            local_dssd_dty[index] += local_dssd_dty[index + pairOffset];
        }
        workgroupBarrier();
        pairOffset = pairOffset / 2u;
    }

    if(index == 0u) {
        atomicAddF32(&output.ssd, local_ssd[0]);
        atomicAddF32(&output.dssd_dtheta, local_dssd_dtheta[0]);
        atomicAddF32(&output.dssd_dtx, local_dssd_dtx[0]);
        atomicAddF32(&output.dssd_dty, local_dssd_dty[0]);
    }
}
