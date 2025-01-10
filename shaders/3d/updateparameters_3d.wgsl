// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct Parameters {
    alpha: f32, // rotation around z-axis
    beta: f32,  // rotation around y-axis
    gamma: f32, // rotation around x-axis
    tx: f32,
    ty: f32,
    tz: f32
};

struct Output {
    ssd: atomic<u32>,
    dssd_dalpha: atomic<u32>,
    dssd_dbeta: atomic<u32>,
    dssd_dgamma: atomic<u32>,
    dssd_dtx: atomic<u32>,
    dssd_dty: atomic<u32>,
    dssd_dtz: atomic<u32>,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var targetImage: texture_3d<f32>;
@group(0) @binding(2) var movingImage: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> output: Output;


const workgroupSize = vec3<u32>({{workgroup_size}});

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

fn finiteDiffX(image: texture_3d<f32>, id: vec3<u32>) -> f32 {
    var sum = 0.0;
    sum += textureLoad(image, vec3<u32>(id.x + 1u, id.y, id.z), 0).r - textureLoad(image, vec3<u32>(id.x - 1u, id.y, id.z), 0).r;
    return sum;
}

fn finiteDiffY(image: texture_3d<f32>, id: vec3<u32>) -> f32 {
    var sum = 0.0;
    sum += textureLoad(image, vec3<u32>(id.x, id.y + 1u, id.z), 0).r - textureLoad(image, vec3<u32>(id.x, id.y - 1u, id.z), 0).r;
    return sum;
}

fn finiteDiffZ(image: texture_3d<f32>, id: vec3<u32>) -> f32 {
    var sum = 0.0;
    sum += textureLoad(image, vec3<u32>(id.x, id.y, id.z + 1u), 0).r - textureLoad(image, vec3<u32>(id.x, id.y, id.z - 1u), 0).r;
    return sum;
}

var<workgroup> local_ssd : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dalpha : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dbeta : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dgamma : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dtx : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dty : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;
var<workgroup> local_dssd_dtz : array<f32, workgroupSize.x * workgroupSize.y * workgroupSize.z>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
)
{
    let index = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let dim = vec3<f32>(textureDimensions(targetImage, 0));

    if(f32(id.x) >= dim.x || f32(id.y) >= dim.y || f32(id.z) >= dim.z) {
        local_ssd[index] = 0.0;
        local_dssd_dalpha[index] = 0.0;
        local_dssd_dbeta[index] = 0.0;
        local_dssd_dgamma[index] = 0.0;
        local_dssd_dtx[index] = 0.0;
        local_dssd_dty[index] = 0.0;
        local_dssd_dtz[index] = 0.0;
    }
    else {
        let gradMovingX = finiteDiffX(movingImage, id);
        let gradMovingY = finiteDiffY(movingImage, id);
        let gradMovingZ = finiteDiffZ(movingImage, id);
        let targetValue = textureLoad(targetImage, id, 0).r;
        let movingValue = textureLoad(movingImage, id, 0).r;
        let error = movingValue - targetValue;
        let sinAlpha = sin(params.alpha);
        let cosAlpha = cos(params.alpha);
        let sinBeta = sin(params.beta);
        let cosBeta = cos(params.beta);
        let sinGamma = sin(params.gamma);
        let cosGamma = cos(params.gamma);


        // Column-major rotation matrix
        // let mat = mat3x3<f32>(
        //     cosAlpha * cosBeta, sinAlpha * cosBeta, -sinBeta, // first column
        //     cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma, cosBeta * sinGamma,
        //     cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma
        // );
        // x' = column 1 dotted with (x, y, z)
        // we need dx'/dalpha, dx'/dbeta, dx'/dgamma, dy'/dalpha, dy'/dbeta, dy'/dgamma, dz'/dalpha, dz'/dbeta, dz'/dgamma
        let dmatDalpha = mat3x3<f32>(
            -sinAlpha * cosBeta, cosAlpha * cosBeta, 0.0,
            -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, 0.0,
            -sinAlpha * sinBeta * cosGamma + cosAlpha * sinGamma, cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, 0.0
        );
        let dmatDbeta = mat3x3<f32>(
            -cosAlpha * sinBeta, -sinAlpha * sinBeta, -cosBeta,
            cosAlpha * cosBeta * sinGamma, sinAlpha * cosBeta * sinGamma, -sinBeta * sinGamma,
            cosAlpha * cosBeta * cosGamma, sinAlpha * cosBeta * cosGamma, -sinBeta * cosGamma
        );
        let dmatDgamma = mat3x3<f32>(
            0.0, 0.0, 0.0,
            cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma,
            -cosAlpha * sinBeta * sinGamma + sinAlpha * cosGamma, -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, -cosBeta * sinGamma
        );

        let gradXYZalpha = dmatDalpha * vec3<f32>(id.xyz);
        let gradXYZbeta = dmatDbeta * vec3<f32>(id.xyz);
        let gradXYZgamma = dmatDgamma * vec3<f32>(id.xyz);

        let gradAlpha = 2 * error * (gradMovingX * gradXYZalpha.x + gradMovingY * gradXYZalpha.y + gradMovingZ * gradXYZalpha.z);
        let gradBeta = 2 * error * (gradMovingX * gradXYZbeta.x + gradMovingY * gradXYZbeta.y + gradMovingZ * gradXYZbeta.z);
        let gradGamma = 2 * error * (gradMovingX * gradXYZgamma.x + gradMovingY * gradXYZgamma.y + gradMovingZ * gradXYZgamma.z);
        let gradTx = 2 * error * gradMovingX;
        let gradTy = 2 * error * gradMovingY;
        let gradTz = 2 * error * gradMovingZ;

        local_ssd[index] = error * error;
        local_dssd_dalpha[index] = gradAlpha;
        local_dssd_dbeta[index] = gradBeta;
        local_dssd_dgamma[index] = gradGamma;
        local_dssd_dtx[index] = gradTx;
        local_dssd_dty[index] = gradTy;
        local_dssd_dtz[index] = gradTz;
    }

    workgroupBarrier();

    // Perform tree based reduction
    var pairOffset = workgroupSize.x * workgroupSize.y * workgroupSize.z / 2;
    while(pairOffset > 0u) {
        if(index < pairOffset) {
            local_ssd[index] += local_ssd[index + pairOffset];
            local_dssd_dalpha[index] += local_dssd_dalpha[index + pairOffset];
            local_dssd_dbeta[index] += local_dssd_dbeta[index + pairOffset];
            local_dssd_dgamma[index] += local_dssd_dgamma[index + pairOffset];
            local_dssd_dtx[index] += local_dssd_dtx[index + pairOffset];
            local_dssd_dty[index] += local_dssd_dty[index + pairOffset];
            local_dssd_dtz[index] += local_dssd_dtz[index + pairOffset];
        }
        workgroupBarrier();
        pairOffset /= 2u;
    }

    if(index == 0u) {
        atomicAddF32(&output.ssd, local_ssd[0]);
        atomicAddF32(&output.dssd_dalpha, local_dssd_dalpha[0]);
        atomicAddF32(&output.dssd_dbeta, local_dssd_dbeta[0]);
        atomicAddF32(&output.dssd_dgamma, local_dssd_dgamma[0]);
        atomicAddF32(&output.dssd_dtx, local_dssd_dtx[0]);
        atomicAddF32(&output.dssd_dty, local_dssd_dty[0]);
        atomicAddF32(&output.dssd_dtz, local_dssd_dtz[0]);
    }
}
