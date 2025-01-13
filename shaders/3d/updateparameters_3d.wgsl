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

struct SSDGradients {
    ssd: f32,
    dssd_dalpha: f32,
    dssd_dbeta: f32,
    dssd_dgamma: f32,
    dssd_dtx: f32,
    dssd_dty: f32,
    dssd_dtz: f32,
};

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var targetImage: texture_3d<f32>;
@group(0) @binding(2) var movingImage: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> ssdGrads: array<SSDGradients>;

const workgroupSize = vec3<u32>({{workgroup_size}});
const workgroupInvocations = workgroupSize.x * workgroupSize.y * workgroupSize.z;

fn finiteDiff(image: texture_3d<f32>, id: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        textureLoad(image, vec3<u32>(id.x + 1u, id.y, id.z), 0).r - textureLoad(image, vec3<u32>(id.x - 1u, id.y, id.z), 0).r,
        textureLoad(image, vec3<u32>(id.x, id.y + 1u, id.z), 0).r - textureLoad(image, vec3<u32>(id.x, id.y - 1u, id.z), 0).r,
        textureLoad(image, vec3<u32>(id.x, id.y, id.z + 1u), 0).r - textureLoad(image, vec3<u32>(id.x, id.y, id.z - 1u), 0).r
    );
}

var<workgroup> local_gradients : array<SSDGradients, workgroupInvocations>;

@compute @workgroup_size(workgroupSize.x, workgroupSize.y, workgroupSize.z)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>,
    @builtin(num_workgroups) numWorkgroups: vec3<u32>
)
{
    let index = localId.x + localId.y * workgroupSize.x + localId.z * workgroupSize.x * workgroupSize.y;
    let dim = vec3<f32>(textureDimensions(targetImage, 0));

    if(f32(id.x) >= dim.x || f32(id.y) >= dim.y || f32(id.z) >= dim.z) {
        local_gradients[index] = SSDGradients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    else {
        let gradMoving = finiteDiff(movingImage, id);
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

        let gradAlpha = 2 * error * (gradMoving.x * gradXYZalpha.x + gradMoving.y * gradXYZalpha.y + gradMoving.z * gradXYZalpha.z);
        let gradBeta = 2 * error * (gradMoving.x * gradXYZbeta.x + gradMoving.y * gradXYZbeta.y + gradMoving.z * gradXYZbeta.z);
        let gradGamma = 2 * error * (gradMoving.x * gradXYZgamma.x + gradMoving.y * gradXYZgamma.y + gradMoving.z * gradXYZgamma.z);
        let gradTx = 2 * error * gradMoving.x;
        let gradTy = 2 * error * gradMoving.y;
        let gradTz = 2 * error * gradMoving.z;

        local_gradients[index] = SSDGradients(error * error, gradAlpha, gradBeta, gradGamma, gradTx, gradTy, gradTz);
    }

    workgroupBarrier();

    // Perform tree based reduction
    var pairOffset = workgroupInvocations / 2;
    while(pairOffset > 0u) {
        if(index < pairOffset) {
            let data1 = local_gradients[index];
            let data2 = local_gradients[index + pairOffset];
            local_gradients[index] = SSDGradients(
                data1.ssd + data2.ssd,
                data1.dssd_dalpha + data2.dssd_dalpha,
                data1.dssd_dbeta + data2.dssd_dbeta,
                data1.dssd_dgamma + data2.dssd_dgamma,
                data1.dssd_dtx + data2.dssd_dtx,
                data1.dssd_dty + data2.dssd_dty,
                data1.dssd_dtz + data2.dssd_dtz
            );
        }
        workgroupBarrier();
        pairOffset /= 2u;
    }

    if(index == 0u) {
        let wgIndex = workgroupId.x + workgroupId.y * numWorkgroups.x + workgroupId.z * numWorkgroups.x * numWorkgroups.y;
        let localData = local_gradients[0];
        ssdGrads[wgIndex].ssd = localData.ssd;
        ssdGrads[wgIndex].dssd_dalpha = localData.dssd_dalpha;
        ssdGrads[wgIndex].dssd_dbeta = localData.dssd_dbeta;
        ssdGrads[wgIndex].dssd_dgamma = localData.dssd_dgamma;
        ssdGrads[wgIndex].dssd_dtx = localData.dssd_dtx;
        ssdGrads[wgIndex].dssd_dty = localData.dssd_dty;
        ssdGrads[wgIndex].dssd_dtz = localData.dssd_dtz;
    }
}
