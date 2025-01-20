// This shader updates the transform of the input 3D texture using the SSD and gradients from previous shader.
// To do this we will employ an Adam optimizer to update the parameters of the transform.

const MAX_ITERATIONS: u32 = 500;
const NUM_ADAM_PARAMETERS: u32 = 6;
const ADAM_BETA1: f32 = 0.7;
const ADAM_BETA2: f32 = 0.9999;
const ADAM_EPSILON: f32 = 1e-8;

struct TransformationParameters {
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


struct AdamState {
    learning_rates: array<f32, NUM_ADAM_PARAMETERS>,
    firstMoments: array<f32, NUM_ADAM_PARAMETERS>,
    secondMoments: array<f32, NUM_ADAM_PARAMETERS>,
};

@group(0) @binding(0) var<storage, read_write> inputGradientValues: SSDGradients;
@group(0) @binding(1) var<storage, read_write> adamState: AdamState;
@group(0) @binding(2) var<storage, read_write> transformParams: TransformationParameters;
@group(0) @binding(3) var<storage, read_write> minSSD: f32;
@group(0) @binding(4) var<storage, read_write> minTransformParams: TransformationParameters;
@group(0) @binding(5) var<storage, read_write> ssdHistory: array<SSDGradients, MAX_ITERATIONS>;
@group(0) @binding(6) var<storage, read_write> currentIteration: u32;
@group(0) @binding(7) var<storage, read_write> stop: u32;


// A single thread will update the parameters using Adam optimizer
@compute @workgroup_size(1)
fn main() {
    if(stop > 0u) {
        return;
    }

    let ssd = inputGradientValues.ssd;
    let dssd_dalpha = inputGradientValues.dssd_dalpha;
    let dssd_dbeta = inputGradientValues.dssd_dbeta;
    let dssd_dgamma = inputGradientValues.dssd_dgamma;
    let dssd_dtx = inputGradientValues.dssd_dtx;
    let dssd_dty = inputGradientValues.dssd_dty;
    let dssd_dtz = inputGradientValues.dssd_dtz;

    if(ssd < minSSD || currentIteration == 0u) {
        minSSD = ssd;
        minTransformParams = transformParams;
    }
    // Update the ssd history
    currentIteration += 1u;

    let gradients = array<f32, NUM_ADAM_PARAMETERS>(dssd_dalpha, dssd_dbeta, dssd_dgamma, dssd_dtx, dssd_dty, dssd_dtz);
    var newTransformParams = array<f32, NUM_ADAM_PARAMETERS>(
        transformParams.alpha, transformParams.beta, transformParams.gamma,
        transformParams.tx, transformParams.ty, transformParams.tz
    );
    for(var i = 0u; i < NUM_ADAM_PARAMETERS; i = i + 1u) {
        adamState.firstMoments[i] = ADAM_BETA1 * adamState.firstMoments[i] + (1.0 - ADAM_BETA1) * gradients[i];
        adamState.secondMoments[i] = ADAM_BETA2 * adamState.secondMoments[i] + (1.0 - ADAM_BETA2) * gradients[i] * gradients[i];

        let firstMomentBiasCorrected = adamState.firstMoments[i] / (1.0 - pow(ADAM_BETA1, f32(currentIteration)));
        let secondMomentBiasCorrected = adamState.secondMoments[i] / (1.0 - pow(ADAM_BETA2, f32(currentIteration)));
        newTransformParams[i] -= adamState.learning_rates[i] * firstMomentBiasCorrected / (sqrt(secondMomentBiasCorrected) + ADAM_EPSILON);
    }

    transformParams.alpha = newTransformParams[0];
    transformParams.beta = newTransformParams[1];
    transformParams.gamma = newTransformParams[2];
    transformParams.tx = newTransformParams[3];
    transformParams.ty = newTransformParams[4];
    transformParams.tz = newTransformParams[5];

    ssdHistory[currentIteration].ssd = ssd;
    ssdHistory[currentIteration].dssd_dalpha = dssd_dalpha;
    ssdHistory[currentIteration].dssd_dbeta = dssd_dbeta;
    ssdHistory[currentIteration].dssd_dgamma = dssd_dgamma;
    ssdHistory[currentIteration].dssd_dtx = dssd_dtx;
    ssdHistory[currentIteration].dssd_dty = dssd_dty;
    ssdHistory[currentIteration].dssd_dtz = dssd_dtz;

    // Reset the gradients
    inputGradientValues.ssd = 0.0;
    inputGradientValues.dssd_dalpha = 0.0;
    inputGradientValues.dssd_dbeta = 0.0;
    inputGradientValues.dssd_dgamma = 0.0;
    inputGradientValues.dssd_dtx = 0.0;
    inputGradientValues.dssd_dty = 0.0;
    inputGradientValues.dssd_dtz = 0.0;


    // Check if the last 10 iterations have not improved the SSD
    var hasNotImproved = false;
    if(currentIteration > 10) {
        // Compute mean of the last 10 SSD values
        var meanSSD = 0.0;
        for(var i = 0u; i < 10u; i = i + 1u) {
            meanSSD += ssdHistory[currentIteration - i].ssd;
        }
        meanSSD = meanSSD / 10.0;

        // Check if the mean SSD has not improved
        if(abs(meanSSD - ssd) < 0.01) {
            hasNotImproved = true;
        }
    }
    if(currentIteration >= MAX_ITERATIONS || hasNotImproved) {
        stop = 1u;
    }
}
