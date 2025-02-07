struct TransformationParameters {
    alpha: f32,
    beta: f32,
    gamma: f32,
    tx: f32,
    ty: f32,
    tz: f32
};


fn rotationMatrix(params: TransformationParameters) -> mat3x3<f32> {
    let sinAlpha = sin(params.alpha);
    let cosAlpha = cos(params.alpha);
    let sinBeta  = sin(params.beta);
    let cosBeta  = cos(params.beta);
    let sinGamma = sin(params.gamma);
    let cosGamma = cos(params.gamma);

    // WebGPU uses column-major matrices
    return mat3x3<f32>(
        cosAlpha * cosBeta, sinAlpha * cosBeta, -sinBeta,
        cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, sinAlpha * sinBeta * sinGamma + cosAlpha * cosGamma, cosBeta * sinGamma,
        cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma
    );
}


// Derivative of the transformation matrix with respect to alpha
fn dmatDalpha(params: TransformationParameters) -> mat3x3<f32> {
    let sinAlpha = sin(params.alpha);
    let cosAlpha = cos(params.alpha);
    let sinBeta  = sin(params.beta);
    let cosBeta  = cos(params.beta);
    let sinGamma = sin(params.gamma);
    let cosGamma = cos(params.gamma);

    return mat3x3<f32>(
        -sinAlpha * cosBeta, cosAlpha * cosBeta, 0.0,
        -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, cosAlpha * sinBeta * sinGamma - sinAlpha * cosGamma, 0.0,
        -sinAlpha * sinBeta * cosGamma + cosAlpha * sinGamma, cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, 0.0
    );
}

// Derivative of the transformation matrix with respect to beta
fn dmatDbeta(params: TransformationParameters) -> mat3x3<f32> {
    let sinAlpha = sin(params.alpha);
    let cosAlpha = cos(params.alpha);
    let sinBeta  = sin(params.beta);
    let cosBeta  = cos(params.beta);
    let sinGamma = sin(params.gamma);
    let cosGamma = cos(params.gamma);

    return mat3x3<f32>(
        -cosAlpha * sinBeta, -sinAlpha * sinBeta, -cosBeta,
        cosAlpha * cosBeta * sinGamma, sinAlpha * cosBeta * sinGamma, -sinBeta * sinGamma,
        cosAlpha * cosBeta * cosGamma, sinAlpha * cosBeta * cosGamma, -sinBeta * cosGamma
    );
}

// Derivative of the transformation matrix with respect to gamma
fn dmatDgamma(params: TransformationParameters) -> mat3x3<f32> {
    let sinAlpha = sin(params.alpha);
    let cosAlpha = cos(params.alpha);
    let sinBeta  = sin(params.beta);
    let cosBeta  = cos(params.beta);
    let sinGamma = sin(params.gamma);
    let cosGamma = cos(params.gamma);
    return mat3x3<f32>(
        0.0, 0.0, 0.0,
        cosAlpha * sinBeta * cosGamma + sinAlpha * sinGamma, sinAlpha * sinBeta * cosGamma - cosAlpha * sinGamma, cosBeta * cosGamma,
        -cosAlpha * sinBeta * sinGamma + sinAlpha * cosGamma, -sinAlpha * sinBeta * sinGamma - cosAlpha * cosGamma, -cosBeta * sinGamma
    );
}
