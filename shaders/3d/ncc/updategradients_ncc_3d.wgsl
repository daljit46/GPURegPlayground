// NOTE: the total workgroup size of this shader must be 32 <= 2^N <= 1024
// need to enable the chromium_internal_graphite feature to use r8unorm
// as a storage format for the output texture
enable chromium_internal_graphite;

struct TransformationParameters {
    alpha: f32, // rotation around z-axis
    beta: f32,  // rotation around y-axis
    gamma: f32, // rotation around x-axis
    tx: f32,
    ty: f32,
    tz: f32
};

// Let I=I(x,y,z) be the target image and J = J(T(x, y, z)) be the moving image
// (the latter is a function of the transformation parameters)
// NCC = A / sqrt(B * C) where
// A = sum((I - mean(I)) * (J - mean(J)))
// B = sum((I - mean(I))^2)
// C = sum((J - mean(J))^2)
// NOTE: J = J(T(x,y,z)) where T is the transformation function
// The sum is over all voxels in the images.
// dNCC/dp_k = 1/[sqrt(B) * C^3/2] * [dA/dp_k * C - 0.5 A * C * dC/dp_k] where
// p_k is the k-th transformation parameter
// which after substituting the expressions for the derivatives of A and C becomes
// dNCC/dp_k = 1/[sqrt(B) * C^3/2] * sum[(CI' - AJ') * (grad(J) * dT/dp_k - d/dp_k(mean(J)))]
// where I' = I - mean(I) and J' = J - mean(J)
// d/dp_k = sum(grad(J) * dT/dp_k) over all voxels

struct NCCPartialSums {

};

@group(0) @binding(0) var<storage, read> params: TransformationParameters;
@group(0) @binding(1) var<storage, read> targetImageMean: f32;
@group(0) @binding(2) var<storage, read> movingImageMean: f32;
@group(0) @binding(3) var targetImage: texture_3d<f32>;
@group(0) @binding(4) var movingImage: texture_3d<f32>;
@group(0) @binding(5) var<storage, read_write> nccGrads: array<NCCGradients>;
@group(0) @binding(6) var linearSampler: sampler;
