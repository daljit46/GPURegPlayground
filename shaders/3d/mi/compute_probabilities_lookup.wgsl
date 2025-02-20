const numBins: u32 = {{numBins}};
const totalBins: u32 = numBins * numBins;

// Each voxel in the image provides a "soft" contribution to the joint histogram.
// h(v) = B_spline(b_i(v) - i) * B_spline(b_j(v) - j) where
// b_i and b_j are the target and (transformed) moving intensities mapped to the bin space
// i and j are the bin indices in the joint histogram.
// Taking the derivative of h(v) with respect to the transformation parameters q gives
// dh(v)/dq = B_spline'(b_i(v) - i) * B_spline(b_j(v) - j) * db_j(v)/dq
//
// dMI/dq = sum_ij [1/T * dh(v)/dq * log(p_ij / (p_i * p_j))]
// where T is the total mass of the joint histogram.
// The factor log(p_ij / (p_i * p_j)) is precomputed and stored in the lookup table.
@group(0) @binding(0) var<storage, read> histogram: array<u32>;
// Lookup table for L_ij = log(p_ij / (p_i * p_j))
@group(0) @binding(1) var<storage, read_write> miLookup: array<f32>;
@group(0) @binding(2) var<storage, read_write> miResult: f32;
@group(0) @binding(3) var<storage, read_write> totalMass: f32;

@compute @workgroup_size(1)
fn main() {
    var total: f32 = 0.0;
    // Marginal sums
    var p_target: array<f32, numBins>;
    var p_moving: array<f32, numBins>;
    for (var i: u32 = 0u; i < numBins; i = i + 1u) {
        p_target[i] = 0.0;
        p_moving[i] = 0.0;
    }
    for (var i: u32 = 0u; i < numBins; i = i + 1u) {
        for (var j: u32 = 0u; j < numBins; j = j + 1u) {
            let idx: u32 = i * numBins + j;
            let count: f32 = bitcast<f32>(histogram[idx]);
            total = total + count;
            p_target[i] = p_target[i] + count;
            p_moving[j] = p_moving[j] + count;
        }
    }


    // Normalize to get probabilities.
    for (var i: u32 = 0u; i < numBins; i = i + 1u) {
        p_target[i] = p_target[i] / total;
        p_moving[i] = p_moving[i] / total;
    }

    // Compute MI and fill in the lookup table.
    // MI = sum_i sum_j p_ij log(p_ij / (p_i * p_j))
    var MI: f32 = 0.0;
    for (var i: u32 = 0u; i < numBins; i = i + 1u) {
        for (var j: u32 = 0u; j < numBins; j = j + 1u) {
            let idx: u32 = i * numBins + j;
            let p_ij: f32 = bitcast<f32>(histogram[idx]) / total;
            if (p_ij > 0.0 && p_target[i] > 0.0 && p_moving[j] > 0.0) {
                let L: f32 = log(p_ij) - log(p_target[i]) - log(p_moving[j]);
                miLookup[idx] = L;
                MI = MI + p_ij * L;
            } else {
                miLookup[idx] = 0.0;
            }
        }
    }
    miResult = MI;
    totalMass = total;
}
