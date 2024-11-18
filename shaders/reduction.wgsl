@group(0) @binding(0) var<storage, read> inputArray: array<i32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<i32>;

const workgroupSize = 256;
var<workgroup> localSums: array<i32, workgroupSize>;

// This shader computes the sum of all elements in the input array.
@compute @workgroup_size(workgroupSize, 1, 1)
fn main(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
) {
    let inputSize = arrayLength(&inputArray);

    if (globalId.x < inputSize) {
        localSums[localId.x] = inputArray[globalId.x];
    } else {
        localSums[localId.x] = 0;
    }
    workgroupBarrier();

    // Perform tree based reduction
    var pairOffset = workgroupSize / 2u;
    while (pairOffset > 0u) {
        if (localId.x < pairOffset) {
            localSums[localId.x] += localSums[localId.x + pairOffset];
        }
        workgroupBarrier();
        pairOffset = pairOffset / 2u;
    }

    if (localId.x == 0u) {
        atomicAdd(&result, localSums[0]);
    }
}
