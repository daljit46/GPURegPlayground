@group(0) @binding(0) var<storage, read> inputArray: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: atomic<u32>;

const workgroupSize = 256;
var<workgroup> localSums: array<f32, workgroupSize>;

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
        atomicAddF32(&result, localSums[0]);
    }
}
