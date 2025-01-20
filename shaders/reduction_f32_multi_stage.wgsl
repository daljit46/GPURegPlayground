// NOTE: This shader requires that:
//   1. The number of threads launched in total (workgroup_size.x * num_workgroups)
//      times unit_size equals the inputArray length. i.e. a multiple of
//      (workgroup_size.x * unit_size)
//   2. The workgroup size in X is a power of two
//   3. partialSums has length >= the number of workgroups
//
// This kernel performs a partial sum of the input array, outputting one sum
// per workgroup in partialSums[].

const wgSize = vec3u({{workgroup_size}});
const unitSize = {{unit_size}};

@group(0) @binding(0) var<storage, read> inputArray: array<f32>;
// Must have at least as many elements as the number of workgroups.
@group(0) @binding(1) var<storage, read_write> partialSums: array<f32>;

// Shared memory: one slice of size unitSize per thread.
var<workgroup> localSums: array<f32, wgSize.x * unitSize>;


fn reduceLocalSums(index: u32, offset: u32) {
    // Each thread index in the local array is (localId.x * unitSize).
    if (index < offset) {
        let dstBase = index * unitSize;
        let srcBase = (index + offset) * unitSize;
        for (var i = 0u; i < unitSize; i += 1) {
            localSums[dstBase + i] += localSums[srcBase + i];
        }
    }
}

@compute @workgroup_size(wgSize.x, 1, 1)
fn main(
    @builtin(global_invocation_id) globalId: vec3<u32>,
    @builtin(local_invocation_id) localId: vec3<u32>,
    @builtin(workgroup_id) workgroupId: vec3<u32>,
)
{
    let baseGlobalIndex = globalId.x * unitSize;
    let baseLocalIndex  = localId.x * unitSize;
    let inputSize = arrayLength(&inputArray);
    // Load 'unitSize' elements from global memory into local workgroup memory.
    // TODO: we could require that the input size is a multiple of the workgroup size
    // and avoid this conditional.
    if (baseGlobalIndex + unitSize <= inputSize) {
        for (var i = 0u; i < unitSize; i += 1) {
            localSums[baseLocalIndex + i] = inputArray[baseGlobalIndex + i];
        }
    } else {
        for (var i = 0u; i < unitSize; i += 1) {
            localSums[baseLocalIndex + i] = 0.0;
        }
    }
    workgroupBarrier();

    // Perform tree-based parallel reduction.

    if (wgSize.x >= 1024u) { reduceLocalSums(localId.x, 512u); workgroupBarrier(); }
    if (wgSize.x >= 512u)  { reduceLocalSums(localId.x, 256u); workgroupBarrier(); }
    if (wgSize.x >= 256u)  { reduceLocalSums(localId.x, 128u); workgroupBarrier(); }
    if (wgSize.x >= 128u)  { reduceLocalSums(localId.x, 64u); workgroupBarrier(); }
    if (wgSize.x >= 64u)   { reduceLocalSums(localId.x, 32u); }
    if (wgSize.x >= 32u)   { reduceLocalSums(localId.x, 16u); }
    if (wgSize.x >= 16u)   { reduceLocalSums(localId.x, 8u); }
    if (wgSize.x >= 8u)    { reduceLocalSums(localId.x, 4u); }
    if (wgSize.x >= 4u)    { reduceLocalSums(localId.x, 2u); }
    if (wgSize.x >= 2u)    { reduceLocalSums(localId.x, 1u); }

    if (localId.x == 0u) {
        for (var i = 0u; i < unitSize; i += 1) {
            partialSums[workgroupId.x * unitSize + i] = localSums[i];
        }
    }
}
