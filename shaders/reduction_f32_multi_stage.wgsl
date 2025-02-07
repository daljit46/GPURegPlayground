// NOTE: This shader requires that:
//   1. The number of threads launched in total (workgroup_size.x * num_workgroups)
//      times group_size equals the inputArray length. i.e. a multiple of
//      (workgroup_size.x * group_size)
//   2. The workgroup size in X is a power of two
//   3. partialSums has length >= the number of workgroups
//
// This kernel performs a partial sum of the input array, outputting one sum
// per workgroup in partialSums[].

const wgSize = vec3u({{workgroup_size}});
const groupSize = {{group_size}};
// Each item in a single unit will be subject to a given operation.
// e.g. for groupSize = 4, operations = [0, 1, 2, 0] would mean that the
// first item is summed, the second is min'd, the third is max'd, and the
// fourth is summed.
const operations = array<u32, {{group_size}}>(
    {{operations}}
);

@group(0) @binding(0) var<storage, read> inputArray: array<f32>;
// Must have at least as many elements as the number of workgroups.
@group(0) @binding(1) var<storage, read_write> partialSums: array<f32>;
// Shared memory: one slice of size groupSize per thread.
var<workgroup> localSums: array<f32, wgSize.x * groupSize>;

fn reductionOperation(a: f32, b: f32, operation: u32) -> f32 {
    switch (operation) {
        case 0: { return a + b; }
        case 1: { return min(a, b); }
        case 2: { return max(a, b); }
        default: { return 0.0; }
    }
}

fn reduceLocalSums(index: u32, offset: u32) {
    // Each thread index in the local array is (localId.x * groupSize).
    if (index < offset) {
        let dstBase = index * groupSize;
        let srcBase = (index + offset) * groupSize;
        for (var i = 0u; i < groupSize; i += 1) {
            let operation = operations[i];
            localSums[dstBase + i] = reductionOperation(localSums[dstBase + i], localSums[srcBase + i], operation);
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
    let baseGlobalIndex = globalId.x * groupSize;
    let baseLocalIndex  = localId.x * groupSize;
    let inputSize = arrayLength(&inputArray);
    // Load 'groupSize' elements from global memory into local workgroup memory.
    // TODO: we could require that the input size is a multiple of the workgroup size
    // and avoid this conditional.
    if (baseGlobalIndex + groupSize <= inputSize) {
        for (var i = 0u; i < groupSize; i += 1) {
            localSums[baseLocalIndex + i] = inputArray[baseGlobalIndex + i];
        }
    } else {
        for (var i = 0u; i < groupSize; i += 1) {
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
        for (var i = 0u; i < groupSize; i += 1) {
            partialSums[workgroupId.x * groupSize + i] = localSums[i];
        }
    }
}
