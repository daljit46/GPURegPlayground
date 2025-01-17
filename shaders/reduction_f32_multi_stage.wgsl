// NOTE: This shader requires that the inputArray is a multiple of the workgroup size
// Additionally, it only supports 1D workgroup sizes that are powers of 2

const wgSize = vec3u({{workgroup_size}});
const unitSize = {{unit_size}};

@group(0) @binding(0) var<storage, read> inputArray: array<f32>;
@group(0) @binding(1) var<storage, read_write> partialSums: array<f32>;

var<workgroup> localSums: array<f32, wgSize.x * unitSize>;

fn reduceLocalSums(index: u32, offset: u32) {
    if (index < offset) {
        for(var i = 0u; i < unitSize; i += 1) {
            localSums[index + i] += localSums[index + offset + i];
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
    localSums[localId.x] = inputArray[globalId.x];
    workgroupBarrier();

    if(wgSize.x >= 1024u) { reduceLocalSums(localId.x, 512u); workgroupBarrier(); }
    if(wgSize.x >= 512u)  { reduceLocalSums(localId.x, 256u); workgroupBarrier(); }
    if(wgSize.x >= 256u)  { reduceLocalSums(localId.x, 128u); workgroupBarrier(); }
    if(wgSize.x >= 128u)  { reduceLocalSums(localId.x, 64u); workgroupBarrier();  }

    // No need for barriers as threads are in lockstep in a warp (size 32)
    if (localId.x < 32u) {
        if(wgSize.x >= 64u) { reduceLocalSums(localId.x, 32u); }
        if(wgSize.x >= 32u) { reduceLocalSums(localId.x, 16u); }
        if(wgSize.x >= 16u) { reduceLocalSums(localId.x, 8u);  }
        if(wgSize.x >= 8u)  { reduceLocalSums(localId.x, 4u);  }
        if(wgSize.x >= 4u)  { reduceLocalSums(localId.x, 2u);  }
        if(wgSize.x >= 2u)  { reduceLocalSums(localId.x, 1u);  }
    }

    if (localId.x == 0u) {
        let wgIndex = workgroupId.x;
        partialSums[wgIndex] = localSums[0];
    }
}
