fn atomicAddF32InGlobalMemory(sum: ptr<storage, atomic<u32>, read_write>, value: f32) -> f32 {
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

fn atomicAddF32InWorkgroupMemory(sum: ptr<workgroup, atomic<u32>>, value: f32) -> f32 {
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
