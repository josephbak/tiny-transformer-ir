module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<32xf32>, %input1: tensor<4xf32>) -> (%output0: tensor<32xf32>, %output1: tensor<32xf32>)"}} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c32]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<32xf32> in !stream.resource<external>{%c128}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)
    %1 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) with(%0 as %arg2: !stream.resource<external>{%c128}, %1 as %arg3: !stream.resource<external>{%c16}) {
      stream.cmd.copy %arg3[%c0], %arg2[%c0], %c16 : !stream.resource<external>{%c16} -> !stream.resource<external>{%c128}
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %0 : !stream.resource<external>{%c128}
    %4 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %3 : tensor<32xf32> in !stream.resource<external>{%c128} -> !hal.buffer_view
    util.return %4, %arg0 : !hal.buffer_view, !hal.buffer_view
  }
}