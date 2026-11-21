hal.executable public @main_dispatch_8 {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @main_dispatch_8_batch_matmul_4x5x32x16_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_8_batch_matmul_4x5x32x16_f32() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.000000e+00 : f32
        %cst_1 = arith.constant 0xFF800000 : f32
        %c3840 = arith.constant 3840 : index
        %c5120 = arith.constant 5120 : index
        %c1280 = arith.constant 1280 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c3840) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x16xf32>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c5120) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c1280) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x32xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 5, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x16xf32>> -> tensor<4x5x16xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>> -> tensor<4x16x32xf32>
        %5 = tensor.empty() : tensor<4x5x32xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x5x32xf32>) -> tensor<4x5x32xf32>
        %7 = linalg.batch_matmul ins(%3, %4 : tensor<4x5x16xf32>, tensor<4x16x32xf32>) outs(%6 : tensor<4x5x32xf32>) -> tensor<4x5x32xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<4x5x32xf32>) outs(%5 : tensor<4x5x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          %9 = arith.divf %in, %cst_0 : f32
          %10 = linalg.index 1 : index
          %11 = arith.index_cast %10 : index to i32
          %12 = linalg.index 2 : index
          %13 = arith.index_cast %12 : index to i32
          %14 = arith.cmpi sle, %13, %11 : i32
          %15 = arith.select %14, %9, %cst_1 : f32
          linalg.yield %15 : f32
        } -> tensor<4x5x32xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0, 0], sizes = [4, 5, 32], strides = [1, 1, 1] : tensor<4x5x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x32xf32>>
        return
      }
    }
  }
}
