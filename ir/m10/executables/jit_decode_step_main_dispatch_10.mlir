hal.executable public @main_dispatch_10 {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @main_dispatch_10_batch_matmul_4x1x32x16_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_10_batch_matmul_4x1x32x16_f32() {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.000000e+00 : f32
        %cst_1 = arith.constant 0xFF800000 : f32
        %c256 = arith.constant 256 : index
        %c1280 = arith.constant 1280 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c256) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x16xf32>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c1280) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
        %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c512) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x32xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 1, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x16xf32>> -> tensor<4x1x16xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>> -> tensor<4x16x32xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
        %7 = tensor.empty() : tensor<4x1x32xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<4x1x32xf32>) -> tensor<4x1x32xf32>
        %9 = linalg.batch_matmul ins(%4, %5 : tensor<4x1x16xf32>, tensor<4x16x32xf32>) outs(%8 : tensor<4x1x32xf32>) -> tensor<4x1x32xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %9 : tensor<i32>, tensor<4x1x32xf32>) outs(%7 : tensor<4x1x32xf32>) {
        ^bb0(%in: i32, %in_2: f32, %out: f32):
          %11 = arith.divf %in_2, %cst_0 : f32
          %12 = linalg.index 2 : index
          %13 = arith.index_cast %12 : index to i32
          %14 = arith.cmpi sle, %13, %in : i32
          %15 = arith.select %14, %11, %cst_1 : f32
          linalg.yield %15 : f32
        } -> tensor<4x1x32xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0, 0], sizes = [4, 1, 32], strides = [1, 1, 1] : tensor<4x1x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x32xf32>>
        return
      }
    }
  }
}
