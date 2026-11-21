hal.executable public @main_dispatch_12 {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @main_dispatch_12_matmul_5x64x64_f32 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_12_matmul_5x64x64_f32() {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c2560 = arith.constant 2560 : index
        %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c1280) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c2560) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %9 = linalg.matmul ins(%4, %7 : tensor<5x64xf32>, tensor<64x64xf32>) outs(%8 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %9 : tensor<5x64xf32>, tensor<5x64xf32>) outs(%6 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          linalg.yield %11 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
}
