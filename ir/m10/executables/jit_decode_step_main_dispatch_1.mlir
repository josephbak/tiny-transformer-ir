hal.executable public @main_dispatch_1 {
  hal.executable.variant public @embedded_elf_arm_64 target(<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>) {
    hal.executable.export public @main_dispatch_1_reduction_64_f32 ordinal(0) layout(#hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_1_reduction_64_f32() {
        %cst = arith.constant 6.400000e+01 : f32
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = arith.index_castui %0 : i32 to index
        %3 = arith.index_castui %1 : i32 to index
        %4:2 = util.assume.int 
            %2[<umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>], 
            %3[<umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>]
          : index, index
        %5 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%4#0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %6 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%4#1) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %8 = tensor.empty() : tensor<64xf32>
        %9 = tensor.empty() : tensor<f32>
        %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<f32>) -> tensor<f32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%7 : tensor<64xf32>) outs(%10 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %13 = arith.addf %out, %in : f32
          linalg.yield %13 : f32
        } -> tensor<f32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %11 : tensor<64xf32>, tensor<f32>) outs(%8 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %13 = arith.divf %in_1, %cst : f32
          %14 = arith.subf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %6, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
}
