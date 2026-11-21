module @jit_prefill attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>]> : !hal.device
  util.global private @__hoisted_tensor_5xi32 : !stream.resource<constant>
  util.initializer {
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %0 = stream.timepoint.immediate => !stream.timepoint
    %buffer_cst = util.buffer.constant <alignment = 64> : !util.buffer = #util.composite<64xi8, [
    dense<[0, 1, 2, 3, 4]> : tensor<5xi32>,
    dense<0> : vector<44xi8>,
]>
    %c64 = arith.constant 64 : index
    %did_map, %result = stream.resource.try_map on(#hal.device.affinity<@__device_0>) %buffer_cst[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c64}
    cf.cond_br %did_map, ^bb2(%0, %result : !stream.timepoint, !stream.resource<constant>), ^bb1
  ^bb1:  // pred: ^bb0
    %1 = stream.resource.alloc uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<constant>{%c64}
    %file = stream.file.constant on(#hal.device.affinity<@__device_0>) %buffer_cst[%c0 for %c64] : !util.buffer{%c64} -> !stream.file
    %2 = stream.file.read on(#hal.device.affinity<@__device_0>) await(%0) => %file[%c0_i64], %1[%c0], %c64 : !stream.file -> !stream.resource<constant>{%c64} => !stream.timepoint
    cf.br ^bb2(%2, %1 : !stream.timepoint, !stream.resource<constant>)
  ^bb2(%3: !stream.timepoint, %4: !stream.resource<constant>):  // 2 preds: ^bb0, ^bb1
    %5 = stream.timepoint.await sync %3 => %4 : !stream.resource<constant>{%c64}
    util.global.store %5, @__hoisted_tensor_5xi32 : !stream.resource<constant>
    util.return
  }
  stream.executable private @main_dispatch_0 {
    stream.executable.export public @main_dispatch_0_elementwise_broadcast_5x64_i32xi32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_0_elementwise_broadcast_5x64_i32xi32xf32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: !stream.binding {stream.alignment = 64 : index}) {
        %c0_i32 = arith.constant 0 : i32
        %c100_i32 = arith.constant 100 : i32
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<100x64xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xi32>>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xi32>>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>> -> tensor<32x64xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [100, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<100x64xf32>> -> tensor<100x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [5], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xi32>> -> tensor<5xi32>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [5], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5xi32>> -> tensor<5xi32>
        %9 = tensor.empty() : tensor<5x64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%7, %8 : tensor<5xi32>, tensor<5xi32>) outs(%9 : tensor<5x64xf32>) {
        ^bb0(%in: i32, %in_0: i32, %out: f32):
          %11 = arith.addi %in, %c100_i32 : i32
          %12 = arith.cmpi slt, %in, %c0_i32 : i32
          %13 = arith.select %12, %11, %in : i32
          %14 = linalg.index 1 : index
          %15 = arith.index_cast %in_0 : i32 to index
          %extracted = tensor.extract %5[%15, %14] : tensor<32x64xf32>
          %16 = arith.index_cast %13 : i32 to index
          %extracted_1 = tensor.extract %6[%16, %14] : tensor<100x64xf32>
          %17 = arith.addf %extracted_1, %extracted : f32
          linalg.yield %17 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_1 {
    stream.executable.export public @main_dispatch_1_reduction_5x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_1_reduction_5x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 0, umax = 0>, <umin = 2560, umax = 2560, udiv = 2560>, <umin = 0, umax = 0>, <umin = 2560, umax = 2560, udiv = 2560>, <umin = 0, umax = 0>, <umin = 2560, umax = 2560, udiv = 2560>], 
            %1[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>, <umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>, <umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>]
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg1[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = tensor.empty() : tensor<5xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<5xf32>) -> tensor<5xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%5 : tensor<5x64xf32>) outs(%8 : tensor<5xf32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.addf %out, %in : f32
          linalg.yield %11 : f32
        } -> tensor<5xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5, %9 : tensor<5x64xf32>, tensor<5xf32>) outs(%6 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %11 = arith.divf %in_1, %cst_0 : f32
          %12 = arith.subf %in, %11 : f32
          linalg.yield %12 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_2 {
    stream.executable.export public @main_dispatch_2_reduction_5x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_2_reduction_5x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>], 
            %1[<umin = 2560, umax = 2560, udiv = 2560>, <umin = 1280, umax = 1280, udiv = 1280>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %8 = tensor.empty() : tensor<5x64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<5xf32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<5xf32>) -> tensor<5xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<5x64xf32>) outs(%12 : tensor<5xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<5xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<5x64xf32>, tensor<5xf32>, tensor<64xf32>) outs(%8 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_3 {
    stream.executable.export public @main_dispatch_3_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_3_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c2560 = arith.constant 2560 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>, <umin = 0, umax = 0>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %9 = linalg.matmul ins(%5, %7 : tensor<5x64xf32>, tensor<64x64xf32>) outs(%8 : tensor<5x64xf32>) -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_6 {
    stream.executable.export public @main_dispatch_6_transpose_5x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_6_transpose_5x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %c1280 = arith.constant 1280 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x4x16xf32>>
        %1 = stream.binding.subspan %arg1[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x16xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [5, 4, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x4x16xf32>> -> tensor<5x4x16xf32>
        %3 = tensor.empty() : tensor<4x5x16xf32>
        %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<5x4x16xf32>) outs(%3 : tensor<4x5x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x5x16xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [4, 5, 16], strides = [1, 1, 1] : tensor<4x5x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x16xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_7 {
    stream.executable.export public @main_dispatch_7_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_7_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = util.assume.int %0[<umin = 5120, umax = 5120, udiv = 5120>, <umin = 6400, umax = 6400, udiv = 6400>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>>
        %3 = stream.binding.subspan %arg1[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        %4 = tensor.empty() : tensor<4x16x32xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 32, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>> -> tensor<32x4x16xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<32x4x16xf32>) outs(%4 : tensor<4x16x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x16x32xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %3, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : tensor<4x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_8 {
    stream.executable.export public @main_dispatch_8_batch_matmul_4x5x32x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_8_batch_matmul_4x5x32x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.000000e+00 : f32
        %cst_1 = arith.constant 0xFF800000 : f32
        %c3840 = arith.constant 3840 : index
        %c5120 = arith.constant 5120 : index
        %c1280 = arith.constant 1280 : index
        %0 = stream.binding.subspan %arg0[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x16xf32>>
        %1 = stream.binding.subspan %arg0[%c5120] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %2 = stream.binding.subspan %arg1[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x32xf32>>
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
  stream.executable private @main_dispatch_9 {
    stream.executable.export public @main_dispatch_9_softmax_4x5x32xf32_dispatch_tensor_store workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_9_softmax_4x5x32xf32_dispatch_tensor_store(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %c1280 = arith.constant 1280 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x32xf32>>
        %1 = stream.binding.subspan %arg1[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x32xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 5, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x32xf32>> -> tensor<4x5x32xf32>
        %3 = tensor.empty() : tensor<4x5x32xf32>
        %4 = linalg.softmax dimension(2) ins(%2 : tensor<4x5x32xf32>) outs(%3 : tensor<4x5x32xf32>) -> tensor<4x5x32xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [4, 5, 32], strides = [1, 1, 1] : tensor<4x5x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x5x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_11 {
    stream.executable.export public @main_dispatch_11_batch_matmul_4x16x5x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_11_batch_matmul_4x16x5x32_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c6400 = arith.constant 6400 : index
        %c3840 = arith.constant 3840 : index
        %c1280 = arith.constant 1280 : index
        %0 = stream.binding.subspan %arg0[%c6400] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %1 = stream.binding.subspan %arg0[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x32xf32>>
        %2 = stream.binding.subspan %arg1[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x4x16xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>> -> tensor<4x16x32xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 5, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x5x32xf32>> -> tensor<4x5x32xf32>
        %5 = tensor.empty() : tensor<5x4x16xf32>
        %6 = tensor.empty() : tensor<4x16x5xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%6 : tensor<4x16x5xf32>) -> tensor<4x16x5xf32>
        %8 = linalg.batch_matmul indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>] ins(%3, %4 : tensor<4x16x32xf32>, tensor<4x5x32xf32>) outs(%7 : tensor<4x16x5xf32>) -> tensor<4x16x5xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d2, d0, d1)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<4x16x5xf32>) outs(%5 : tensor<5x4x16xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<5x4x16xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %2, offsets = [0, 0, 0], sizes = [5, 4, 16], strides = [1, 1, 1] : tensor<5x4x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x4x16xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_12 {
    stream.executable.export public @main_dispatch_12_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_12_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c2560 = arith.constant 2560 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg2[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
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
  stream.executable private @main_dispatch_15 {
    stream.executable.export public @main_dispatch_15_matmul_5x256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_15_matmul_5x256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg3[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = tensor.empty() : tensor<5x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.matmul ins(%4, %8 : tensor<5x64xf32>, tensor<64x256xf32>) outs(%7 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<5x256xf32>, tensor<256xf32>) outs(%5 : tensor<5x256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<5x256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : tensor<5x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_16 {
    stream.executable.export public @main_dispatch_16_matmul_5x64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_16_matmul_5x64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c3840 = arith.constant 3840 : index
        %c2560 = arith.constant 2560 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %2 = stream.binding.subspan %arg2[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>>
        %3 = stream.binding.subspan %arg2[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>> -> tensor<5x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %7 = tensor.empty() : tensor<5x64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %11 = linalg.matmul ins(%5, %9 : tensor<5x256xf32>, tensor<256x64xf32>) outs(%10 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %11, %8 : tensor<5x64xf32>, tensor<5x64xf32>, tensor<64xf32>) outs(%7 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %13 = arith.addf %in_0, %in_1 : f32
          %14 = arith.addf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_18 {
    stream.executable.export public @main_dispatch_18_reduction_5x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_18_reduction_5x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>], 
            %1[<umin = 2560, umax = 2560, udiv = 2560>, <umin = 1280, umax = 1280, udiv = 1280>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %8 = tensor.empty() : tensor<5x64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<5xf32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<5xf32>) -> tensor<5xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<5x64xf32>) outs(%12 : tensor<5xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<5xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<5x64xf32>, tensor<5xf32>, tensor<64xf32>) outs(%8 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_19 {
    stream.executable.export public @main_dispatch_19_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_19_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c2560 = arith.constant 2560 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 8192, umax = 8192, udiv = 8192>, <umin = 32768, umax = 32768, udiv = 32768>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [1, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %9 = linalg.matmul ins(%5, %7 : tensor<5x64xf32>, tensor<64x64xf32>) outs(%8 : tensor<5x64xf32>) -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_23 {
    stream.executable.export public @main_dispatch_23_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_23_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 0, umax = 0>, <umin = 24576, umax = 24576, udiv = 24576>], 
            %1[<umin = 5120, umax = 5120, udiv = 5120>, <umin = 6400, umax = 6400, udiv = 6400>]
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>>
        %4 = stream.binding.subspan %arg1[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        %5 = tensor.empty() : tensor<4x16x32xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [1, 0, 0, 0], sizes = [1, 32, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>> -> tensor<32x4x16xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<32x4x16xf32>) outs(%5 : tensor<4x16x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x16x32xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %4, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : tensor<4x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_28 {
    stream.executable.export public @main_dispatch_28_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_28_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c2560 = arith.constant 2560 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg2[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
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
  stream.executable private @main_dispatch_31 {
    stream.executable.export public @main_dispatch_31_matmul_5x256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_31_matmul_5x256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg3[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = tensor.empty() : tensor<5x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [1, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.matmul ins(%4, %8 : tensor<5x64xf32>, tensor<64x256xf32>) outs(%7 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<5x256xf32>, tensor<256xf32>) outs(%5 : tensor<5x256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<5x256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : tensor<5x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_32 {
    stream.executable.export public @main_dispatch_32_matmul_5x64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_32_matmul_5x64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c3840 = arith.constant 3840 : index
        %c2560 = arith.constant 2560 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %2 = stream.binding.subspan %arg2[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>>
        %3 = stream.binding.subspan %arg2[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>> -> tensor<5x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %7 = tensor.empty() : tensor<5x64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [1, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %11 = linalg.matmul ins(%5, %9 : tensor<5x256xf32>, tensor<256x64xf32>) outs(%10 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%6, %11, %8 : tensor<5x64xf32>, tensor<5x64xf32>, tensor<64xf32>) outs(%7 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %13 = arith.addf %in_0, %in_1 : f32
          %14 = arith.addf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_34 {
    stream.executable.export public @main_dispatch_34_reduction_5x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_34_reduction_5x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 0, umax = 0>], 
            %1[<umin = 2560, umax = 2560, udiv = 2560>, <umin = 1280, umax = 1280, udiv = 1280>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %8 = tensor.empty() : tensor<5x64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<5xf32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<5xf32>) -> tensor<5xf32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%7 : tensor<5x64xf32>) outs(%12 : tensor<5xf32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<5xf32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<5x64xf32>, tensor<5xf32>, tensor<64xf32>) outs(%8 : tensor<5x64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_35 {
    stream.executable.export public @main_dispatch_35_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_35_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c2560 = arith.constant 2560 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 16384, umax = 16384, udiv = 16384>, <umin = 40960, umax = 40960, udiv = 40960>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [2, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %9 = linalg.matmul ins(%5, %7 : tensor<5x64xf32>, tensor<64x64xf32>) outs(%8 : tensor<5x64xf32>) -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_39 {
    stream.executable.export public @main_dispatch_39_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_39_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 0, umax = 0>, <umin = 24576, umax = 24576, udiv = 24576>], 
            %1[<umin = 5120, umax = 5120, udiv = 5120>, <umin = 6400, umax = 6400, udiv = 6400>]
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>>
        %4 = stream.binding.subspan %arg1[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        %5 = tensor.empty() : tensor<4x16x32xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [2, 0, 0, 0], sizes = [1, 32, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>> -> tensor<32x4x16xf32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6 : tensor<32x4x16xf32>) outs(%5 : tensor<4x16x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x16x32xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %4, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : tensor<4x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_44 {
    stream.executable.export public @main_dispatch_44_matmul_5x64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_44_matmul_5x64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c2560 = arith.constant 2560 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg2[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %6 = tensor.empty() : tensor<5x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
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
  stream.executable private @main_dispatch_47 {
    stream.executable.export public @main_dispatch_47_matmul_5x256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_47_matmul_5x256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c1280 = arith.constant 1280 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg3[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<5x64xf32>
        %5 = tensor.empty() : tensor<5x256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [2, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.matmul ins(%4, %8 : tensor<5x64xf32>, tensor<64x256xf32>) outs(%7 : tensor<5x256xf32>) -> tensor<5x256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %6 : tensor<5x256xf32>, tensor<256xf32>) outs(%5 : tensor<5x256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<5x256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : tensor<5x256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_48 {
    stream.executable.export public @main_dispatch_48_matmul_5x64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_48_matmul_5x64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c3840 = arith.constant 3840 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %1 = stream.binding.subspan %arg1[%c3840] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [5, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x256xf32>> -> tensor<5x256xf32>
        %4 = tensor.empty() : tensor<5x64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%4 : tensor<5x64xf32>) -> tensor<5x64xf32>
        %7 = linalg.matmul ins(%3, %5 : tensor<5x256xf32>, tensor<256x64xf32>) outs(%6 : tensor<5x64xf32>) -> tensor<5x64xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [5, 64], strides = [1, 1] : tensor<5x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<5x64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_49 {
    stream.executable.export public @main_dispatch_49_elementwise_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_49_elementwise_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %c0 = arith.constant 0 : index
        %c2560 = arith.constant 2560 : index
        %c1280 = arith.constant 1280 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %2 = stream.binding.subspan %arg1[%c2560] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>>
        %3 = stream.binding.subspan %arg2[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %4 = tensor.empty() : tensor<64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [4, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [4, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x64xf32>> -> tensor<64xf32>
        %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %6, %5 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%4 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %9 = arith.addf %in_0, %in_1 : f32
          %10 = arith.addf %in, %9 : f32
          linalg.yield %10 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %8, %3, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_50 {
    stream.executable.export public @main_dispatch_50_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_50_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %c1280 = arith.constant 1280 : index
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %3 = tensor.empty() : tensor<64xf32>
        %4 = tensor.empty() : tensor<f32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<f32>) -> tensor<f32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2 : tensor<64xf32>) outs(%5 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %8 = arith.addf %out, %in : f32
          linalg.yield %8 : f32
        } -> tensor<f32>
        %7 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2, %6 : tensor<64xf32>, tensor<f32>) outs(%3 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %8 = arith.divf %in_1, %cst_0 : f32
          %9 = arith.subf %in, %8 : f32
          linalg.yield %9 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %1, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_51 {
    stream.executable.export public @main_dispatch_51_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_51_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg3[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %7 = tensor.empty() : tensor<64xf32>
        %8 = tensor.empty() : tensor<f32>
        %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<f32>) -> tensor<f32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%4 : tensor<64xf32>) outs(%9 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %12 = arith.mulf %in, %in : f32
          %13 = arith.addf %out, %12 : f32
          linalg.yield %13 : f32
        } -> tensor<f32>
        %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %4, %10, %6 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %12 = arith.divf %in_3, %cst_0 : f32
          %13 = arith.addf %12, %cst_1 : f32
          %14 = math.sqrt %13 : f32
          %15 = arith.divf %in_2, %14 : f32
          %16 = arith.mulf %in, %15 : f32
          %17 = arith.addf %16, %in_4 : f32
          linalg.yield %17 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %11, %3, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_52 {
    stream.executable.export public @main_dispatch_52_vecmat_100x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_52_vecmat_100x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c49152 = arith.constant 49152 : index
        %0 = stream.binding.subspan %arg0[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x100xf32>>
        %2 = stream.binding.subspan %arg2[%c49152] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<100xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 100], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x100xf32>> -> tensor<64x100xf32>
        %5 = tensor.empty() : tensor<100xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<100xf32>) -> tensor<100xf32>
        %7 = linalg.vecmat ins(%3, %4 : tensor<64xf32>, tensor<64x100xf32>) outs(%6 : tensor<100xf32>) -> tensor<100xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0], sizes = [100], strides = [1] : tensor<100xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<100xf32>>
        return
      }
    }
  }
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<3x64x64xf32>, %input1: tensor<3x64x64xf32>, %input2: tensor<3x64x64xf32>, %input3: tensor<3x64x64xf32>, %input4: tensor<3x256xf32>, %input5: tensor<3x64xf32>, %input6: tensor<3x64x256xf32>, %input7: tensor<3x256x64xf32>, %input8: tensor<3x64xf32>, %input9: tensor<3x64xf32>, %input10: tensor<3x64xf32>, %input11: tensor<3x64xf32>, %input12: tensor<100x64xf32>, %input13: tensor<64xf32>, %input14: tensor<64xf32>, %input15: tensor<32x64xf32>, %input16: tensor<64x100xf32>, %input17: tensor<5xi32>, %input18: tensor<3x32x4x16xf32>, %input19: tensor<3x32x4x16xf32>) -> (%output0: tensor<1x100xf32>, %output1: tensor<3x32x4x16xf32>, %output2: tensor<3x32x4x16xf32>)"}} {
    %c40960_i32 = arith.constant 40960 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c24576_i32 = arith.constant 24576 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c6400_i32 = arith.constant 6400 : i32
    %c5120_i32 = arith.constant 5120 : i32
    %c2560_i32 = arith.constant 2560 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c0_i32 = arith.constant 0 : i32
    %c400 = arith.constant 400 : index
    %c24576 = arith.constant 24576 : index
    %c20 = arith.constant 20 : index
    %c8192 = arith.constant 8192 : index
    %c25600 = arith.constant 25600 : index
    %c196608 = arith.constant 196608 : index
    %c768 = arith.constant 768 : index
    %c3072 = arith.constant 3072 : index
    %c49152 = arith.constant 49152 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c49600 = arith.constant 49600 : index
    %c14592 = arith.constant 14592 : index
    %c0 = arith.constant 0 : index
    %__hoisted_tensor_5xi32 = util.global.load immutable @__hoisted_tensor_5xi32 : !stream.resource<constant>
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c3, %c64, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<3x64x64xf32> in !stream.resource<external>{%c49152}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c3, %c64, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %1 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<3x64x64xf32> in !stream.resource<external>{%c49152}
    hal.buffer_view.assert<%arg2 : !hal.buffer_view> message("input2") shape([%c3, %c64, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg2 : !hal.buffer_view -> tensor<3x64x64xf32> in !stream.resource<external>{%c49152}
    hal.buffer_view.assert<%arg3 : !hal.buffer_view> message("input3") shape([%c3, %c64, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %3 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg3 : !hal.buffer_view -> tensor<3x64x64xf32> in !stream.resource<external>{%c49152}
    hal.buffer_view.assert<%arg4 : !hal.buffer_view> message("input4") shape([%c3, %c256]) type(%element_type_f32) encoding(%dense_row_major)
    %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg4 : !hal.buffer_view -> tensor<3x256xf32> in !stream.resource<external>{%c3072}
    hal.buffer_view.assert<%arg5 : !hal.buffer_view> message("input5") shape([%c3, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %5 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg5 : !hal.buffer_view -> tensor<3x64xf32> in !stream.resource<external>{%c768}
    hal.buffer_view.assert<%arg6 : !hal.buffer_view> message("input6") shape([%c3, %c64, %c256]) type(%element_type_f32) encoding(%dense_row_major)
    %6 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg6 : !hal.buffer_view -> tensor<3x64x256xf32> in !stream.resource<external>{%c196608}
    hal.buffer_view.assert<%arg7 : !hal.buffer_view> message("input7") shape([%c3, %c256, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %7 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg7 : !hal.buffer_view -> tensor<3x256x64xf32> in !stream.resource<external>{%c196608}
    hal.buffer_view.assert<%arg8 : !hal.buffer_view> message("input8") shape([%c3, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %8 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg8 : !hal.buffer_view -> tensor<3x64xf32> in !stream.resource<external>{%c768}
    hal.buffer_view.assert<%arg9 : !hal.buffer_view> message("input9") shape([%c3, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %9 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg9 : !hal.buffer_view -> tensor<3x64xf32> in !stream.resource<external>{%c768}
    hal.buffer_view.assert<%arg10 : !hal.buffer_view> message("input10") shape([%c3, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %10 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg10 : !hal.buffer_view -> tensor<3x64xf32> in !stream.resource<external>{%c768}
    hal.buffer_view.assert<%arg11 : !hal.buffer_view> message("input11") shape([%c3, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %11 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg11 : !hal.buffer_view -> tensor<3x64xf32> in !stream.resource<external>{%c768}
    hal.buffer_view.assert<%arg12 : !hal.buffer_view> message("input12") shape([%c100, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %12 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg12 : !hal.buffer_view -> tensor<100x64xf32> in !stream.resource<external>{%c25600}
    hal.buffer_view.assert<%arg13 : !hal.buffer_view> message("input13") shape([%c64]) type(%element_type_f32) encoding(%dense_row_major)
    %13 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg13 : !hal.buffer_view -> tensor<64xf32> in !stream.resource<external>{%c256}
    hal.buffer_view.assert<%arg14 : !hal.buffer_view> message("input14") shape([%c64]) type(%element_type_f32) encoding(%dense_row_major)
    %14 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg14 : !hal.buffer_view -> tensor<64xf32> in !stream.resource<external>{%c256}
    hal.buffer_view.assert<%arg15 : !hal.buffer_view> message("input15") shape([%c32, %c64]) type(%element_type_f32) encoding(%dense_row_major)
    %15 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg15 : !hal.buffer_view -> tensor<32x64xf32> in !stream.resource<external>{%c8192}
    hal.buffer_view.assert<%arg16 : !hal.buffer_view> message("input16") shape([%c64, %c100]) type(%element_type_f32) encoding(%dense_row_major)
    %16 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg16 : !hal.buffer_view -> tensor<64x100xf32> in !stream.resource<external>{%c25600}
    %element_type_i32 = hal.element_type<i32> : i32
    hal.buffer_view.assert<%arg17 : !hal.buffer_view> message("input17") shape([%c5]) type(%element_type_i32) encoding(%dense_row_major)
    %17 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg17 : !hal.buffer_view -> tensor<5xi32> in !stream.resource<external>{%c20}
    hal.buffer_view.assert<%arg18 : !hal.buffer_view> message("input18") shape([%c3, %c32, %c4, %c16]) type(%element_type_f32) encoding(%dense_row_major)
    %18 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg18 : !hal.buffer_view -> tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576}
    hal.buffer_view.assert<%arg19 : !hal.buffer_view> message("input19") shape([%c3, %c32, %c4, %c16]) type(%element_type_f32) encoding(%dense_row_major)
    %19 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg19 : !hal.buffer_view -> tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c49600} => !stream.timepoint
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c14592} => !stream.timepoint
    %20 = stream.timepoint.join max(%result_timepoint, %result_timepoint_1) => !stream.timepoint
    %21 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%20) => with(%15 as %arg20: !stream.resource<external>{%c8192}, %12 as %arg21: !stream.resource<external>{%c25600}, %17 as %arg22: !stream.resource<external>{%c20}, %__hoisted_tensor_5xi32 as %arg23: !stream.resource<constant>{%c64}, %8 as %arg24: !stream.resource<external>{%c768}, %9 as %arg25: !stream.resource<external>{%c768}, %2 as %arg26: !stream.resource<external>{%c49152}, %0 as %arg27: !stream.resource<external>{%c49152}, %18 as %arg28: !stream.resource<external>{%c24576}, %3 as %arg29: !stream.resource<external>{%c49152}, %19 as %arg30: !stream.resource<external>{%c24576}, %1 as %arg31: !stream.resource<external>{%c49152}, %10 as %arg32: !stream.resource<external>{%c768}, %11 as %arg33: !stream.resource<external>{%c768}, %4 as %arg34: !stream.resource<external>{%c3072}, %6 as %arg35: !stream.resource<external>{%c196608}, %5 as %arg36: !stream.resource<external>{%c768}, %7 as %arg37: !stream.resource<external>{%c196608}, %14 as %arg38: !stream.resource<external>{%c256}, %13 as %arg39: !stream.resource<external>{%c256}, %16 as %arg40: !stream.resource<external>{%c25600}, %result as %arg41: !stream.resource<external>{%c49600}, %result_0 as %arg42: !stream.resource<transient>{%c14592}) {
      stream.cmd.dispatch @main_dispatch_0::@main_dispatch_0_elementwise_broadcast_5x64_i32xi32xf32 {
        ro %arg20[%c0 for %c8192] : !stream.resource<external>{%c8192},
        ro %arg21[%c0 for %c25600] : !stream.resource<external>{%c25600},
        ro %arg22[%c0 for %c20] : !stream.resource<external>{%c20},
        ro %arg23[%c0 for %c64] : !stream.resource<constant>{%c64},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_2::@main_dispatch_2_reduction_5x64_f32(%c1280_i32, %c2560_i32 : i32, i32) {
        ro %arg24[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_matmul_5x64x64_f32(%c1280_i32 : i32) {
          ro %arg26[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_matmul_5x64x64_f32(%c0_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg28[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_matmul_5x64x64_f32(%c0_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
        stream.cmd.dispatch @main_dispatch_6::@main_dispatch_6_transpose_5x4x16_f32 {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_7::@main_dispatch_7_transpose_32x4x16_f32(%c5120_i32 : i32) {
          ro %arg28[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_8::@main_dispatch_8_batch_matmul_4x5x32x16_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_9::@main_dispatch_9_softmax_4x5x32xf32_dispatch_tensor_store {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_7::@main_dispatch_7_transpose_32x4x16_f32(%c6400_i32 : i32) {
          ro %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_batch_matmul_4x16x5x32_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_12::@main_dispatch_12_matmul_5x64x64_f32 {
        ro %arg31[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c2560_i32, %c0_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_2::@main_dispatch_2_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg32[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_15::@main_dispatch_15_matmul_5x256x64_f32 {
        ro %arg34[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg35[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_16::@main_dispatch_16_matmul_5x64x256_f32 {
        ro %arg36[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg37[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_18::@main_dispatch_18_reduction_5x64_f32(%c1280_i32, %c2560_i32 : i32, i32) {
        ro %arg24[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_19::@main_dispatch_19_matmul_5x64x64_f32(%c1280_i32 : i32) {
          ro %arg26[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.copy %arg28[%c0], %arg41[%c0], %c24576 : !stream.resource<external>{%c24576} -> !stream.resource<external>{%c49600}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_19::@main_dispatch_19_matmul_5x64x64_f32(%c8192_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600}
        }
        stream.cmd.copy %arg30[%c0], %arg41[%c24576], %c24576 : !stream.resource<external>{%c24576} -> !stream.resource<external>{%c49600}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_19::@main_dispatch_19_matmul_5x64x64_f32(%c32768_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600}
        }
        stream.cmd.dispatch @main_dispatch_6::@main_dispatch_6_transpose_5x4x16_f32 {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_23::@main_dispatch_23_transpose_32x4x16_f32(%c0_i32, %c5120_i32 : i32, i32) {
          ro %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_8::@main_dispatch_8_batch_matmul_4x5x32x16_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_9::@main_dispatch_9_softmax_4x5x32xf32_dispatch_tensor_store {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_23::@main_dispatch_23_transpose_32x4x16_f32(%c24576_i32, %c6400_i32 : i32, i32) {
          ro %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_batch_matmul_4x16x5x32_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_28::@main_dispatch_28_matmul_5x64x64_f32 {
        ro %arg31[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c2560_i32, %c0_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_18::@main_dispatch_18_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg32[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_31::@main_dispatch_31_matmul_5x256x64_f32 {
        ro %arg34[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg35[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_32::@main_dispatch_32_matmul_5x64x256_f32 {
        ro %arg36[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg37[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_34::@main_dispatch_34_reduction_5x64_f32(%c1280_i32, %c2560_i32 : i32, i32) {
        ro %arg24[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_35::@main_dispatch_35_matmul_5x64x64_f32(%c1280_i32 : i32) {
          ro %arg26[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_35::@main_dispatch_35_matmul_5x64x64_f32(%c16384_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_35::@main_dispatch_35_matmul_5x64x64_f32(%c40960_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          rw %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600}
        }
        stream.cmd.dispatch @main_dispatch_6::@main_dispatch_6_transpose_5x4x16_f32 {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_39::@main_dispatch_39_transpose_32x4x16_f32(%c0_i32, %c5120_i32 : i32, i32) {
          ro %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_8::@main_dispatch_8_batch_matmul_4x5x32x16_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_9::@main_dispatch_9_softmax_4x5x32xf32_dispatch_tensor_store {
          ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
        stream.cmd.dispatch @main_dispatch_39::@main_dispatch_39_transpose_32x4x16_f32(%c24576_i32, %c6400_i32 : i32, i32) {
          ro %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600},
          wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
        }
      }
      stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_batch_matmul_4x16x5x32_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_44::@main_dispatch_44_matmul_5x64x64_f32 {
        ro %arg31[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_5x64_f32(%c2560_i32, %c0_i32 : i32, i32) {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_34::@main_dispatch_34_reduction_5x64_f32(%c0_i32, %c1280_i32 : i32, i32) {
        ro %arg32[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_47::@main_dispatch_47_matmul_5x256x64_f32 {
        ro %arg34[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg35[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_48::@main_dispatch_48_matmul_5x64x256_f32 {
        ro %arg37[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_49::@main_dispatch_49_elementwise_64_f32 {
        ro %arg36[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_50::@main_dispatch_50_reduction_64_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_51::@main_dispatch_51_reduction_64_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        ro %arg38[%c0 for %c256] : !stream.resource<external>{%c256},
        ro %arg39[%c0 for %c256] : !stream.resource<external>{%c256},
        wo %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592}
      }
      stream.cmd.dispatch @main_dispatch_52::@main_dispatch_52_vecmat_100x64_f32 {
        ro %arg42[%c0 for %c14592] : !stream.resource<transient>{%c14592},
        ro %arg40[%c0 for %c25600] : !stream.resource<external>{%c25600},
        wo %arg41[%c0 for %c49600] : !stream.resource<external>{%c49600}
      }
    } => !stream.timepoint
    %22 = stream.resource.dealloca on(#hal.device.affinity<@__device_0>) await(%21) => %result_0 : !stream.resource<transient>{%c14592} => !stream.timepoint
    %23 = stream.timepoint.await %22 => %result : !stream.resource<external>{%c49600}
    %24 = stream.resource.subview %23[%c49152] : !stream.resource<external>{%c49600} -> !stream.resource<external>{%c400}
    %25 = stream.resource.subview %23[%c0] : !stream.resource<external>{%c49600} -> !stream.resource<external>{%c24576}
    %26 = stream.resource.subview %23[%c24576] : !stream.resource<external>{%c49600} -> !stream.resource<external>{%c24576}
    %27 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %24 : tensor<1x100xf32> in !stream.resource<external>{%c400} -> !hal.buffer_view
    %28 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %25 : tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576} -> !hal.buffer_view
    %29 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %26 : tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576} -> !hal.buffer_view
    util.return %27, %28, %29 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}