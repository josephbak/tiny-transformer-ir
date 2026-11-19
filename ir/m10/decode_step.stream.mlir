module @jit_decode_step attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu = "apple-m4", cpu_features = "+v8.7a,+aes,+bf16,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+ras,+rcpc,+rdm,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+reserve-x18", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>, max_stack_allocation_size = 32768 : i64, native_vector_size = 16 : i64, target_triple = "arm64-unknown-unknown-eabi-elf"}>]> : !hal.device
  stream.executable private @main_dispatch_0 {
    stream.executable.export public @main_dispatch_0_elementwise_broadcast_64_i32xi32xf32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_0_elementwise_broadcast_64_i32xi32xf32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: !stream.binding {stream.alignment = 64 : index}) {
        %c0_i32 = arith.constant 0 : i32
        %c100_i32 = arith.constant 100 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<100x64xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
        %4 = stream.binding.subspan %arg4[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x64xf32>> -> tensor<32x64xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [100, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<100x64xf32>> -> tensor<100x64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
        %9 = tensor.empty() : tensor<64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%7, %8 : tensor<i32>, tensor<i32>) outs(%9 : tensor<64xf32>) {
        ^bb0(%in: i32, %in_0: i32, %out: f32):
          %11 = arith.addi %in, %c100_i32 : i32
          %12 = arith.addi %in_0, %c32_i32 : i32
          %13 = arith.cmpi slt, %in, %c0_i32 : i32
          %14 = arith.select %13, %11, %in : i32
          %15 = arith.cmpi slt, %in_0, %c0_i32 : i32
          %16 = arith.select %15, %12, %in_0 : i32
          %17 = linalg.index 0 : index
          %18 = arith.index_cast %16 : i32 to index
          %extracted = tensor.extract %5[%18, %17] : tensor<32x64xf32>
          %19 = arith.index_cast %14 : i32 to index
          %extracted_1 = tensor.extract %6[%19, %17] : tensor<100x64xf32>
          %20 = arith.addf %extracted_1, %extracted : f32
          linalg.yield %20 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_1 {
    stream.executable.export public @main_dispatch_1_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_1_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>, <umin = 512, umax = 512, udiv = 512>, <umin = 0, umax = 0>], 
            %1[<umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>, <umin = 256, umax = 256, udiv = 256>]
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg1[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = tensor.empty() : tensor<f32>
        %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<f32>) -> tensor<f32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%5 : tensor<64xf32>) outs(%8 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %11 = arith.addf %out, %in : f32
          linalg.yield %11 : f32
        } -> tensor<f32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %9 : tensor<64xf32>, tensor<f32>) outs(%6 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_1: f32, %out: f32):
          %11 = arith.divf %in_1, %cst_0 : f32
          %12 = arith.subf %in, %11 : f32
          linalg.yield %12 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_2 {
    stream.executable.export public @main_dispatch_2_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_2_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>], 
            %1[<umin = 512, umax = 512, udiv = 512>, <umin = 256, umax = 256, udiv = 256>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %8 = tensor.empty() : tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<f32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<f32>) -> tensor<f32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%7 : tensor<64xf32>) outs(%12 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<f32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>) outs(%8 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_3 {
    stream.executable.export public @main_dispatch_3_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_3_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 256, umax = 256, udiv = 256>, <umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%5, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_6 {
    stream.executable.export public @main_dispatch_6_elementwise workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_6_elementwise(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %c0_i32 = arith.constant 0 : i32
        %c32_i32 = arith.constant 32 : i32
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>> -> tensor<i32>
        %3 = tensor.empty() : tensor<i32>
        %4 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2 : tensor<i32>) outs(%3 : tensor<i32>) {
        ^bb0(%in: i32, %out: i32):
          %5 = arith.addi %in, %c32_i32 : i32
          %6 = arith.cmpi slt, %in, %c0_i32 : i32
          %7 = arith.select %6, %5, %in : i32
          linalg.yield %7 : i32
        } -> tensor<i32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<i32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<i32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_7 {
    stream.executable.export public @main_dispatch_7_slow_memcpy workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_7_slow_memcpy(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>], 
            %1<umin = 0, umax = 31>
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        %5 = iree_tensor_ext.dispatch.workload.ordinal %2#1, 0 : index
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [4, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>> -> tensor<4x16xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, %5, 0, 0], sizes = [1, 1, 4, 16], strides = [1, 1, 1, 1] : tensor<4x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_9 {
    stream.executable.export public @main_dispatch_9_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_9_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 1536, umax = 1536, udiv = 1536>] : index
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
  stream.executable private @main_dispatch_10 {
    stream.executable.export public @main_dispatch_10_batch_matmul_4x1x32x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_10_batch_matmul_4x1x32x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 4.000000e+00 : f32
        %cst_1 = arith.constant 0xFF800000 : f32
        %c256 = arith.constant 256 : index
        %c1280 = arith.constant 1280 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x16xf32>>
        %1 = stream.binding.subspan %arg0[%c1280] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<i32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x32xf32>>
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
  stream.executable private @main_dispatch_11 {
    stream.executable.export public @main_dispatch_11_softmax_4x1x32xf32_dispatch_tensor_store workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_11_softmax_4x1x32xf32_dispatch_tensor_store(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %c512 = arith.constant 512 : index
        %c1024 = arith.constant 1024 : index
        %0 = stream.binding.subspan %arg0[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x32xf32>>
        %1 = stream.binding.subspan %arg1[%c1024] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x32xf32>>
        %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 1, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x1x32xf32>> -> tensor<4x1x32xf32>
        %3 = tensor.empty() : tensor<4x1x32xf32>
        %4 = linalg.softmax dimension(2) ins(%2 : tensor<4x1x32xf32>) outs(%3 : tensor<4x1x32xf32>) -> tensor<4x1x32xf32>
        iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [4, 1, 32], strides = [1, 1, 1] : tensor<4x1x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x1x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_13 {
    stream.executable.export public @main_dispatch_13_batch_matmul_4x16x1x32_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_13_batch_matmul_4x16x1x32_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c1536 = arith.constant 1536 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %0 = stream.binding.subspan %arg0[%c1536] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>>
        %1 = stream.binding.subspan %arg0[%c1024] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x1xf32>>
        %2 = stream.binding.subspan %arg1[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x1xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16x32xf32>> -> tensor<4x16x32xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [4, 32, 1], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x32x1xf32>> -> tensor<4x32x1xf32>
        %5 = tensor.empty() : tensor<4x16x1xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<4x16x1xf32>) -> tensor<4x16x1xf32>
        %7 = linalg.batch_matmul ins(%3, %4 : tensor<4x16x32xf32>, tensor<4x32x1xf32>) outs(%6 : tensor<4x16x1xf32>) -> tensor<4x16x1xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0], sizes = [4, 16, 1], strides = [1, 1, 1] : tensor<4x16x1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x1xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_14 {
    stream.executable.export public @main_dispatch_14_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_14_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%4, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %9 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          linalg.yield %11 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_17 {
    stream.executable.export public @main_dispatch_17_vecmat_256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_17_vecmat_256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c768 = arith.constant 768 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg3[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = tensor.empty() : tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.vecmat ins(%4, %8 : tensor<64xf32>, tensor<64x256xf32>) outs(%7 : tensor<256xf32>) -> tensor<256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9, %6 : tensor<256xf32>, tensor<256xf32>) outs(%5 : tensor<256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [256], strides = [1] : tensor<256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_18 {
    stream.executable.export public @main_dispatch_18_vecmat_64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_18_vecmat_64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c768 = arith.constant 768 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %2 = stream.binding.subspan %arg2[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %7 = tensor.empty() : tensor<64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %11 = linalg.vecmat ins(%5, %9 : tensor<256xf32>, tensor<256x64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %11, %8 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %13 = arith.addf %in_0, %in_1 : f32
          %14 = arith.addf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_20 {
    stream.executable.export public @main_dispatch_20_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_20_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>], 
            %1[<umin = 512, umax = 512, udiv = 512>, <umin = 256, umax = 256, udiv = 256>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %8 = tensor.empty() : tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<f32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<f32>) -> tensor<f32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%7 : tensor<64xf32>) outs(%12 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<f32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>) outs(%8 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_21 {
    stream.executable.export public @main_dispatch_21_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_21_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 256, umax = 256, udiv = 256>, <umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [1, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%5, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_24 {
    stream.executable.export public @main_dispatch_24_slow_memcpy workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_24_slow_memcpy(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>], 
            %1<umin = 0, umax = 31>
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        %5 = iree_tensor_ext.dispatch.workload.ordinal %2#1, 0 : index
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [4, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>> -> tensor<4x16xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [1, %5, 0, 0], sizes = [1, 1, 4, 16], strides = [1, 1, 1, 1] : tensor<4x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_26 {
    stream.executable.export public @main_dispatch_26_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_26_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 1536, umax = 1536, udiv = 1536>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>>
        %3 = stream.binding.subspan %arg1[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        %4 = tensor.empty() : tensor<4x16x32xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [1, 0, 0, 0], sizes = [1, 32, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>> -> tensor<32x4x16xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<32x4x16xf32>) outs(%4 : tensor<4x16x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x16x32xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %3, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : tensor<4x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_31 {
    stream.executable.export public @main_dispatch_31_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_31_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%4, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %9 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          linalg.yield %11 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_34 {
    stream.executable.export public @main_dispatch_34_vecmat_256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_34_vecmat_256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c768 = arith.constant 768 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg3[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = tensor.empty() : tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [1, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.vecmat ins(%4, %8 : tensor<64xf32>, tensor<64x256xf32>) outs(%7 : tensor<256xf32>) -> tensor<256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9, %6 : tensor<256xf32>, tensor<256xf32>) outs(%5 : tensor<256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [256], strides = [1] : tensor<256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_35 {
    stream.executable.export public @main_dispatch_35_vecmat_64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_35_vecmat_64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c768 = arith.constant 768 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %2 = stream.binding.subspan %arg2[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %7 = tensor.empty() : tensor<64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [1, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [1, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %11 = linalg.vecmat ins(%5, %9 : tensor<256xf32>, tensor<256x64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %11, %8 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %13 = arith.addf %in_0, %in_1 : f32
          %14 = arith.addf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_37 {
    stream.executable.export public @main_dispatch_37_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_37_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}, %arg4: i32, %arg5: i32) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %0 = arith.index_castui %arg4 : i32 to index
        %1 = arith.index_castui %arg5 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 256, umax = 256, udiv = 256>, <umin = 0, umax = 0>], 
            %1[<umin = 512, umax = 512, udiv = 512>, <umin = 256, umax = 256, udiv = 256>]
          : index, index
        %3 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %5 = stream.binding.subspan %arg2[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %6 = stream.binding.subspan %arg3[%2#1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %8 = tensor.empty() : tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %10 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %11 = tensor.empty() : tensor<f32>
        %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<f32>) -> tensor<f32>
        %13 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%7 : tensor<64xf32>) outs(%12 : tensor<f32>) {
        ^bb0(%in: f32, %out: f32):
          %15 = arith.mulf %in, %in : f32
          %16 = arith.addf %out, %15 : f32
          linalg.yield %16 : f32
        } -> tensor<f32>
        %14 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%10, %7, %13, %9 : tensor<64xf32>, tensor<64xf32>, tensor<f32>, tensor<64xf32>) outs(%8 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_2: f32, %in_3: f32, %in_4: f32, %out: f32):
          %15 = arith.divf %in_3, %cst_0 : f32
          %16 = arith.addf %15, %cst_1 : f32
          %17 = math.sqrt %16 : f32
          %18 = arith.divf %in_2, %17 : f32
          %19 = arith.mulf %in, %18 : f32
          %20 = arith.addf %19, %in_4 : f32
          linalg.yield %20 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %14, %6, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_38 {
    stream.executable.export public @main_dispatch_38_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_38_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: i32) {
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = arith.index_castui %arg3 : i32 to index
        %1 = util.assume.int %0[<umin = 256, umax = 256, udiv = 256>, <umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %3 = stream.binding.subspan %arg1[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg2[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [2, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%5, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %9, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_41 {
    stream.executable.export public @main_dispatch_41_slow_memcpy workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%arg0)
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_41_slow_memcpy(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32, %arg3: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = arith.index_castui %arg3 : i32 to index
        %2:2 = util.assume.int 
            %0[<umin = 768, umax = 768, udiv = 768>, <umin = 1024, umax = 1024, udiv = 1024>], 
            %1<umin = 0, umax = 31>
          : index, index
        %3 = stream.binding.subspan %arg0[%2#0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>>
        %4 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        %5 = iree_tensor_ext.dispatch.workload.ordinal %2#1, 0 : index
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [4, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x16xf32>> -> tensor<4x16xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [2, %5, 0, 0], sizes = [1, 1, 4, 16], strides = [1, 1, 1, 1] : tensor<4x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<3x32x4x16xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_43 {
    stream.executable.export public @main_dispatch_43_transpose_32x4x16_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_43_transpose_32x4x16_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: i32) {
        %c0 = arith.constant 0 : index
        %0 = arith.index_castui %arg2 : i32 to index
        %1 = util.assume.int %0[<umin = 1280, umax = 1280, udiv = 1280>, <umin = 1536, umax = 1536, udiv = 1536>] : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>>
        %3 = stream.binding.subspan %arg1[%1] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        %4 = tensor.empty() : tensor<4x16x32xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [2, 0, 0, 0], sizes = [1, 32, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x32x4x16xf32>> -> tensor<32x4x16xf32>
        %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2, d0)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<32x4x16xf32>) outs(%4 : tensor<4x16x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<4x16x32xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %3, offsets = [0, 0, 0], sizes = [4, 16, 32], strides = [1, 1, 1] : tensor<4x16x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x16x32xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_48 {
    stream.executable.export public @main_dispatch_48_vecmat_64x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_48_vecmat_64x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>>
        %1 = stream.binding.subspan %arg1[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %6 = tensor.empty() : tensor<64xf32>
        %7 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0, 0], sizes = [1, 64, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x64xf32>> -> tensor<64x64xf32>
        %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<64xf32>) -> tensor<64xf32>
        %9 = linalg.vecmat ins(%4, %7 : tensor<64xf32>, tensor<64x64xf32>) outs(%8 : tensor<64xf32>) -> tensor<64xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5, %9 : tensor<64xf32>, tensor<64xf32>) outs(%6 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          linalg.yield %11 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_51 {
    stream.executable.export public @main_dispatch_51_vecmat_256x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_51_vecmat_256x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c768 = arith.constant 768 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>>
        %2 = stream.binding.subspan %arg2[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg3[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %5 = tensor.empty() : tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0], sizes = [1, 256], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256xf32>> -> tensor<256xf32>
        %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [2, 0, 0], sizes = [1, 64, 256], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64x256xf32>> -> tensor<64x256xf32>
        %9 = linalg.vecmat ins(%4, %8 : tensor<64xf32>, tensor<64x256xf32>) outs(%7 : tensor<256xf32>) -> tensor<256xf32>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9, %6 : tensor<256xf32>, tensor<256xf32>) outs(%5 : tensor<256xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %11 = arith.addf %in, %in_0 : f32
          %12 = arith.maximumf %11, %cst : f32
          linalg.yield %12 : f32
        } -> tensor<256xf32>
        iree_tensor_ext.dispatch.tensor.store %10, %3, offsets = [0], sizes = [256], strides = [1] : tensor<256xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_52 {
    stream.executable.export public @main_dispatch_52_vecmat_64x256_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_52_vecmat_64x256_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c768 = arith.constant 768 : index
        %c512 = arith.constant 512 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>>
        %2 = stream.binding.subspan %arg2[%c768] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>>
        %3 = stream.binding.subspan %arg2[%c512] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %4 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [256], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256xf32>> -> tensor<256xf32>
        %6 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0], sizes = [64], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
        %7 = tensor.empty() : tensor<64xf32>
        %8 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [2, 0], sizes = [1, 64], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x64xf32>> -> tensor<64xf32>
        %9 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [2, 0, 0], sizes = [1, 256, 64], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x256x64xf32>> -> tensor<256x64xf32>
        %10 = linalg.fill ins(%cst : f32) outs(%7 : tensor<64xf32>) -> tensor<64xf32>
        %11 = linalg.vecmat ins(%5, %9 : tensor<256xf32>, tensor<256x64xf32>) outs(%10 : tensor<64xf32>) -> tensor<64xf32>
        %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %11, %8 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%7 : tensor<64xf32>) {
        ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
          %13 = arith.addf %in_0, %in_1 : f32
          %14 = arith.addf %in, %13 : f32
          linalg.yield %14 : f32
        } -> tensor<64xf32>
        iree_tensor_ext.dispatch.tensor.store %12, %4, offsets = [0], sizes = [64], strides = [1] : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
        return
      }
    }
  }
  stream.executable private @main_dispatch_54 {
    stream.executable.export public @main_dispatch_54_reduction_64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_54_reduction_64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}, %arg3: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_0 = arith.constant 6.400000e+01 : f32
        %cst_1 = arith.constant 9.99999974E-6 : f32
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c256] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
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
  stream.executable private @main_dispatch_55 {
    stream.executable.export public @main_dispatch_55_vecmat_100x64_f32 workgroups() -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @main_dispatch_55_vecmat_100x64_f32(%arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x100xf32>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<100xf32>>
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
  util.func public @main(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.buffer_view, %arg4: !hal.buffer_view, %arg5: !hal.buffer_view, %arg6: !hal.buffer_view, %arg7: !hal.buffer_view, %arg8: !hal.buffer_view, %arg9: !hal.buffer_view, %arg10: !hal.buffer_view, %arg11: !hal.buffer_view, %arg12: !hal.buffer_view, %arg13: !hal.buffer_view, %arg14: !hal.buffer_view, %arg15: !hal.buffer_view, %arg16: !hal.buffer_view, %arg17: !hal.buffer_view, %arg18: !hal.buffer_view, %arg19: !hal.buffer_view, %arg20: !hal.buffer_view) -> (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @main(%input0: tensor<3x64x64xf32>, %input1: tensor<3x64x64xf32>, %input2: tensor<3x64x64xf32>, %input3: tensor<3x64x64xf32>, %input4: tensor<3x256xf32>, %input5: tensor<3x64xf32>, %input6: tensor<3x64x256xf32>, %input7: tensor<3x256x64xf32>, %input8: tensor<3x64xf32>, %input9: tensor<3x64xf32>, %input10: tensor<3x64xf32>, %input11: tensor<3x64xf32>, %input12: tensor<100x64xf32>, %input13: tensor<64xf32>, %input14: tensor<64xf32>, %input15: tensor<32x64xf32>, %input16: tensor<64x100xf32>, %input17: tensor<1xi32>, %input18: tensor<3x32x4x16xf32> {iree.abi.output = 1 : index}, %input19: tensor<3x32x4x16xf32> {iree.abi.output = 2 : index}, %input20: tensor<i32>) -> (%output0: tensor<1x100xf32>, %output1: tensor<3x32x4x16xf32>, %output2: tensor<3x32x4x16xf32>)"}} {
    %c1536_i32 = arith.constant 1536 : i32
    %c1280_i32 = arith.constant 1280 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c768_i32 = arith.constant 768 : i32
    %c512_i32 = arith.constant 512 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c9728 = arith.constant 9728 : index
    %c400 = arith.constant 400 : index
    %c24576 = arith.constant 24576 : index
    %c8192 = arith.constant 8192 : index
    %c25600 = arith.constant 25600 : index
    %c196608 = arith.constant 196608 : index
    %c768 = arith.constant 768 : index
    %c3072 = arith.constant 3072 : index
    %c49152 = arith.constant 49152 : index
    %c16 = arith.constant 16 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c100 = arith.constant 100 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c31 = arith.constant 31 : index
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
    hal.buffer_view.assert<%arg17 : !hal.buffer_view> message("input17") shape([%c1]) type(%element_type_i32) encoding(%dense_row_major)
    %17 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg17 : !hal.buffer_view -> tensor<1xi32> in !stream.resource<external>{%c4}
    hal.buffer_view.assert<%arg18 : !hal.buffer_view> message("input18") shape([%c3, %c32, %c4, %c16]) type(%element_type_f32) encoding(%dense_row_major)
    %18 = stream.tensor.import on(#hal.device.affinity<@__device_0>) consume %arg18 : !hal.buffer_view -> tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576}
    hal.buffer_view.assert<%arg19 : !hal.buffer_view> message("input19") shape([%c3, %c32, %c4, %c16]) type(%element_type_f32) encoding(%dense_row_major)
    %19 = stream.tensor.import on(#hal.device.affinity<@__device_0>) consume %arg19 : !hal.buffer_view -> tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576}
    hal.buffer_view.assert<%arg20 : !hal.buffer_view> message("input20") shape([]) type(%element_type_i32) encoding(%dense_row_major)
    %20 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg20 : !hal.buffer_view -> tensor<i32> in !stream.resource<external>{%c4}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<staging>{%c4} => !stream.timepoint
    %result_0, %result_timepoint_1 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c64} => !stream.timepoint
    %21 = stream.timepoint.join max(%result_timepoint, %result_timepoint_1) => !stream.timepoint
    %22 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%21) => with(%20 as %arg21: !stream.resource<external>{%c4}, %result as %arg22: !stream.resource<staging>{%c4}, %result_0 as %arg23: !stream.resource<transient>{%c64}) {
      stream.cmd.dispatch @main_dispatch_6::@main_dispatch_6_elementwise {
        ro %arg21[%c0 for %c4] : !stream.resource<external>{%c4},
        wo %arg23[%c0 for %c64] : !stream.resource<transient>{%c64}
      }
      stream.cmd.copy %arg23[%c0], %arg22[%c0], %c4 : !stream.resource<transient>{%c64} -> !stream.resource<staging>{%c4}
      stream.cmd.flush %arg22[%c0 for %c4] : !stream.resource<staging>{%c4}
    } => !stream.timepoint
    %23 = stream.resource.dealloca on(#hal.device.affinity<@__device_0>) await(%22) => %result_0 : !stream.resource<transient>{%c64} => !stream.timepoint
    %24 = stream.timepoint.await %23 => %result : !stream.resource<staging>{%c4}
    %25 = stream.resource.load %24[%c0] : !stream.resource<staging>{%c4} -> i32
    %26 = arith.index_cast %25 : i32 to index
    %27 = arith.maxsi %26, %c0 : index
    %28 = arith.minui %27, %c31 : index
    %result_2, %result_timepoint_3 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c400} => !stream.timepoint
    %result_4, %result_timepoint_5 = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<transient>{%c9728} => !stream.timepoint
    %29 = stream.timepoint.join max(%result_timepoint_3, %result_timepoint_5) => !stream.timepoint
    %30 = arith.index_castui %28 : index to i32
    %31 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%29) => with(%15 as %arg21: !stream.resource<external>{%c8192}, %12 as %arg22: !stream.resource<external>{%c25600}, %17 as %arg23: !stream.resource<external>{%c4}, %20 as %arg24: !stream.resource<external>{%c4}, %8 as %arg25: !stream.resource<external>{%c768}, %9 as %arg26: !stream.resource<external>{%c768}, %2 as %arg27: !stream.resource<external>{%c49152}, %0 as %arg28: !stream.resource<external>{%c49152}, %3 as %arg29: !stream.resource<external>{%c49152}, %18 as %arg30: !stream.resource<external>{%c24576}, %19 as %arg31: !stream.resource<external>{%c24576}, %1 as %arg32: !stream.resource<external>{%c49152}, %10 as %arg33: !stream.resource<external>{%c768}, %11 as %arg34: !stream.resource<external>{%c768}, %4 as %arg35: !stream.resource<external>{%c3072}, %6 as %arg36: !stream.resource<external>{%c196608}, %5 as %arg37: !stream.resource<external>{%c768}, %7 as %arg38: !stream.resource<external>{%c196608}, %14 as %arg39: !stream.resource<external>{%c256}, %13 as %arg40: !stream.resource<external>{%c256}, %16 as %arg41: !stream.resource<external>{%c25600}, %result_2 as %arg42: !stream.resource<external>{%c400}, %result_4 as %arg43: !stream.resource<transient>{%c9728}) {
      stream.cmd.dispatch @main_dispatch_0::@main_dispatch_0_elementwise_broadcast_64_i32xi32xf32 {
        ro %arg21[%c0 for %c8192] : !stream.resource<external>{%c8192},
        ro %arg22[%c0 for %c25600] : !stream.resource<external>{%c25600},
        ro %arg23[%c0 for %c4] : !stream.resource<external>{%c4},
        ro %arg24[%c0 for %c4] : !stream.resource<external>{%c4},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_2::@main_dispatch_2_reduction_64_f32(%c256_i32, %c512_i32 : i32, i32) {
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg26[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_vecmat_64x64_f32(%c256_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_vecmat_64x64_f32(%c768_i32 : i32) {
          ro %arg28[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_3::@main_dispatch_3_vecmat_64x64_f32(%c1024_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_7::@main_dispatch_7_slow_memcpy[%28](%c768_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_7::@main_dispatch_7_slow_memcpy[%28](%c1024_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
        stream.cmd.dispatch @main_dispatch_9::@main_dispatch_9_transpose_32x4x16_f32(%c1280_i32 : i32) {
          ro %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_10::@main_dispatch_10_batch_matmul_4x1x32x16_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        ro %arg24[%c0 for %c4] : !stream.resource<external>{%c4},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_softmax_4x1x32xf32_dispatch_tensor_store {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_9::@main_dispatch_9_transpose_32x4x16_f32(%c1536_i32 : i32) {
          ro %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_13::@main_dispatch_13_batch_matmul_4x16x1x32_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_14::@main_dispatch_14_vecmat_64x64_f32 {
        ro %arg32[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c512_i32, %c0_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_2::@main_dispatch_2_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg34[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_17::@main_dispatch_17_vecmat_256x64_f32 {
        ro %arg35[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg36[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_18::@main_dispatch_18_vecmat_64x256_f32 {
        ro %arg37[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg38[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_20::@main_dispatch_20_reduction_64_f32(%c256_i32, %c512_i32 : i32, i32) {
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg26[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_21::@main_dispatch_21_vecmat_64x64_f32(%c256_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_21::@main_dispatch_21_vecmat_64x64_f32(%c768_i32 : i32) {
          ro %arg28[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_21::@main_dispatch_21_vecmat_64x64_f32(%c1024_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_24::@main_dispatch_24_slow_memcpy[%28](%c768_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_24::@main_dispatch_24_slow_memcpy[%28](%c1024_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
        stream.cmd.dispatch @main_dispatch_26::@main_dispatch_26_transpose_32x4x16_f32(%c1280_i32 : i32) {
          ro %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_10::@main_dispatch_10_batch_matmul_4x1x32x16_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        ro %arg24[%c0 for %c4] : !stream.resource<external>{%c4},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_softmax_4x1x32xf32_dispatch_tensor_store {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_26::@main_dispatch_26_transpose_32x4x16_f32(%c1536_i32 : i32) {
          ro %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_13::@main_dispatch_13_batch_matmul_4x16x1x32_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_31::@main_dispatch_31_vecmat_64x64_f32 {
        ro %arg32[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c512_i32, %c0_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_20::@main_dispatch_20_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg34[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_34::@main_dispatch_34_vecmat_256x64_f32 {
        ro %arg35[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg36[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_35::@main_dispatch_35_vecmat_64x256_f32 {
        ro %arg37[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg38[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_37::@main_dispatch_37_reduction_64_f32(%c256_i32, %c512_i32 : i32, i32) {
        ro %arg25[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg26[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_38::@main_dispatch_38_vecmat_64x64_f32(%c256_i32 : i32) {
          ro %arg27[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_38::@main_dispatch_38_vecmat_64x64_f32(%c768_i32 : i32) {
          ro %arg28[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_38::@main_dispatch_38_vecmat_64x64_f32(%c1024_i32 : i32) {
          ro %arg29[%c0 for %c49152] : !stream.resource<external>{%c49152},
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_41::@main_dispatch_41_slow_memcpy[%28](%c768_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_41::@main_dispatch_41_slow_memcpy[%28](%c1024_i32, %30 : i32, i32) {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          rw %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576}
        }
        stream.cmd.dispatch @main_dispatch_43::@main_dispatch_43_transpose_32x4x16_f32(%c1280_i32 : i32) {
          ro %arg30[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_10::@main_dispatch_10_batch_matmul_4x1x32x16_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        ro %arg24[%c0 for %c4] : !stream.resource<external>{%c4},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.concurrent {
        stream.cmd.dispatch @main_dispatch_11::@main_dispatch_11_softmax_4x1x32xf32_dispatch_tensor_store {
          ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
        stream.cmd.dispatch @main_dispatch_43::@main_dispatch_43_transpose_32x4x16_f32(%c1536_i32 : i32) {
          ro %arg31[%c0 for %c24576] : !stream.resource<external>{%c24576},
          wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
        }
      }
      stream.cmd.dispatch @main_dispatch_13::@main_dispatch_13_batch_matmul_4x16x1x32_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_48::@main_dispatch_48_vecmat_64x64_f32 {
        ro %arg32[%c0 for %c49152] : !stream.resource<external>{%c49152},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c512_i32, %c0_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_37::@main_dispatch_37_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg33[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg34[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_51::@main_dispatch_51_vecmat_256x64_f32 {
        ro %arg35[%c0 for %c3072] : !stream.resource<external>{%c3072},
        ro %arg36[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_52::@main_dispatch_52_vecmat_64x256_f32 {
        ro %arg37[%c0 for %c768] : !stream.resource<external>{%c768},
        ro %arg38[%c0 for %c196608] : !stream.resource<external>{%c196608},
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_1::@main_dispatch_1_reduction_64_f32(%c0_i32, %c256_i32 : i32, i32) {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_54::@main_dispatch_54_reduction_64_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        ro %arg39[%c0 for %c256] : !stream.resource<external>{%c256},
        ro %arg40[%c0 for %c256] : !stream.resource<external>{%c256},
        wo %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728}
      }
      stream.cmd.dispatch @main_dispatch_55::@main_dispatch_55_vecmat_100x64_f32 {
        ro %arg43[%c0 for %c9728] : !stream.resource<transient>{%c9728},
        ro %arg41[%c0 for %c25600] : !stream.resource<external>{%c25600},
        wo %arg42[%c0 for %c400] : !stream.resource<external>{%c400}
      }
    } => !stream.timepoint
    %32 = stream.resource.dealloca on(#hal.device.affinity<@__device_0>) await(%31) => %result_4 : !stream.resource<transient>{%c9728} => !stream.timepoint
    %33:3 = stream.timepoint.await %32 => %result_2, %18, %19 : !stream.resource<external>{%c400}, !stream.resource<external>{%c24576}, !stream.resource<external>{%c24576}
    %34 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %33#0 : tensor<1x100xf32> in !stream.resource<external>{%c400} -> !hal.buffer_view
    %35 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %33#1 : tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576} -> !hal.buffer_view
    %36 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %33#2 : tensor<3x32x4x16xf32> in !stream.resource<external>{%c24576} -> !hal.buffer_view
    util.return %34, %35, %36 : !hal.buffer_view, !hal.buffer_view, !hal.buffer_view
  }
}