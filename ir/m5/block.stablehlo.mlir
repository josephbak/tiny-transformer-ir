module @jit_transformer_block attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>, %arg5: tensor<256xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64x256xf32>, %arg8: tensor<256x64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<64xf32>, %arg11: tensor<64xf32>, %arg12: tensor<64xf32>) -> (tensor<8x64xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_0 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %3 = stablehlo.divide %1, %2 : tensor<8x1xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %5 = stablehlo.subtract %arg0, %4 : tensor<8x64xf32>
    %6 = stablehlo.multiply %5, %5 : tensor<8x64xf32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %7 = stablehlo.reduce(%6 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_2 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %9 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %10 = stablehlo.divide %8, %9 : tensor<8x1xf32>
    %11 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %12 = stablehlo.subtract %arg0, %11 : tensor<8x64xf32>
    %cst_3 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %13 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %14 = stablehlo.add %10, %13 : tensor<8x1xf32>
    %15 = stablehlo.sqrt %14 : tensor<8x1xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %17 = stablehlo.divide %12, %16 : tensor<8x64xf32>
    %18 = stablehlo.broadcast_in_dim %arg10, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<8x64xf32>
    %20 = stablehlo.multiply %19, %17 : tensor<8x64xf32>
    %21 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<8x64xf32>
    %23 = stablehlo.add %20, %22 : tensor<8x64xf32>
    %24 = stablehlo.dot_general %23, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %25 = stablehlo.reshape %24 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %26 = stablehlo.dot_general %23, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %27 = stablehlo.reshape %26 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %28 = stablehlo.dot_general %23, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %29 = stablehlo.reshape %28 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %30 = stablehlo.transpose %27, dims = [1, 2, 0] : (tensor<8x4x16xf32>) -> tensor<4x16x8xf32>
    %31 = stablehlo.dot_general %25, %30, batching_dims = [1] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4x16xf32>, tensor<4x16x8xf32>) -> tensor<4x8x8xf32>
    %cst_4 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %32 = stablehlo.sqrt %cst_4 : tensor<f32>
    %33 = stablehlo.convert %32 : tensor<f32>
    %34 = stablehlo.broadcast_in_dim %33, dims = [] : (tensor<f32>) -> tensor<4x8x8xf32>
    %35 = stablehlo.divide %31, %34 : tensor<4x8x8xf32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %36 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<8x8xi1>
    %37 = call @tril(%36) : (tensor<8x8xi1>) -> tensor<8x8xi1>
    %cst_5 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %38 = call @_where(%37, %35, %cst_5) : (tensor<8x8xi1>, tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8x8xf32>
    %cst_6 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %39 = stablehlo.reduce(%38 init: %cst_6) applies stablehlo.maximum across dimensions = [2] : (tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8x1xf32>
    %41 = stablehlo.broadcast_in_dim %40, dims = [0, 1, 2] : (tensor<4x8x1xf32>) -> tensor<4x8x8xf32>
    %42 = stablehlo.subtract %38, %41 : tensor<4x8x8xf32>
    %43 = stablehlo.exponential %42 : tensor<4x8x8xf32>
    %cst_7 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%43 init: %cst_7) applies stablehlo.add across dimensions = [2] : (tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8x1xf32>
    %46 = stablehlo.broadcast_in_dim %45, dims = [0, 1, 2] : (tensor<4x8x1xf32>) -> tensor<4x8x8xf32>
    %47 = stablehlo.divide %43, %46 : tensor<4x8x8xf32>
    %48 = stablehlo.dot_general %47, %29, batching_dims = [0] x [1], contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x8x8xf32>, tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
    %49 = stablehlo.transpose %48, dims = [1, 0, 2] : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %50 = stablehlo.reshape %49 : (tensor<8x4x16xf32>) -> tensor<8x64xf32>
    %51 = stablehlo.dot_general %50, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %52 = stablehlo.add %arg0, %51 : tensor<8x64xf32>
    %cst_8 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %53 = stablehlo.reduce(%52 init: %cst_8) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %54 = stablehlo.broadcast_in_dim %53, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_9 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %55 = stablehlo.broadcast_in_dim %cst_9, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %56 = stablehlo.divide %54, %55 : tensor<8x1xf32>
    %57 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %58 = stablehlo.subtract %52, %57 : tensor<8x64xf32>
    %59 = stablehlo.multiply %58, %58 : tensor<8x64xf32>
    %cst_10 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %60 = stablehlo.reduce(%59 init: %cst_10) applies stablehlo.add across dimensions = [1] : (tensor<8x64xf32>, tensor<f32>) -> tensor<8xf32>
    %61 = stablehlo.broadcast_in_dim %60, dims = [0] : (tensor<8xf32>) -> tensor<8x1xf32>
    %cst_11 = stablehlo.constant dense<6.400000e+01> : tensor<f32>
    %62 = stablehlo.broadcast_in_dim %cst_11, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %63 = stablehlo.divide %61, %62 : tensor<8x1xf32>
    %64 = stablehlo.broadcast_in_dim %56, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %65 = stablehlo.subtract %52, %64 : tensor<8x64xf32>
    %cst_12 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %66 = stablehlo.broadcast_in_dim %cst_12, dims = [] : (tensor<f32>) -> tensor<8x1xf32>
    %67 = stablehlo.add %63, %66 : tensor<8x1xf32>
    %68 = stablehlo.sqrt %67 : tensor<8x1xf32>
    %69 = stablehlo.broadcast_in_dim %68, dims = [0, 1] : (tensor<8x1xf32>) -> tensor<8x64xf32>
    %70 = stablehlo.divide %65, %69 : tensor<8x64xf32>
    %71 = stablehlo.broadcast_in_dim %arg12, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %72 = stablehlo.broadcast_in_dim %71, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<8x64xf32>
    %73 = stablehlo.multiply %72, %70 : tensor<8x64xf32>
    %74 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %75 = stablehlo.broadcast_in_dim %74, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<8x64xf32>
    %76 = stablehlo.add %73, %75 : tensor<8x64xf32>
    %77 = stablehlo.dot_general %76, %arg7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x256xf32>) -> tensor<8x256xf32>
    %78 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %79 = stablehlo.broadcast_in_dim %78, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<8x256xf32>
    %80 = stablehlo.add %77, %79 : tensor<8x256xf32>
    %cst_13 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %81 = stablehlo.broadcast_in_dim %cst_13, dims = [] : (tensor<f32>) -> tensor<8x256xf32>
    %82 = stablehlo.maximum %81, %80 : tensor<8x256xf32>
    %83 = stablehlo.dot_general %82, %arg8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x256xf32>, tensor<256x64xf32>) -> tensor<8x64xf32>
    %84 = stablehlo.broadcast_in_dim %arg6, dims = [1] : (tensor<64xf32>) -> tensor<1x64xf32>
    %85 = stablehlo.broadcast_in_dim %84, dims = [0, 1] : (tensor<1x64xf32>) -> tensor<8x64xf32>
    %86 = stablehlo.add %83, %85 : tensor<8x64xf32>
    %87 = stablehlo.add %52, %86 : tensor<8x64xf32>
    return %87 : tensor<8x64xf32>
  }
  func.func private @tril(%arg0: tensor<8x8xi1>) -> tensor<8x8xi1> {
    %0 = stablehlo.iota dim = 0 : tensor<8x8xi32>
    %c = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<8x8xi32>
    %2 = stablehlo.add %0, %1 : tensor<8x8xi32>
    %3 = stablehlo.iota dim = 1 : tensor<8x8xi32>
    %4 = stablehlo.compare GE, %2, %3, SIGNED : (tensor<8x8xi32>, tensor<8x8xi32>) -> tensor<8x8xi1>
    %c_0 = stablehlo.constant dense<false> : tensor<i1>
    %5 = stablehlo.broadcast_in_dim %c_0, dims = [] : (tensor<i1>) -> tensor<8x8xi1>
    %6 = stablehlo.select %4, %arg0, %5 : tensor<8x8xi1>, tensor<8x8xi1>
    return %6 : tensor<8x8xi1>
  }
  func.func private @_where(%arg0: tensor<8x8xi1>, %arg1: tensor<4x8x8xf32>, %arg2: tensor<f32>) -> tensor<4x8x8xf32> {
    %0 = stablehlo.convert %arg2 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
    %2 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<8x8xi1>) -> tensor<4x8x8xi1>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1, 2] : (tensor<8x8xf32>) -> tensor<4x8x8xf32>
    %4 = stablehlo.select %2, %arg1, %3 : tensor<4x8x8xi1>, tensor<4x8x8xf32>
    return %4 : tensor<4x8x8xf32>
  }
}
