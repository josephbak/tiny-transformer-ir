module @jit_multi_head_attention attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>, %arg4: tensor<64x64xf32>) -> (tensor<8x64xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %1 = stablehlo.reshape %0 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %2 = stablehlo.dot_general %arg0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %3 = stablehlo.reshape %2 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %4 = stablehlo.dot_general %arg0, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    %5 = stablehlo.reshape %4 : (tensor<8x64xf32>) -> tensor<8x4x16xf32>
    %6 = stablehlo.transpose %3, dims = [1, 2, 0] : (tensor<8x4x16xf32>) -> tensor<4x16x8xf32>
    %7 = stablehlo.dot_general %1, %6, batching_dims = [1] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x4x16xf32>, tensor<4x16x8xf32>) -> tensor<4x8x8xf32>
    %cst = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %8 = stablehlo.sqrt %cst : tensor<f32>
    %9 = stablehlo.convert %8 : tensor<f32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [] : (tensor<f32>) -> tensor<4x8x8xf32>
    %11 = stablehlo.divide %7, %10 : tensor<4x8x8xf32>
    %c = stablehlo.constant dense<true> : tensor<i1>
    %12 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<8x8xi1>
    %13 = call @tril(%12) : (tensor<8x8xi1>) -> tensor<8x8xi1>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %14 = call @_where(%13, %11, %cst_0) : (tensor<8x8xi1>, tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8x8xf32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %15 = stablehlo.reduce(%14 init: %cst_1) applies stablehlo.maximum across dimensions = [2] : (tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    %16 = stablehlo.broadcast_in_dim %15, dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8x1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<4x8x1xf32>) -> tensor<4x8x8xf32>
    %18 = stablehlo.subtract %14, %17 : tensor<4x8x8xf32>
    %19 = stablehlo.exponential %18 : tensor<4x8x8xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %20 = stablehlo.reduce(%19 init: %cst_2) applies stablehlo.add across dimensions = [2] : (tensor<4x8x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8x1xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1, 2] : (tensor<4x8x1xf32>) -> tensor<4x8x8xf32>
    %23 = stablehlo.divide %19, %22 : tensor<4x8x8xf32>
    %24 = stablehlo.dot_general %23, %5, batching_dims = [0] x [1], contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x8x8xf32>, tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
    %25 = stablehlo.transpose %24, dims = [1, 0, 2] : (tensor<4x8x16xf32>) -> tensor<8x4x16xf32>
    %26 = stablehlo.reshape %25 : (tensor<8x4x16xf32>) -> tensor<8x64xf32>
    %27 = stablehlo.dot_general %26, %arg4, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8x64xf32>, tensor<64x64xf32>) -> tensor<8x64xf32>
    return %27 : tensor<8x64xf32>
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
