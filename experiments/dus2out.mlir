func.func @main(%x: tensor<32xf32>, %y: tensor<4xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
  %c0 = stablehlo.constant dense<0> : tensor<i32>
  %r = stablehlo.dynamic_update_slice %x, %y, %c0 : (tensor<32xf32>, tensor<4xf32>, tensor<i32>) -> tensor<32xf32>
  return %r, %x : tensor<32xf32>, tensor<32xf32>
}
