func.func @main(%x: tensor<64xf32>) -> tensor<64xf32> {
  %y = stablehlo.add %x, %x : tensor<64xf32>
  return %y : tensor<64xf32>
}
