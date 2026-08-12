module {
  func.func @tensor_i16(%arg0: tensor<10xi16>) -> i16 {
    %c0_i16 = arith.constant 0 : i16
    return %c0_i16 : i16
  }
  func.func @func1() -> f32 {
    %idx0 = index.constant 0
    %idx1 = index.constant 1
    %idx2 = index.constant 2
    %idx5 = index.constant 5
    %true = arith.constant true
    %c0_i16 = arith.constant 0 : i16
    %c1_i16 = arith.constant 1 : i16
        
    // %49 can not be replaced with %c0_i16
    %alloc_76 = memref.alloc() : memref<24xi16>
    linalg.fill ins(%c0_i16 : i16) outs(%alloc_76 : memref<24xi16>)
    %49 = affine.load %alloc_76[%idx0] : memref<24xi16>
    %from_elements_117 = tensor.from_elements %49, %49, %49, %49, %49, %49, %49, %49, %49, %49 : tensor<5x2xi16>
      
    // scf.if can not be removed
    %171 = scf.if %true -> (tensor<5x2xi16>) {
      scf.yield %from_elements_117 : tensor<5x2xi16>
    } else {
      scf.yield %from_elements_117 : tensor<5x2xi16>
    }
    
    %cast_258 = tensor.splat %c1_i16[%idx5, %idx2] : tensor<?x?xi16>
    %212 = linalg.copy ins(%cast_258 : tensor<?x?xi16>) outs(%from_elements_117 : tensor<5x2xi16>) -> tensor<5x2xi16>
    
    %collapsed_217 = tensor.collapse_shape %171 [[0, 1]] : tensor<5x2xi16> into tensor<10xi16>
    %dim = index.constant 10
    %0 = scf.for %arg1 = %idx0 to %dim step %idx1 iter_args(%arg2 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %collapsed_217[%arg1] : tensor<10xi16>
      %1 = arith.addi %arg2, %extracted : i16
      vector.print %extracted : i16
      scf.yield %1 : i16
    }

    // can not be removed
    %collapsed_213 = tensor.collapse_shape %from_elements_117 [[0, 1]] : tensor<5x2xi16> into tensor<10xi16>
    %collapsed_222 = tensor.collapse_shape %212 [[0, 1]] : tensor<5x2xi16> into tensor<10xi16>
    %490 = call @tensor_i16(%collapsed_222) : (tensor<10xi16>) -> i16

    %1 = arith.sitofp %0 : i16 to f32
    return %1 : f32
  }
}
