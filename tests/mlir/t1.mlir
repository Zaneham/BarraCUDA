module {

func.func @matrix_add(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
    %result = memref.alloc() : memref<10x20xf32>

	affine.for %i = 0 to 10 {
		affine.for %j = 0 to 20 {
			%lhs = memref.load %arg0[%i, %j] : memref<10x20xf32>
			%rhs = memref.load %arg1[%i, %j] : memref<10x20xf32>
			%sum = arith.addf %lhs, %rhs : f32
			memref.store %sum, %result[%i, %j] : memref<10x20xf32>
		}
	}
    
    func.return %result : memref<10x20xf32>
}

}
