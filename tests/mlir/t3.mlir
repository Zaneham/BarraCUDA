#map = affine_map<(d0, d1) -> (d0, d1)>

module {
    func.func @avg(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
        %alloc = memref.alloc() : memref<10x20xf32>
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %0 = arith.subi %c10, %c0 : index
        %c1 = arith.constant 1 : index
        %c0_0 = arith.constant 0 : index
        %c20 = arith.constant 20 : index
        %1 = arith.subi %c20, %c0_0 : index
        %c1_1 = arith.constant 1 : index
        %c1_2 = arith.constant 1 : index
        
        gpu.launch blocks(%arg2, %arg3, %arg4) in (%arg8 = %0, %arg9 = %c1_2, %arg10 = %c1_2) threads(%arg5, %arg6, %arg7) in (%arg11 = %1, %arg12 = %c1_2, %arg13 = %c1_2) {
            %2 = arith.addi %c0, %arg2 : index
            %3 = arith.addi %c0_0, %arg5 : index
            %4 = memref.load %arg0[%2, %3] : memref<10x20xf32>
            %5 = memref.load %arg1[%2, %3] : memref<10x20xf32>
            %6 = arith.addf %4, %5 : f32
            memref.store %4, %alloc[%2, %3] : memref<10x20xf32>
            gpu.terminator
        }
        
        return %alloc : memref<10x20xf32>
    }
}
