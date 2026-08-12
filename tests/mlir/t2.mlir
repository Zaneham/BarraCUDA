#map = affine_map<(d0, d1) -> (d0, d1)>

module {
    func.func @avg(%arg0: memref<10x20xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
        %alloc = memref.alloc() : memref<10x20xf32>
        %c0 = arith.constant 0 : index
        %c10 = arith.constant 10 : index
        %c1 = arith.constant 1 : index
        
        scf.for %arg2 = %c0 to %c10 step %c1 {
            %c0_0 = arith.constant 0 : index
            %c20 = arith.constant 20 : index
            %c1_1 = arith.constant 1 : index
            
            scf.for %arg3 = %c0_0 to %c20 step %c1_1 {
                %0 = memref.load %arg0[%arg2, %arg3] : memref<10x20xf32>
                %1 = memref.load %arg1[%arg2, %arg3] : memref<10x20xf32>
                %2 = arith.addf %0, %1 : f32
                memref.store %2, %alloc[%arg2, %arg3] : memref<10x20xf32>
            }
        }
        
        return %alloc : memref<10x20xf32>
    }
}
