module {
  tt.func public @kernel_with_label_0d1d2d3(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/data/users/oulgen/pytorch/test/dynamo/test_triton_kernels.py":1228:0), %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/data/users/oul
gen/pytorch/test/dynamo/test_triton_kernels.py":1228:0), %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} loc("/data/users/oulgen/pytorch/test/dynamo/test_triton_kernels.py":1228:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc2)
    %1 = arith.cmpi sgt, %0, %c1_i32 : i32 loc(#loc2)
    cf.cond_br %1, ^bb1, ^bb2 loc(#loc2)
  ^bb1:  // pred: ^bb0
    tt.return loc(#loc3)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3 loc(#loc4)
  ^bb3:  // pred: ^bb2
    %c4_i32 = arith.constant 4 : i32 loc(#loc5)
    %2 = arith.muli %0, %c4_i32 : i32 loc(#loc5)
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32> loc(#loc6)
    %4 = tt.splat %2 : (i32) -> tensor<4xi32> loc(#loc7)
    %5 = arith.addi %4, %3 : tensor<4xi32> loc(#loc7)
    %c4_i32_0 = arith.constant 4 : i32 loc(#loc8)
    %cst = arith.constant dense<4> : tensor<4xi32> loc(#loc8)
    %6 = arith.cmpi slt, %5, %cst : tensor<4xi32> loc(#loc8)
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<4x!tt.ptr<f32, 1>> loc(#loc9)
    %8 = tt.addptr %7, %5 : tensor<4x!tt.ptr<f32, 1>>, tensor<4xi32> loc(#loc9)
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4xf32> loc(#loc10)
    %10 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<4x!tt.ptr<f32, 1>> loc(#loc11)
    %11 = tt.addptr %10, %5 : tensor<4x!tt.ptr<f32, 1>>, tensor<4xi32> loc(#loc11)
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<4xf32> loc(#loc12)
    %13 = arith.addf %9, %12 : tensor<4xf32> loc(#loc13)
    %14 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<4x!tt.ptr<f32, 1>> loc(#loc14)
    %15 = tt.addptr %14, %5 : tensor<4x!tt.ptr<f32, 1>>, tensor<4xi32> loc(#loc14)
    tt.store %15, %13, %6 {cache = 1 : i32, evict = 1 : i32} : tensor<4xf32> loc(#loc15)
    tt.return loc(#loc16)
  } loc(#loc)
} loc(#loc)
