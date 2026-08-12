module {
  tt.func public @load_reduce_kernel_0d1d2de3c4c(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %2 = arith.extsi %1 : tensor<128xi32> to tensor<128xi64>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<128xi64>) -> tensor<128x1xi64>
    %4 = tt.splat %0 : (i64) -> tensor<128x1xi64>
    %5 = arith.muli %3, %4 : tensor<128x1xi64>
    %6 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>>
    %7 = tt.addptr %6, %5 : tensor<128x1x!tt.ptr<f16, 1>>, tensor<128x1xi64>
    %8 = tt.broadcast %7 : (tensor<128x1x!tt.ptr<f16, 1>>) -> tensor<128x64x!tt.ptr<f16, 1>>
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %10 = arith.extsi %9 : tensor<64xi32> to tensor<64xi64>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : (tensor<64xi64>) -> tensor<1x64xi64>
    %12 = tt.broadcast %11 : (tensor<1x64xi64>) -> tensor<128x64xi64>
    %13 = tt.addptr %8, %12 : tensor<128x64x!tt.ptr<f16, 1>>, tensor<128x64xi64>
    %14 = tt.load %13 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x64xf16>
    %15 = arith.extf %14 : tensor<128x64xf16> to tensor<128x64xf32>
    %16 = "tt.reduce"(%15) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %20 = arith.maxf %arg3, %arg4 : f32
      tt.reduce.return %20 : f32
    }) : (tensor<128x64xf32>) -> tensor<128xf32>
    %17 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<128x!tt.ptr<f16, 1>>
    %18 = tt.addptr %17, %1 : tensor<128x!tt.ptr<f16, 1>>, tensor<128xi32>
    %19 = arith.truncf %16 : tensor<128xf32> to tensor<128xf16>
    tt.store %18, %19 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf16>
    tt.return
  }
}
