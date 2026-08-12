module {
  func.func @c() -> f32 {
    %0 = arith.constant 1 : f32
    %1 = arith.constant 2.5 : f32
    %2 = arith.mulf %0, %1 : f32
    return %2 : f32
  }
}
