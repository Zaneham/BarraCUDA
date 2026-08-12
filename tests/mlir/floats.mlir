module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = arith.subf %0, %b : f32
    %2 = arith.mulf %1, %b : f32
    %3 = arith.divf %2, %b : f32
    return %3 : f32
  }
}
