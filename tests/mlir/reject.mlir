module {
  func.func @f(%a: f32, %b: f32) -> f32 {
    %0 = arith.addf %a, %b : f32
    %1 = math.exp %0 : f32
    return %1 : f32
  }
}
