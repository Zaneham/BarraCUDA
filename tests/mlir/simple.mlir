module {
  func.func private @putchar(i64) -> i64
  func.func private @main() -> i64 {
    %c0_i64 = arith.constant 0 : i64
    cf.br ^bb1(%c0_i64 : i64)
  ^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
    %c10_i64 = arith.constant 10 : i64
    %1 = arith.cmpi slt, %0, %c10_i64 : i64
    %2 = arith.extui %1 : i1 to i64
    %c0_i64_0 = arith.constant 0 : i64
    %3 = arith.cmpi ne, %2, %c0_i64_0 : i64
    cf.cond_br %3, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %c48_i64 = arith.constant 48 : i64
    %4 = arith.addi %0, %c48_i64 : i64
    %5 = call @putchar(%4) : (i64) -> i64
    %c1_i64 = arith.constant 1 : i64
    %6 = arith.addi %0, %c1_i64 : i64
    cf.br ^bb1(%6 : i64)
  ^bb3:  // pred: ^bb1
    return %0 : i64
  }
}
