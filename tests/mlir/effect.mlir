module {
  func.func private @main() -> i64 {
    %c0_i64 = arith.constant 0 : i64
    %c0_i64_0 = arith.constant 0 : i64
    %c0_i64_1 = arith.constant 0 : i64
    %c0_i64_2 = arith.constant 0 : i64
    %0 = arith.cmpi ne, %c0_i64_1, %c0_i64_2 : i64
    %false = arith.constant false
    cf.cond_br %0, ^bb1, ^bb2(%false, %c0_i64_0 : i1, i64)
  ^bb1:  // pred: ^bb0
    %1 = call @effect() : () -> i64
    %c0_i64_3 = arith.constant 0 : i64
    %2 = arith.cmpi ne, %1, %c0_i64_3 : i64
    cf.br ^bb2(%2, %1 : i1, i64)
  ^bb2(%3: i1, %4: i64):  // 2 preds: ^bb0, ^bb1
    %5 = arith.extui %3 : i1 to i64
    %c0_i64_4 = arith.constant 0 : i64
    %6 = arith.cmpi ne, %5, %c0_i64_4 : i64
    cf.cond_br %6, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %c1_i64 = arith.constant 1 : i64
    return %c1_i64 : i64
  ^bb4:  // pred: ^bb2
    cf.br ^bb5
  ^bb5:  // pred: ^bb4
    %c0_i64_5 = arith.constant 0 : i64
    %7 = arith.cmpi ne, %4, %c0_i64_5 : i64
    cf.cond_br %7, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %c2_i64 = arith.constant 2 : i64
    return %c2_i64 : i64
  ^bb7:  // pred: ^bb5
    cf.br ^bb8
  ^bb8:  // pred: ^bb7
    %c1_i64_6 = arith.constant 1 : i64
    %c0_i64_7 = arith.constant 0 : i64
    %8 = arith.cmpi ne, %c1_i64_6, %c0_i64_7 : i64
    %false_8 = arith.constant false
    cf.cond_br %8, ^bb9, ^bb10(%false_8, %4 : i1, i64)
  ^bb9:  // pred: ^bb8
    %9 = call @effect() : () -> i64
    %c0_i64_9 = arith.constant 0 : i64
    %10 = arith.cmpi ne, %9, %c0_i64_9 : i64
    cf.br ^bb10(%10, %9 : i1, i64)
  ^bb10(%11: i1, %12: i64):  // 2 preds: ^bb8, ^bb9
    %13 = arith.extui %11 : i1 to i64
    %c0_i64_10 = arith.constant 0 : i64
    %14 = arith.cmpi ne, %13, %c0_i64_10 : i64
    cf.cond_br %14, ^bb11, ^bb15
  ^bb11:  // pred: ^bb10
    %c1_i64_11 = arith.constant 1 : i64
    %15 = arith.cmpi ne, %12, %c1_i64_11 : i64
    %16 = arith.extui %15 : i1 to i64
    %c0_i64_12 = arith.constant 0 : i64
    %17 = arith.cmpi ne, %16, %c0_i64_12 : i64
    cf.cond_br %17, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %c3_i64 = arith.constant 3 : i64
    return %c3_i64 : i64
  ^bb13:  // pred: ^bb11
    cf.br ^bb14
  ^bb14:  // pred: ^bb13
    cf.br ^bb16
  ^bb15:  // pred: ^bb10
    %c4_i64 = arith.constant 4 : i64
    return %c4_i64 : i64
  ^bb16:  // pred: ^bb14
    %c0_i64_13 = arith.constant 0 : i64
    %c1_i64_14 = arith.constant 1 : i64
    %c0_i64_15 = arith.constant 0 : i64
    %18 = arith.cmpi ne, %c1_i64_14, %c0_i64_15 : i64
    %true = arith.constant true
    cf.cond_br %18, ^bb18(%true, %c0_i64_13 : i1, i64), ^bb17
  ^bb17:  // pred: ^bb16
    %19 = call @effect() : () -> i64
    %c0_i64_16 = arith.constant 0 : i64
    %20 = arith.cmpi ne, %19, %c0_i64_16 : i64
    cf.br ^bb18(%20, %19 : i1, i64)
  ^bb18(%21: i1, %22: i64):  // 2 preds: ^bb16, ^bb17
    %23 = arith.extui %21 : i1 to i64
    %c0_i64_17 = arith.constant 0 : i64
    %24 = arith.cmpi ne, %23, %c0_i64_17 : i64
    cf.cond_br %24, ^bb19, ^bb23
  ^bb19:  // pred: ^bb18
    %c0_i64_18 = arith.constant 0 : i64
    %25 = arith.cmpi ne, %22, %c0_i64_18 : i64
    cf.cond_br %25, ^bb20, ^bb21
  ^bb20:  // pred: ^bb19
    %c5_i64 = arith.constant 5 : i64
    return %c5_i64 : i64
  ^bb21:  // pred: ^bb19
    cf.br ^bb22
  ^bb22:  // pred: ^bb21
    cf.br ^bb24
  ^bb23:  // pred: ^bb18
    %c6_i64 = arith.constant 6 : i64
    return %c6_i64 : i64
  ^bb24:  // pred: ^bb22
    %c0_i64_19 = arith.constant 0 : i64
    %c0_i64_20 = arith.constant 0 : i64
    %26 = arith.cmpi ne, %c0_i64_19, %c0_i64_20 : i64
    %true_21 = arith.constant true
    cf.cond_br %26, ^bb26(%true_21, %22 : i1, i64), ^bb25
  ^bb25:  // pred: ^bb24
    %27 = call @effect() : () -> i64
    %c0_i64_22 = arith.constant 0 : i64
    %28 = arith.cmpi ne, %27, %c0_i64_22 : i64
    cf.br ^bb26(%28, %27 : i1, i64)
  ^bb26(%29: i1, %30: i64):  // 2 preds: ^bb24, ^bb25
    %31 = arith.extui %29 : i1 to i64
    %c0_i64_23 = arith.constant 0 : i64
    %32 = arith.cmpi ne, %31, %c0_i64_23 : i64
    cf.cond_br %32, ^bb27, ^bb31
  ^bb27:  // pred: ^bb26
    %c1_i64_24 = arith.constant 1 : i64
    %33 = arith.cmpi ne, %30, %c1_i64_24 : i64
    %34 = arith.extui %33 : i1 to i64
    %c0_i64_25 = arith.constant 0 : i64
    %35 = arith.cmpi ne, %34, %c0_i64_25 : i64
    cf.cond_br %35, ^bb28, ^bb29
  ^bb28:  // pred: ^bb27
    %c7_i64 = arith.constant 7 : i64
    return %c7_i64 : i64
  ^bb29:  // pred: ^bb27
    cf.br ^bb30
  ^bb30:  // pred: ^bb29
    cf.br ^bb32
  ^bb31:  // pred: ^bb26
    %c8_i64 = arith.constant 8 : i64
    return %c8_i64 : i64
  ^bb32:  // pred: ^bb30
    %c0_i64_26 = arith.constant 0 : i64
    return %c0_i64_26 : i64
  }
  func.func private @effect() -> i64 {
    %c1_i64 = arith.constant 1 : i64
    return %c1_i64 : i64
  }
}
