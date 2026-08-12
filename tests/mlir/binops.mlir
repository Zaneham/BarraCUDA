module {
  func.func @allbin(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    %1 = arith.subi %0, %b : i32
    %2 = arith.muli %1, %b : i32
    %3 = arith.divsi %2, %b : i32
    %4 = arith.divui %3, %b : i32
    %5 = arith.remsi %4, %b : i32
    %6 = arith.remui %5, %b : i32
    %7 = arith.andi %6, %b : i32
    %8 = arith.ori %7, %b : i32
    %9 = arith.xori %8, %b : i32
    %10 = arith.shli %9, %b : i32
    %11 = arith.shrui %10, %b : i32
    %12 = arith.shrsi %11, %b : i32
    return %12 : i32
  }
}
