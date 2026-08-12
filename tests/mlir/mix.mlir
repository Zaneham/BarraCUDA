module {
  func.func @mix(%a: i32, %b: i32) -> i32 {
    %c = arith.constant 7 : i32
    %0 = arith.addi %a, %b : i32
    %1 = arith.muli %0, %c : i32
    %2 = arith.subi %1, %a : i32
    %3 = arith.cmpi slt, %2, %c : i32
    %4 = arith.extui %3 : i1 to i32
    %5 = arith.xori %2, %4 : i32
    return %5 : i32
  }
}
