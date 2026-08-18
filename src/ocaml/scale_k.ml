(* Literals reach BIR as immediate operands, not instructions. *)

open Kernel

let scale (a : f32 garray) (b : f32 garray) (n : i32) =
  let i = block_id () * block_dim () + thread_id () in
  if i < n then set b i (get a i *. float 2.5 +. float 1.)
