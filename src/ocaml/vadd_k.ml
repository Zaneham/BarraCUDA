(* vadd again, but as an ordinary OCaml function rather than a program that
   builds one. kcomp lowers this file's .cmt to the same BIR vadd.ml emits. *)

open Kernel

let vadd (a : f32 garray) (b : f32 garray) (c : f32 garray) (n : i32) =
  let i = block_id () * block_dim () + thread_id () in
  if i < n then set c i (get a i +. get b i)
