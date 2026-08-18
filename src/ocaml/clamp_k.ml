(* if/else with a float comparison. Ordered, so a NaN takes the else. *)

open Kernel

let clamp (a : f32 garray) (b : f32 garray) (n : i32) =
  let i = block_id () * block_dim () + thread_id () in
  if i < n then
    if get a i <. float 0. then set b i (float 0.) else set b i (get a i)
