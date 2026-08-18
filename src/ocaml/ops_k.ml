(* Exercises the arithmetic the finance kernel will need: 2D indexing by
   div and mod, an int turned into a float, and the transcendentals. *)

open Kernel

let ops (a : f32 garray) (b : f32 garray) (w : i32) (n : i32) =
  let t = block_id () * block_dim () + thread_id () in
  if t < n then begin
    let row = t / w in
    let col = t mod w in
    let scale = to_f32 (row lxor col) in
    let x = get a t in
    let y = sqrtf (fabsf x) +. exp2f (x /. scale) -. log2f (fmaxf x (float 1.)) in
    set b t (fminf y (float 1e6))
  end
