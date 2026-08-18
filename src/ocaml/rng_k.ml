(* Device functions plus a kernel with a loop, which is the shape that caught
   mem2reg moving one function's body into the next. *)

open Kernel

let[@device] hash (x : i32) =
  let a = x lxor (x lsr int 16) in
  let b = a * int 0x85ebca6b in
  let c = b lxor (b lsr int 13) in
  let d = c * int 0xc2b2ae35 in
  d lxor (d lsr int 16)

let[@device] unif (x : i32) =
  to_f32 ((x lsr int 8) land int 8388607) /. float 8388608.

let[@device] expf (x : f32) = exp2f (x *. float 1.4426950408889634)

let[@device] normal (c : i32) =
  let u1 = unif (hash (c * int 2)) in
  let u2 = unif (hash (c * int 2 + int 1)) in
  let lu = log2f (fmaxf u1 (float 1e-7)) *. float 0.69314718 in
  sqrtf (float (-2.) *. lu) *. cosf (float 6.2831853 *. u2)

let draws (out : f32 garray) (n : i32) (m : i32) =
  let tid = block_id () * block_dim () + thread_id () in
  if tid < n then begin
    let acc = ref (float 0.) in
    loop (int 0) m (fun j ->
      let s = ref (float 1.) in
      loop (int 0) m (fun i ->
        s := !s *. expf (normal ((tid * m + j) * m + i) *. float 0.01));
      acc := !acc +. !s);
    set out tid (!acc /. to_f32 m)
  end
