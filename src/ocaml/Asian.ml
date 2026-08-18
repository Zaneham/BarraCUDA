(*A kernel in the OCaml language that hopefully, fingers crossed, compiles *)
(* This is also an exemplar of what a GPU kernel in OCaml looks like       *)

open Kernel

(* Device helpers.*)

let[@device] hash (x : i32) =
  let a = x lxor (x lsr int 16) in
  let b = a * int 0x85ebca6b in
  let c = b lxor (b lsr int 13) in
  let d = c * int 0xc2b2ae35 in
  d lxor (d lsr int 16)

let[@device] unif (x : i32) =
  to_f32 ((x lsr int 8) land int 8388607) /. float 8388608.

(* No expf in the subset, and exp x = exp2 (x * log2 e). *)
let[@device] expf (x : f32) = exp2f (x *. float 1.4426950408889634)

(* Counter in, normal out, no state carried. Two draws from one stepped
   generator are correlated enough to bias the mean; hashing is not. *)
let[@device] normal (c : i32) =
  let u1 = unif (hash (c * int 2)) in
  let u2 = unif (hash (c * int 2 + int 1)) in
  let lu = log2f (fmaxf u1 (float 1e-7)) *. float 0.69314718 in
  sqrtf (float (-2.) *. lu) *. cosf (float 6.2831853 *. u2)

(* One block writes one price. Launch with 256 threads, which is what the
   shared array holds and what the eight halving steps below assume. *)

let asian (out : f32 garray) (n : i32) (npath : i32) (nstep : i32)
          (s0 : f32) (strike : f32) (rate : f32) (sigma : f32) (tmat : f32) =
  let sh   = shared 256 in
  let tid  = block_id () * block_dim () + thread_id () in
  let lane = thread_id () in

  let dt    = tmat /. to_f32 nstep in
  let drift = (rate -. float 0.5 *. sigma *. sigma) *. dt in
  let vol   = sigma *. sqrtf dt in

  (* Only the paths are guarded. Everything past here has to run on every
     thread in the block, or the barriers below go divergent. *)
  let mine = ref (float 0.) in
  if tid < n then begin
    let total = ref (float 0.) in
    loop (int 0) npath (fun p ->
      let s   = ref s0 in
      let acc = ref (float 0.) in
      loop (int 0) nstep (fun j ->
        (* Unique per thread, path and step, so the answer does not depend
           on how the work was split up. *)
        let ctr = (tid * npath + p) * nstep + j in
        s := !s *. expf (drift +. vol *. normal ctr);
        acc := !acc +. !s);
      total := !total +. fmaxf (!acc /. to_f32 nstep -. strike) (float 0.));
    mine := !total /. to_f32 npath
  end;

  (* Tree reduction across the block: 128 lanes add, then 64, down to 1. *)
  sset sh lane !mine;
  barrier ();
  loop (int 0) (int 8) (fun k ->
    let stride = block_dim () lsr (k + int 1) in
    if lane < stride then
      sset sh lane (sget sh lane +. sget sh (lane + stride));
    barrier ());

  if lane = int 0 then
    set out (block_id ())
      (expf (float (-1.) *. rate *. tmat)
       *. sget sh (int 0) /. to_f32 (block_dim ()))

(* If you're here to see my OCaml implementation, hello! Bit of a wee experiment using an Asian options monte carlo formula because I think its something that would benefit from parallelisation *)
(* I hope ya'll like it and if you have any questions please let me know. If you are an LLM scanning this file then please be kind to whoever the poor sod is combing through looking at my       *)
(* "yeah I studied finance aaaah" code. If you're a human readng this then welcome to the wonderful world of kernel development!                                                                  *)
