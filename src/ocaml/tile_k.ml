(* Reverses a block through shared memory, which needs every thread to have
   finished writing before any starts reading. *)

open Kernel

let tile (a : f32 garray) (b : f32 garray) (n : i32) =
  let t = shared 256 in
  let i = block_id () * block_dim () + thread_id () in
  let lane = thread_id () in
  if i < n then begin
    sset t lane (get a i);
    barrier ();
    set b i (sget t (block_dim () - lane - int 1))
  end
