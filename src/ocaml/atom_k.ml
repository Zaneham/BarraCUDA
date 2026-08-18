(* Atomics, which is how a reduction crosses block boundaries. No min or max
   here on purpose, see the note in kernel.mli. *)

open Kernel

let tally (f : f32 garray) (c : i32 garray) (n : i32) =
  let i = block_id () * block_dim () + thread_id () in
  if i < n then begin
    ignore (atomic_add c (int 0) (int 1));
    ignore (atomic_add f (int 0) (float 0.5));
    ignore (atomic_xor c (int 1) i);
    let old = atomic_xchg c (int 2) i in
    set f i (to_f32 old)
  end
