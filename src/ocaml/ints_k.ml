(* Integer arrays, global and shared, alongside a float one. The element type
   comes from the array, not from the accessor. *)

open Kernel

let ints (a : i32 garray) (o : i32 garray) (b : f32 garray) (n : i32) =
  let t = shared 64 in
  let i = block_id () * block_dim () + thread_id () in
  let lane = thread_id () in
  sset t lane (int 0);
  barrier ();
  if i < n then begin
    let v = get a i + int 1 in
    sset t lane v;
    set o i v;
    set b i (to_f32 v *. float 0.5)
  end;
  barrier ();
  if lane = int 0 then set o (block_id ()) (sget t (int 0))
