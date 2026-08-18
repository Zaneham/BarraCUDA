(* Thread t sums a[t..n), so every thread gets a different answer and a wrong
   index shows up immediately. Needs the two things straight-line code cannot
   express, a loop and something to accumulate into. *)

open Kernel

let reduce (a : f32 garray) (o : f32 garray) (n : i32) =
  let t = block_id () * block_dim () + thread_id () in
  let s = ref (float 0.) in
  if t < n then begin
    loop t n (fun i -> s := !s +. get a i);
    set o t !s
  end
