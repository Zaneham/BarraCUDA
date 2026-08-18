(* See kernel.mli. These are the shapes ocamlc checks against, not an
   implementation: a kernel is compiled from its .cmt, never run here. *)

type i32 = int
type f32 = float
type 'a garray = 'a array

let device () = failwith "Kernel: this runs on the device, not the host"

let thread_id () = device ()
let block_id  () = device ()
let block_dim () = device ()
let grid_dim  () = device ()

let int (_ : int) = device ()
let float (_ : float) = device ()

let get _ _ = device ()
let set _ _ _ = device ()

let ( + ) _ _ = device ()
let ( - ) _ _ = device ()
let ( * ) _ _ = device ()

let ( +. ) _ _ = device ()
let ( -. ) _ _ = device ()
let ( *. ) _ _ = device ()

let ( <  ) _ _ = device ()
let ( <= ) _ _ = device ()
let ( >  ) _ _ = device ()
let ( >= ) _ _ = device ()
let ( =  ) _ _ = device ()
let ( <> ) _ _ = device ()
