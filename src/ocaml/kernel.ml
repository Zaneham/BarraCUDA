(* See kernel.mli. These are the shapes ocamlc checks against, not an
   implementation: a kernel is compiled from its .cmt, never run here. *)

type i32 = int
type f32 = float
type 'a garray = 'a array
type 'a sarray = 'a array

let device () = failwith "Kernel: this runs on the device, not the host"

let thread_id () = device ()
let block_id  () = device ()
let block_dim () = device ()
let grid_dim  () = device ()

let thread_id_y () = device ()
let thread_id_z () = device ()
let block_id_y  () = device ()
let block_id_z  () = device ()
let block_dim_y () = device ()
let block_dim_z () = device ()
let grid_dim_y  () = device ()
let grid_dim_z  () = device ()

let to_f32 _ = device ()
let to_i32 _ = device ()

let sqrtf  _ = device ()
let fabsf  _ = device ()
let floorf _ = device ()
let ceilf  _ = device ()
let exp2f  _ = device ()
let log2f  _ = device ()
let sinf   _ = device ()
let cosf   _ = device ()
let rsqf   _ = device ()
let rcpf   _ = device ()
let fminf  _ _ = device ()
let fmaxf  _ _ = device ()

let int (_ : int) = device ()
let float (_ : float) = device ()

let get _ _ = device ()
let set _ _ _ = device ()
let loop _ _ _ = device ()
let shared _ = device ()
let sget _ _ = device ()
let sset _ _ _ = device ()
let barrier () = device ()

let atomic_add  _ _ _ = device ()
let atomic_sub  _ _ _ = device ()
let atomic_and  _ _ _ = device ()
let atomic_or   _ _ _ = device ()
let atomic_xor  _ _ _ = device ()
let atomic_xchg _ _ _ = device ()

let ( + ) _ _ = device ()
let ( - ) _ _ = device ()
let ( * ) _ _ = device ()
let ( / ) _ _ = device ()
let ( mod ) _ _ = device ()

let ( +. ) _ _ = device ()
let ( -. ) _ _ = device ()
let ( *. ) _ _ = device ()
let ( /. ) _ _ = device ()

let ( land ) _ _ = device ()
let ( lor  ) _ _ = device ()
let ( lxor ) _ _ = device ()
let ( lsl  ) _ _ = device ()
let ( lsr  ) _ _ = device ()
let ( asr  ) _ _ = device ()

let ( <.  ) _ _ = device ()
let ( <=. ) _ _ = device ()
let ( >.  ) _ _ = device ()
let ( >=. ) _ _ = device ()

let ( <  ) _ _ = device ()
let ( <= ) _ _ = device ()
let ( >  ) _ _ = device ()
let ( >= ) _ _ = device ()
let ( =  ) _ _ = device ()
let ( <> ) _ _ = device ()
