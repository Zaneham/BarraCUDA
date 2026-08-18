(** The OCaml subset that compiles to a kernel.

    Kernel code is ordinary OCaml and ocamlc type-checks it. None of it runs on
    the host; every function here raises. [kcomp] lowers the .cmt instead. *)

type i32
type f32

type 'a garray
(** An array in global memory. Indexing is unchecked, as on the device. *)

val thread_id : unit -> i32
val block_id  : unit -> i32
val block_dim : unit -> i32
val grid_dim  : unit -> i32

val int : int -> i32
val float : float -> f32

val ( + ) : i32 -> i32 -> i32
val ( - ) : i32 -> i32 -> i32
val ( * ) : i32 -> i32 -> i32

val ( +. ) : f32 -> f32 -> f32
val ( -. ) : f32 -> f32 -> f32
val ( *. ) : f32 -> f32 -> f32

val ( <  ) : i32 -> i32 -> bool
val ( <= ) : i32 -> i32 -> bool
val ( >  ) : i32 -> i32 -> bool
val ( >= ) : i32 -> i32 -> bool
val ( =  ) : i32 -> i32 -> bool
val ( <> ) : i32 -> i32 -> bool

val get : f32 garray -> i32 -> f32
val set : f32 garray -> i32 -> f32 -> unit
