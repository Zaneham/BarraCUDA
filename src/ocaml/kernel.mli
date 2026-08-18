(** The OCaml subset that compiles to a kernel.

    Kernel code is ordinary OCaml and ocamlc type-checks it. None of it runs on
    the host; every function here raises. [kcomp] lowers the .cmt instead. *)

type i32
type f32

type 'a garray
(** An array in global memory. Indexing is unchecked, as on the device. *)

type 'a sarray
(** Per block, not per thread, and a different type from [garray] so the two
    cannot be confused at a call site. Every thread in the block sees the same
    storage, which is why [barrier] exists. *)

(** The unsuffixed forms are x, which is what most kernels want. *)
val thread_id : unit -> i32
val block_id  : unit -> i32
val block_dim : unit -> i32
val grid_dim  : unit -> i32

val thread_id_y : unit -> i32
val thread_id_z : unit -> i32
val block_id_y  : unit -> i32
val block_id_z  : unit -> i32
val block_dim_y : unit -> i32
val block_dim_z : unit -> i32
val grid_dim_y  : unit -> i32
val grid_dim_z  : unit -> i32

val int : int -> i32
val float : float -> f32

val ( + ) : i32 -> i32 -> i32
val ( - ) : i32 -> i32 -> i32
val ( * ) : i32 -> i32 -> i32
val ( / ) : i32 -> i32 -> i32
val ( mod ) : i32 -> i32 -> i32

val ( +. ) : f32 -> f32 -> f32
val ( -. ) : f32 -> f32 -> f32
val ( *. ) : f32 -> f32 -> f32
val ( /. ) : f32 -> f32 -> f32

val ( land ) : i32 -> i32 -> i32
val ( lor  ) : i32 -> i32 -> i32
val ( lxor ) : i32 -> i32 -> i32
val ( lsl  ) : i32 -> i32 -> i32
val ( lsr  ) : i32 -> i32 -> i32
val ( asr  ) : i32 -> i32 -> i32

val to_f32 : i32 -> f32
val to_i32 : f32 -> i32
(** Truncates toward zero, as C does. *)

val sqrtf  : f32 -> f32
val fabsf  : f32 -> f32
val floorf : f32 -> f32
val ceilf  : f32 -> f32
val exp2f  : f32 -> f32
val log2f  : f32 -> f32
val sinf   : f32 -> f32
val cosf   : f32 -> f32
val rsqf   : f32 -> f32
val rcpf   : f32 -> f32
val fminf  : f32 -> f32 -> f32
val fmaxf  : f32 -> f32 -> f32

(* Ordered comparisons, so anything involving a NaN is false. *)
val ( <.  ) : f32 -> f32 -> bool
val ( <=. ) : f32 -> f32 -> bool
val ( >.  ) : f32 -> f32 -> bool
val ( >=. ) : f32 -> f32 -> bool

val ( <  ) : i32 -> i32 -> bool
val ( <= ) : i32 -> i32 -> bool
val ( >  ) : i32 -> i32 -> bool
val ( >= ) : i32 -> i32 -> bool
val ( =  ) : i32 -> i32 -> bool
val ( <> ) : i32 -> i32 -> bool

val get : f32 garray -> i32 -> f32
val set : f32 garray -> i32 -> f32 -> unit

val shared : int -> f32 sarray
(** [shared n] is one block-wide array of n floats. The size is a literal, so
    it is fixed when the kernel is built rather than at launch. *)

val sget : f32 sarray -> i32 -> f32
val sset : f32 sarray -> i32 -> f32 -> unit

val barrier : unit -> unit
(** Every thread in the block arrives before any leaves. Reaching one from
    inside a branch that not all threads take is undefined, as it is in CUDA. *)

val loop : i32 -> i32 -> (i32 -> unit) -> unit
(** [loop lo hi f] runs [f i] for lo up to but not including hi. The body is
    inlined, so it may not escape or be passed on. *)

(** Accumulators are ordinary [ref] cells. They become one stack slot each and
    mem2reg promotes them, so nothing reaches the device as memory. *)
