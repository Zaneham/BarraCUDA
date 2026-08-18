(** Emit BIR text for [kath --bir-in].

    Do not nest the emitting calls: each appends, and OCaml does not specify
    argument evaluation order. Bind each to a [let]. *)

type addrspace = Private | Shared | Global | Constant | Generic

type ty =
  | Void
  | Int of int                      (** [Int 32] prints as [i32] *)
  | Float of int
  | Ptr of addrspace * ty

type value

type block
(** Reserve every block before emitting: a branch names blocks with no body. *)

type t

type dim = X | Y | Z

type pred = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge

val create : unit -> t

val const_int : int -> value
val const_float : float -> value
(** Immediates are operands rather than instructions, so these emit nothing. *)

val func : t -> string -> params:ty list -> kernel:bool -> value list
(** Open a function, returning its parameters. One at a time. *)

val block : t -> string -> block
val entry : t -> block -> unit
(** Start filling a reserved block. *)

val line : t -> int -> unit
(** Source line stamped on instructions emitted after this. *)

val thread_id : t -> dim -> value
val block_id  : t -> dim -> value
val block_dim : t -> dim -> value
val grid_dim  : t -> dim -> value

val add  : t -> ty -> value -> value -> value
val sub  : t -> ty -> value -> value -> value
val mul  : t -> ty -> value -> value -> value
val fadd : t -> ty -> value -> value -> value
val fsub : t -> ty -> value -> value -> value
val fmul : t -> ty -> value -> value -> value

val icmp : t -> pred -> ty -> value -> value -> value

val gep   : t -> ty -> value -> value -> value
val load  : t -> ty -> value -> value
val store : t -> ty -> value -> value -> unit

val br      : t -> block -> unit
val br_cond : t -> value -> block -> block -> merge:block -> unit
val ret     : t -> unit

val finish : t -> unit

val to_string : t -> string
val write : t -> string -> unit
