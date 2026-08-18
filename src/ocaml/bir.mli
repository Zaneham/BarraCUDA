(** Emit BIR text for [kath --bir-in].

    Do not nest the emitting calls: each appends, and OCaml does not specify
    argument evaluation order. Bind each to a [let]. *)

type addrspace = Private | Shared | Global | Constant | Generic

type ty =
  | Void
  | Int of int                      (** [Int 32] prints as [i32] *)
  | Float of int
  | Ptr of addrspace * ty
  | Arr of int * ty                 (** [Arr (64, Float 32)] prints as [[64 x f32]] *)

type value

type block
(** Reserve every block before emitting: a branch names blocks with no body. *)

type t

type dim = X | Y | Z

type pred = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge
          | Oeq | One | Olt | Ole | Ogt | Oge

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

val sdiv : t -> ty -> value -> value -> value
val udiv : t -> ty -> value -> value -> value
val srem : t -> ty -> value -> value -> value
val urem : t -> ty -> value -> value -> value
val fdiv : t -> ty -> value -> value -> value
val frem : t -> ty -> value -> value -> value

val band : t -> ty -> value -> value -> value
val bor  : t -> ty -> value -> value -> value
val bxor : t -> ty -> value -> value -> value
val shl  : t -> ty -> value -> value -> value
val lshr : t -> ty -> value -> value -> value
val ashr : t -> ty -> value -> value -> value

(** Intrinsics carry no type in the text; the operand supplies it. *)
val sqrt  : t -> value -> value
val rsq   : t -> value -> value
val rcp   : t -> value -> value
val exp2  : t -> value -> value
val log2  : t -> value -> value
val sin   : t -> value -> value
val cos   : t -> value -> value
val fabs  : t -> value -> value
val floor : t -> value -> value
val ceil  : t -> value -> value
val fmax  : t -> value -> value -> value
val fmin  : t -> value -> value -> value

val sitofp : t -> src:ty -> value -> dst:ty -> value
val fptosi : t -> src:ty -> value -> dst:ty -> value

val select : t -> ty -> value -> value -> value -> value

val icmp : t -> pred -> ty -> value -> value -> value
val fcmp : t -> pred -> ty -> value -> value -> value

val alloca : t -> ty -> value
(** Takes the pointer type, not the pointee. mem2reg promotes these away. *)

val shared_alloc : t -> ty -> value
(** Per block rather than per thread, so mem2reg leaves it alone. *)

val barrier : t -> unit

val atomic : t -> string -> ty -> value -> value -> value
(** Relaxed only for now. Named rather than enumerated because min and max
    have no settled signedness in BIR and must not be reachable. *)

val gep   : t -> ty -> value -> value -> value
val load  : t -> ty -> value -> value
val store : t -> ty -> value -> value -> unit

val br      : t -> block -> unit
val br_cond : t -> value -> block -> block -> merge:block -> unit
val ret     : t -> unit
val retv    : t -> ty -> value -> unit
val call    : t -> ty -> string -> value list -> value

val finish : t -> unit

val to_string : t -> string
val write : t -> string -> unit
