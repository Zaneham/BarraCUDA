(* See bir.mli. *)

type addrspace = Private | Shared | Global | Constant | Generic

type ty =
  | Void
  | Int of int
  | Float of int
  | Ptr of addrspace * ty

type value = V of int | Imm of string
type block = B of { name : string; mutable body : string list }

type dim = X | Y | Z
type pred = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge

type func = {
  fname   : string;
  fparams : ty list;
  kernel  : bool;
  mutable blocks : block list;   (* reverse order while building *)
}

type t = {
  mutable funcs : func list;     (* reverse order *)
  mutable cur   : func option;
  mutable here  : block option;
  mutable next  : int;           (* next %N *)
  mutable src   : int;           (* current source line, 0 = none *)
}

let create () = { funcs = []; cur = None; here = None; next = 0; src = 0 }

(* ---- Printing types ---- *)

let addrspace_str = function
  | Private -> "private" | Shared -> "shared" | Global -> "global"
  | Constant -> "constant" | Generic -> "generic"

let rec ty_str = function
  | Void -> "void"
  | Int n -> Printf.sprintf "i%d" n
  | Float n -> Printf.sprintf "f%d" n
  | Ptr (a, inner) -> Printf.sprintf "ptr<%s, %s>" (addrspace_str a) (ty_str inner)

let pred_str = function
  | Eq -> "eq" | Ne -> "ne"
  | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
  | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

let dim_str = function X -> "x" | Y -> "y" | Z -> "z"

let vstr = function
  | V n -> Printf.sprintf "%%%d" n
  | Imm s -> s

(* Immediates are operands, so no instruction and no value number. Printed
   the way bir_print.c writes them, which is %d and %g. *)
let const_int n = Imm (string_of_int n)
let const_float f = Imm (Printf.sprintf "%g" f)

(* ---- Emission ---- *)

let fail msg = invalid_arg ("Bir: " ^ msg)

(* Every instruction burns a value number, including the void ones that print
   no result: bir_print.c numbers by position in the function. *)
let inst t fmt =
  match t.here with
  | None -> fail "no block is open"
  | Some (B b) ->
    let n = t.next in
    t.next <- n + 1;
    let line = if t.src > 0 then Printf.sprintf "  ; line %d" t.src else "" in
    b.body <- ("    " ^ fmt n ^ line) :: b.body;
    V n

let emit t text = ignore (inst t (fun _ -> text))

let fresh t =
  let n = t.next in
  t.next <- n + 1;
  V n


let def t text = inst t (fun n -> Printf.sprintf "%%%d = %s" n text)

let line t n = t.src <- n

(* ---- Functions and blocks ---- *)

let func t name ~params ~kernel =
  if t.cur <> None then fail "a function is already open";
  let f = { fname = name; fparams = params; kernel; blocks = [] } in
  t.cur <- Some f;
  t.next <- 0;
  (* Numbered first, so signature and body agree. *)
  List.map (fun _ -> fresh t) params

let block t name =
  match t.cur with
  | None -> fail "no function is open"
  | Some f ->
    let b = B { name; body = [] } in
    f.blocks <- b :: f.blocks;
    b

let entry t b =
  if t.cur = None then fail "no function is open";
  t.here <- Some b

let finish t =
  match t.cur with
  | None -> fail "no function is open"
  | Some f ->
    f.blocks <- List.rev f.blocks;
    t.funcs <- f :: t.funcs;
    t.cur <- None;
    t.here <- None

(* ---- Instructions ---- *)

let thread_id t d = def t (Printf.sprintf "thread_id.%s" (dim_str d))
let block_id  t d = def t (Printf.sprintf "block_id.%s"  (dim_str d))
let block_dim t d = def t (Printf.sprintf "block_dim.%s" (dim_str d))
let grid_dim  t d = def t (Printf.sprintf "grid_dim.%s"  (dim_str d))

let bin op t ty a b =
  def t (Printf.sprintf "%s %s %s, %s" op (ty_str ty) (vstr a) (vstr b))

let add  = bin "add"
let sub  = bin "sub"
let mul  = bin "mul"
let fadd = bin "fadd"
let fsub = bin "fsub"
let fmul = bin "fmul"

let icmp t p ty a b =
  def t (Printf.sprintf "icmp %s %s %s, %s"
           (pred_str p) (ty_str ty) (vstr a) (vstr b))

let gep t ty base idx =
  def t (Printf.sprintf "gep %s, %s, %s" (ty_str ty) (vstr base) (vstr idx))

let load t ty addr =
  def t (Printf.sprintf "load %s, %s" (ty_str ty) (vstr addr))

let store t ty v addr =
  emit t (Printf.sprintf "store %s %s, %s" (ty_str ty) (vstr v) (vstr addr))

let bname (B b) = b.name

let br t b = emit t (Printf.sprintf "br %s" (bname b))

let br_cond t c bt bf ~merge =
  emit t (Printf.sprintf "br_cond %s, %s, %s merge %s"
            (vstr c) (bname bt) (bname bf) (bname merge))

let ret t = emit t "ret void"

(* ---- Output ---- *)

let func_str f =
  let buf = Buffer.create 512 in
  let params =
    List.mapi (fun i ty -> Printf.sprintf "%s %%%d" (ty_str ty) i) f.fparams
  in
  Buffer.add_string buf
    (Printf.sprintf "func @%s(%s)%s {\n"
       f.fname (String.concat ", " params)
       (if f.kernel then " __global__" else " __device__"));
  List.iteri
    (fun i (B b) ->
       if i > 0 then Buffer.add_char buf '\n';
       Buffer.add_string buf (b.name ^ ":\n");
       List.iter (fun l -> Buffer.add_string buf (l ^ "\n")) (List.rev b.body))
    f.blocks;
  Buffer.add_string buf "}\n";
  Buffer.contents buf

let to_string t =
  if t.cur <> None then fail "a function is still open";
  let buf = Buffer.create 1024 in
  Buffer.add_string buf "; Booth IR\n";
  List.iter
    (fun f -> Buffer.add_char buf '\n'; Buffer.add_string buf (func_str f))
    (List.rev t.funcs);
  Buffer.contents buf

let write t path =
  let oc = open_out path in
  output_string oc (to_string t);
  close_out oc
