(* See bir.mli. *)

type addrspace = Private | Shared | Global | Constant | Generic

type ty =
  | Void
  | Int of int
  | Float of int
  | Ptr of addrspace * ty
  | Arr of int * ty

type value = V of int | Imm of string
type block = B of { name : string; mutable body : string list;
                    mutable opened : bool }

type dim = X | Y | Z
type pred = Eq | Ne | Slt | Sle | Sgt | Sge | Ult | Ule | Ugt | Uge
          | Oeq | One | Olt | Ole | Ogt | Oge

type func = {
  fname   : string;
  fparams : ty list;
  kernel  : bool;
  mutable blocks : block list;   (* reverse creation order, for the fill check *)
  mutable order  : block list;   (* reverse order of first opening *)
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
  | Arr (n, e) -> Printf.sprintf "[%d x %s]" n (ty_str e)

let pred_str = function
  | Eq -> "eq" | Ne -> "ne"
  | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
  | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"
  | Oeq -> "oeq" | One -> "one"
  | Olt -> "olt" | Ole -> "ole" | Ogt -> "ogt" | Oge -> "oge"

let dim_str = function X -> "x" | Y -> "y" | Z -> "z"

let vstr = function
  | V n -> Printf.sprintf "%%%d" n
  | Imm s -> s

(* Immediates are operands, so no instruction and no value number. Printed
   the way bir_print.c writes them, which is %d and %g. *)
let const_int n = Imm (string_of_int n)

(* Nine digits, which is what a float32 needs to read back as itself, and
   OCaml writes e+006 on Windows where C writes e+06, so the exponent is
   trimmed to the C99 minimum or the text depends on the host. *)
let trim_exp s =
  match String.index_opt s 'e' with
  | None -> s
  | Some i ->
    let n = String.length s in
    let j = if i + 1 < n && (s.[i + 1] = '+' || s.[i + 1] = '-') then i + 2 else i + 1 in
    let d = n - j in
    let k = ref 0 in
    while !k < d - 2 && s.[j + !k] = '0' do incr k done;
    String.sub s 0 j ^ String.sub s (j + !k) (d - !k)

let const_float f = Imm (trim_exp (Printf.sprintf "%.9g" f))

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
  let f = { fname = name; fparams = params; kernel; blocks = []; order = [] } in
  t.cur <- Some f;
  t.next <- 0;
  (* Numbered first, so signature and body agree. *)
  List.map (fun _ -> fresh t) params

let block t name =
  match t.cur with
  | None -> fail "no function is open"
  | Some f ->
    let b = B { name; body = []; opened = false } in
    f.blocks <- b :: f.blocks;
    b

(* Blocks print in the order they are first opened, not the order they are
   reserved, because bir_print numbers instructions by position and the numbers
   here are handed out as they are emitted. *)
let entry t b =
  match t.cur with
  | None -> fail "no function is open"
  | Some f ->
    (match b with
     | B bb ->
       if not bb.opened then begin
         bb.opened <- true;
         f.order <- b :: f.order
       end);
    t.here <- Some b

let finish t =
  match t.cur with
  | None -> fail "no function is open"
  | Some f ->
    List.iter
      (fun (B b) -> if not b.opened then fail ("block " ^ b.name ^ " was never filled"))
      f.blocks;
    f.blocks <- List.rev f.order;
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

let sdiv = bin "sdiv"
let udiv = bin "udiv"
let srem = bin "srem"
let urem = bin "urem"
let fdiv = bin "fdiv"
let frem = bin "frem"
let band = bin "and"
let bor  = bin "or"
let bxor = bin "xor"
let shl  = bin "shl"
let lshr = bin "lshr"
let ashr = bin "ashr"

(* Intrinsics carry no type in the text. the operand gives it. *)
let fn1 op t a = def t (Printf.sprintf "%s %s" op (vstr a))
let fn2 op t a b = def t (Printf.sprintf "%s %s %s" op (vstr a) (vstr b))

let sqrt  = fn1 "sqrt"
let rsq   = fn1 "rsq"
let rcp   = fn1 "rcp"
let exp2  = fn1 "exp2"
let log2  = fn1 "log2"
let sin   = fn1 "sin"
let cos   = fn1 "cos"
let fabs  = fn1 "fabs"
let floor = fn1 "floor"
let ceil  = fn1 "ceil"
let fmax  = fn2 "fmax"
let fmin  = fn2 "fmin"

let cvt op t ~src a ~dst =
  def t (Printf.sprintf "%s %s %s to %s" op (ty_str src) (vstr a) (ty_str dst))

let sitofp = cvt "sitofp"
let fptosi = cvt "fptosi"

let select t ty c a b =
  def t (Printf.sprintf "select %s %s, %s, %s"
           (ty_str ty) (vstr c) (vstr a) (vstr b))

let cmp op t p ty a b =
  def t (Printf.sprintf "%s %s %s %s, %s"
           op (pred_str p) (ty_str ty) (vstr a) (vstr b))

let icmp = cmp "icmp"
let fcmp = cmp "fcmp"

let alloca t ty = def t (Printf.sprintf "alloca %s" (ty_str ty))
let shared_alloc t ty = def t (Printf.sprintf "shared_alloc %s" (ty_str ty))
let barrier t = emit t "barrier"

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
let retv t ty v = emit t (Printf.sprintf "ret %s %s" (ty_str ty) (vstr v))

let call t ty name args =
  def t (Printf.sprintf "call %s @%s(%s)" (ty_str ty) name
           (String.concat ", " (List.map vstr args)))

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
