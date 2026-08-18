(* kcomp -- lower the kernel functions in a .cmt to BIR text.
 * ocamlc has already type-checked the file; what is left is the subset check.
 *
 *   kcomp vadd_k.cmt -o vadd.bir
 *)

open Typedtree

exception Reject of Location.t * string

let reject loc fmt = Printf.ksprintf (fun s -> raise (Reject (loc, s))) fmt

(* ---- Types ---- *)

let rec ty_of loc t =
  match Types.get_desc t with
  | Types.Tconstr (p, args, _) ->
    (match Path.name p, args with
     | "Kernel.i32", [] -> Bir.Int 32
     | "Kernel.f32", [] -> Bir.Float 32
     | "Kernel.garray", [e] -> Bir.Ptr (Bir.Global, ty_of loc e)
     | n, _ -> reject loc "type %s is not in the kernel subset" n)
  | _ -> reject loc "this type is not in the kernel subset"

let elem loc = function
  | Bir.Ptr (_, e) -> e
  | _ -> reject loc "expected an array"

(* ---- Context ---- *)

type ctx = {
  m : Bir.t;
  mutable vars : (Ident.t * Bir.value) list;
  mutable nblk : int;                  (* names the if blocks apart *)
}

let bind c id v = c.vars <- (id, v) :: c.vars

let lookup c loc id =
  match List.find_opt (fun (i, _) -> Ident.same i id) c.vars with
  | Some (_, v) -> v
  | None -> reject loc "%s is not bound in the kernel" (Ident.name id)

let stamp c loc = Bir.line c.m loc.Location.loc_start.Lexing.pos_lnum

let callee loc f =
  match f.exp_desc with
  | Texp_ident (p, _, _) -> Path.name p
  | _ -> reject loc "only Kernel primitives may be applied"

let given args =
  List.filter_map (function (_, Arg x) -> Some x | (_, Omitted _) -> None) args

(* ---- Expressions ---- *)

let rec expr c e =
  match e.exp_desc with
  | Texp_ident (Path.Pident id, _, _) -> lookup c e.exp_loc id
  | Texp_ident (p, _, _) ->
    reject e.exp_loc "%s is not a kernel value" (Path.name p)
  | Texp_apply (f, args) -> apply c e.exp_loc f args
  | Texp_let (Asttypes.Nonrecursive, vbs, body) -> binds c vbs; expr c body
  | _ -> reject e.exp_loc "this expression is not in the kernel subset"

and apply c loc f args =
  let i32 = Bir.Int 32 and f32 = Bir.Float 32 in
  match callee loc f, given args with
  | "Kernel.thread_id", [_] -> stamp c loc; Bir.thread_id c.m Bir.X
  | "Kernel.block_id",  [_] -> stamp c loc; Bir.block_id  c.m Bir.X
  | "Kernel.block_dim", [_] -> stamp c loc; Bir.block_dim c.m Bir.X
  | "Kernel.grid_dim",  [_] -> stamp c loc; Bir.grid_dim  c.m Bir.X

  | "Kernel.+",  [x; y] -> arith c loc Bir.add  i32 x y
  | "Kernel.-",  [x; y] -> arith c loc Bir.sub  i32 x y
  | "Kernel.*",  [x; y] -> arith c loc Bir.mul  i32 x y
  | "Kernel.+.", [x; y] -> arith c loc Bir.fadd f32 x y
  | "Kernel.-.", [x; y] -> arith c loc Bir.fsub f32 x y
  | "Kernel.*.", [x; y] -> arith c loc Bir.fmul f32 x y

  | "Kernel.<",  [x; y] -> cmp c loc Bir.Slt x y
  | "Kernel.<=", [x; y] -> cmp c loc Bir.Sle x y
  | "Kernel.>",  [x; y] -> cmp c loc Bir.Sgt x y
  | "Kernel.>=", [x; y] -> cmp c loc Bir.Sge x y
  | "Kernel.=",  [x; y] -> cmp c loc Bir.Eq  x y
  | "Kernel.<>", [x; y] -> cmp c loc Bir.Ne  x y

  | "Kernel.int", [x] ->
    (match x.exp_desc with
     | Texp_constant (Asttypes.Const_int n) -> Bir.const_int n
     | _ -> reject x.exp_loc "int takes a literal")
  | "Kernel.float", [x] ->
    (match x.exp_desc with
     | Texp_constant (Asttypes.Const_float s) -> Bir.const_float (float_of_string s)
     | _ -> reject x.exp_loc "float takes a literal")

  | "Kernel.get", [arr; idx] ->
    let base = expr c arr in
    let i = expr c idx in
    let pty = ty_of arr.exp_loc arr.exp_type in
    stamp c loc;
    let p = Bir.gep c.m pty base i in
    Bir.load c.m (elem loc pty) p

  | n, _ -> reject loc "%s is not a kernel primitive" n

and arith c loc op ty x y =
  let a = expr c x in
  let b = expr c y in
  stamp c loc;
  op c.m ty a b

and cmp c loc p x y =
  let a = expr c x in
  let b = expr c y in
  stamp c loc;
  Bir.icmp c.m p (Bir.Int 32) a b

and binds c vbs =
  List.iter
    (fun vb ->
       match vb.vb_pat.pat_desc with
       | Tpat_var (id, _, _) -> let v = expr c vb.vb_expr in bind c id v
       | _ -> reject vb.vb_pat.pat_loc "only a plain name may be bound")
    vbs

(* ---- Statements ---- *)

let rec stmt c e =
  match e.exp_desc with
  | Texp_sequence (a, b) -> stmt c a; stmt c b
  | Texp_let (Asttypes.Nonrecursive, vbs, body) -> binds c vbs; stmt c body
  | Texp_ifthenelse (cond, t, None) -> ifthen c e.exp_loc cond t
  | Texp_ifthenelse (_, _, Some _) ->
    reject e.exp_loc "an if with an else branch is not in the subset yet"
  | Texp_apply (f, args) -> effect_ c e.exp_loc f args
  | _ -> reject e.exp_loc "this statement is not in the kernel subset"

and ifthen c loc cond t =
  let v = expr c cond in
  let n = c.nblk in
  c.nblk <- n + 1;
  let tag s = if n = 0 then s else Printf.sprintf "%s.%d" s n in
  let then_ = Bir.block c.m (tag "if.then") in
  let endb = Bir.block c.m (tag "if.end") in
  stamp c loc;
  Bir.br_cond c.m v then_ endb ~merge:endb;
  Bir.entry c.m then_;
  stmt c t;
  Bir.br c.m endb;
  Bir.entry c.m endb

and effect_ c loc f args =
  match callee loc f, given args with
  | "Kernel.set", [arr; idx; v] ->
    let base = expr c arr in
    let i = expr c idx in
    let pty = ty_of arr.exp_loc arr.exp_type in
    stamp c loc;
    let p = Bir.gep c.m pty base i in
    let value = expr c v in
    stamp c loc;
    Bir.store c.m (elem loc pty) value p
  | n, _ -> reject loc "%s is not a kernel statement" n

(* ---- Functions ---- *)

let kernel c name ps body =
  c.vars <- [];
  c.nblk <- 0;
  let pats =
    List.map
      (fun p ->
         match p.fp_kind with
         | Tparam_pat pat -> pat
         | Tparam_optional_default (pat, _) ->
           reject pat.pat_loc "an optional argument is not a kernel parameter")
      ps
  in
  let tys = List.map (fun p -> ty_of p.pat_loc p.pat_type) pats in
  let vals = Bir.func c.m name ~params:tys ~kernel:true in
  List.iter2
    (fun p v ->
       match p.pat_desc with
       | Tpat_var (id, _, _) -> bind c id v
       | _ -> reject p.pat_loc "a kernel parameter must be a plain name")
    pats vals;
  let entry = Bir.block c.m "entry" in
  Bir.entry c.m entry;
  stmt c body;
  Bir.ret c.m;
  Bir.finish c.m

let structure c st =
  List.iter
    (fun it ->
       match it.str_desc with
       | Tstr_value (Asttypes.Nonrecursive, vbs) ->
         List.iter
           (fun vb ->
              match vb.vb_pat.pat_desc, vb.vb_expr.exp_desc with
              | Tpat_var (id, _, _), Texp_function (ps, Tfunction_body body) ->
                kernel c (Ident.name id) ps body
              | _ -> reject vb.vb_loc "only kernel functions may be defined")
           vbs
       | Tstr_value (Asttypes.Recursive, _) ->
         reject it.str_loc "a kernel may not recurse"
       | Tstr_open _ | Tstr_attribute _ -> ()
       | _ -> reject it.str_loc "only kernel definitions are allowed")
    st.str_items

(* ---- Driver ---- *)

let () =
  let cmt, out =
    match Array.to_list Sys.argv with
    | [ _; f; "-o"; o ] -> f, o
    | [ _; f ] -> f, "a.bir"
    | _ -> prerr_endline "usage: kcomp file.cmt [-o out.bir]"; exit 2
  in
  let info = Cmt_format.read_cmt cmt in
  let c = { m = Bir.create (); vars = []; nblk = 0 } in
  match info.Cmt_format.cmt_annots with
  | Cmt_format.Implementation st ->
    (try
       structure c st;
       Bir.write c.m out;
       Printf.printf "wrote %s\n%!" out
     with Reject (loc, msg) ->
       let p = loc.Location.loc_start in
       Printf.eprintf "%s:%d:%d: %s\n%!" p.Lexing.pos_fname p.Lexing.pos_lnum
         (p.Lexing.pos_cnum - p.Lexing.pos_bol + 1) msg;
       exit 1)
  | _ -> prerr_endline "kcomp: not an implementation .cmt"; exit 2
