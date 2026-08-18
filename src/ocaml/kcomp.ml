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
     | "Kernel.sarray", [e] -> Bir.Ptr (Bir.Shared, ty_of loc e)
     | n, _ -> reject loc "type %s is not in the kernel subset" n)
  | _ -> reject loc "this type is not in the kernel subset"

(* A trailing () closes a sequence in ordinary OCaml and does nothing here. *)
let is_unit t =
  match Types.get_desc t with
  | Types.Tconstr (p, [], _) -> Path.name p = "unit"
  | _ -> false

let elem loc = function
  | Bir.Ptr (_, e) -> e
  | _ -> reject loc "expected an array"

(* ---- Context ---- *)

(* A name is either a value or the address of a ref cell. *)
type bnd = Val of Bir.value | Cell of Bir.value * Bir.ty
            | Shared of Bir.value

type ctx = {
  m : Bir.t;
  mutable vars : (Ident.t * bnd) list;
  mutable nblk : int;                  (* keeps generated block names apart *)
  mutable devs : (string * Bir.ty) list;   (* device functions and their result *)
}

let bind c id b = c.vars <- (id, b) :: c.vars

let lookup c loc id =
  match List.find_opt (fun (i, _) -> Ident.same i id) c.vars with
  | Some (_, b) -> b
  | None -> reject loc "%s is not bound in the kernel" (Ident.name id)

(* One suffix per construct, so the blocks of a nested if or loop stay apart. *)
let fresh_tag c =
  let n = c.nblk in
  c.nblk <- n + 1;
  fun base -> if n = 0 then base else Printf.sprintf "%s.%d" base n

let stamp c loc = Bir.line c.m loc.Location.loc_start.Lexing.pos_lnum

let callee loc f =
  match f.exp_desc with
  | Texp_ident (p, _, _) -> Path.name p
  | _ -> reject loc "only Kernel primitives may be applied"

let callee_is f name =
  match f.exp_desc with
  | Texp_ident (p, _, _) -> Path.name p = name
  | _ -> false

let given args =
  List.filter_map (function (_, Arg x) -> Some x | (_, Omitted _) -> None) args

(* ---- Expressions ---- *)

let rec expr c e =
  match e.exp_desc with
  | Texp_ident (Path.Pident id, _, _) ->
    (match lookup c e.exp_loc id with
     | Val v -> v
     | Cell _ -> reject e.exp_loc "%s is a ref, read it with !" (Ident.name id)
     | Shared _ ->
       reject e.exp_loc "%s is shared memory, index it with sget" (Ident.name id))
  | Texp_ident (p, _, _) ->
    reject e.exp_loc "%s is not a kernel value" (Path.name p)
  | Texp_apply (f, args) -> apply c e.exp_loc f args
  | Texp_let (Asttypes.Nonrecursive, vbs, body) -> binds c vbs; expr c body
  | _ -> reject e.exp_loc "this expression is not in the kernel subset"

and apply c loc f args =
  let i32 = Bir.Int 32 and f32 = Bir.Float 32 in
  match callee loc f, given args with
  | "Kernel.thread_id",   [_] -> stamp c loc; Bir.thread_id c.m Bir.X
  | "Kernel.thread_id_y", [_] -> stamp c loc; Bir.thread_id c.m Bir.Y
  | "Kernel.thread_id_z", [_] -> stamp c loc; Bir.thread_id c.m Bir.Z
  | "Kernel.block_id",    [_] -> stamp c loc; Bir.block_id  c.m Bir.X
  | "Kernel.block_id_y",  [_] -> stamp c loc; Bir.block_id  c.m Bir.Y
  | "Kernel.block_id_z",  [_] -> stamp c loc; Bir.block_id  c.m Bir.Z
  | "Kernel.block_dim",   [_] -> stamp c loc; Bir.block_dim c.m Bir.X
  | "Kernel.block_dim_y", [_] -> stamp c loc; Bir.block_dim c.m Bir.Y
  | "Kernel.block_dim_z", [_] -> stamp c loc; Bir.block_dim c.m Bir.Z
  | "Kernel.grid_dim",    [_] -> stamp c loc; Bir.grid_dim  c.m Bir.X
  | "Kernel.grid_dim_y",  [_] -> stamp c loc; Bir.grid_dim  c.m Bir.Y
  | "Kernel.grid_dim_z",  [_] -> stamp c loc; Bir.grid_dim  c.m Bir.Z

  | "Kernel.+",  [x; y] -> arith c loc Bir.add  i32 x y
  | "Kernel.-",  [x; y] -> arith c loc Bir.sub  i32 x y
  | "Kernel.*",  [x; y] -> arith c loc Bir.mul  i32 x y
  | "Kernel./",  [x; y] -> arith c loc Bir.sdiv i32 x y
  | "Kernel.mod",[x; y] -> arith c loc Bir.srem i32 x y
  | "Kernel.land", [x; y] -> arith c loc Bir.band i32 x y
  | "Kernel.lor",  [x; y] -> arith c loc Bir.bor  i32 x y
  | "Kernel.lxor", [x; y] -> arith c loc Bir.bxor i32 x y
  | "Kernel.lsl",  [x; y] -> arith c loc Bir.shl  i32 x y
  | "Kernel.lsr",  [x; y] -> arith c loc Bir.lshr i32 x y
  | "Kernel.asr",  [x; y] -> arith c loc Bir.ashr i32 x y
  | "Kernel.+.", [x; y] -> arith c loc Bir.fadd f32 x y
  | "Kernel.-.", [x; y] -> arith c loc Bir.fsub f32 x y
  | "Kernel.*.", [x; y] -> arith c loc Bir.fmul f32 x y
  | "Kernel./.", [x; y] -> arith c loc Bir.fdiv f32 x y

  | "Kernel.sqrtf",  [x] -> fn1 c loc Bir.sqrt  x
  | "Kernel.rsqf",   [x] -> fn1 c loc Bir.rsq   x
  | "Kernel.rcpf",   [x] -> fn1 c loc Bir.rcp   x
  | "Kernel.exp2f",  [x] -> fn1 c loc Bir.exp2  x
  | "Kernel.log2f",  [x] -> fn1 c loc Bir.log2  x
  (* BIR_SIN and BIR_COS take turns, not radians, because that is what AMD's
     hardware takes and the NVIDIA backend multiplies back. Divide on the way
     in exactly as the CUDA frontend does, so sinf means what C's sinf means. *)
  | "Kernel.sinf",   [x] -> turns c loc Bir.sin x
  | "Kernel.cosf",   [x] -> turns c loc Bir.cos x
  | "Kernel.fabsf",  [x] -> fn1 c loc Bir.fabs  x
  | "Kernel.floorf", [x] -> fn1 c loc Bir.floor x
  | "Kernel.ceilf",  [x] -> fn1 c loc Bir.ceil  x
  | "Kernel.fminf",  [x; y] -> fn2 c loc Bir.fmin x y
  | "Kernel.fmaxf",  [x; y] -> fn2 c loc Bir.fmax x y

  | "Kernel.to_f32", [x] ->
    let v = expr c x in
    stamp c loc;
    Bir.sitofp c.m ~src:i32 v ~dst:f32
  | "Kernel.to_i32", [x] ->
    let v = expr c x in
    stamp c loc;
    Bir.fptosi c.m ~src:f32 v ~dst:i32

  | "Kernel.<.",  [x; y] -> fcmp c loc Bir.Olt x y
  | "Kernel.<=.", [x; y] -> fcmp c loc Bir.Ole x y
  | "Kernel.>.",  [x; y] -> fcmp c loc Bir.Ogt x y
  | "Kernel.>=.", [x; y] -> fcmp c loc Bir.Oge x y

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

  | "Kernel.sget", [arr; idx] ->
    let base = sbase c loc arr in
    let i = expr c idx in
    let pty = ty_of arr.exp_loc arr.exp_type in
    stamp c loc;
    let p = Bir.gep c.m pty base i in
    Bir.load c.m (elem loc pty) p

  | "Stdlib.!", [r] ->
    (match cell c loc r with
     | (p, ty) -> stamp c loc; Bir.load c.m ty p)

  | n, a when List.mem_assoc n c.devs ->
    if List.length a > 5 then
      reject loc "%s takes more arguments than one call can carry" n;
    let vs = List.map (fun x -> expr c x) a in
    stamp c loc;
    Bir.call c.m (List.assoc n c.devs) n vs

  | n, _ -> reject loc "%s is not a kernel primitive" n

and turns c loc op x =
  let a = expr c x in
  stamp c loc;
  let t = Bir.fmul c.m (Bir.Float 32) a (Bir.const_float 0.15915494309189535) in
  op c.m t

and fn1 c loc op x =
  let a = expr c x in
  stamp c loc;
  op c.m a

and fn2 c loc op x y =
  let a = expr c x in
  let b = expr c y in
  stamp c loc;
  op c.m a b

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

and fcmp c loc p x y =
  let a = expr c x in
  let b = expr c y in
  stamp c loc;
  Bir.fcmp c.m p (Bir.Float 32) a b

(* The base of a shared array, which is all sget and sset accept. *)
and sbase c loc r =
  match r.exp_desc with
  | Texp_ident (Path.Pident id, _, _) ->
    (match lookup c r.exp_loc id with
     | Shared p -> p
     | _ -> reject loc "%s is not shared memory" (Ident.name id))
  | _ -> reject loc "only a named shared array may be indexed"

(* The address behind a ref, which is the only thing ! and := accept. *)
and cell c loc r =
  match r.exp_desc with
  | Texp_ident (Path.Pident id, _, _) ->
    (match lookup c r.exp_loc id with
     | Cell (p, ty) -> (p, ty)
     | Val _ | Shared _ -> reject loc "%s is not a ref" (Ident.name id))
  | _ -> reject loc "only a named ref may be read or assigned"

and binds c vbs =
  List.iter
    (fun vb ->
       match vb.vb_pat.pat_desc with
       | Tpat_var (id, _, _) -> bind c id (bound c vb.vb_expr)
       | _ -> reject vb.vb_pat.pat_loc "only a plain name may be bound")
    vbs

(* A ref becomes one stack slot. mem2reg turns it back into a value. *)
and bound c e =
  match e.exp_desc with
  | Texp_apply (f, args) when callee_is f "Stdlib.ref" ->
    (match given args with
     | [init] ->
       let ty = ty_of init.exp_loc init.exp_type in
       stamp c e.exp_loc;
       let p = Bir.alloca c.m (Bir.Ptr (Bir.Private, ty)) in
       let v = expr c init in
       stamp c e.exp_loc;
       Bir.store c.m ty v p;
       Cell (p, ty)
     | _ -> reject e.exp_loc "ref takes one argument")
  | Texp_apply (f, args) when callee_is f "Kernel.shared" ->
    (match given args with
     | [n] ->
       (match n.exp_desc with
        | Texp_constant (Asttypes.Const_int k) when k > 0 ->
          (* The element type comes from how the array is used, not from here. *)
          let ety = elem e.exp_loc (ty_of e.exp_loc e.exp_type) in
          stamp c e.exp_loc;
          Shared (Bir.shared_alloc c.m (Bir.Ptr (Bir.Shared, Bir.Arr (k, ety))))
        | _ -> reject n.exp_loc "shared takes a positive literal size")
     | _ -> reject e.exp_loc "shared takes one argument")
  | _ -> Val (expr c e)

(* ---- Statements ---- *)

let rec stmt c e =
  match e.exp_desc with
  | Texp_sequence (a, b) -> stmt c a; stmt c b
  | Texp_let (Asttypes.Nonrecursive, vbs, body) -> binds c vbs; stmt c body
  | Texp_ifthenelse (cond, t, els) -> ifthen c e.exp_loc cond t els
  | Texp_apply (f, args) -> effect_ c e.exp_loc f args
  | Texp_construct (_, _, []) when is_unit e.exp_type -> ()
  | _ -> reject e.exp_loc "this statement is not in the kernel subset"

and ifthen c loc cond t els =
  let v = expr c cond in
  let tag = fresh_tag c in
  let bt = Bir.block c.m (tag "if.then") in
  let bf = match els with
    | None -> None
    | Some _ -> Some (Bir.block c.m (tag "if.else"))
  in
  let be = Bir.block c.m (tag "if.end") in
  stamp c loc;
  Bir.br_cond c.m v bt (match bf with Some b -> b | None -> be) ~merge:be;

  Bir.entry c.m bt;
  stmt c t;
  Bir.br c.m be;

  (match els, bf with
   | Some e, Some b ->
     Bir.entry c.m b;
     stmt c e;
     Bir.br c.m be
   | _ -> ());

  Bir.entry c.m be

(* The counter is a stack slot for the same reason a ref is. *)
and lower_loop c loc lo hi body =
  let i32 = Bir.Int 32 in
  let lov = expr c lo in
  let hiv = expr c hi in
  stamp c loc;
  let cnt = Bir.alloca c.m (Bir.Ptr (Bir.Private, i32)) in
  Bir.store c.m i32 lov cnt;

  let tag = fresh_tag c in
  let bcond = Bir.block c.m (tag "loop.cond") in
  let bbody = Bir.block c.m (tag "loop.body") in
  let bend  = Bir.block c.m (tag "loop.end") in
  Bir.br c.m bcond;

  Bir.entry c.m bcond;
  let iv = Bir.load c.m i32 cnt in
  let go = Bir.icmp c.m Bir.Slt i32 iv hiv in
  Bir.br_cond c.m go bbody bend ~merge:bend;

  Bir.entry c.m bbody;
  let (pid, inner) =
    match body.exp_desc with
    | Texp_function ([p], Tfunction_body b) ->
      (match p.fp_kind with
       | Tparam_pat pat ->
         (match pat.pat_desc with
          | Tpat_var (id, _, _) -> (Some id, b)
          | Tpat_any -> (None, b)     (* fun _ -> ignores the index *)
          | _ -> reject pat.pat_loc "the loop index must be a plain name")
       | Tparam_optional_default (pat, _) ->
         reject pat.pat_loc "the loop index may not be optional")
    | _ -> reject body.exp_loc "the loop body must be written as a fun here"
  in
  let saved = c.vars in
  (match pid with Some id -> bind c id (Val iv) | None -> ());
  stmt c inner;
  c.vars <- saved;
  stamp c loc;
  let nx = Bir.add c.m i32 iv (Bir.const_int 1) in
  Bir.store c.m i32 nx cnt;
  Bir.br c.m bcond;

  Bir.entry c.m bend

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

  | "Kernel.sset", [arr; idx; v] ->
    let base = sbase c loc arr in
    let i = expr c idx in
    let pty = ty_of arr.exp_loc arr.exp_type in
    stamp c loc;
    let p = Bir.gep c.m pty base i in
    let value = expr c v in
    stamp c loc;
    Bir.store c.m (elem loc pty) value p

  | "Kernel.barrier", [_] -> stamp c loc; Bir.barrier c.m

  | "Kernel.loop", [lo; hi; body] -> lower_loop c loc lo hi body

  | "Stdlib.:=", [r; v] ->
    let (p, ty) = cell c loc r in
    let x = expr c v in
    stamp c loc;
    Bir.store c.m ty x p

  | n, _ -> reject loc "%s is not a kernel statement" n

(* ---- Functions ---- *)

let params_of ps =
  List.map
    (fun p ->
       match p.fp_kind with
       | Tparam_pat pat -> pat
       | Tparam_optional_default (pat, _) ->
         reject pat.pat_loc "an optional argument is not a kernel parameter")
    ps

(* A device function returns a value, so its body is an expression and Booth
   inlines it for the backends with no calling convention. *)
let device c name ps body rty =
  c.vars <- [];
  c.nblk <- 0;
  let pats = params_of ps in
  let tys = List.map (fun p -> ty_of p.pat_loc p.pat_type) pats in
  let vals = Bir.func c.m name ~params:tys ~kernel:false in
  List.iter2
    (fun p v ->
       match p.pat_desc with
       | Tpat_var (id, _, _) -> bind c id (Val v)
       | _ -> reject p.pat_loc "a device parameter must be a plain name")
    pats vals;
  let entry = Bir.block c.m "entry" in
  Bir.entry c.m entry;
  let v = expr c body in
  Bir.retv c.m rty v;
  Bir.finish c.m;
  c.devs <- (name, rty) :: c.devs

let kernel c name ps body =
  c.vars <- [];
  c.nblk <- 0;
  let pats = params_of ps in
  let tys = List.map (fun p -> ty_of p.pat_loc p.pat_type) pats in
  let vals = Bir.func c.m name ~params:tys ~kernel:true in
  List.iter2
    (fun p v ->
       match p.pat_desc with
       | Tpat_var (id, _, _) -> bind c id (Val v)
       | _ -> reject p.pat_loc "a kernel parameter must be a plain name")
    pats vals;
  let entry = Bir.block c.m "entry" in
  Bir.entry c.m entry;
  stmt c body;
  Bir.ret c.m;
  Bir.finish c.m

(* let[@device] f x = ... is a device function; anything else is a kernel. *)
let is_device attrs =
  List.exists
    (fun (a : Parsetree.attribute) -> a.Parsetree.attr_name.Location.txt = "device")
    attrs

let structure c st =
  List.iter
    (fun it ->
       match it.str_desc with
       | Tstr_value (Asttypes.Nonrecursive, vbs) ->
         List.iter
           (fun vb ->
              match vb.vb_pat.pat_desc, vb.vb_expr.exp_desc with
              | Tpat_var (id, _, _), Texp_function (ps, Tfunction_body body) ->
                if is_device vb.vb_attributes then
                  device c (Ident.name id) ps body
                    (ty_of body.exp_loc body.exp_type)
                else
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
  let c = { m = Bir.create (); vars = []; nblk = 0; devs = [] } in
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
