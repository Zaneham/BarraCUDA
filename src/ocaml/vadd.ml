(* vadd, the kernel examples/cmake/vadd.cu compiles, written in OCaml.
 *
 *   vadd out.bir            write the BIR
 *   vadd out.bir out.o      and compile it for the CPU backend work
 *)

let kath = try Sys.getenv "KATH" with Not_found -> "./kath"

let build () =
  let m = Bir.create () in
  let f32 = Bir.Float 32 in
  let i32 = Bir.Int 32 in
  let pf32 = Bir.Ptr (Bir.Global, f32) in

  let params = [ pf32; pf32; pf32; i32 ] in
  match Bir.func m "vadd" ~params ~kernel:true with
  | [ a; b; c; n ] ->
    (* Reserved up front: the branch below names blocks with no body yet. *)
    let entry = Bir.block m "entry" in
    let then_ = Bir.block m "if.then" in
    let endb  = Bir.block m "if.end" in

    Bir.line m 3;
    Bir.entry m entry;
    (* One binding each; nesting would leave the order unspecified. *)
    let bid  = Bir.block_id m Bir.X in
    let bdim = Bir.block_dim m Bir.X in
    let prod = Bir.mul m i32 bid bdim in
    let tid  = Bir.thread_id m Bir.X in
    let i    = Bir.add m i32 prod tid in

    Bir.line m 4;
    let cond = Bir.icmp m Bir.Slt i32 i n in
    Bir.br_cond m cond then_ endb ~merge:endb;

    Bir.entry m then_;
    let pc = Bir.gep m pf32 c i in
    let pa = Bir.gep m pf32 a i in
    let va = Bir.load m f32 pa in
    let pb = Bir.gep m pf32 b i in
    let vb = Bir.load m f32 pb in
    let s  = Bir.fadd m f32 va vb in
    Bir.store m f32 s pc;
    Bir.br m endb;

    Bir.entry m endb;
    Bir.ret m;

    Bir.finish m;
    m
  | _ -> failwith "vadd: parameter count changed"

let () =
  let bir_path = if Array.length Sys.argv > 1 then Sys.argv.(1) else "vadd.bir" in
  let m = build () in
  Bir.write m bir_path;
  Printf.printf "wrote %s\n%!" bir_path;

  if Array.length Sys.argv > 2 then begin
    let obj = Sys.argv.(2) in
    let cmd = Printf.sprintf "%s --bir-in --cpu %s -o %s" kath bir_path obj in
    Printf.printf "%s\n%!" cmd;
    exit (Sys.command cmd)
  end
