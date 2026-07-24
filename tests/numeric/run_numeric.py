#!/usr/bin/env python3
"""Numeric regression CI for Booth, checked against SLATEC known-good values.

Chain per kernel:  .f90  --lfortran-->  .cu  --kath-->  backend  -->  run  -->  value

Two axes decide pass/fail, because GPU IEEE float is not CPU IEEE float and a
mismatch between them is expected, not a bug:

  accuracy   worst output element vs the known-good value. Loose ceiling — this
             only catches a kernel that is actually broken (the saxpy tail bug
             read 3.0 where 5.0 was due: 40% off, caught).
  regression this run's mean vs the committed baseline for the SAME backend.
             Like-for-like, so it factors the float differences out and catches
             codegen drift. This is the real gate.

A backend with no baseline entry yet is recorded, not failed. Bless new numbers
with --update-baseline once you have eyeballed them.

Env: KATH (kath binary), LFORTRAN (lfortran binary), TINYGRAD_PATH (for the emu).
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path

HERE     = Path(__file__).resolve().parent
BOOTH    = HERE.parent.parent
KATH     = os.environ.get("KATH", str(BOOTH / "kath"))
LFORTRAN = os.environ.get("LFORTRAN", "lfortran")

GATING = ("cpu", "rdna3", "rdna4")   # fail the build; others record only

CTYPE = {"f32": "float", "i32": "int"}


# ---- Frontend: Fortran -> stripped CUDA ----

def fortran_to_cu(src, workdir):
    """Run LFortran's do-concurrent offload, return the kernel .cu text with the
    CUDA-runtime registrar stripped (kath parses kernels, not host glue)."""
    obj = workdir / "k.o"
    subprocess.run([LFORTRAN, "--gpu=cuda", "-c", str(src), "-o", str(obj)],
                   cwd=workdir, capture_output=True, text=True)
    sidecar = Path(str(obj) + ".cuda.cu")
    if not sidecar.exists():
        raise RuntimeError(f"lfortran emitted no kernel for {src} "
                           f"(offload did not fire?)")
    text = sidecar.read_text()
    cut = text.find("// Auto-generated kernel registration")
    return text[:cut] if cut >= 0 else text


# ---- CPU backend ----

def run_cpu(cu_path, k, n, workdir):
    """kath --cpu -> object, link a generated harness, run, read the outputs.
    The CPU kernel takes a hidden trailing nthreads; one call is one block, so
    nthreads = n and there is no ragged edge to lose."""
    obj = workdir / "cpu.o"
    r = subprocess.run([KATH, "--cpu", "-o", str(obj), str(cu_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kath --cpu failed: {r.stderr.strip()}")

    decl_args = [f"{CTYPE[a['type']]}{' *' if a['kind']=='buffer' else ' '}"
                 for a in k["args"]] + ["int"]          # trailing nthreads
    call, seeds, outs = [], [], []
    for i, a in enumerate(k["args"]):
        if a["kind"] == "buffer":
            seeds.append(f"  {CTYPE[a['type']]} *b{i} = "
                         f"malloc((size_t)n*sizeof({CTYPE[a['type']]}));")
            seeds.append(f"  for (int j=0;j<n;j++) b{i}[j] = "
                         f"{float(a.get('init',0.0))};")
            call.append(f"b{i}")
            if a["role"] in ("out", "inout"):
                outs.append(i)
        else:
            v = n if a["value"] == "@n" else a["value"]
            call.append(f"({CTYPE[a['type']]})({v})")
    call.append("n")                                    # nthreads

    if len(outs) != 1:
        raise RuntimeError("harness supports exactly one output buffer for now")
    outi = outs[0]

    harness = workdir / "h.c"
    harness.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n"
        f"extern void {k['entry']}({', '.join(decl_args)});\n"
        "int main(int argc, char **argv){\n"
        "  int n = atoi(argv[1]);\n"
        + "\n".join(seeds) + "\n"
        f"  {k['entry']}({', '.join(call)});\n"
        f"  for (int j=0;j<n;j++) printf(\"%.9g\\n\", (double)b{outi}[j]);\n"
        "  return 0;\n}\n")

    exe = workdir / "h.exe"
    r = subprocess.run(["gcc", "-no-pie", "-O2", str(harness), str(obj),
                        "-o", str(exe), "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpu harness link failed: {r.stderr.strip()}")
    r = subprocess.run([str(exe), str(n)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpu harness run failed: {r.stderr.strip()}")
    return [float(x) for x in r.stdout.split()]


# ---- RDNA emulator backend ----

def run_rdna(cu_path, k, n, workdir, gfx):
    """kath --amdgpu-bin -> hsaco, execute on tinygrad's mockgpu, read outputs."""
    sys.path.insert(0, str(BOOTH / "tests" / "emu"))
    import ctypes, struct
    from run_emu import lo_mem, prs_elf, prs_kd, prs_arch
    from test.mockgpu.amd.emu import run_asm

    hsaco = workdir / f"{gfx}.hsaco"
    cmd = [KATH, "--amdgpu-bin", "-o", str(hsaco), str(cu_path)]
    if gfx == "rdna4":
        cmd.insert(1, "--gfx1200")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kath {gfx} failed: {r.stderr.strip()}")

    data = hsaco.read_bytes()
    arch = prs_arch(data)
    txt = prs_elf(data)
    _rsrc1, rsrc2, _ka, _lds, scrsz, kprop = prs_kd(txt)
    code = txt[256:]
    caddr = lo_mem(len(code))
    ctypes.memmove(caddr, code, len(code))

    bksz  = 64
    ngrps = (n + bksz - 1) // bksz

    # One arena for all buffers plus the 64-byte kernarg block.
    bufsz = n * 4
    nbuf  = sum(1 for a in k["args"] if a["kind"] == "buffer")
    base  = lo_mem(bufsz * nbuf + 128)
    kaoff = base + bufsz * nbuf

    karg = (ctypes.c_uint8 * 64).from_address(kaoff)
    slot, outaddr, bi = 0, None, 0
    for a in k["args"]:
        if a["kind"] == "buffer":
            addr = base + bufsz * bi
            buf = (ctypes.c_float * n).from_address(addr)
            for j in range(n):
                buf[j] = float(a.get("init", 0.0))
            struct.pack_into("<Q", karg, slot, addr)
            if a["role"] in ("out", "inout"):
                outaddr = addr
            bi += 1
        elif a["type"] == "f32":
            v = n if a["value"] == "@n" else a["value"]
            struct.pack_into("<f", karg, slot, float(v))
        else:
            v = n if a["value"] == "@n" else a["value"]
            struct.pack_into("<i", karg, slot, int(v))
        slot += 8                                        # 8-byte param slots

    struct.pack_into("<I", karg, slot,      ngrps)       # hidden block_count
    struct.pack_into("<I", karg, slot + 4,  1)
    struct.pack_into("<I", karg, slot + 8,  1)
    struct.pack_into("<H", karg, slot + 12, bksz)        # hidden group_size
    struct.pack_into("<H", karg, slot + 14, 1)
    struct.pack_into("<H", karg, slot + 16, 1)

    usr = []
    if kprop & (1 << 1):
        usr += [0, 0]
    if kprop & (1 << 3):
        usr += [kaoff & 0xFFFFFFFF, (kaoff >> 32) & 0xFFFFFFFF]

    rc = run_asm(caddr, len(code), ngrps, 1, 1, bksz, 1, 1,
                 kaoff, rsrc2, scrsz, arch, usr)
    if rc != 0:
        raise RuntimeError(f"{gfx} emulator rc={rc}")
    out = (ctypes.c_float * n).from_address(outaddr)
    return [out[j] for j in range(n)]


BACKENDS = {
    "cpu":   lambda cu, k, n, w: run_cpu(cu, k, n, w),
    "rdna3": lambda cu, k, n, w: run_rdna(cu, k, n, w, "rdna3"),
    "rdna4": lambda cu, k, n, w: run_rdna(cu, k, n, w, "rdna4"),
}


# ---- Scoring ----

def score(vals, n, expect):
    """Mean over the in-range elements (the baseline number) and the worst
    element's relative error against the known-good value (the accuracy axis)."""
    live = vals[:n]
    mean = sum(live) / len(live)
    denom = abs(expect) if expect != 0 else 1.0
    worst = max(abs(v - expect) for v in live) / denom
    return mean, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="cpu,rdna3,rdna4")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--kernel", help="run only this kernel")
    args = ap.parse_args()

    man = json.loads((HERE / "manifest.json").read_text())
    cfg = man["config"]
    acc_ceiling = cfg["accuracy_ceiling_rel"]
    reg_gate    = cfg["regression_gate_rel"]

    bl_path = HERE / "baseline.json"
    baseline = json.loads(bl_path.read_text()) if bl_path.exists() else {}

    want = args.backends.split(",")
    failures, measured = [], {}

    for k in man["kernels"]:
        if args.kernel and k["name"] != args.kernel:
            continue
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td)
            try:
                cu = wd / "kern.cu"
                cu.write_text(fortran_to_cu(HERE / k["source"], wd))
            except Exception as e:
                print(f"FAIL {k['name']}: frontend: {e}")
                failures.append(k["name"])
                continue

            for be in want:
                if be not in BACKENDS:
                    print(f"  skip {be}: no runner")
                    continue
                for case in k["cases"]:
                    n, expect = case["n"], case["expect"]
                    key = f"{k['name']}/n={n}"
                    try:
                        vals = BACKENDS[be](cu, k, n, wd)
                    except Exception as e:
                        tag = "FAIL" if be in GATING else "warn"
                        print(f"  {tag} {be} {key}: {e}")
                        if be in GATING:
                            failures.append(f"{be} {key}")
                        continue

                    mean, worst = score(vals, n, expect)
                    measured.setdefault(be, {})[key] = round(mean, 9)

                    verdict, notes = "ok", []
                    if worst > acc_ceiling:
                        notes.append(f"accuracy {worst*100:.1f}% > "
                                     f"{acc_ceiling*100:.0f}%")
                        if be in GATING:
                            verdict = "FAIL"
                    prev = baseline.get(be, {}).get(key)
                    if prev is None:
                        notes.append("no baseline (recorded)")
                    else:
                        denom = abs(prev) if prev else 1.0
                        drift = abs(mean - prev) / denom
                        if drift > reg_gate:
                            notes.append(f"regression {drift*100:.1f}% vs "
                                         f"baseline {prev:g}")
                            if be in GATING:
                                verdict = "FAIL"
                    if verdict == "FAIL":
                        failures.append(f"{be} {key}")
                    tail = ("  [" + "; ".join(notes) + "]") if notes else ""
                    print(f"  {verdict:4s} {be:6s} {key}: "
                          f"mean={mean:.6g} worst_err={worst*100:.2f}%{tail}")

    if args.update_baseline:
        merged = baseline
        for be, d in measured.items():
            merged.setdefault(be, {}).update(d)
        bl_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"\nbaseline updated: {bl_path}")

    if failures:
        print(f"\n{len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print("\nall gating checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
