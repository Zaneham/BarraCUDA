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


# ---- Output plumbing ----
# A kernel may write more than one buffer (sswap swaps x and y). Every runner
# prints "argidx value" per element and returns {argidx: [values]}, so a
# single- and a two-output kernel travel the same path.

def out_indices(k):
    return [i for i, a in enumerate(k["args"])
            if a["kind"] == "buffer" and a["role"] in ("out", "inout")]


def parse_multi(text):
    """Collect "argidx value" lines into {argidx: [values]}. Anything that isn't
    that shape is skipped, so a runtime that chats on stdout (nv_rt's device
    banner) doesn't derail the parse."""
    res = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            idx, val = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        res.setdefault(idx, []).append(val)
    return res


def key_expect(k, case, argidx):
    """Baseline key and known-good value for one output of one case. Single-output
    kernels keep the flat kernel/n=N key; multi-output tag each by arg name."""
    n = case["n"]
    outs = out_indices(k)
    if len(outs) == 1:
        return f"{k['name']}/n={n}", case["expect"]
    a = k["args"][argidx]
    return f"{k['name']}/n={n}/{a['name']}", a["expect"]


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

    if not outs:
        raise RuntimeError("kernel has no output buffer")

    prints = "".join(
        f"  for (int j=0;j<n;j++) printf(\"%d %.9g\\n\", {i}, (double)b{i}[j]);\n"
        for i in outs)
    harness = workdir / "h.c"
    harness.write_text(
        "#include <stdio.h>\n#include <stdlib.h>\n"
        f"extern void {k['entry']}({', '.join(decl_args)});\n"
        "int main(int argc, char **argv){\n"
        "  int n = atoi(argv[1]);\n"
        + "\n".join(seeds) + "\n"
        f"  {k['entry']}({', '.join(call)});\n"
        + prints +
        "  return 0;\n}\n")

    exe = workdir / "h.exe"
    r = subprocess.run(["gcc", "-no-pie", "-O2", str(harness), str(obj),
                        "-o", str(exe), "-lm"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpu harness link failed: {r.stderr.strip()}")
    r = subprocess.run([str(exe), str(n)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpu harness run failed: {r.stderr.strip()}")
    return parse_multi(r.stdout)


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

    # One arena for all buffers plus the kernarg block. Each parameter takes an
    # 8-byte slot with 18 bytes of hidden block and group dims behind them, so
    # size it from the kernel; a fixed 64 runs off the end past five arguments.
    bufsz = n * 4
    nbuf  = sum(1 for a in k["args"] if a["kind"] == "buffer")
    kasz  = 8 * len(k["args"]) + 24
    base  = lo_mem(bufsz * nbuf + kasz)
    kaoff = base + bufsz * nbuf

    karg = (ctypes.c_uint8 * kasz).from_address(kaoff)
    slot, bi = 0, 0
    out_addr = {}
    for i, a in enumerate(k["args"]):
        if a["kind"] == "buffer":
            addr = base + bufsz * bi
            buf = (ctypes.c_float * n).from_address(addr)
            for j in range(n):
                buf[j] = float(a.get("init", 0.0))
            struct.pack_into("<Q", karg, slot, addr)
            if a["role"] in ("out", "inout"):
                out_addr[i] = addr
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
    res = {}
    for i, addr in out_addr.items():
        ob = (ctypes.c_float * n).from_address(addr)
        res[i] = [ob[j] for j in range(n)]
    return res


# ---- NVIDIA backend (real hardware, via nv_rt + CUDA driver) ----

def run_nvidia(cu_path, k, n, workdir):
    """kath --nvidia-ptx -> PTX, run it on the GPU through nv_rt. Needs a CUDA
    driver (works under WSL via the passthrough libcuda). Record-only: there is
    no free NVIDIA CI runner, so this fires on a self-hosted box behind a label."""
    ptx = workdir / "k.ptx"
    r = subprocess.run([KATH, "--nvidia-ptx", "-o", str(ptx), str(cu_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"kath --nvidia-ptx failed: {r.stderr.strip()}")

    nv_rt_o = workdir / "nv_rt.o"
    r = subprocess.run(["gcc", "-c", "-O2", "-D_POSIX_C_SOURCE=200809L",
                        "-I", str(BOOTH / "src" / "nvidia"),
                        str(BOOTH / "src" / "nvidia" / "nv_rt.c"),
                        "-o", str(nv_rt_o)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nv_rt build failed: {r.stderr.strip()}")

    seeds, args, outs = [], [], []
    for i, a in enumerate(k["args"]):
        if a["kind"] == "buffer":
            ct = CTYPE[a["type"]]
            seeds.append(f"  {ct} *h{i} = malloc((size_t)n*sizeof({ct}));")
            seeds.append(f"  for (int j=0;j<n;j++) h{i}[j] = {float(a.get('init',0.0))};")
            seeds.append(f"  CUdevptr d{i} = nv_rt_alloc(&dev, (size_t)n*sizeof({ct}));")
            seeds.append(f"  nv_rt_h2d(&dev, d{i}, h{i}, (size_t)n*sizeof({ct}));")
            args.append(f"&d{i}")
            if a["role"] in ("out", "inout"):
                outs.append(i)
        else:
            v = n if a["value"] == "@n" else a["value"]
            seeds.append(f"  {CTYPE[a['type']]} s{i} = ({CTYPE[a['type']]})({v});")
            args.append(f"&s{i}")
    if not outs:
        raise RuntimeError("kernel has no output buffer")

    readback = "".join(
        f"  nv_rt_d2h(&dev, h{i}, d{i}, (size_t)n*sizeof(*h{i}));\n"
        f"  for (int j=0;j<n;j++) printf(\"%d %.9g\\n\", {i}, (double)h{i}[j]);\n"
        for i in outs)
    harness = workdir / "nvh.c"
    harness.write_text(
        "#include \"nv_rt.h\"\n#include <stdio.h>\n#include <stdlib.h>\n"
        "int main(int argc, char **argv){\n"
        "  int n = atoi(argv[1]);\n"
        "  nv_dev_t dev; nv_kern_t kern;\n"
        "  if (nv_rt_init(&dev)) { fprintf(stderr,\"init\\n\"); return 2; }\n"
        f"  if (nv_rt_load(&dev, argv[2], \"{k['entry']}\", &kern)) "
        "{ fprintf(stderr,\"load\\n\"); return 3; }\n"
        + "\n".join(seeds) + "\n"
        f"  void *args[] = {{ {', '.join(args)} }};\n"
        "  int bs = 256, gs = (n + 255) / 256;\n"
        "  nv_rt_launch(&dev, &kern, (unsigned)gs,1,1, (unsigned)bs,1,1, 0, args);\n"
        "  nv_rt_sync(&dev);\n"
        + readback +
        "  return 0;\n}\n")

    exe = workdir / "nvh.exe"
    r = subprocess.run(["gcc", "-O2", "-I", str(BOOTH / "src" / "nvidia"),
                        str(harness), str(nv_rt_o), "-o", str(exe), "-ldl"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia harness link failed: {r.stderr.strip()}")
    r = subprocess.run([str(exe), str(n), str(ptx)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia run failed: {r.stderr.strip()}")
    return parse_multi(r.stdout)


BACKENDS = {
    "cpu":    lambda cu, k, n, w: run_cpu(cu, k, n, w),
    "rdna3":  lambda cu, k, n, w: run_rdna(cu, k, n, w, "rdna3"),
    "rdna4":  lambda cu, k, n, w: run_rdna(cu, k, n, w, "rdna4"),
    "nvidia": lambda cu, k, n, w: run_nvidia(cu, k, n, w),
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
                    n = case["n"]
                    try:
                        vals = BACKENDS[be](cu, k, n, wd)
                    except Exception as e:
                        tag = "FAIL" if be in GATING else "warn"
                        print(f"  {tag} {be} {k['name']}/n={n}: {e}")
                        if be in GATING:
                            failures.append(f"{be} {k['name']}/n={n}")
                        continue

                    for oi in out_indices(k):
                        key, expect = key_expect(k, case, oi)
                        mean, worst = score(vals[oi], n, expect)
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
