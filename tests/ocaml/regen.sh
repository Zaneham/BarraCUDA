#!/bin/sh
# Regenerate the golden modules after a deliberate change to a kernel or to
# the lowering. Read the diff before committing it.
set -e
cd "$(dirname "$0")/../.."
(cd src/ocaml && opam exec -- dune build)
K=src/ocaml/_build/default/kcomp.exe
O=src/ocaml/_build/default/.kernels.objs/byte
for k in vadd_k scale_k reduce_k clamp_k ops_k tile_k rng_k ints_k atom_k; do
    "$K" "$O/$k.cmt" -o "tests/ocaml/$k.bir"
done
