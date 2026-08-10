#!/bin/sh
# Same input twice, same bytes out. Separate processes, so ASLR and any
# pointer-ordered container shows up as a mismatch.
#
# Each fixture may carry a sidecar tests/NAME.opt listing the modes it is known
# not to survive, one "xfail MODE reason" per line. That is z390 keeping
# TESTDCB1.OPT next to TESTDCB1.MLC rather than parking a table of special
# cases off in the runner. This script used to read any refusal as a skip and
# say nothing about it, so eight of them sat quiet, and a backend that started
# refusing every fixture would still have come out green.
set -u

root=$(git rev-parse --show-toplevel)
cd "$root"

kath=./kath.exe
[ -x "$kath" ] || kath=./kath
[ -x "$kath" ] || { echo "reprocheck: no kath binary, run make first" >&2; exit 2; }

modes="--amdgpu --nvidia-ptx --ir"
fail=0
checked=0
declared=0

EMPTY=$(printf '' | sha256sum | cut -d' ' -f1)

# Prints the declared reason when fixture $1 is xfail for mode $2, nothing
# otherwise. Always prints something on a match, so callers can test for it.
xfail_text() {
  opt="${1%.cu}.opt"
  [ -f "$opt" ] || return 0
  awk -v m="$2" '
    { sub(/#.*/, "") }
    $1 == "xfail" && ($2 == "all" || $2 == m) {
      $1 = ""; $2 = ""; sub(/^ +/, "")
      print ($0 == "" ? "(no reason given)" : $0)
      exit
    }' "$opt"
}

for f in tests/*.cu; do
  for m in $modes; do
    why=$(xfail_text "$f" "$m")

    if [ -n "$why" ]; then
      # Declared broken. Confirm it still is, so a fix does not slip past.
      if "$kath" $m "$f" >/dev/null 2>&1; then
        fail=$((fail + 1))
        echo "XPASS: $m $f succeeds now, drop its xfail from ${f%.cu}.opt" >&2
      else
        declared=$((declared + 1))
      fi
      continue
    fi

    a=$("$kath" $m "$f" 2>/dev/null | sha256sum 2>/dev/null | cut -d' ' -f1)
    if [ -z "$a" ] || [ "$a" = "$EMPTY" ]; then
      fail=$((fail + 1))
      echo "NO OUTPUT: $m $f emitted nothing and declares no xfail" >&2
      echo "  add one to ${f%.cu}.opt if that is deliberate" >&2
      continue
    fi

    b=$("$kath" $m "$f" 2>/dev/null | sha256sum 2>/dev/null | cut -d' ' -f1)
    checked=$((checked + 1))
    if [ "$a" != "$b" ]; then
      fail=$((fail + 1))
      echo "NOT REPRODUCIBLE: $m $f" >&2
      echo "  $a" >&2
      echo "  $b" >&2
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "reprocheck: $fail problem(s) alongside $checked reproducible runs" >&2
  exit 1
fi

echo "reprocheck: $checked runs reproducible, $declared declared xfail"
