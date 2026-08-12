#!/bin/sh
# mutate.sh -- mutation harness
#
# Bends one line of Booth, rebuilds, and checks the suite notices. A test suite
# that passes on code you have deliberately broken is not testing that code,
# and nothing short of breaking it on purpose tells you which.
#
# The mutants live in tests/mutants.tbl, one row each, and this file holds no
# tests of its own. Adding a mutant is adding a row.
#
# Everything happens in a scratch copy. The working tree is never touched, so
# an interrupted run leaves nothing behind to clean up or explain.
#
#   sh tests/mutate.sh              check every mutant against its table row
#   sh tests/mutate.sh --discover   print what each mutant actually kills
#   sh tests/mutate.sh --only mlr01 one row
#
# Ported from Kahu's, which is the same harness. Four things differ, all of
# them because Booth is bigger: source paths are kept whole rather than
# basenamed, since src/ has five emit.c and three isel.c between the backends;
# objects live under build/<host>/ rather than beside their source; lang/ and
# runtime/ come along because the build and the tests want them; and kath is
# built as well as trunner, because most families shell out to the binary and
# rebuilding only the runner tests the compiler from the row before.

set -u

SCRATCH=".mutscratch"
TABLE="tests/mutants.tbl"
MODE="verify"
ONLY=""

# Families asserting properties of the harness rather than of Booth. No
# mutation to src/ can reach them, so they are excluded from the never-killed
# report instead of sitting in it forever.
EXEMPT="smk ord"

while [ $# -gt 0 ]; do
    case "$1" in
        --discover) MODE="discover" ;;
        --only)     shift; ONLY="$1" ;;
        *) echo "usage: mutate.sh [--discover] [--only ID]" >&2; exit 2 ;;
    esac
    shift
done

[ -f "$TABLE" ] || { echo "mutate: no $TABLE" >&2; exit 2; }

# ---- Scratch ----

cleanup() { rm -rf "$SCRATCH"; }
trap cleanup EXIT INT TERM

rm -rf "$SCRATCH"
mkdir -p "$SCRATCH"
cp -r src tests runtime lang Makefile "$SCRATCH/" 2>/dev/null
cp -r "$SCRATCH/src" "$SCRATCH/src.pristine"

echo "mutation harness"
echo "===================="

# ---- Baseline ----
# A suite that is already red tells you nothing about what a mutation did.

( cd "$SCRATCH" && make >/dev/null 2>&1 && make trunner >/dev/null 2>&1 ) || {
    echo "baseline: build failed, refusing to run" >&2; exit 1; }
BASE=$( cd "$SCRATCH" && ./trunner --all 2>/dev/null | grep -cE '[[:space:]]FAIL$' )
if [ "$BASE" -ne 0 ]; then
    echo "baseline: $BASE tests already failing, refusing to run" >&2
    exit 1
fi
echo "baseline green"
echo

# ---- Literal replacement ----
# awk with index/substr, so a pattern full of punctuation is matched as text
# rather than as a regex nobody meant to write.

apply() {
    awk -v f="$2" -v r="$3" '
    {
        while ((i = index($0, f)) > 0) {
            $0 = substr($0, 1, i-1) r substr($0, i + length(f))
            if (r == "") break
        }
        print
    }' "$1" > "$1.mut" && mv "$1.mut" "$1"
}

# src/foo/bar.c -> src.pristine/foo/bar.c. Kahu can basename this because its
# src/ is flat. Booth cannot.
pristine() { echo "$SCRATCH/src.pristine/${1#src/}"; }

# ---- Run ----

n_ok=0; n_survived=0; n_drift=0; n_broke=0; n_equiv=0
KILLED_ALL=""
PREV=""

# Put back whatever the last row bent. Restoring only the file about to be
# mutated leaves the previous one bent, and the kill sets then accumulate
# down the table until every mutant looks like it breaks everything.
restore_prev() {
    [ -n "$PREV" ] || return 0
    cp "$(pristine "$PREV")" "$SCRATCH/$PREV"
    rm -f "$SCRATCH"/build/*/"${PREV%.c}.o"
    PREV=""
}

while IFS='|' read -r id file find repl expect note; do
    # The table pads its columns for reading, so every field arrives with the
    # padding still attached. find and repl keep their internal spacing and
    # lose only the padding, since C lines are full of deliberate runs of it.
    #
    # Trim before deciding what to skip, or a comment line that happens to
    # start with a space stops looking like one and gets run as a row.
    id=$(echo "$id" | tr -d ' \t\r')
    case "$id" in ''|\#*) continue ;; esac
    file=$(echo "$file" | sed 's/^ *//;s/ *$//')
    [ -n "$file" ] || continue
    find=$(echo "$find" | sed 's/^ *//;s/ *$//')
    repl=$(echo "$repl" | sed 's/^ *//;s/ *$//')
    expect=$(echo "$expect" | tr -d ' ')
    note=$(echo "$note" | sed 's/^ *//;s/ *$//')

    [ -n "$ONLY" ] && [ "$ONLY" != "$id" ] && continue

    # Exactly one deliberate defect in the tree at a time, so a kill set names
    # this row's mutation and nothing else.
    restore_prev
    cp "$(pristine "$file")" "$SCRATCH/$file"

    # Ask whether the pattern is there rather than inferring it from the file
    # changing, because awk rewrites the last newline either way and that is
    # enough to make an unmatched row look applied.
    if ! grep -Fq "$find" "$SCRATCH/$file"; then
        printf '  %-6s %-9s %s\n' "$id" "NOMATCH" "$note"
        n_broke=$((n_broke + 1))
        continue
    fi
    apply "$SCRATCH/$file" "$find" "$repl"
    PREV="$file"

    # Drop the object rather than trusting mtimes. Copying the file and
    # rewriting it both land inside the same filesystem second, so make sees
    # nothing newer and skips it, and every row then quietly tests the binary
    # the row before it built. Both binaries go too, or the tests that shell
    # out to kath keep running the previous row's compiler.
    rm -f "$SCRATCH"/build/*/"${file%.c}.o" \
          "$SCRATCH/kath" "$SCRATCH/kath.exe" \
          "$SCRATCH/trunner" "$SCRATCH/trunner.exe"

    if ! ( cd "$SCRATCH" && make >/dev/null 2>&1 && make trunner >/dev/null 2>&1 ); then
        printf '  %-6s %-9s %s\n' "$id" "NOBUILD" "$note"
        n_broke=$((n_broke + 1))
        continue
    fi

    killed=$( cd "$SCRATCH" && ./trunner --all 2>/dev/null \
              | grep -E '[[:space:]]FAIL$' \
              | awk '{ print substr($1, 1, 3) }' | sort -u | tr '\n' ',' \
              | sed 's/,$//' )
    KILLED_ALL="$KILLED_ALL,$killed"

    if [ "$MODE" = "discover" ]; then
        printf '  %-6s %-28s %s\n' "$id" "${killed:-<nothing>}" "$note"
        continue
    fi

    if [ "$expect" = "EQUIV" ]; then
        if [ -z "$killed" ]; then
            printf '  %-6s %-9s %s\n' "$id" "EQUIV" "$note"
            n_equiv=$((n_equiv + 1))
        else
            printf '  %-6s %-9s %s\n' "$id" "NOTEQUIV" "$note"
            printf '         declared no observable change, killed: %s\n' "$killed"
            n_drift=$((n_drift + 1))
        fi
    elif [ -z "$killed" ]; then
        printf '  %-6s %-9s %s\n' "$id" "SURVIVED" "$note"
        printf '         nothing noticed; expected %s\n' "$expect"
        n_survived=$((n_survived + 1))
    elif [ "$killed" = "$expect" ]; then
        printf '  %-6s %-9s %s\n' "$id" "ok" "$note"
        n_ok=$((n_ok + 1))
    else
        printf '  %-6s %-9s %s\n' "$id" "DRIFT" "$note"
        printf '         expected %s\n         killed   %s\n' "$expect" "$killed"
        n_drift=$((n_drift + 1))
    fi
done < "$TABLE"

# ---- Never killed ----
# Coverage inverted. A family no mutation can break is either testing nothing
# or testing something there is no mutant for, and both are worth knowing.

if [ "$MODE" != "discover" ] && [ -z "$ONLY" ]; then
    echo
    echo "families no mutant reached:"
    fams=$( cd "$SCRATCH" && ./trunner --families 2>/dev/null | awk '{ print $1 }' )
    none=1
    for f in $fams; do
        case " $EXEMPT " in *" $f "*) continue ;; esac
        case "$KILLED_ALL" in *"$f"*) continue ;; esac
        echo "  $f"
        none=0
    done
    [ "$none" = 1 ] && echo "  (none)"
fi

if [ "$MODE" = "discover" ]; then
    echo
    echo "discover only, nothing checked"
    exit 0
fi

echo
echo "===================="
echo "$((n_ok + n_survived + n_drift + n_broke + n_equiv)) mutants: $n_ok ok, \
$n_survived survived, $n_drift drifted, $n_equiv equivalent, $n_broke unusable"

[ "$n_survived" -eq 0 ] && [ "$n_drift" -eq 0 ] && [ "$n_broke" -eq 0 ] && exit 0
exit 1
