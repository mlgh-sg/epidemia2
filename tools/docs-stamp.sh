#!/usr/bin/env bash
# Are the published tutorials still the ones the current code produces?
#
# The tutorials ship PRE-BAKED: the fitted numbers and figures are committed so
# the docs site builds without CmdStan. That is fast, but it means the published
# results can drift silently away from what the code now produces -- which is
# exactly what happened when the inf2death kernel was corrected and three
# tutorials kept showing pre-correction numbers.
#
# Re-fitting to find out is expensive (hours). Hashing the INPUTS is not. This
# records a fingerprint of everything that can change a fitted number, so
# "are the docs stale?" is answerable in about a second, anywhere, including CI.
#
#   tools/docs-stamp.sh compute r|python    print the current fingerprint
#   tools/docs-stamp.sh write   r|python    record it (do this after baking)
#   tools/docs-stamp.sh check   r|python    exit 1 if the docs are stale
#   tools/docs-stamp.sh check   all
#
# The signal is deliberately CONSERVATIVE: any change under R/, inst/stan/ or
# data/ marks the docs stale, even one that cannot move a number. A false
# "stale" costs a re-bake; a false "fresh" ships wrong numbers.
#
# Only git-TRACKED files are hashed, so build artefacts, caches and mtimes are
# irrelevant and the fingerprint is identical on every machine.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAMP="$ROOT/.docs-stamp"

# Inputs that can change a fitted number. Kept explicit rather than "everything"
# so that editing README or tests does not spuriously invalidate the docs.
r_paths() {
  git -C "$ROOT" ls-files \
    'R/*.R' 'inst/stan/*' 'data/*' \
    'vignettes/*.Rmd.orig' 'vignettes/precompute.R' 'DESCRIPTION'
}

python_paths() {
  git -C "$ROOT" ls-files \
    'python/src/epidemia/*' 'python/notebooks/*.py' \
    'python/scripts/precompute.py' 'python/pyproject.toml'
}

compute() {
  local side=$1 files
  files=$("${side}_paths")
  [ -n "$files" ] || { echo "no input files matched for '$side'" >&2; exit 2; }
  # hash file CONTENTS, in a stable order, with the path included so a rename
  # counts as a change
  ( cd "$ROOT" && printf '%s\n' "$files" | LC_ALL=C sort | while read -r f; do
      printf '%s  %s\n' "$(git hash-object "$f")" "$f"
    done ) | shasum -a 256 | cut -c1-16
}

read_stamp() {   # read_stamp <side>; empty if absent
  [ -f "$STAMP" ] || return 0
  awk -v k="$1" '$1 == k { print $2 }' "$STAMP"
}

write_stamp() {
  local side=$1 val
  val=$(compute "$side")
  local tmp="$STAMP.tmp"
  : > "$tmp"
  [ -f "$STAMP" ] && awk -v k="$side" '$1 != k' "$STAMP" >> "$tmp"
  printf '%s %s\n' "$side" "$val" >> "$tmp"
  LC_ALL=C sort -o "$tmp" "$tmp"
  mv "$tmp" "$STAMP"
  echo "recorded $side docs stamp: $val"
}

check_one() {
  local side=$1 now was
  now=$(compute "$side")
  was=$(read_stamp "$side")
  if [ -z "$was" ]; then
    echo "STALE  $side  (no stamp recorded yet; current $now)"
    return 1
  fi
  if [ "$now" != "$was" ]; then
    echo "STALE  $side  (recorded $was, now $now)"
    return 1
  fi
  echo "fresh  $side  ($now)"
}

cmd=${1:-}; side=${2:-all}
case "$cmd" in
  compute) compute "$side" ;;
  write)
    if [ "$side" = all ]; then write_stamp r; write_stamp python
    else write_stamp "$side"; fi ;;
  check)
    rc=0
    if [ "$side" = all ]; then
      check_one r || rc=1
      check_one python || rc=1
    else
      check_one "$side" || rc=1
    fi
    if [ "$rc" -ne 0 ]; then
      cat >&2 <<'MSG'

The published tutorials were baked from different inputs than the current tree,
so their numbers may no longer be what this code produces.

Re-bake LOCALLY -- these are real model fits and are far better suited to your
machine than to a 4-core CI runner:

    make tutorials-clean          # R vignettes  (drops the knitr cache first)
    make tutorials-python         # Python notebooks
    make docs-stamp               # record the new fingerprint
    git add -A && git commit

MSG
    fi
    exit "$rc" ;;
  *)
    sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 2 ;;
esac
