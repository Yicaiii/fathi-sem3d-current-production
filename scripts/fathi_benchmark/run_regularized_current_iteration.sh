#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <parent-iteration-k> [extra Python driver options...]" >&2
  exit 2
fi

K="$1"
shift

if [ "$K" -ne 0 ]; then
  echo "BLOCK_GATE4C_SCOPE: only regularized iter000 -> iter001 is authorized" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${FATHI_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME="${FATHI_RUNTIME_ROOT:-$SOURCE}"

SOURCE="$(cd "$SOURCE" && pwd)"
RUNTIME="$(cd "$RUNTIME" && pwd)"

cd "$SOURCE"

if [ "$(git rev-parse --show-toplevel)" != "$SOURCE" ]; then
  echo "BLOCK_SOURCE_GUARD: git top-level mismatch" >&2
  exit 3
fi

git merge-base --is-ancestor 917c721 HEAD || {
  echo "BLOCK_SOURCE_GUARD: HEAD is not descended from 917c721" >&2
  exit 3
}

export FATHI_BENCHMARK_ROOT="$SOURCE"
export FATHI_RUNTIME_ROOT="$RUNTIME"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

RUN="fathi_s43_repro_tv_p20_t052"
CHILD=$((K + 1))
TRANS="$(printf 'iter_%03d_to_iter_%03d' "$K" "$CHILD")"
LOG="$RUNTIME/results/$RUN/$TRANS/regularized_iteration_driver.log"

mkdir -p "$(dirname "$LOG")"

printf \
'SOURCE=%s\nRUNTIME=%s\nRUN=%s\nPARENT_ITERATION=%s\nTRANSITION=%s\nLOG=%s\n' \
"$SOURCE" "$RUNTIME" "$RUN" "$K" "$TRANS" "$LOG"

python -u -m scripts.fathi_benchmark.run_regularized_current_iteration \
  --repo "$SOURCE" \
  --parent-iteration "$K" \
  "$@" 2>&1 | tee -a "$LOG"

exit "${PIPESTATUS[0]}"
