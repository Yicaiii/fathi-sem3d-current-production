#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <parent-iteration-k> [extra run_current_iteration.py options...]" >&2
  exit 2
fi

K="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${FATHI_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME="${FATHI_RUNTIME_ROOT:-$SOURCE}"

SOURCE="$(cd "$SOURCE" && pwd)"
RUNTIME="$(cd "$RUNTIME" && pwd)"

cd "$SOURCE"

export FATHI_BENCHMARK_ROOT="$SOURCE"
export FATHI_RUNTIME_ROOT="$RUNTIME"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

RUN="fathi_s43_repro_p20_t052"
CHILD=$((K + 1))
TRANS="$(printf 'iter_%03d_to_iter_%03d' "$K" "$CHILD")"
LOG="$RUNTIME/results/$RUN/$TRANS/current_iteration_driver.log"

mkdir -p "$(dirname "$LOG")"

printf \
'SOURCE=%s\nRUNTIME=%s\nRUN=%s\nPARENT_ITERATION=%s\nTRANSITION=%s\nLOG=%s\n' \
"$SOURCE" "$RUNTIME" "$RUN" "$K" "$TRANS" "$LOG"

python -u -m scripts.fathi_benchmark.run_current_iteration \
  --repo "$SOURCE" \
  --parent-iteration "$K" \
  "$@" 2>&1 | tee -a "$LOG"

exit "${PIPESTATUS[0]}"
