#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 PARENT_ITERATION" >&2
    exit 2
fi

K="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${FATHI_BENCHMARK_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNTIME="${FATHI_RUNTIME_ROOT:-$SOURCE}"

SOURCE="$(cd "$SOURCE" && pwd)"
RUNTIME="$(cd "$RUNTIME" && pwd)"

cd "$SOURCE"

export FATHI_BENCHMARK_ROOT="$SOURCE"
export FATHI_RUNTIME_ROOT="$RUNTIME"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"

printf \
'SOURCE=%s\nRUNTIME=%s\nPARENT_ITERATION=%s\n' \
"$SOURCE" "$RUNTIME" "$K"

exec python \
    "$SOURCE/scripts/fathi_benchmark/audit_current_iteration.py" \
    --repo "$RUNTIME" \
    --run "fathi_s43_repro_p20_t052" \
    --parent-iteration "$K"
