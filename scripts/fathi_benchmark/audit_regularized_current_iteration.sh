#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <parent-iteration-k>" >&2
  exit 2
fi

K="$1"
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

python -u -m scripts.fathi_benchmark.audit_regularized_current_iteration \
  --repo "$SOURCE" \
  --parent-iteration "$K"
