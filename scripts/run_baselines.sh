#!/bin/bash
# Train, mesh and evaluate every method that has a config for a dataset.
#
#   bash scripts/run_baselines.sh <dataset> [gpu] [method ...]
#   bash scripts/run_baselines.sh truck                 # base, ours and all baselines
#   bash scripts/run_baselines.sh truck 1 regsdf digs   # just these two
#
# Set SKIP_EVAL=1 to only train. A failed method is reported and the loop
# moves on to the next one.
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
DATASET=$1; GPU=${2:-${CUDA_VISIBLE_DEVICES:-0}}; shift; [ $# -ge 1 ] && shift
LIST=${*:-$METHODS}
DIR="$REPO_ROOT/$(config_dir "$DATASET")" || exit 1

mkdir -p "$REPO_ROOT/experiments/logs"
failed=""
for m in $LIST; do
    if [ ! -f "$DIR/$m.yaml" ]; then
        echo "[nieo] skip $m: no config for $DATASET"
        continue
    fi
    log="$REPO_ROOT/experiments/logs/${DATASET}_${m}.log"
    echo "[nieo] === $DATASET / $m  (log: $log)"
    if bash "$REPO_ROOT/scripts/train.sh" "$DATASET" "$m" "$GPU" > "$log" 2>&1 \
       && { [ -n "$SKIP_EVAL" ] || bash "$REPO_ROOT/scripts/evaluate.sh" "$DATASET" "$m" "" "$GPU" >> "$log" 2>&1; }; then
        echo "[nieo]     done"
    else
        echo "[nieo]     FAILED (see $log)"
        failed="$failed $m"
    fi
done

[ -z "$SKIP_EVAL" ] && echo "[nieo] metrics: experiments/evaluation/"
[ -z "$failed" ] || { echo "[nieo] failed:$failed"; exit 1; }
