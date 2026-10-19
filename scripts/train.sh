#!/bin/bash
# Train one method on one dataset.
#
#   bash scripts/train.sh <dataset> <method> [gpu]
#   bash scripts/train.sh truck ours 0
#   bash scripts/train.sh ncd base
#
# datasets: ncd maicity truck ignatius caterpillar barn courthouse redwood
# methods:  base ours regsdf neuralssd gradhess singular_hessian digs siren diffcd stoch_precond
# Output lands in experiments/<dataset>_<method>_<timestamp>/.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || { sed -n 2,10p "$0"; exit 1; }
DATASET=$1; METHOD=$2; GPU=${3:-${CUDA_VISIBLE_DEVICES:-0}}

CFG="$(config_dir "$DATASET")/$METHOD.yaml"
cd "$REPO_ROOT"
if [ ! -f "$CFG" ]; then
    echo "No config $CFG. Available for $DATASET:"
    ls "$(config_dir "$DATASET")" | sed 's/\.yaml$//; s/^/  /'
    exit 1
fi

echo "[nieo] $DATASET / $METHOD on GPU $GPU  ($CFG)"
CUDA_VISIBLE_DEVICES=$GPU "$PYTHON" train.py "$CFG"
