#!/bin/bash
# Extract a mesh from a trained run and score it: mesh metrics against the
# dataset's ground truth (eval/eval_run.py) and field metrics on the SDF itself
# (eval/field_metrics.py: Dirichlet / Hessian energy, shape-operator norm).
#
#   bash scripts/evaluate.sh <dataset> <method> [run_dir] [gpu]
#   bash scripts/evaluate.sh truck ours                 # newest truck_ours run
#   bash scripts/evaluate.sh ncd base experiments/ncd_base_2026-01-01_00-00-00
#
# Results: experiments/evaluation/*_<method>_eval.csv (mesh) and
#          experiments/evaluation/<dataset>_<method>_field_eval.csv (field)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 2 ] || { sed -n 2,11p "$0"; exit 1; }
DATASET=$1; METHOD=$2; RUN=${3:-}; GPU=${4:-${CUDA_VISIBLE_DEVICES:-0}}
cd "$REPO_ROOT"
[ -n "$RUN" ] || RUN=$(latest_run "$DATASET" "$METHOD")
[ -d "$RUN" ] || { echo "No run found for $DATASET/$METHOD; train it first or pass run_dir"; exit 1; }

CKPT=$(ls "$RUN"/model/model_iter_*.pth 2>/dev/null | grep -v '_decoders.pth' | sort -V | tail -1)
[ -n "$CKPT" ] || { echo "No checkpoint in $RUN/model"; exit 1; }
RES=$(mc_res "$DATASET")
OUT="$RUN/mesh"
export CUDA_VISIBLE_DEVICES=$GPU

echo "[nieo] meshing $CKPT at ${RES} m"
EXTRA=""
case "$DATASET" in
    # whole-map bounding-box marching cubes, tiled so it fits in memory
    ncd|maicity) EXTRA="--bbx_map_ply $RUN/map/pc_map_down.ply --tile_voxels 25e6" ;;
esac
"$PYTHON" eval/reconstruct_mesh.py --checkpoint "$CKPT" --mc_res_m "$RES" \
    --output_dir "$OUT" --tag final $EXTRA
MESH=$(ls -t "$OUT"/final_res*.ply | grep -v '_map.ply' | head -1)

echo "[nieo] mesh metrics for $MESH"
"$PYTHON" eval/eval_run.py --dataset "$(eval_key "$DATASET")" --method_name "$METHOD" \
    --pred_mesh "$MESH" $(eval_flags "$DATASET")

echo "[nieo] field metrics for $CKPT"
"$PYTHON" eval/field_metrics.py --checkpoint "$CKPT" \
    --dataset "$DATASET" --method_name "$METHOD"
