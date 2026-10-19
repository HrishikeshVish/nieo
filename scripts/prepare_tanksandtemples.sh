#!/bin/bash
# Turn a Tanks-and-Temples Training scene (images + published camera
# trajectory) into per-frame point clouds via COLMAP dense MVS, in the
# KITTI-style layout the configs expect: data/tanksandtemples/<scene>_kitti_format/.
#
#   bash scripts/prepare_tanksandtemples.sh <scene> [gpu]
#   bash scripts/prepare_tanksandtemples.sh truck
#
# Expects data/tanksandtemples/<scene>/images/ and
# data/tanksandtemples/<scene>/gt/{<Scene>.ply,<Scene>.json,<Scene>_trans.txt,<Scene>_COLMAP_SfM.log}
# (scripts/download_data.sh tanksandtemples <scene>). Needs a CUDA COLMAP; the
# first call sets one up via scripts/tanksandtemples/setup_mvs_env.sh.
#
# Settings match the paper: fx = 1156 px for all five scenes (all were shot with
# the same body and 21 mm lens; T&T's generic 0.7*W guess is ~16% off and
# corrupts the depth maps), and Barn/Courthouse are subsampled to 250 frames.
# Check fx for a new scene with: python dataset/estimate_tandt_focal.py --scene <scene>
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ $# -ge 1 ] || { sed -n 2,17p "$0"; exit 1; }
SCENE=$1; export CUDA_VISIBLE_DEVICES=${2:-${CUDA_VISIBLE_DEVICES:-0}}
cd "$REPO_ROOT"

[ -f scripts/tanksandtemples/.mvs_backend ] || bash scripts/tanksandtemples/setup_mvs_env.sh

EXTRA=""
case "$SCENE" in barn|courthouse) EXTRA="--max_frames 250" ;; esac

"$PYTHON" dataset/tandt_to_kitti_format.py --scene "$SCENE" --fx 1156 $EXTRA
echo "Done -> data/tanksandtemples/${SCENE}_kitti_format. Train with: bash scripts/train.sh $SCENE ours"
