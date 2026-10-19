#!/bin/bash
# Convert the Redwood bedroom RGB-D sequence to per-frame point clouds, and
# bring the laser-scan ground truth into the camera trajectory's frame.
#
#   bash scripts/prepare_redwood.sh
#   REGISTER=1 bash scripts/prepare_redwood.sh   # re-estimate the GT alignment
#
# Redwood's GT scan and RGB-D trajectory are independent recordings with no
# published alignment. By default the transform used in the paper
# (dataset/redwood_bedroom_gt_registration.txt, 2.4 cm median frame-to-GT
# residual) is applied; REGISTER=1 re-runs FPFH-RANSAC + coarse-to-fine ICP
# (dataset/register_redwood_gt.py) instead, which is not deterministic.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"

BASE=data/redwood/bedroom
[ -f $BASE/bedroom.log ] || { echo "Missing $BASE; see: bash scripts/download_data.sh redwood"; exit 1; }

# .log poses carry an index line per matrix and are already in open3d's
# convention; empty intrinsics select the PrimeSense 640x480 fx=fy=525 model
"$PYTHON" dataset/rgbd_to_kitti_format.py \
    --output_root ${BASE}_kitti_format \
    --depth_img_folder $BASE/depth/ --rgb_img_folder $BASE/image/ \
    --intrinsic_file '' --pose_file $BASE/bedroom.log \
    --is_focal_file False --already_kitti_format_pose False \
    --pose_with_head True --flip_extrinsic False --vis_on False

# the loader pairs scans with poses by list position, so the counts must match
n_ply=$(ls ${BASE}_kitti_format/rgbd_ply | wc -l)
n_pose=$(wc -l < ${BASE}_kitti_format/poses.txt)
[ "$n_ply" -eq "$n_pose" ] || { echo "ERROR: $n_ply frames vs $n_pose poses"; exit 1; }

if [ -n "$REGISTER" ]; then
    "$PYTHON" dataset/register_redwood_gt.py --root_path data/redwood --sequence_name bedroom
else
    "$PYTHON" - <<'EOF'
import numpy as np, open3d as o3d
pc = o3d.io.read_point_cloud("data/redwood/gt_mesh/bedroom.ply")
pc.transform(np.loadtxt("dataset/redwood_bedroom_gt_registration.txt"))
o3d.io.write_point_cloud("data/redwood/gt_mesh/bedroom_registered.ply", pc)
print("wrote data/redwood/gt_mesh/bedroom_registered.ply")
EOF
fi
echo "Done. Train with: bash scripts/train.sh redwood ours"
