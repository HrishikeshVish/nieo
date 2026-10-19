#!/bin/bash
# Download a dataset into data/.
#
#   bash scripts/download_data.sh ncd                # Newer College Quad subset + TLS ground truth
#   bash scripts/download_data.sh maicity            # MaiCity sequence 01 + ground truth
#   bash scripts/download_data.sh kitti              # KITTI 00, first 100 frames (smoke test)
#   bash scripts/download_data.sh tanksandtemples truck ignatius caterpillar barn courthouse
#   bash scripts/download_data.sh redwood            # prints manual-download instructions
#
# NCD / MaiCity / KITTI are the subsets released by SHINE-Mapping.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO_ROOT"; mkdir -p data

fetch() {  # fetch <url> <archive>
    wget -c -O "data/$2" "$1"
    tar -xf "data/$2" -C data && rm "data/$2"
}

case "$1" in
    ncd)
        fetch https://uni-bonn.sciebo.de/s/ZKTMubNY9mqbfwN/download ncd_example.tar.gz
        echo "-> data/ncd_example/quad" ;;
    maicity)
        fetch https://www.ipb.uni-bonn.de/html/projects/mai_city/mai_city.tar.gz mai_city.tar.gz
        wget -c -O data/mai_city/gt_map_pc_mai.ply https://uni-bonn.sciebo.de/s/DAMWVCC1Kxkfkyz/download
        echo "-> data/mai_city" ;;
    kitti)
        fetch https://uni-bonn.sciebo.de/s/KwOuBiPZi8vSz2O/download kitti_example.tar.gz
        echo "-> data/kitti_example" ;;
    tanksandtemples)
        shift
        [ $# -ge 1 ] || set -- truck ignatius caterpillar barn courthouse
        "$PYTHON" dataset/download_tanks_and_temples.py --scenes "$@" --output_root data/tanksandtemples
        echo "Next: bash scripts/prepare_tanksandtemples.sh <scene>" ;;
    redwood)
        cat <<'EOF'
Redwood (Park, Zhou & Koltun, ICCV 2017) is hosted on Google Drive behind a login,
so it cannot be fetched headlessly. From
    http://redwood-data.org/indoor_lidar_rgbd/download.html
download, for the bedroom scene:
    1. the RGB-D sequence          -> data/redwood/bedroom/depth/, data/redwood/bedroom/image/
    2. the camera poses            -> data/redwood/bedroom/bedroom.log
    3. the merged lidar scan (GT)  -> data/redwood/gt_mesh/bedroom.ply
then run:  bash scripts/prepare_redwood.sh
EOF
        mkdir -p data/redwood/bedroom data/redwood/gt_mesh ;;
    *)
        sed -n 2,10p "$0"; exit 1 ;;
esac
