#!/bin/bash
# Installs and verifies a CUDA dense-MVS backend for the Tanks-and-Temples
# pipeline. Run once, on a machine with a GPU, before
# dataset/tandt_to_kitti_format.py.
#
# Tries pycolmap first. pip wheels are usually CPU-only even though they expose
# patch_match_stereo, so the probe actually calls it and looks for COLMAP's
# "requires CUDA" error rather than trusting hasattr(). If pycolmap cannot run
# dense stereo, a CUDA build of the native COLMAP CLI is installed from
# conda-forge into a separate conda env named "colmap".
#
# Writes scripts/tanksandtemples/.mvs_backend:
#   "pycolmap"              -- use pycolmap's dense-MVS API
#   "native_colmap:<path>"  -- shell out to this colmap binary
#
# Usage: bash scripts/tanksandtemples/setup_mvs_env.sh
#        PYTHON=/path/to/python bash scripts/tanksandtemples/setup_mvs_env.sh
set -eo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARKER_FILE="$REPO_ROOT/scripts/tanksandtemples/.mvs_backend"

echo "==== Installing pycolmap into $($PYTHON -c 'import sys; print(sys.prefix)') ===="
"$PYTHON" -m pip install -U pycolmap

echo "==== Probing pycolmap for CUDA dense MVS ===="
PROBE_OUT=$("$PYTHON" - <<'EOF'
import sys
try:
    import pycolmap
except ImportError as e:
    print("PYCOLMAP_DENSE_MVS:FAIL (import: %s)" % e)
    sys.exit(0)
if not (hasattr(pycolmap, "patch_match_stereo") and hasattr(pycolmap, "stereo_fusion")):
    print("PYCOLMAP_DENSE_MVS:FAIL (no dense-MVS API)")
    sys.exit(0)
try:
    pycolmap.patch_match_stereo("/nonexistent_probe_workspace_dir")
    print("PYCOLMAP_DENSE_MVS:PASS")
except ValueError as e:
    # a bad-path error means the CUDA check itself passed
    print("PYCOLMAP_DENSE_MVS:FAIL (%s)" % e if "CUDA" in str(e) else "PYCOLMAP_DENSE_MVS:PASS")
except Exception as e:
    print("PYCOLMAP_DENSE_MVS:FAIL (%s)" % e)
EOF
)
echo "$PROBE_OUT"

if echo "$PROBE_OUT" | grep -q "PYCOLMAP_DENSE_MVS:PASS"; then
    echo "pycolmap" > "$MARKER_FILE"
    echo "==== Backend: pycolmap ===="
    exit 0
fi

echo "==== Falling back to the native COLMAP CLI (conda-forge, CUDA build) ===="
command -v conda &>/dev/null || { echo "ERROR: conda not on PATH"; exit 1; }

if conda env list | grep -q "^colmap "; then
    echo "conda env 'colmap' already exists"
else
    # conda-forge's colmap links libOpenImageIO without declaring it, so add it explicitly
    conda create -y -n colmap -c conda-forge colmap openimageio
fi

COLMAP_BIN="$(conda run -n colmap which colmap || true)"
if [ -z "$COLMAP_BIN" ]; then
    echo "ERROR: colmap CLI not found after install (try: conda activate colmap; colmap -h)"
    exit 1
fi
"$COLMAP_BIN" -h | head -1 || true
echo "native_colmap:$COLMAP_BIN" > "$MARKER_FILE"
echo "==== Backend: native COLMAP at $COLMAP_BIN ===="
