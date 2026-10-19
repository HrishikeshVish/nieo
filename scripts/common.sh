#!/bin/bash
# Shared by the scripts in this folder: resolves a short dataset name to its
# config directory and evaluation settings. Source it, don't run it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

DATASETS="ncd maicity truck ignatius caterpillar barn courthouse redwood"
METHODS="base ours regsdf neuralssd gradhess singular_hessian digs siren diffcd stoch_precond"
BASELINES="regsdf neuralssd gradhess singular_hessian digs siren diffcd stoch_precond"

# config_dir <dataset>
config_dir() {
    case "$1" in
        ncd|maicity|redwood) echo "config/$1" ;;
        truck|ignatius|caterpillar|barn|courthouse) echo "config/tanksandtemples/$1" ;;
        *) echo "unknown dataset '$1' (choose from: $DATASETS)" >&2; return 1 ;;
    esac
}

# eval_key <dataset>: the DATASETS key in eval/eval_run.py
eval_key() {
    case "$1" in
        ncd) echo "ncd_quad" ;;
        maicity|redwood) echo "$1" ;;
        *) echo "tanksandtemples_$1" ;;
    esac
}

# mc_res <dataset>: marching-cubes resolution (m) the paper's meshes were extracted at
mc_res() {
    case "$1" in
        ncd|maicity) echo 0.1 ;;
        truck)       echo 0.008 ;;
        ignatius)    echo 0.005 ;;
        caterpillar) echo 0.01 ;;
        barn)        echo 0.02 ;;
        courthouse)  echo 0.05 ;;
        redwood)     echo 0.01 ;;
    esac
}

# eval_flags <dataset>: protocol flags for eval/eval_run.py
eval_flags() {
    case "$1" in
        # SHINE's released protocol: score a surface sample of the mesh
        ncd|maicity) echo "--resample_surface" ;;
        # exact point-to-surface recall + normal metrics
        *)           echo "--with_normals" ;;
    esac
}

# latest_run <dataset> <method>: newest experiments/ dir for this config
latest_run() {
    local name
    name="$(config_dir "$1" | sed 's|^config/||; s|/|_|g')_$2"
    ls -d "$REPO_ROOT"/experiments/"${name}"_20* 2>/dev/null | sort | tail -1
}
