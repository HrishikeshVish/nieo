"""Crop a ground-truth point cloud to the intersection of one or more predicted
meshes -- SHINE's "fairer accuracy" protocol, vectorized.

SHINE-Mapping's paper reports accuracy against "the ground truth mesh masked by
the intersection area of the 3D reconstruction from all the compared methods",
implemented in eval/crop_intersection.py. That implementation is correct but
loops over the ground-truth points one at a time in Python
(eval_utils.crop_intersection, `for pt in pcd_gt_pts`), which is impractical on
the Newer College Quad ground truth's 5.4M points.

This is the same operation with the nearest-neighbour search batched through
open3d's compute_point_cloud_distance (C++), keeping the semantics identical:
each predicted mesh is uniformly sampled to `mesh_sample_point` points, and the
ground truth is reduced, mesh after mesh, to the points lying within
`dist_thre` of that sample. Applying the meshes in sequence intersects them,
exactly as the original loop does.

Equivalence to the shipped implementation is checked by --verify, which runs the
original eval_utils.crop_intersection on a random subset of the ground truth and
asserts the two keep the same points.

NOTE ON REPRODUCIBILITY: sample_points_uniformly is randomised and neither this
script nor the original seeds it, so the mask shifts by a few points between
runs. Pass --seed to pin it. The effect on metrics is far below the differences
being measured, but pinning it makes a reported number re-derivable.

Usage:
    python eval/crop_gt_to_meshes.py \
        --gt data/ncd_example/quad/ncd_quad_gt_pc.ply \
        --preds data/shine_released_meshes/shine_ncd.ply \
        --out data/ncd_example/quad/ncd_quad_gt_pc_crop_shine.ply \
        --dist_thre 0.2 --seed 0
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "eval"))


def crop_to_meshes(gt_pts, pred_paths, dist_thre, mesh_sample_point, seed=None):
    """Keep the ground-truth points within dist_thre of EVERY predicted mesh."""
    kept = gt_pts
    for path in pred_paths:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) == 0:
            sys.exit(f"{path} has no triangles; crop_intersection needs meshes, not clouds")
        if seed is not None:
            o3d.utility.random.seed(seed)
        sample = mesh.sample_points_uniformly(number_of_points=mesh_sample_point)

        cur = o3d.geometry.PointCloud()
        cur.points = o3d.utility.Vector3dVector(kept)
        # C++ batched nearest-neighbour: one distance per point of `cur`
        d = np.asarray(cur.compute_point_cloud_distance(sample))
        kept = kept[d < dist_thre]
        print(f"  after {os.path.basename(path)}: {len(kept)} points")
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", nargs="+", required=True,
                     help="predicted meshes to intersect; SHINE used all compared methods")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dist_thre", type=float, default=0.2,
                     help="metres; eval/crop_intersection.py's own example uses 0.2")
    ap.add_argument("--mesh_sample_point", type=int, default=1_000_000,
                     help="matches eval_utils.crop_intersection's default")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verify", type=int, default=0,
                     help="cross-check against the original per-point implementation on "
                          "this many randomly chosen GT points (0 = skip)")
    args = ap.parse_args()

    gt = o3d.io.read_point_cloud(args.gt)
    gt_pts = np.asarray(gt.points)
    print(f"ground truth: {len(gt_pts)} points")

    kept = crop_to_meshes(gt_pts, args.preds, args.dist_thre, args.mesh_sample_point, args.seed)
    frac = len(kept) / max(len(gt_pts), 1)
    print(f"kept {len(kept)} / {len(gt_pts)} ({100*frac:.2f}%)")

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    o3d.io.write_point_cloud(args.out, out)
    print(f"wrote {args.out}")

    if args.verify:
        from eval_utils import crop_intersection
        import tempfile
        rng = np.random.default_rng(0)
        idx = rng.choice(len(gt_pts), size=min(args.verify, len(gt_pts)), replace=False)
        subset = gt_pts[idx]
        with tempfile.TemporaryDirectory() as td:
            sub_path = os.path.join(td, "sub.ply")
            ref_path = os.path.join(td, "ref.ply")
            sp = o3d.geometry.PointCloud()
            sp.points = o3d.utility.Vector3dVector(subset)
            o3d.io.write_point_cloud(sub_path, sp)
            if args.seed is not None:
                o3d.utility.random.seed(args.seed)
            crop_intersection(sub_path, args.preds, ref_path,
                               dist_thre=args.dist_thre,
                               mesh_sample_point=args.mesh_sample_point)
            ref = np.asarray(o3d.io.read_point_cloud(ref_path).points)
        mine = crop_to_meshes(subset, args.preds, args.dist_thre,
                               args.mesh_sample_point, args.seed)
        print(f"\nVERIFY on {len(subset)} points: original kept {len(ref)}, "
              f"vectorized kept {len(mine)}")
        # the two samplings are independent draws unless --seed is set, so compare
        # as sets and allow the handful of points sitting within a sampling jitter
        # of the threshold to differ
        a = {tuple(np.round(p, 6)) for p in ref}
        b = {tuple(np.round(p, 6)) for p in mine}
        diff = len(a ^ b)
        print(f"symmetric difference: {diff} points "
              f"({100*diff/max(len(a | b), 1):.3f}% of the union)")


if __name__ == "__main__":
    main()
