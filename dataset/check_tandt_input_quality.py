"""Gate a Tanks-and-Temples conversion BEFORE spending a training run on it.

Checks whether the *input points* can support the detail being reconstructed.
The typical failure is a wrong focal length (e.g. T&T's generic 0.7*W guess,
~16% too large on Ignatius), which corrupts COLMAP's dense stereo itself rather
than merely misplacing the back-projected points. Running this takes seconds; a
training run takes an hour.

Two measurements, both against data that is independent of the intrinsics being
tested (T&T's own GT cloud and its published poses):

  1. ACCURACY -- distance from input points to the true GT surface, restricted
     to points inside the GT object's bounding box so that ground/surroundings
     the GT legitimately excludes do not confound it. A fused surface can never
     be sharper than this, so compare it against the scene's tau.

  2. MULTI-VIEW CONSISTENCY -- do two frames seeing the same surface put it in
     the same world position? This is the measurement that identifies a *wrong
     camera model* specifically: with a bad focal, near-identical viewpoints
     still agree with each other (they are wrong in the same way) while
     separated viewpoints diverge. Plain depth noise degrades both equally.
     E.g. Ignatius with fx=0.7*W: adjacent frames 7.2mm, six apart 119.7mm.

Usage:
    python dataset/check_tandt_input_quality.py --scene ignatius
    python dataset/check_tandt_input_quality.py --scene ignatius --tau 0.003
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tandt_to_kitti_format import SCENE_DISPLAY_NAME  # noqa: E402

# T&T's own per-scene F-score thresholds (python_toolbox/evaluation/config.py)
SCENE_TAU = {
    "ignatius": 0.003, "truck": 0.005, "caterpillar": 0.005,
    "barn": 0.01, "courthouse": 0.025,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENE_DISPLAY_NAME))
    ap.add_argument("--data_root", default="data/tanksandtemples")
    ap.add_argument("--tau", type=float, default=0.0,
                     help="inlier threshold; defaults to T&T's official per-scene tau")
    ap.add_argument("--num_frames", type=int, default=9)
    ap.add_argument("--pad_m", type=float, default=0.03,
                     help="how far outside the GT bbox a point may sit and still "
                          "count as 'on the object'")
    ap.add_argument("--frames_dir", default="",
                     help="converted output to check (dir holding poses.txt + "
                          "tandt_ply); defaults to <data_root>/<scene>_kitti_format. "
                          "Point this at a pilot conversion to gate it before "
                          "committing to the full run.")
    args = ap.parse_args()

    display = SCENE_DISPLAY_NAME[args.scene]
    tau = args.tau or SCENE_TAU.get(args.scene, 0.005)
    fr = args.frames_dir or os.path.join(args.data_root, f"{args.scene}_kitti_format")
    gt_path = os.path.join(args.data_root, args.scene, "gt", f"{display}.ply")
    for p in (gt_path, os.path.join(fr, "poses.txt"), os.path.join(fr, "tandt_ply")):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run tandt_to_kitti_format.py first")

    gt = o3d.io.read_point_cloud(gt_path)
    g = np.asarray(gt.points)
    lo, hi = g.min(0), g.max(0)
    gt_tree = o3d.geometry.KDTreeFlann(gt.voxel_down_sample(max(tau / 2, 0.001)))
    poses = np.loadtxt(os.path.join(fr, "poses.txt")).reshape(-1, 3, 4)

    row_norms = np.linalg.norm(poses[:, :3, :3], axis=2)
    print(f"{display}: {len(poses)} poses, tau = {tau*1000:.0f} mm")
    print(f"poses rigid? mean row norm {row_norms.mean():.6f} "
          f"{'OK' if abs(row_norms.mean()-1) < 1e-3 else '<-- NOT RIGID, scale not folded into points'}")

    def world(fid):
        p = np.asarray(o3d.io.read_point_cloud(
            os.path.join(fr, "tandt_ply", f"{fid:06d}.ply")).points)
        return p @ poses[fid, :3, :3].T + poses[fid, :3, 3] if len(p) else p

    def on_object(w):
        return w[np.all((w >= lo - args.pad_m) & (w <= hi + args.pad_m), axis=1)]

    print(f"\n=== 1. input accuracy vs GT (points on the object only) ===")
    frames = np.linspace(0, len(poses) - 1, args.num_frames, dtype=int)
    pooled = []
    for fid in frames:
        w = world(int(fid))
        if len(w) == 0:
            continue
        s = on_object(w)
        if len(s) < 100:
            print(f"  frame {fid:>4}: only {len(s)} pts on object, skipped")
            continue
        sub = s[np.random.choice(len(s), min(8000, len(s)), replace=False)]
        d = np.sqrt([gt_tree.search_knn_vector_3d(p, 1)[2][0] for p in sub])
        pooled.append(d)
        print(f"  frame {fid:>4}: {len(s):>7} pts | median {np.median(d)*1000:7.1f} mm "
              f"| within tau {(d < tau).mean():6.1%}")

    if not pooled:
        sys.exit("no frame had points on the object -- conversion or alignment is broken")
    d = np.concatenate(pooled)
    med = np.median(d)
    print(f"\n  POOLED median {med*1000:.1f} mm, mean {d.mean()*1000:.1f} mm, "
          f"within tau {(d < tau).mean():.1%}")

    print(f"\n=== 2. multi-view consistency (adjacent vs separated) ===")
    # Averaged over many pairs at each baseline, not one hand-picked pair: a
    # single pair is dominated by whether those two frames happen to see the
    # same part of the object at all. Within a pair we take the 25th percentile
    # of nearest-neighbour distances rather than the median, for the same reason
    # -- it reads the co-visible surface instead of being inflated by the parts
    # one frame sees and the other does not.
    def pair_disagreement(a, b):
        wa, wb = on_object(world(a)), on_object(world(b))
        if len(wa) < 200 or len(wb) < 200:
            return None
        pb = o3d.geometry.PointCloud()
        pb.points = o3d.utility.Vector3dVector(wb)
        tb = o3d.geometry.KDTreeFlann(pb)
        sub = wa[np.random.choice(len(wa), min(4000, len(wa)), replace=False)]
        dd = np.sqrt([tb.search_knn_vector_3d(p, 1)[2][0] for p in sub])
        return float(np.percentile(dd, 25))

    by_delta = {}
    starts = np.linspace(0, len(poses) - 10, 10, dtype=int)
    for delta in (1, 6):
        vals = [v for a in starts
                if (v := pair_disagreement(int(a), int(a) + delta)) is not None]
        if vals:
            by_delta[delta] = float(np.median(vals))
            print(f"  baseline delta={delta}: median over {len(vals)} pairs = "
                  f"{by_delta[delta]*1000:7.1f} mm")
    adj = [by_delta[1]] if 1 in by_delta else []
    sep = [by_delta[6]] if 6 in by_delta else []

    print("\n=== verdict ===")
    ok = True
    if med > 5 * tau:
        print(f"  FAIL accuracy: median {med*1000:.1f} mm is {med/tau:.0f}x tau. "
              f"No SDF/octree setting can recover detail the input lacks.")
        ok = False
    else:
        print(f"  OK accuracy: median {med*1000:.1f} mm ({med/tau:.1f}x tau)")

    if adj and sep:
        ra, rs = float(np.median(adj)), float(np.median(sep))
        print(f"  adjacent-frame agreement {ra*1000:.1f} mm vs "
              f"separated-frame {rs*1000:.1f} mm (ratio {rs/max(ra,1e-9):.1f}x)")
        if rs > 4 * ra and rs > 5 * tau:
            print("  FAIL camera model: separated views disagree far more than adjacent "
                  "ones. That is the signature of wrong intrinsics, not depth noise -- "
                  "run dataset/estimate_tandt_focal.py and re-run MVS with --fx.")
            ok = False
        else:
            print("  OK camera model: no view-separation-dependent disagreement")

    print(f"\n  => {'PASS -- safe to train on' if ok else 'DO NOT TRAIN ON THIS CONVERSION'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
