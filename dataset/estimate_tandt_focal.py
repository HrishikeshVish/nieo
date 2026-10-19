"""Measure a Tanks-and-Temples scene's true focal length from its own data.

Why this exists: tandt_to_kitti_format.py needs per-image intrinsics, but T&T's
evaluation bundle ships only the extrinsic trajectory. Its documented fallback
("fx = fy = 0.7 * W") turns out to be ~20% too large for every Training scene --
the footage is 21mm on a full-frame Sony A7S II, i.e. an 81-degree horizontal
field of view, not the 71 degrees that 0.7*W implies.

That error is not cosmetic. The camera model is handed to COLMAP for dense
stereo, so a wrong focal corrupts the depth maps themselves rather than merely
misplacing the back-projected points. Measured on Ignatius at fx=0.7*W: input
points sat a median 79.5mm from the GT surface (1.8% within T&T's 3mm tau), and
frames six apart disagreed with each other by 120mm while adjacent frames agreed
to 7mm -- a wrong camera model, not depth noise.

Method: GT geometry and T&T's published poses are both independent of
intrinsics, so projecting the GT cloud into a real image is a direct test of a
candidate focal. The correct focal puts the GT silhouette on the object; a wrong
one scales it about the principal point. Scored by the mean image gradient along
the projected silhouette's boundary (the object's outline is a strong edge), and
the score is computed per frame, normalised, then averaged -- so no single
viewpoint dominates. Reports the per-frame spread too: a trustworthy estimate
has most frames agreeing, which is the actual evidence the peak is real.

EXIF is used only as a cross-check, not the answer: FocalLengthIn35mmFilm is an
integer, so it cannot resolve better than ~5% (Ignatius: EXIF says 21mm ->
fx=1120, this measurement says 1156, i.e. a ~21.7mm lens).

Usage:
    python dataset/estimate_tandt_focal.py --scene ignatius
    python dataset/estimate_tandt_focal.py --scene truck --save_overlays /tmp/ov

Then pass the result to the converter, BEFORE dense stereo runs:
    python dataset/tandt_to_kitti_format.py --scene ignatius --fx 1156
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tandt_to_kitti_format import SCENE_DISPLAY_NAME, focal_from_exif  # noqa: E402

try:
    import cv2
except ImportError:
    sys.exit("this script needs opencv (cv2)")


def project(points_world, pose, fx, fy, cx, cy):
    """world -> camera -> pixels; pose is camera-to-world and must be rigid."""
    R, t = pose[:3, :3], pose[:3, 3]
    cam = (points_world - t) @ R          # R^T (x - t)
    z = cam[:, 2]
    ok = z > 1e-6
    return (fx * cam[ok, 0] / z[ok] + cx,
            fy * cam[ok, 1] / z[ok] + cy)


def silhouette_edge_score(grad_mag, u, v, W, H, min_pts=2000, min_edge=200):
    mask = np.zeros((H, W), np.uint8)
    uu, vv = np.round(u).astype(int), np.round(v).astype(int)
    ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    if ok.sum() < min_pts:
        return np.nan, mask
    mask[vv[ok], uu[ok]] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    sel = edge > 0
    if sel.sum() < min_edge:
        return np.nan, mask
    return grad_mag[sel].mean(), mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENE_DISPLAY_NAME))
    ap.add_argument("--data_root", default="data/tanksandtemples")
    ap.add_argument("--fx_min", type=float, default=1000.0)
    ap.add_argument("--fx_max", type=float, default=1400.0)
    ap.add_argument("--fx_step", type=float, default=4.0)
    ap.add_argument("--num_frames", type=int, default=21)
    ap.add_argument("--gt_voxel_m", type=float, default=0.008,
                     help="GT downsample before projecting; smaller = denser "
                          "silhouette but slower")
    ap.add_argument("--save_overlays", default="",
                     help="directory to write overlay PNGs at the best/EXIF/0.7W "
                          "focals, so the alignment can be seen and not only scored")
    args = ap.parse_args()

    display = SCENE_DISPLAY_NAME[args.scene]
    scene_dir = os.path.join(args.data_root, args.scene)
    images_dir = os.path.join(scene_dir, "images")
    gt_ply = os.path.join(scene_dir, "gt", f"{display}.ply")
    poses_txt = os.path.join(args.data_root, f"{args.scene}_kitti_format", "poses.txt")
    for p in (images_dir, gt_ply, poses_txt):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    gt = o3d.io.read_point_cloud(gt_ply)
    g = np.asarray(gt.voxel_down_sample(args.gt_voxel_m).points)
    poses = np.loadtxt(poses_txt).reshape(-1, 3, 4)

    row_norms = np.linalg.norm(poses[:, :3, :3], axis=2)
    if abs(row_norms.mean() - 1.0) > 1e-3:
        sys.exit(f"poses.txt is not rigid (mean row norm {row_norms.mean():.4f}) -- "
                 "re-run tandt_to_kitti_format.py, which folds trans.txt's scale "
                 "into the points instead of the poses")

    names = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png")))
    probe = cv2.imread(os.path.join(images_dir, names[0]))
    H, W = probe.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    print(f"{display}: {len(names)} images {W}x{H}, {len(poses)} poses, "
          f"{len(g)} GT points after downsample")

    exif_fx = focal_from_exif(os.path.join(images_dir, names[0]), W)
    candidates = np.arange(args.fx_min, args.fx_max + 1e-9, args.fx_step)
    frame_ids = np.linspace(0, min(len(poses), len(names)) - 1,
                            args.num_frames, dtype=int)

    rows = []
    for fid in frame_ids:
        img = cv2.imread(os.path.join(images_dir, names[fid]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.hypot(gx, gy)
        row = np.array([silhouette_edge_score(gm, *project(g, poses[fid], f, f, cx, cy),
                                              W, H)[0] for f in candidates])
        if np.isfinite(row).sum() < 0.8 * len(candidates):
            continue                      # object barely visible in this frame
        rows.append(row / np.nanmax(row))

    if not rows:
        sys.exit("no frame had the object sufficiently in view -- widen --num_frames")

    rows = np.array(rows)
    mean_score = np.nanmean(rows, axis=0)
    best = float(candidates[int(np.nanargmax(mean_score))])
    per_frame = np.array([candidates[int(np.nanargmax(r))] for r in rows])
    within = np.mean(np.abs(per_frame - best) <= 3 * args.fx_step)

    print(f"\nframes scored: {len(rows)}")
    print(f"per-frame argmax median {np.median(per_frame):.0f}, "
          f"{within:.0%} within +-{3*args.fx_step:.0f}px of the pooled peak")
    print(f"\n  BEST fx = {best:.0f} px")
    print(f"  EXIF               = {exif_fx:.0f} px" if exif_fx else "  EXIF = unavailable")
    print(f"  T&T generic 0.7*W  = {0.7*W:.0f} px "
          f"({100*(0.7*W/best-1):+.1f}% vs measured)")
    if within < 0.6:
        print("\n  WARNING: frames disagree a lot -- treat this estimate with caution "
              "and inspect --save_overlays before committing to an MVS run.")
    print(f"\n  -> dataset/tandt_to_kitti_format.py --scene {args.scene} --fx {best:.0f}")

    if args.save_overlays:
        os.makedirs(args.save_overlays, exist_ok=True)
        fid = int(frame_ids[len(frame_ids) // 2])
        img = cv2.imread(os.path.join(images_dir, names[fid]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = [(best, "measured")] + ([(exif_fx, "exif")] if exif_fx else []) \
                   + [(0.7 * W, "generic0.7W")]
        for f, tag in variants:
            _, mask = silhouette_edge_score(gray, *project(g, poses[fid], f, f, cx, cy), W, H)
            ov = img.copy()
            ov[mask > 0] = (0.45 * ov[mask > 0] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            out = os.path.join(args.save_overlays, f"{args.scene}_f{fid}_{tag}_fx{int(f)}.png")
            cv2.imwrite(out, ov)
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
