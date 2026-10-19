"""CLI wrapper around eval_mesh for scoring one predicted mesh against a
dataset's ground truth, with per-dataset ground-truth paths and thresholds.

Usage:
    python eval/eval_run.py \
        --dataset redwood --method_name regsdf \
        --pred_mesh experiments/<run_dir>/mesh/mesh_iter_10000.ply
"""
import argparse
import csv
import os
import sys

from eval_utils import eval_mesh, normal_and_mesh_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Per-dataset evaluation settings. angle_thre is only used
# for --with_normals (RegSDF-style normal-augmented F-score); 30 degrees
# follows [1] RegSDF's own DTU/BlendedMVS/Tanks and Temples setup.
DATASETS = {
    "redwood": dict(
        dataset_name="redwood_",  # output CSV prefix: f"{dataset_name}{method_name}_eval.csv"
        gt_pcd_path=os.path.join(ROOT, "data/redwood/gt_mesh/bedroom_registered.ply"),
        # The registered GT has a 2.4 cm median frame-to-GT residual (81% of
        # points within 5 cm; see dataset/register_redwood_gt.py). dist_thre
        # 0.05 is 2x that residual and 5x the 0.01 m mc_res_m the meshes are
        # extracted at; truncations are 2x / 10x dist_thre.
        down_sample_vox=0.01,
        dist_thre=0.05,
        angle_thre=30.0,
        truncation_dist_acc=0.10,
        truncation_dist_com=0.50,
    ),
    # MaiCity 01 -- synthetic, noise-free 64-beam sequence; GT is the merged
    # point cloud generated from sequence 02 (gt_map_pc_mai.ply). Every value
    # is copied from SHINE's released evaluation settings, and dist_thre is the
    # 10 cm threshold used for this dataset. Their evaluator also offers
    # gt_map_pc_mai_inter_croped.ply, the
    # GT masked by the intersection of all compared methods -- that is a SECOND
    # protocol, not this one; build it with eval/crop_gt_to_meshes.py and pass
    # --gt_pcd if reproducing their intersection column.
    "maicity": dict(
        dataset_name="maicity_01_",
        gt_pcd_path=os.path.join(ROOT, "data/mai_city/gt_map_pc_mai.ply"),
        down_sample_vox=0.02,
        dist_thre=0.1,
        angle_thre=30.0,
        truncation_dist_acc=0.2,
        truncation_dist_com=2.0,
    ),
    # Newer College Quad -- SHINE's OWN example subset, shipped with its own
    # TLS ground truth (ncd_quad_gt_pc.ply, Leica BLK360). Every value here is
    # copied from SHINE's released evaluation settings, NOT rederived, so that
    # their protocol is reproduced exactly:
    #   * pass --resample_surface, since their eval_mesh call leaves
    #     use_mesh_vertices at its False default and scores a surface sample;
    #   * sample_density_per_m2 is deliberately ABSENT so that path falls back
    #     to eval_mesh's flat mesh_sample_point=10_000_000, which is what they
    #     ran. Adding a density here would silently change the protocol being
    #     reproduced.
    #   * pass --no_exact_recall for their column set; the exact-BVH recall is
    #     the better metric and is still worth reporting alongside, but it is
    #     not what their published number is.
    "ncd_quad": dict(
        dataset_name="ncd_quad_",
        gt_pcd_path=os.path.join(ROOT, "data/ncd_example/quad/ncd_quad_gt_pc.ply"),
        down_sample_vox=0.02,
        dist_thre=0.2,
        angle_thre=30.0,
        truncation_dist_acc=0.4,
        truncation_dist_com=2.0,
    ),
    "tanksandtemples_ignatius": dict(
        dataset_name="tanksandtemples_ignatius_",
        gt_pcd_path=os.path.join(ROOT, "data/tanksandtemples/ignatius/gt/Ignatius.ply"),
        # dist_thre is T&T's own official tau for this scene
        # (python_toolbox/evaluation/config.py's scenes_tau_dict) -- the finest
        # of all seven Training scenes, since Ignatius is a small close-range
        # object scan rather than a building. Truncation distances are ~3x/~10x
        # of tau.
        down_sample_vox=0.002,
        dist_thre=0.003,
        # T&T's own evaluation toolkit crops the reconstruction with this
        # volume before scoring, so it is the protocol their published numbers
        # use. It shifts every method equally (about +2.3pp F-score versus the
        # gt-bbox fallback), so it changes absolute numbers, not rankings.
        crop_json=os.path.join(ROOT, "data/tanksandtemples/ignatius/gt/Ignatius.json"),
        # Points sampled per m^2 of predicted surface, replacing the flat 10M
        # samples per mesh (eval_utils._sample_count_for_mesh). A smoother mesh
        # has less area, so a flat count silently samples it denser than the
        # baseline -- and the 2 mm downsample grid is not saturated at 10M, so
        # density still moves recall. 2.27e6 /m^2 gives ~40M samples on a
        # 17.65 m^2 base mesh, past the knee of the saturation curve.
        # Every method must be scored at the same density for numbers to be
        # comparable, so this is a dataset-level constant, not a per-run flag.
        sample_density_per_m2=2.27e6,
        angle_thre=30.0,
        truncation_dist_acc=0.01,
        truncation_dist_com=0.03,
    ),
    "tanksandtemples_caterpillar": dict(
        dataset_name="tanksandtemples_caterpillar_",
        gt_pcd_path=os.path.join(ROOT, "data/tanksandtemples/caterpillar/gt/Caterpillar.ply"),
        # T&T's own official tau for this scene (scenes_tau_dict), 1.67x
        # Ignatius's -- Caterpillar is an excavator at standoff range, not a
        # close-range statue. Every other threshold here is Ignatius's value
        # scaled by that same 1.67, preserving this dict's established
        # ~0.67x/~3x/~10x-of-tau pattern.
        down_sample_vox=0.003,
        dist_thre=0.005,
        # T&T's evaluation toolkit crops the reconstruction with this volume
        # before scoring, so it is the protocol their published numbers use.
        crop_json=os.path.join(ROOT, "data/tanksandtemples/caterpillar/gt/Caterpillar.json"),
        # Ignatius's 2.27e6/m^2 was chosen to sit past the knee of the
        # saturation curve for ITS 2mm downsample grid; saturation density goes
        # as 1/vox^2, so the 3mm grid here scales it to 2.27e6 * (2/3)^2 = 1.0e6.
        # The knee was measured on Ignatius, not here, and Caterpillar's mesh has
        # several times the surface area; to verify, score one mesh at
        # 0.5e6/1.0e6/2.0e6 and confirm recall has flattened. Rankings are unaffected as
        # long as every method is scored at the same density, which this
        # dataset-level constant guarantees.
        sample_density_per_m2=1.0e6,
        angle_thre=30.0,
        truncation_dist_acc=0.017,
        truncation_dist_com=0.05,
    ),
    "tanksandtemples_truck": dict(
        dataset_name="tanksandtemples_truck_",
        gt_pcd_path=os.path.join(ROOT, "data/tanksandtemples/truck/gt/Truck.ply"),
        # T&T's own tau for this scene (scenes_tau_dict), same as Caterpillar's
        down_sample_vox=0.003,
        dist_thre=0.005,
        crop_json=os.path.join(ROOT, "data/tanksandtemples/truck/gt/Truck.json"),
        # only used with --resample_surface; the default vertex protocol needs
        # no density (see eval_utils.eval_mesh's use_mesh_vertices)
        sample_density_per_m2=1.0e6,
        angle_thre=30.0,
        truncation_dist_acc=0.017,
        truncation_dist_com=0.05,
    ),
    # Barn and Courthouse use T&T's own scenes_tau_dict values (0.01 and 0.025),
    # the same source as Ignatius's 0.003 and Caterpillar's 0.005. Every other
    # threshold is Ignatius's value scaled by tau/0.003, preserving this dict's
    # established ~0.67x / ~3x / ~10x-of-tau pattern. sample_density_per_m2 is
    # deliberately omitted: recall is scored exactly against the surface (BVH),
    # so the sampled-recall density knob does not affect the reported columns.
    "tanksandtemples_barn": dict(
        dataset_name="tanksandtemples_barn_",
        gt_pcd_path=os.path.join(ROOT, "data/tanksandtemples/barn/gt/Barn.ply"),
        down_sample_vox=0.0067,
        dist_thre=0.01,
        crop_json=os.path.join(ROOT, "data/tanksandtemples/barn/gt/Barn.json"),
        angle_thre=30.0,
        truncation_dist_acc=0.033,
        truncation_dist_com=0.10,
    ),
    "tanksandtemples_courthouse": dict(
        dataset_name="tanksandtemples_courthouse_",
        gt_pcd_path=os.path.join(ROOT, "data/tanksandtemples/courthouse/gt/Courthouse.ply"),
        down_sample_vox=0.0167,
        dist_thre=0.025,
        crop_json=os.path.join(ROOT, "data/tanksandtemples/courthouse/gt/Courthouse.json"),
        angle_thre=30.0,
        truncation_dist_acc=0.083,
        truncation_dist_com=0.25,
    ),
}

CSV_COLUMNS = ['MAE_accuracy (m)', 'MAE_completeness (m)', 'Chamfer_L1 (m)', 'Chamfer_L2 (m)',
               'Precision [Accuracy] (%)', 'Recall [Completeness] (%)', 'F-score (%)', 'Spacing (m)',
               'Inlier_threshold (m)', 'Outlier_truncation_acc (m)', 'Outlier_truncation_com (m)']

NORMAL_CSV_COLUMNS = ['Triangle_count', 'Surface_area (m2)', 'Normal_consistency (%)', 'Chamfer_angle (deg)',
                       'Normal_aug_Precision (%)', 'Normal_aug_Recall (%)', 'Normal_aug_F-score (%)',
                       'Dist_threshold (m)', 'Angle_threshold (deg)']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    ap.add_argument("--method_name", required=True, help="e.g. regsdf, neuralssd, ours_iter8000")
    ap.add_argument("--pred_mesh", required=True)
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "experiments/evaluation"))
    ap.add_argument("--dist_thre", type=float, default=None,
                     help="override the dataset's inlier threshold (tau), in metres. "
                          "The truncation distances scale with it by the same ratio the "
                          "dataset entry uses, because they are NOT independent: "
                          "nn_correspondance drops predicted points beyond "
                          "truncation_dist_acc before computing precision, so any tau at "
                          "or near that truncation reads ~100%% by construction. Use "
                          "--truncation_acc/--truncation_com to set them explicitly.")
    ap.add_argument("--truncation_acc", type=float, default=None,
                     help="override truncation_dist_acc (metres). Pass a large value to "
                          "disable truncation entirely and score every predicted point.")
    ap.add_argument("--truncation_com", type=float, default=None)
    ap.add_argument("--resample_surface", action="store_true",
                     help="OLD protocol: score a uniform surface sample of the predicted "
                          "mesh (at the dataset's sample_density_per_m2) instead of the "
                          "mesh's own vertices, with both clouds voxel-downsampled to "
                          "down_sample_vox. Off by default -- the vertices ARE the "
                          "reconstruction, so resampling only picks a density, invents "
                          "points across genuine holes, and costs time proportional to "
                          "surface area (226M points on a Caterpillar mesh).")
    ap.add_argument("--sample_density", type=float, default=None,
                     help="points sampled per m^2 of predicted surface, overriding the "
                          "dataset default; only for probing the protocol -- comparable "
                          "numbers require every method scored at one density. Requires "
                          "--resample_surface.")
    ap.add_argument("--gt_pcd", default=None,
                     help="override the dataset's ground-truth cloud. Needed to reproduce "
                          "SHINE's Newer College protocol, which is NOT one protocol for "
                          "the whole row: their paper reports completeness against the "
                          "full ground truth but ACCURACY against a ground truth masked "
                          "by the intersection of all compared methods' meshes (see "
                          "eval/crop_gt_to_meshes.py). Those have to be two runs. Note a "
                          "ground truth cropped to within tau of ONE method's own mesh "
                          "forces that method's recall to 100%% by construction.")
    ap.add_argument("--with_normals", action="store_true",
                     help="also compute Triangle_count/Normal_consistency/Chamfer_angle/"
                          "Normal_aug_F-score (eval_utils.normal_and_mesh_metrics) -- slower, "
                          "since it does a second mesh sampling + NN search pass for normals")
    ap.add_argument("--no_exact_recall", action="store_true",
                     help="skip the tessellation-free recall/F-score columns "
                          "(GT vs the predicted SURFACE via BVH). They are on by "
                          "default; eval_mesh's sampled recall has a ~2.4mm "
                          "discretisation floor that dominates a 3mm tau.")
    ap.add_argument("--exact_recall_points", type=int, default=1_000_000,
                     help="GT points used for the exact recall columns")
    args = ap.parse_args()

    if not os.path.exists(args.pred_mesh):
        sys.exit(f"predicted mesh not found: {args.pred_mesh}")

    d = DATASETS[args.dataset]
    if args.sample_density and not args.resample_surface:
        sys.exit("--sample_density only applies with --resample_surface")
    use_vertices = not args.resample_surface

    dist_thre = args.dist_thre or d["dist_thre"]
    scale = dist_thre / d["dist_thre"]   # keep the dataset's tau:truncation ratios
    truncation_acc = args.truncation_acc or d["truncation_dist_acc"] * scale
    truncation_com = args.truncation_com or d["truncation_dist_com"] * scale
    if truncation_acc < 2.0 * dist_thre:
        print(f"WARNING: truncation_acc {truncation_acc:.4f} is close to tau "
              f"{dist_thre:.4f}; precision is inflated toward 100% because points "
              f"beyond the truncation are dropped rather than counted as misses.")

    gt_pcd_path = args.gt_pcd or d["gt_pcd_path"]
    if args.gt_pcd:
        print(f"ground truth OVERRIDDEN: {gt_pcd_path}")
    metrics = eval_mesh(args.pred_mesh, gt_pcd_path,
                         # no voxel grid in vertex mode: the mesh's own vertex
                         # spacing is already set by mc_res_m, which is the same
                         # for every method being compared
                         down_sample_res=0.0 if use_vertices else d["down_sample_vox"],
                         threshold=dist_thre,
                         truncation_acc=truncation_acc, truncation_com=truncation_com,
                         gt_bbx_mask_on=True, crop_json=d.get("crop_json"),
                         sample_density_per_m2=args.sample_density or d.get("sample_density_per_m2"),
                         use_mesh_vertices=use_vertices)
    csv_columns = list(CSV_COLUMNS)
    # recorded per row so a CSV written under one protocol is never silently
    # compared against one written under another
    metrics['GT_cloud'] = os.path.basename(gt_pcd_path)
    metrics['Pred_point_source'] = 'mesh_vertices' if use_vertices else 'surface_sample'
    metrics['Sample_density (pts/m2)'] = None if use_vertices else \
        (args.sample_density or d.get("sample_density_per_m2"))
    csv_columns += ['GT_cloud', 'Pred_point_source', 'Sample_density (pts/m2)']

    if args.with_normals:
        normal_metrics = normal_and_mesh_metrics(
            args.pred_mesh, gt_pcd_path, down_sample_res=d["down_sample_vox"],
            dist_threshold=dist_thre, angle_threshold_deg=d["angle_thre"],
            truncation_acc=truncation_acc, truncation_com=truncation_com,
            gt_bbx_mask_on=True)
        metrics.update(normal_metrics)
        csv_columns += NORMAL_CSV_COLUMNS

    if not args.no_exact_recall:
        # eval_mesh's recall compares GT against a point SAMPLE of the
        # prediction, which carries a discretisation floor of ~0.5/sqrt(density)
        # (~2.4 mm on Ignatius, i.e. most of T&T's 3 mm tau). That floor also
        # scales with the mesh's own area and tessellation, so a
        # smoother/smaller-area mesh is scored on a different yardstick.
        # Point-to-triangle distance against the surface itself has no floor
        # and cannot be gamed by tessellation.
        from eval_gradhess_protocol import exact_surface_distance
        import numpy as np
        import open3d as o3d

        gt_pts = np.asarray(o3d.io.read_point_cloud(gt_pcd_path).points)
        if gt_pts.shape[0] > args.exact_recall_points:
            sel = np.random.default_rng(0).choice(
                gt_pts.shape[0], args.exact_recall_points, replace=False)
            gt_pts = gt_pts[sel]
        # measured in metric units: pass an identity normalization
        dist = exact_surface_distance(args.pred_mesh, np.zeros(3), 1.0,
                                      d.get("crop_json"), gt_pts)
        rec = float(np.mean(dist < dist_thre)) * 100.0
        prec = metrics['Precision [Accuracy] (%)']
        metrics['Recall_exact (%)'] = rec
        metrics['F-score_exact (%)'] = 0.0 if prec + rec == 0 else \
            2 * prec * rec / (prec + rec)
        # threshold-free companions: the whole effect being measured on this
        # scene is ~0.07 mm, too small to rest on one hard threshold
        metrics['GT_to_surface_median (m)'] = float(np.median(dist))
        metrics['GT_to_surface_mean (m)'] = float(np.mean(dist))
        csv_columns += ['Recall_exact (%)', 'F-score_exact (%)',
                        'GT_to_surface_median (m)', 'GT_to_surface_mean (m)']

    print(f"{args.dataset} / {args.method_name}: {metrics}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{d['dataset_name']}{args.method_name}_eval.csv")
    with open(out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(metrics)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
