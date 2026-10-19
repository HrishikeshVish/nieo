# This file is derived from [Atlas](https://github.com/magicleap/Atlas).
# Originating Author: Zak Murez (zak.murez.com)
# Modified for [SHINEMapping] by Yue Pan.

# Original header:
# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import os
import sys

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def eval_mesh(file_pred, file_trgt, down_sample_res=0.02, threshold=0.05, truncation_acc=0.50, truncation_com=0.50, gt_bbx_mask_on= True,
              mesh_sample_point=10000000, possion_sample_init_factor=5, crop_json=None,
              sample_density_per_m2=None, use_mesh_vertices=False):
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        sample_density_per_m2: if set, overrides mesh_sample_point with
            round(density * mesh surface area), so two meshes are always sampled
            at the same points-per-square-metre regardless of how much surface
            each one has. See _sample_count_for_mesh for why this matters.
            Ignored when use_mesh_vertices is True.
        use_mesh_vertices: score the mesh's OWN vertices instead of resampling
            its surface. Preferred: the vertices are the reconstruction, so no
            sampling density has to be chosen, nothing is invented inside the
            holes a marching-cubes mesh legitimately has, and the cost is set by
            the mesh rather than by its surface area. Resampling needs
            area-proportional density to stop a smoother (smaller-area) mesh
            being sampled denser than the baseline; scoring vertices directly
            removes that failure mode, and avoids the scaling problem of large
            scenes (Caterpillar's 218 m^2 mesh needs 226M samples at matched
            density).
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """

    mesh_pred = o3d.io.read_triangle_mesh(file_pred)

    pcd_trgt = o3d.io.read_point_cloud(file_trgt)

    def _pred_points(mesh):
        """The predicted point set: the mesh's own vertices, or a uniform
        surface sample when the caller explicitly asks for the old protocol."""
        if use_mesh_vertices:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
            return pcd
        return mesh.sample_points_uniformly(
            number_of_points=_sample_count_for_mesh(mesh, mesh_sample_point,
                                                    sample_density_per_m2))

    # Tanks-and-Temples publishes a per-scene crop volume (<Scene>.json,
    # an open3d SelectionPolygonVolume) and its own evaluation toolkit crops the
    # reconstruction with it before scoring. Use that when available, since it
    # is the protocol the benchmark's published numbers are computed under --
    # the gt-bbox fallback below is a different, and for Ignatius a stricter,
    # crop: the official polygon follows the statue outline in XY, so it
    # excludes surrounding ground that the axis-aligned bbox keeps (measured
    # +2.3pp F-score on both a vanilla and a regularized mesh). Note this
    # shifts every method's numbers equally -- it is a protocol correction, not
    # a quality improvement.
    if crop_json is not None:
        if not os.path.exists(crop_json):
            sys.exit(f"crop volume {crop_json} not found")
        vol = o3d.visualization.read_selection_polygon_volume(crop_json)
        # crop_point_cloud only accepts a point cloud, so build the predicted
        # points first and skip the mesh-level crop entirely
        pcd_sample_pred = vol.crop_point_cloud(_pred_points(mesh_pred))
        return _eval_sampled(pcd_sample_pred, pcd_trgt, down_sample_res, threshold,
                             truncation_acc, truncation_com)

    # (optional) filter the prediction outside the gt bounding box (since gt sometimes is not complete enough)
    if gt_bbx_mask_on:
        trgt_bbx = pcd_trgt.get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound()
        min_bound[2]-=down_sample_res
        max_bound = trgt_bbx.get_max_bound()
        max_bound[2]+=down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound) 
        mesh_pred = mesh_pred.crop(trgt_bbx)
        # pcd_sample_pred = pcd_sample_pred.crop(trgt_bbx)

    # pcd_sample_pred = mesh_pred.sample_points_poisson_disk(number_of_points=mesh_sample_point, init_factor=possion_sample_init_factor)
    pcd_sample_pred = _pred_points(mesh_pred)

    return _eval_sampled(pcd_sample_pred, pcd_trgt, down_sample_res, threshold,
                         truncation_acc, truncation_com)


def _sample_count_for_mesh(mesh, mesh_sample_point, sample_density_per_m2):
    """How many points to draw from a mesh before the voxel downsample.

    A fixed count gives every mesh the same *number* of samples, which means a
    smoother method -- one whose mesh has less surface area -- gets sampled at a
    higher density than the baseline it is compared against. Density is not
    neutral here: the 2 mm voxel grid this eval downsamples onto is not
    saturated at the usual 10M, so recall keeps climbing with the sample count
    (measured on Ignatius: +0.64pp from 10M to 20M, +0.24pp more at 40M, for
    both vanilla and the regularized mesh). Fixing the density instead of the
    count removes surface area from the comparison.

    Measured caveat, so this is not mistaken for a fix to a ranking: on the
    Ignatius 20k pair, density matching moves the regularized mesh by -0.06pp
    F-score, i.e. the vanilla-vs-ours gap there is geometry (recall/shrinkage),
    not sampling. This makes the protocol fair; it does not make it favourable.
    """
    if sample_density_per_m2 is None:
        return mesh_sample_point
    n = int(round(sample_density_per_m2 * mesh.get_surface_area()))
    return max(n, 1)


def _eval_sampled(pcd_sample_pred, pcd_trgt, down_sample_res, threshold,
                  truncation_acc, truncation_com):
    """Metric computation shared by both crop paths, so the official
    <Scene>.json crop and the gt-bbox fallback differ only in which predicted
    points survive -- never in how they are scored."""
    if down_sample_res > 0:
        pred_pt_count_before = len(pcd_sample_pred.points)
        pcd_pred = pcd_sample_pred.voxel_down_sample(down_sample_res)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample_res)
        pred_pt_count_after = len(pcd_pred.points)
        print("Predicted mesh unifrom sample: ", pred_pt_count_before, " --> ", pred_pt_count_after, " (", down_sample_res, "m)")
    else:
        # no voxel grid on either side -- both clouds are scored exactly as
        # they are
        pcd_pred = pcd_sample_pred
        print("Predicted points: ", len(pcd_pred.points), " GT points: ", len(pcd_trgt.points), " (no downsampling)")


    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist_p = nn_correspondance(verts_trgt, verts_pred, truncation_acc, True) # find nn in ground truth samples for each predict sample -> precision related
    _, dist_r = nn_correspondance(verts_pred, verts_trgt, truncation_com, False) # find nn in predict samples for each ground truth sample -> recall related
    
    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)

    dist_p_s = np.square(dist_p)
    dist_r_s = np.square(dist_r)

    dist_p_mean = np.mean(dist_p)
    dist_r_mean = np.mean(dist_r) 

    dist_p_s_mean = np.mean(dist_p_s)
    dist_r_s_mean = np.mean(dist_r_s) 

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

    precision = np.mean((dist_p < threshold).astype('float')) * 100.0 # %
    recall = np.mean((dist_r < threshold).astype('float')) * 100.0 # %
    fscore = 2 * precision * recall / (precision + recall) # %
    
    metrics = {'MAE_accuracy (m)': dist_p_mean,
               'MAE_completeness (m)': dist_r_mean,
               'Chamfer_L1 (m)': chamfer_l1,
               'Chamfer_L2 (m)': chamfer_l2, 
               'Precision [Accuracy] (%)': precision, 
               'Recall [Completeness] (%)': recall,
               'F-score (%)': fscore, 
               'Spacing (m)': down_sample_res,  # evlaution setup
               'Inlier_threshold (m)': threshold,  # evlaution setup
               'Outlier_truncation_acc (m)': truncation_acc, # evlaution setup
               'Outlier_truncation_com (m)': truncation_com  # evlaution setup
               }
    return metrics


def nn_correspondance(verts1, verts2, truncation_dist, ignore_outlier=True):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
        scalar truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
    Returns:
        ([indices], [distances])
    """

    if len(verts1) == 0 or len(verts2) == 0:
        return [], []

    # Vectorized cKDTree query; same nearest neighbour, truncation rule and
    # output order as the original per-point KDTreeFlann loop, but feasible
    # past ~1e7 points.
    tree = cKDTree(verts1.astype(np.float64))
    dist, inds = tree.query(verts2.astype(np.float64), k=1, workers=-1)

    keep = dist < truncation_dist
    if ignore_outlier:
        # outliers are dropped entirely, from BOTH returned lists
        return inds[keep].tolist(), dist[keep].tolist()
    # ...otherwise they are kept, with the distance clamped to the truncation
    dist = np.where(keep, dist, truncation_dist)
    return inds.tolist(), dist.tolist()


def _load_points_with_normals(file_path, mesh_sample_point, normal_knn=30):
    """Load a mesh or point-cloud file as an oriented point cloud.

    Meshes get vertex normals computed from face connectivity, then uniformly
    sampled points inherit interpolated normals (open3d does this for free
    once compute_vertex_normals() has run) -- this is how [1] RegSDF derives
    normals ("inherited from mesh triangles"). Files with no faces (e.g. a raw
    scanner point cloud) fall back to PCA-based
    normal estimation, which has no consistent global orientation -- every
    metric below is written to tolerate that (abs(cos) or an explicit
    best-of-two-flips search), never assuming a sign convention.

    Returns:
        (points [N,3] float64, normals [N,3] float64, triangle_count int,
         surface_area float [m^2]; 0.0 for a file with no faces)
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    triangle_count = len(mesh.triangles)
    # area is only defined once there are faces; a raw .pcd reports 0.0
    surface_area = float(mesh.get_surface_area()) if triangle_count > 0 else 0.0

    if triangle_count > 0:
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=mesh_sample_point)
    else:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(norm, 1e-12, None)
    return points, normals, triangle_count, surface_area


def _nn_with_normals(points_ref, normals_ref, points_query, normals_query, truncation_dist):
    """For each point in points_query, find its nearest neighbor in points_ref.

    Returns three parallel arrays, one entry per query point whose nearest
    match falls within truncation_dist: (distances, that query point's own
    normal, its matched reference point's normal). Explicitly re-filtering and
    returning normals_query alongside the match -- rather than letting a
    caller slice the original normals_query array by count -- matters because
    truncation (partial-coverage GT, expected on Redwood) can
    drop any subset of query points, not just a trailing run; slicing by count
    would silently pair unrelated points.
    """
    if len(points_ref) == 0 or len(points_query) == 0:
        empty = np.zeros((0, 3))
        return np.zeros(0), empty, empty

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(points_ref.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd_ref)

    truncation_sq = truncation_dist ** 2
    dists, normals_query_kept, normals_ref_matched = [], [], []
    for i, pt in enumerate(points_query):
        _, inds, dist_sq = kdtree.search_knn_vector_3d(pt, 1)
        if dist_sq[0] < truncation_sq:
            dists.append(math.sqrt(dist_sq[0]))
            normals_query_kept.append(normals_query[i])
            normals_ref_matched.append(normals_ref[inds[0]])

    return np.array(dists), np.array(normals_query_kept), np.array(normals_ref_matched)


def normal_and_mesh_metrics(file_pred, file_trgt, down_sample_res=0.02, dist_threshold=0.05,
                             angle_threshold_deg=30.0, truncation_acc=0.50, truncation_com=0.50,
                             gt_bbx_mask_on=True, mesh_sample_point=10000000):
    """Normal-orientation and mesh-complexity metrics, as defined in the
    compared prior-regularizer papers [1]-[8]. Complements eval_mesh's
    pure-distance Chamfer/precision/recall/F-score, which is largely blind to
    surface smoothness/high-frequency noise.

    Returns:
        'Triangle_count': raw triangle count of the predicted mesh -- a cheap
            proxy for over-triangulated/noisy surfaces; a denser mesh also
            means more marching-cubes/rendering cost downstream.
        'Normal_consistency (%)': mean |cos(angle)| between each point's normal
            and its nearest-neighbor match's normal, both directions averaged
            (pred->trgt, trgt->pred), as percent. This is the "Normal C." metric
            reported by [2] NeuralSSD, [3] GradHess and [4] Neural-Singular-Hessian
            (their "degree of agreement between the normals of the reconstructed
            surface and ... ground truth"). abs() is used rather than raw cosine
            because a mesh/estimated-normal source is not guaranteed to be
            consistently oriented (see _load_points_with_normals) -- [1]-[4]'s own
            benchmarks (DTU/BlendedMVS/ShapeNet/Stanford Scanning) assume
            consistently-oriented meshes on both sides, which real-world lidar
            GT (often a raw scan) does not guarantee.
        'Chamfer_angle (deg)': [7] DiffCD's CA metric -- mean angle in degrees
            between matched normals, bidirectional, with the ground-truth normals
            globally flipped (not per-point) if that reduces the mean -- "compute
            CA using S with and without flipped normals, pick the one yielding the
            smaller value". Unlike Normal_consistency this is signed/not abs(), so
            it is only meaningful when GT has a genuine (if possibly globally
            flipped) orientation, not per-point PCA noise -- expect this to be
            noisier on estimated-normal GT than on a real mesh GT.
        'Normal_aug_Precision (%)' / 'Normal_aug_Recall (%)' / 'Normal_aug_F-score (%)':
            [1] RegSDF's inlier criterion -- a point counts as an inlier only if
            BOTH its nearest-neighbor distance is below dist_threshold AND the
            angular difference to that neighbor's normal is below
            angle_threshold_deg (using abs(cos) for the same orientation-robustness
            reason as Normal_consistency above). Precision/accuracy uses trgt as
            reference, recall/completeness uses pred as reference, matching
            eval_mesh's existing accuracy/completeness convention.
    """
    points_pred, normals_pred, triangle_count, surface_area = \
        _load_points_with_normals(file_pred, mesh_sample_point)

    mesh_trgt_check = o3d.io.read_triangle_mesh(file_trgt)
    if gt_bbx_mask_on:
        if len(mesh_trgt_check.triangles) > 0:
            trgt_bbx = mesh_trgt_check.get_axis_aligned_bounding_box()
        else:
            trgt_bbx = o3d.io.read_point_cloud(file_trgt).get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound() - down_sample_res
        max_bound = trgt_bbx.get_max_bound() + down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        keep = trgt_bbx.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points_pred))
        points_pred = points_pred[keep]
        normals_pred = normals_pred[keep]

    points_trgt, normals_trgt, _, _ = _load_points_with_normals(file_trgt, mesh_sample_point)

    if down_sample_res > 0:
        def _voxel_down(points, normals, res):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)
            pcd = pcd.voxel_down_sample(res)
            n = np.asarray(pcd.normals)
            n = n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-12, None)
            return np.asarray(pcd.points), n

        points_pred, normals_pred = _voxel_down(points_pred, normals_pred, down_sample_res)
        points_trgt, normals_trgt = _voxel_down(points_trgt, normals_trgt, down_sample_res)

    # accuracy direction: each pred point vs. its nearest trgt neighbor
    dist_acc, normals_pred_kept, normals_trgt_for_pred = _nn_with_normals(
        points_trgt, normals_trgt, points_pred, normals_pred, truncation_acc)
    cos_acc = np.abs(np.sum(normals_pred_kept * normals_trgt_for_pred, axis=1)) if len(dist_acc) else np.zeros(0)
    # completeness direction: each trgt point vs. its nearest pred neighbor
    dist_com, normals_trgt_kept, normals_pred_for_trgt = _nn_with_normals(
        points_pred, normals_pred, points_trgt, normals_trgt, truncation_com)
    cos_com = np.abs(np.sum(normals_trgt_kept * normals_pred_for_trgt, axis=1)) if len(dist_com) else np.zeros(0)

    normal_consistency = 100.0 * 0.5 * (np.mean(cos_acc) + np.mean(cos_com)) if len(cos_acc) and len(cos_com) else 0.0

    # [7] DiffCD Chamfer Angle: signed, global-flip-corrected (not abs)
    def _mean_angle_deg(cos_vals):
        return math.degrees(np.mean(np.arccos(np.clip(cos_vals, -1.0, 1.0))))

    cos_acc_signed = np.sum(normals_pred_kept * normals_trgt_for_pred, axis=1) if len(dist_acc) else np.zeros(0)
    cos_com_signed = np.sum(normals_trgt_kept * normals_pred_for_trgt, axis=1) if len(dist_com) else np.zeros(0)
    if len(cos_acc_signed) and len(cos_com_signed):
        angle_normal = 0.5 * (_mean_angle_deg(cos_acc_signed) + _mean_angle_deg(cos_com_signed))
        angle_flipped = 0.5 * (_mean_angle_deg(-cos_acc_signed) + _mean_angle_deg(-cos_com_signed))
        chamfer_angle = min(angle_normal, angle_flipped)
    else:
        chamfer_angle = 0.0

    # [1] RegSDF normal-augmented precision/recall/F-score
    angle_thresh_cos = math.cos(math.radians(angle_threshold_deg))
    acc_inlier = (dist_acc < dist_threshold) & (cos_acc >= angle_thresh_cos) if len(dist_acc) else np.zeros(0, dtype=bool)
    com_inlier = (dist_com < dist_threshold) & (cos_com >= angle_thresh_cos) if len(dist_com) else np.zeros(0, dtype=bool)
    precision = 100.0 * np.mean(acc_inlier) if len(acc_inlier) else 0.0
    recall = 100.0 * np.mean(com_inlier) if len(com_inlier) else 0.0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'Triangle_count': triangle_count,
        'Surface_area (m2)': surface_area,
        'Normal_consistency (%)': normal_consistency,
        'Chamfer_angle (deg)': chamfer_angle,
        'Normal_aug_Precision (%)': precision,
        'Normal_aug_Recall (%)': recall,
        'Normal_aug_F-score (%)': fscore,
        'Dist_threshold (m)': dist_threshold,
        'Angle_threshold (deg)': angle_threshold_deg,
    }


def eval_depth(depth_pred, depth_trgt):
    """ Computes 2d metrics between two depth maps
    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth
    Returns:
        Dict of metrics
    """
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < 10) * (depth_trgt > 0) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics

def crop_intersection(file_gt, files_pred, out_file_crop, dist_thre=0.1, mesh_sample_point=1000000):
    """ Get the cropped ground truth point cloud according to the intersection of the predicted
    mesh by different methods
    Args:
        file_gt: file path of the ground truth (shoud be point cloud)
        files_pred: a list of the paths of different methods's reconstruction (shoud be mesh)
        out_file_crop: output path of the cropped ground truth point cloud
        dist_thre: nearest neighbor distance threshold in meter
        mesh_sample_point: number of the sampling points from the mesh
    """
    print("Load the original ground truth point cloud from:", file_gt)
    pcd_gt = o3d.io.read_point_cloud(file_gt)
    pcd_gt_pts = np.asarray(pcd_gt.points)
    dist_square_thre = dist_thre**2
    for i in range(len(files_pred)):
        cur_file_pred = files_pred[i]
        print("Process", cur_file_pred)
        cur_mesh_pred = o3d.io.read_triangle_mesh(cur_file_pred)

        cur_sample_pred = cur_mesh_pred.sample_points_uniformly(number_of_points=mesh_sample_point)
        
        cur_kdtree = o3d.geometry.KDTreeFlann(cur_sample_pred)
        
        crop_pcd_gt_pts = []
        for pt in pcd_gt_pts:
            _, _, dist_square = cur_kdtree.search_knn_vector_3d(pt, 1)
            
            if dist_square[0] < dist_square_thre:
                crop_pcd_gt_pts.append(pt)
        
        pcd_gt_pts = np.asarray(crop_pcd_gt_pts, dtype=np.float64)

    crop_pcd_gt = o3d.geometry.PointCloud()
    crop_pcd_gt.points = o3d.utility.Vector3dVector(pcd_gt_pts)
    
    print("Output the croped ground truth to:", out_file_crop)
    o3d.io.write_point_cloud(out_file_crop, crop_pcd_gt)

    
