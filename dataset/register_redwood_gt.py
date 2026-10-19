"""Register a Redwood laser-scanner GT point cloud into the RGB-D camera
trajectory's coordinate frame.

The GT archives ("aligned_low_*.zip") are internally self-registered across
the laser scanner's own sub-scans, but are NOT registered to the RGB-D camera
trajectory shipped alongside them (they were captured as independent
recordings). This builds a fused point cloud from the converted RGB-D
sequence (each frame transformed into world space by poses.txt), then runs
FPFH-feature RANSAC global registration followed by point-to-plane ICP to
find the rigid transform that brings the GT cloud into that trajectory frame.

Usage:
    python dataset/register_redwood_gt.py \
        --root_path data/redwood --sequence_name bedroom
"""
import argparse
import os

import numpy as np
import open3d as o3d


def load_poses_kitti(path):
    poses = np.loadtxt(path)
    return poses.reshape(-1, 3, 4)


def build_fused_trajectory_cloud(rgbd_ply_dir, poses, stride, voxel_size):
    # Each raw frame is up to ~300K points; with ~1000+ frames sampled, naively
    # concatenating full-resolution clouds before downsampling can reach 100M+
    # points and exhaust memory. Downsample every frame first, so each
    # contributes only a small amount before the single final concatenation.
    all_points = []
    frame_ids = list(range(0, len(poses), stride))
    for idx in frame_ids:
        frame_path = os.path.join(rgbd_ply_dir, f"{idx:06d}.ply")
        if not os.path.exists(frame_path):
            continue
        local = o3d.io.read_point_cloud(frame_path)
        if len(local.points) == 0:
            continue
        local = local.voxel_down_sample(voxel_size)
        T = np.eye(4)
        T[:3, :] = poses[idx]
        local.transform(T)
        all_points.append(np.asarray(local.points))
    fused = o3d.geometry.PointCloud()
    fused.points = o3d.utility.Vector3dVector(np.concatenate(all_points, axis=0))
    fused = fused.voxel_down_sample(voxel_size)
    return fused


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


# Coarse-to-fine, NOT a single tight pass. RANSAC's output on Redwood bedroom is
# ~70 cm off, so with a single 2 cm threshold every correspondence is already a
# wrong match and ICP sits in that local minimum while reporting a plausible
# fitness/rmse. Running 1.0 -> 0.5 -> 0.25 -> 0.10 -> 0.05 m recovers the
# correction (direct frame-to-GT residual 2.4 cm median, 81% within 5 cm).
# Trust the direct check, never ICP's own fitness/rmse: both are computed only over
# correspondences inside the threshold, so a tight threshold reports a fine rmse
# precisely when it is most wrong.
ICP_SCHEDULE_M = (1.0, 0.5, 0.25, 0.10, 0.05)


def refine_registration(source, target, init_transform, voxel_size):
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    transform = init_transform
    result = None
    for distance_threshold in ICP_SCHEDULE_M:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        transform = result.transformation
        print(f"  ICP thr {distance_threshold:4.2f} m: fitness {result.fitness:.4f} "
              f"inlier_rmse {result.inlier_rmse * 100:.2f} cm")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="data/redwood")
    parser.add_argument("--sequence_name", default="bedroom")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                         help="downsample/feature voxel size in meters")
    parser.add_argument("--traj_frame_stride", type=int, default=10,
                         help="use every Nth RGB-D frame to build the fused trajectory cloud")
    args = parser.parse_args()

    kitti_dir = os.path.join(args.root_path, f"{args.sequence_name}_kitti_format")
    rgbd_ply_dir = os.path.join(kitti_dir, "rgbd_ply")
    poses_path = os.path.join(kitti_dir, "poses.txt")
    gt_path = os.path.join(args.root_path, "gt_mesh", f"{args.sequence_name}.ply")
    out_path = os.path.join(args.root_path, "gt_mesh", f"{args.sequence_name}_registered.ply")

    print("Loading poses...")
    poses = load_poses_kitti(poses_path)

    print(f"Building fused trajectory point cloud (every {args.traj_frame_stride} frames)...")
    target = build_fused_trajectory_cloud(rgbd_ply_dir, poses, args.traj_frame_stride, args.voxel_size)
    print(f"  fused trajectory cloud: {len(target.points)} points")

    print("Loading GT laser scan...")
    source = o3d.io.read_point_cloud(gt_path)
    print(f"  GT cloud: {len(source.points)} points")

    source_down, source_fpfh = preprocess(source, args.voxel_size)
    target_down, target_fpfh = preprocess(target, args.voxel_size)
    print(f"  downsampled source: {len(source_down.points)}, target: {len(target_down.points)}")

    print("Running RANSAC global registration...")
    ransac_result = execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, args.voxel_size)
    print("  RANSAC fitness:", ransac_result.fitness, "inlier_rmse:", ransac_result.inlier_rmse)
    print(ransac_result.transformation)

    print("Refining with point-to-plane ICP...")
    icp_result = refine_registration(source_down, target_down, ransac_result.transformation, args.voxel_size)
    print("  ICP fitness:", icp_result.fitness, "inlier_rmse:", icp_result.inlier_rmse)
    print(icp_result.transformation)

    print(f"Applying final transform to full-resolution GT cloud and writing {out_path} ...")
    source.transform(icp_result.transformation)
    o3d.io.write_point_cloud(out_path, source)

    np.savetxt(os.path.join(args.root_path, "gt_mesh", f"{args.sequence_name}_registration_transform.txt"),
               icp_result.transformation)
    print("Done.")


if __name__ == "__main__":
    main()
