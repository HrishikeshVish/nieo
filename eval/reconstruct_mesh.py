"""Re-run marching-cubes mesh extraction from an already-trained checkpoint at
different mc_res_m / mc_vis_level settings, without retraining -- for probing
whether reconstruction artifacts (floating components, tessellation noise)
are marching-cubes settings rather than the field itself.

mc_res_m is the marching-cubes grid voxel size (metres); mc_vis_level controls
how strict the octree-occupancy mask is (utils/mesher.py's query_points does
check_level = min(octree.featured_level_num, mc_vis_level) - 1 -- a coarser
(smaller) mc_vis_level is a more permissive mask, since fewer, coarser octree
levels have to be occupied for a query point to be considered inside the
reconstructable region; that permissiveness is one plausible source of
floating far-away components).

mc_query_level (which octree level's node positions get iterated over) is
NOT swept here -- it is structurally tied to the octree architecture
(tree_level_world - tree_level_feat + 1) and not an independent knob the way
mc_res_m/mc_vis_level are.

Usage:
    python eval/reconstruct_mesh.py \
        --checkpoint experiments/<run_dir>/model/model_iter_30000.pth \
        --mc_res_m 0.005 0.008 0.012 --mc_vis_level 1 2 3 \
        --output_dir experiments/evaluation/truck_mc_ablation --tag vanilla_iter30000
"""
import argparse
import glob
import itertools
import os
import sys
import time

import numpy as np
import torch
from numpy.linalg import inv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.config import SHINEConfig
from utils.mesher import Mesher
from model.decoder import Decoder


def _find_sibling_config(checkpoint_path):
    run_path = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    matches = glob.glob(os.path.join(run_path, "*.yaml"))
    if len(matches) != 1:
        sys.exit(f"expected exactly one sibling *.yaml next to {run_path}, found {matches}; "
                  "pass --config explicitly")
    return matches[0]


def load_model(config_path, checkpoint_path, device=None):
    config = SHINEConfig()
    config.load(config_path)
    if device is not None:
        config.device = device

    # same weights_only=False rationale as eval/field_metrics.py: these are
    # our own runs' checkpoints, which pickle a whole FeatureOctree.
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
    octree = checkpoint["feature_octree"]
    phys_eng = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned)
    phys_eng.load_state_dict(checkpoint["geo_decoder"])
    phys_eng.to(config.device)
    return config, octree, phys_eng


def recon_bbx_mesh_tiled(mesher, bbx, voxel_size, mesh_path, tile_voxels,
                          estimate_normal=True, filter_isolated_mesh=True):
    """mesher.recon_bbx_mesh, but marching-cubed in slabs so peak memory is set
    by the slab rather than by the scene.

    The monolithic path builds the whole grid at once: Newer College Quad at
    mc_res_m 0.1 is 948x1370x310 = 402.6M points, a 4.5 GiB coord tensor plus a
    3.2 GB float64 sdf array and its mask, which does not fit on a 16 GB card
    alongside other work.

    The grid geometry below is copied from Mesher.get_query_from_bbx so the
    tiled result is the same grid, not merely a similar one: same pad_voxel,
    same extra voxel below ground on z, same [-1,1] world_scale.

    Tiles are split along x and ADJACENT TILES SHARE THEIR BOUNDARY PLANE. A
    marching-cubes cell spans samples i..i+1, so a tile holding samples
    [start, end] owns cells start..end-1; starting the next tile at `end`
    rather than `end+1` means every cell belongs to exactly one tile -- no gap
    along the seam and no cell meshed twice. Vertices that land exactly on a
    shared plane are produced by both neighbours, so the seams are welded with
    merge_close_vertices afterwards.
    """
    import open3d as o3d

    min_bound = bbx.get_min_bound()
    max_bound = bbx.get_max_bound()
    pad = mesher.config.pad_voxel
    voxel_num_xyz = (np.ceil((max_bound - min_bound) / voxel_size) + pad * 2).astype(np.int_)
    voxel_origin = min_bound - pad * voxel_size
    voxel_origin[2] -= voxel_size      # extra voxel underground, as upstream does
    voxel_num_xyz[2] += 1

    nx, ny, nz = (int(v) for v in voxel_num_xyz)
    # samples per slab, chosen so nx_tile*ny*nz stays under the budget; at least
    # 2 so a slab contains one cell
    per_slab = max(2, int(tile_voxels // max(ny * nz, 1)))
    total = nx * ny * nz
    print(f"grid {nx}x{ny}x{nz} = {total/1e6:.1f}M points; "
          f"slabs of {per_slab} samples along x "
          f"(<= {per_slab*ny*nz/1e6:.1f}M points, "
          f"~{per_slab*ny*nz*8/2**30:.2f} GiB per sdf array)")

    all_verts, all_faces, n_verts = [], [], 0
    starts = list(range(0, nx - 1, per_slab - 1))    # -1: share the boundary plane
    for k, start in enumerate(starts):
        end = min(start + per_slab - 1, nx - 1)      # inclusive
        n_slab = end - start + 1
        if n_slab < 2:
            continue
        x = torch.arange(start, end + 1, dtype=torch.int16, device=mesher.cur_device)
        y = torch.arange(ny, dtype=torch.int16, device=mesher.cur_device)
        z = torch.arange(nz, dtype=torch.int16, device=mesher.cur_device)
        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
        coord = torch.stack((gx.flatten(), gy.flatten(), gz.flatten())).transpose(0, 1).float()
        del gx, gy, gz
        coord *= voxel_size
        coord += torch.tensor(voxel_origin, dtype=mesher.dtype, device=mesher.cur_device)
        coord *= mesher.world_scale

        print(f"  slab {k+1}/{len(starts)}: x[{start}..{end}] "
              f"({coord.shape[0]/1e6:.1f}M points)")
        sdf_pred, _, mc_mask = mesher.query_points(
            coord, mesher.config.infer_bs, True, False, mesher.config.mc_mask_on)
        del coord
        mc_sdf, _, mc_mask = mesher.assign_to_bbx(
            sdf_pred, None, mc_mask, np.array([n_slab, ny, nz]))
        # slab origin: the global origin shifted to this slab's first sample
        slab_origin = voxel_origin.copy()
        slab_origin[0] += start * voxel_size
        verts, faces = mesher.mc_mesh(mc_sdf, mc_mask, voxel_size, slab_origin)
        del mc_sdf, mc_mask, sdf_pred
        if len(verts):
            all_verts.append(verts)
            all_faces.append(np.asarray(faces) + n_verts)
            n_verts += len(verts)

    if not all_verts:
        print("WARNING: every slab produced an empty mesh")
        verts = np.zeros((0, 3))
        faces = np.zeros((0, 3), dtype=np.int32)
    else:
        verts = np.concatenate(all_verts, axis=0)
        faces = np.concatenate(all_faces, axis=0)

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces))
    before = len(mesh.vertices)
    # weld the duplicated seam vertices; 1e-6 m is far below mc_res_m and far
    # above the float error between a slab's origin arithmetic and the global's
    mesh = mesh.merge_close_vertices(1e-6)
    print(f"  welded seams: {before} -> {len(mesh.vertices)} vertices")

    if estimate_normal:
        mesh.compute_vertex_normals()
    if filter_isolated_mesh:
        mesh = mesher.filter_isolated_vertices(mesh, mesher.config.min_cluster_vertices)
    mesh.transform(mesher.global_transform)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"save the mesh to {mesh_path}")
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mc_res_m", type=float, nargs="+", required=True)
    ap.add_argument("--mc_vis_level", type=int, nargs="+", default=None,
                     help="required for the octree-masked path; ignored with --bbx_map_ply")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--tag", default="mesh")
    ap.add_argument("--bbx_map_ply", default=None,
                     help="reproduce a mc_with_octree:False run's meshing (mesher."
                          "recon_bbx_mesh over the whole map bounding box) instead of the "
                          "octree-masked path. Pass the run's own map/pc_map_down.ply: "
                          "dataset.map_bbx is exactly that cloud's axis-aligned bounding "
                          "box (see dataset/lidar_dataset.py), so reading it back gives "
                          "the identical box the in-run marching cubes would have used. "
                          "--mc_vis_level is then ignored, because that path applies no "
                          "octree-occupancy mask. Exists so a config whose in-run MC OOMs "
                          "can still be meshed under its own protocol, after training has "
                          "released the sample pool.")
    ap.add_argument("--tile_voxels", type=float, default=25e6,
                     help="with --bbx_map_ply, marching-cube the grid in slabs of at most "
                          "this many query points instead of all at once, so peak memory "
                          "is set by the slab and not by the scene. The default 25e6 is "
                          "~0.2 GiB of float64 sdf per slab. 0 disables tiling and "
                          "restores the single-allocation path.")
    args = ap.parse_args()

    if not args.bbx_map_ply and not args.mc_vis_level:
        ap.error("--mc_vis_level is required unless --bbx_map_ply is given")

    config_path = args.config or _find_sibling_config(args.checkpoint)
    config, octree, phys_eng = load_model(config_path, args.checkpoint)

    mesher = Mesher(config, octree, phys_eng, None)
    # Same rule train.py uses (mesher.global_transform =
    # inv(dataset.begin_pose_inv)): with first_frame_ref: False, begin_pose_inv
    # is identity plus a z-shift of global_shift_default -- reproduced directly
    # here so this doesn't need to load the dataset just to get one constant.
    begin_pose_inv = np.eye(4)
    begin_pose_inv[2, 3] += config.global_shift_default
    mesher.global_transform = inv(begin_pose_inv)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.bbx_map_ply:
        import open3d as o3d
        bbx = o3d.io.read_point_cloud(args.bbx_map_ply).get_axis_aligned_bounding_box()
        print(f"map_bbx from {args.bbx_map_ply}: "
              f"min {np.asarray(bbx.min_bound).round(2)} max {np.asarray(bbx.max_bound).round(2)}")
        for mc_res_m in args.mc_res_m:
            tag = f"{args.tag}_res{mc_res_m}_bbx"
            mesh_path = os.path.join(args.output_dir, f"{tag}.ply")
            map_path = os.path.join(args.output_dir, f"{tag}_map.ply")
            print(f"=== mc_res_m={mc_res_m} (full bbox, no octree mask) ===")
            t0 = time.time()
            if args.tile_voxels:
                recon_bbx_mesh_tiled(mesher, bbx, mc_res_m, mesh_path, args.tile_voxels)
            else:
                mesher.recon_bbx_mesh(bbx, mc_res_m, mesh_path, map_path,
                                       save_map=False, estimate_sem=False)
            print(f"  done in {time.time()-t0:.1f}s")
        return

    for mc_res_m, mc_vis_level in itertools.product(args.mc_res_m, args.mc_vis_level):
        config.mc_vis_level = mc_vis_level
        tag = f"{args.tag}_res{mc_res_m}_lvl{mc_vis_level}"
        mesh_path = os.path.join(args.output_dir, f"{tag}.ply")
        map_path = os.path.join(args.output_dir, f"{tag}_map.ply")
        print(f"=== mc_res_m={mc_res_m} mc_vis_level={mc_vis_level} ===")
        t0 = time.time()
        mesher.recon_octree_mesh(config.mc_query_level, mc_res_m, mesh_path, map_path,
                                  save_map=False, estimate_sem=False)
        print(f"  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
