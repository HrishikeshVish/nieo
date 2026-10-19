"""
This codebase references several components from the SHINE-Mapping project:
https://github.com/PRBonn/SHINE_mapping

SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations
by Xingguang Zhong, Yue Pan, Jens Behley, and Cyrill Stachniss.
Presented at the IEEE International Conference on Robotics and Automation (ICRA), 2023.

Citation:
@inproceedings{zhong2023icra,
  title={SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit NEural Representations},
  author={Zhong, Xingguang and Pan, Yue and Behley, Jens and Stachniss, Cyrill},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}

We thank the authors for making their work open source.
"""

import sys
import numpy as np
from numpy.linalg import inv, norm
from tqdm import tqdm
import wandb
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from utils.config import SHINEConfig
from utils.tools import *
from utils.loss import *
from utils.mesher import Mesher
from utils.bump_mask import BumpMask, OnlineBumpMask, shape_operator_norm
from utils.visualizer import MapVisualizer, random_color_table
from model.feature_octree import FeatureOctree
from model.decoder import Decoder
from dataset.lidar_dataset import LiDARDataset


def run_mapping_batch():
    """
    Runs the mapping process over a batch of LiDAR frames.
    
    This function sets up the experiment, loads data and models, and performs
    training over several iterations to reconstruct a scene from LiDAR point clouds.
    """
    config = SHINEConfig()
    if len(sys.argv) > 1:
        config.load(sys.argv[1])
    else:
        sys.exit("Please provide the path to the config file.\nTry: python train.py config/<dataset>/<method>.yaml")
    
    run_path = setup_experiment(config)
    shutil.copy2(sys.argv[1], run_path)

    dev = config.device

    # Initialize feature octree and decoder
    octree = FeatureOctree(config)
    phys_eng = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned)
    sem_mlp = Decoder(config, is_geo_encoder=False)

    # Load pretrained model if specified
    if config.load_model:
        # weights_only=False: checkpoints pickle a whole FeatureOctree object,
        # which torch>=2.6's default weights_only=True refuses to unpickle.
        loaded_model = torch.load(config.model_path, weights_only=False)
        # save_checkpoint writes the geometry decoder under "geo_decoder";
        # "phys_eng" is accepted as a legacy key name.
        if "geo_decoder" in loaded_model:
            phys_eng.load_state_dict(loaded_model["geo_decoder"])
        elif "phys_eng" in loaded_model:
            phys_eng.load_state_dict(loaded_model["phys_eng"])
        else:
            sys.exit(f"{config.model_path} has no decoder weights "
                     f"(keys: {sorted(loaded_model.keys())})")
        print(f"Pretrained decoder loaded from {config.model_path}"
              + (f" (saved at iter {loaded_model['iters']})"
                 if "iters" in loaded_model else ""))
        if 'feature_octree' in loaded_model:
            octree = loaded_model["feature_octree"]
            octree.print_detail()

    # Load dataset and initialize mesher
    dataset = LiDARDataset(config, octree)
    mesher = Mesher(config, octree, phys_eng, sem_mlp)
    mesher.global_transform = inv(dataset.begin_pose_inv)

    if config.o3d_vis_on:
        vis = MapVisualizer()

    print("Load, preprocess and sample data")
    for frame_id in tqdm(range(dataset.total_pc_count)):
        if (frame_id < config.begin_frame or frame_id > config.end_frame or frame_id % config.every_frame != 0):
            continue
        t0 = get_time()
        dataset.process_frame(frame_id)
        t1 = get_time()

    if config.hessian_intrinsic_loss_on and config.intrinsic_area_weight_on:
        # must run after every frame is in the pool -- it measures how often
        # each patch of surface was observed, which is only known once the whole
        # sequence has contributed
        dataset.build_area_weights(config.intrinsic_area_voxel_m,
                                   config.intrinsic_area_count_floor)

    # Prepare parameters for optimization
    octree_feat = list(octree.parameters())
    decoder_param = list(phys_eng.parameters())
    sem_decoder_param = list(sem_mlp.parameters())
    sigma_size = torch.nn.Parameter(torch.ones(1, device=dev) * 1.0)
    sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale

    pc_map_path = run_path + '/map/pc_map_down.ply'
    dataset.write_merged_pc(pc_map_path)

    opt = setup_optimizer(config, octree_feat, decoder_param, sem_decoder_param, sigma_size)
    octree.print_detail()

    require_gradient = config.require_gradient()
    require_hessian = config.require_hessian()

    bump_mask = online_mask = None
    # the region gate applies to either hessian term (ambient-L2 or intrinsic)
    _mask_wanted = config.hessian_loss_on or config.hessian_intrinsic_loss_on
    if _mask_wanted and config.hessian_mask_path:
        bump_mask = BumpMask(config.hessian_mask_path, dev, config.dtype)
    elif _mask_wanted and config.hessian_mask_online:
        online_mask = OnlineBumpMask(octree, config, dev, config.dtype,
                                     voxel=config.hessian_mask_voxel,
                                     momentum=config.hessian_mask_momentum,
                                     warmup=config.hessian_mask_warmup,
                                     quantile=config.hessian_mask_quantile,
                                     soft=config.hessian_mask_soft,
                                     adaptive=config.hessian_mask_adaptive,
                                     score=config.hessian_mask_score,
                                     probe=config.hessian_mask_probe,
                                     every=config.hessian_mask_every,
                                     sigmas=config.hessian_mask_sigmas)
    if bump_mask is not None or online_mask is not None:
        # report the weights of whichever term the gate actually reaches
        print(f"hessian weight mask: {bump_mask or online_mask}")
        if config.hessian_loss_on:
            print(f"  [ambient]   weight_h  {config.weight_h} on ordinary surface, "
                  f"{config.weight_h_bump} where the mask fires")
        if config.hessian_intrinsic_loss_on:
            print(f"  [intrinsic] weight_hi {config.weight_hi} on ordinary surface, "
                  f"{config.weight_hi_bump} where the mask fires "
                  f"({config.weight_hi_bump / max(config.weight_hi, 1e-30):.2f}x)")

    def query_field(query_coord):
        """Evaluate the sdf (in metres) and its metric gradient at arbitrary
        points. Used by the DiffCD term, which needs the field re-evaluated at
        the projected surface location.

        enable_grad is explicit because the projection loop runs under no_grad:
        the positions are detached there, but the spatial gradient still has to
        be differentiated out of the network.
        """
        with torch.enable_grad():
            query_coord = query_coord.detach().requires_grad_(True)
            query_pred = phys_eng(octree.query_feature(query_coord))
            q_sdf, q_g, _ = get_derivatives(
                query_coord, query_pred, sigma_sigmoid, config.scale, need_hessian=False
            )
        return q_sdf, q_g

    print("Begin mapping")
    cur_base_lr = config.lr
    progress_bar = tqdm(range(config.iters))

    for iter in progress_bar:
        T0 = get_time()
        step_lr_decay(opt, cur_base_lr, iter, config.lr_decay_step, config.lr_iters_reduce_ratio)

        # Get training batch
        if config.ray_loss:
            coord, sample_depth, ray_depth, normal_label, sem_label, weight = dataset.get_batch()
        else:
            coord, sdf_label, origin, ts, normal_label, sem_label, weight = dataset.get_batch()

        # weight applied to every regularizer this iteration; several of the
        # baselines are defined with a coarse-to-fine schedule rather than a
        # constant weight
        anneal_t = anneal_factor(iter, config.iters, config.anneal_schedule,
                                 config.anneal_final_ratio)

        # [8] stochastic preconditioning: blur the field by jittering the query
        # locations. Must happen before requires_grad_ so the derivatives are
        # taken at the perturbed position.
        if config.stoch_precond_on:
            coord = perturb_coord(coord, config.stoch_precond_sigma_m * config.scale * anneal_t)

        if require_gradient:
            coord.requires_grad_(True)

        T1 = get_time()
        feature = octree.query_feature(coord)
        T2 = get_time()
        pred = phys_eng(feature)
        surface_mask = weight > 0
        free_mask = ~surface_mask
        T3 = get_time()

        sdf_m = None
        hessian_m = None
        if require_gradient:
            sdf_m, g, hessian_m = get_derivatives(
                coord, pred, sigma_sigmoid, config.scale, need_hessian=require_hessian
            )

            if config.hessian_loss_on:
                # the original term was tuned against the hessian of the raw
                # logit, so undo the metric conversion for it alone
                hessian = hessian_m / (sigma_sigmoid * config.scale)

        if config.proj_correction_on:
            cos = torch.abs(F.cosine_similarity(g, coord - origin))
            cos[~surface_mask] = 1.0
            sdf_label = sdf_label * cos

        if config.consistency_loss_on:
            near_index = torch.randint(0, coord.shape[0], (min(config.consistency_count, coord.shape[0]),), device=dev)
            shift_scale = config.consistency_range * config.scale
            random_shift = torch.rand_like(coord) * 2 * shift_scale - shift_scale
            coord_near = coord + random_shift
            coord_near = coord_near[near_index, :]
            coord_near.requires_grad_(True)
            feature_near = octree.query_feature(coord_near)
            pred_near = phys_eng(feature_near)
            g_near = get_gradient(coord_near, pred_near) * sigma_sigmoid

        cur_loss = 0.0

        # Compute main loss
        if config.ray_loss:
            pred_occ = torch.sigmoid(pred / sigma_size)
            pred_ray = pred_occ.reshape(config.bs, -1)
            sample_depth = sample_depth.reshape(config.bs, -1)

            if config.main_loss_type == "dr":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=False)
            elif config.main_loss_type == "dr_neus":
                dr_loss = batch_ray_rendering_loss(sample_depth, pred_ray, ray_depth, neus_on=True)

            cur_loss += dr_loss
        else:
            weight = torch.abs(weight)
            if config.main_loss_type == "sdf_bce":
                sdf_loss = sdf_bce_loss(pred, sdf_label, sigma_sigmoid, weight, config.loss_weight_on, config.loss_reduction)
            elif config.main_loss_type == "sdf_l1":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=False)
            elif config.main_loss_type == "sdf_l2":
                sdf_loss = sdf_diff_loss(pred, sdf_label, weight, config.scale, l2_loss=True)

            cur_loss += sdf_loss

        # Optional losses
        eikonal_loss = 0.
        hessian_loss = 0.
        hessian_intrinsic_loss = 0.
        if config.ekional_loss_on:
            eikonal_loss = ((1.0 - g[surface_mask].norm(2, dim=-1)) ** 2).mean()
            cur_loss += config.weight_e * eikonal_loss

        if config.hessian_loss_on:
            # spatially varying weight_h: weight_h on ordinary surface,
            # weight_h_bump where the artifact detector fires
            point_w = None
            if bump_mask is not None:
                point_w = bump_mask.point_weight(coord[surface_mask], config.scale,
                                                 config.weight_h, config.weight_h_bump)
            elif online_mask is not None:
                # 'shape_op' reuses the metric hessian already computed here (not
                # the raw-logit one this term is scaled against); 'monge' re-probes
                # the level set through query_field and needs no hessian at all
                curv = (shape_operator_norm(g[surface_mask], hessian_m[surface_mask])
                        if online_mask.score_mode == "shape_op" else None)
                point_w = online_mask.step(coord[surface_mask], config.scale, iter,
                                           config.weight_h, config.weight_h_bump,
                                           query_fn=query_field, curv=curv)
            hessian_loss = sdf_hessian_loss(g, hessian, surface_mask,
                                            hessian_norm_scale=config.hessian_norm_scale,
                                            grad_gate_tau=config.grad_gate_tau,
                                            point_weight=point_w)
            cur_loss += config.weight_h * anneal_t * hessian_loss

        # Domain for our two hessian terms. ours_volumetric_on puts them on
        # RegSDF's full ray-sample set instead of the near-surface band.
        all_mask = torch.ones_like(surface_mask)
        ours_mask = all_mask if config.ours_volumetric_on else surface_mask

        if config.hessian_intrinsic_loss_on:
            # corrected form of the term above: shape operator rather than the
            # ambient hessian, measure concentrated on the level set rather than
            # spread over the whole sampling band. Uses the metric hessian
            # (hessian_m), not the raw-logit `hessian` the original is scaled
            # against, and sdf_m to locate each sample relative to the surface.
            # sdf_label is in scaled cube units; the band width is in metres.
            # Uses the label rather than the prediction so the weighting is
            # fixed data -- see intrinsic_hessian_loss's `d_m` note.
            area_w = None
            if config.intrinsic_area_weight_on:
                area_w = dataset.area_weight_for(coord[surface_mask])

            # Spatially varying smoothing strength: weight_hi on ordinary
            # surface, weight_hi_bump where the artifact detector fires. The
            # returned multiplier is RELATIVE (1 off-mask, ratio on-mask), so
            # the `config.weight_hi *` below stays correct and nothing is
            # double-counted. Without the gate, a single global weight_hi high
            # enough to remove artifacts also flattens genuine curved geometry.
            point_wi = None
            if bump_mask is not None:
                point_wi = bump_mask.point_weight(coord[ours_mask], config.scale,
                                                  config.weight_hi, config.weight_hi_bump)
            elif online_mask is not None:
                curv_i = (shape_operator_norm(g[ours_mask], hessian_m[ours_mask])
                          if online_mask.score_mode == "shape_op" else None)
                point_wi = online_mask.step(coord[ours_mask], config.scale, iter,
                                            config.weight_hi, config.weight_hi_bump,
                                            query_fn=query_field, curv=curv_i)
            hessian_intrinsic_loss = intrinsic_hessian_loss(
                g, hessian_m, sdf_label.detach() / config.scale, ours_mask,
                band_h_m=config.intrinsic_band_h_m,
                grad_floor=config.intrinsic_grad_floor,
                point_weight=point_wi,
                area_weight=area_w, squared=config.intrinsic_squared)
            cur_loss += config.weight_hi * anneal_t * hessian_intrinsic_loss

        # ---- baseline regularizers from the compared works ------------------
        baseline_losses = {}

        # [1] RegSDF's two terms are both means over R, its full ray-sample
        # set: "we encourage the small second order derivatives EVERYWHERE in
        # space", L_H = (1/|R|) sum_{x in R} ||Hf(x)||_1. L_M in particular
        # must not be restricted to off-surface samples: the regularized Dirac
        # delta_eps(f) peaks at f = 0, where almost all of its mass lies.
        # Diagnostic-only (default False): restricts RegSDF's two terms to
        # surface_mask instead of all_mask, isolating the volumetric-vs-band
        # question against our intrinsic term's Gaussian band weighting.
        # Never affects the reported RegSDF baseline, which stays on all_mask.
        regsdf_mask = surface_mask if config.regsdf_surface_mask_on else all_mask

        if config.hessian_l1_loss_on:  # [1] RegSDF
            baseline_losses['hessian_l1'] = hessian_l1_loss(hessian_m, regsdf_mask)
            cur_loss += config.weight_hl1 * anneal_t * baseline_losses['hessian_l1']

        if config.minimal_surface_loss_on:  # [1] RegSDF
            baseline_losses['minimal_surface'] = minimal_surface_loss(
                sdf_m, regsdf_mask, eps_m=config.minimal_surface_eps_m)
            cur_loss += config.weight_ms * anneal_t * baseline_losses['minimal_surface']

        if config.gaussian_curvature_loss_on:  # [1] RegSDF + developability add-on
            # |kappa_1 kappa_2| on the level set: zero for planes/cylinders/cones,
            # positive for spheres/saddles/noise. Uses the same mask as RegSDF's
            # own terms so the only variable against that baseline is the
            # functional, not the domain.
            baseline_losses['gaussian_curvature'] = gaussian_curvature_loss(
                g, hessian_m, regsdf_mask)
            cur_loss += config.weight_gk * anneal_t * baseline_losses['gaussian_curvature']

        if config.hessian_frob_loss_on:  # [2] NeuralSSD E_H
            # E_H(f) = integral over Omega of ||H(f)||_F^2 dV (their Eq. 4) --
            # squared Frobenius, volumetric, so every sample in the batch.
            baseline_losses['hessian_frob'] = hessian_frobenius_loss(
                hessian_m, all_mask)
            cur_loss += config.weight_hf * anneal_t * baseline_losses['hessian_frob']

        if config.normal_align_loss_on:  # [2] NeuralSSD E_N
            baseline_losses['normal_align'] = normal_alignment_loss(g, normal_label, surface_mask)
            cur_loss += config.weight_na * anneal_t * baseline_losses['normal_align']

        if config.grad_hessian_align_loss_on:  # [3] GradHess
            # sdf_label is in scaled cube units, beta's delta is per metre
            baseline_losses['grad_hess_align'] = grad_hessian_align_loss(
                g, hessian_m, ours_mask,
                d_m=sdf_label.detach() / config.scale,
                delta_per_m=config.grad_hessian_delta_per_m)
            cur_loss += config.weight_gh * anneal_t * baseline_losses['grad_hess_align']

        if config.singular_hessian_loss_on:  # [4] Neural-Singular-Hessian
            baseline_losses['singular_hessian'] = singular_hessian_loss(hessian_m, surface_mask)
            cur_loss += config.weight_sh * anneal_t * baseline_losses['singular_hessian']

        if config.relaxed_eikonal_loss_on:  # [4] Neural-Singular-Hessian
            baseline_losses['relaxed_eikonal'] = relaxed_eikonal_loss(
                g, surface_mask, sigma_min=config.relaxed_eikonal_sigma_min)
            cur_loss += config.weight_re * baseline_losses['relaxed_eikonal']

        if config.divergence_loss_on:  # [5] DiGS
            baseline_losses['divergence'] = divergence_loss(hessian_m, free_mask)
            cur_loss += config.weight_div * anneal_t * baseline_losses['divergence']

        if config.siren_offsurface_loss_on:  # [6] SIREN L_D
            baseline_losses['siren_offsurface'] = siren_offsurface_loss(
                sdf_m, free_mask, alpha=config.siren_alpha)
            cur_loss += config.weight_so * baseline_losses['siren_offsurface']

        if config.diffcd_loss_on:  # [7] DiffCD
            # reference set: the measured hit points of the rays in this batch,
            # recovered by walking each surface sample back along its own ray
            with torch.no_grad():
                ray_dir = F.normalize(coord - origin, dim=-1)
                hit_points = (coord - sdf_label.unsqueeze(-1) * ray_dir)[surface_mask]
                if hit_points.shape[0] > config.diffcd_ref_count:
                    sel = torch.randint(0, hit_points.shape[0],
                                        (config.diffcd_ref_count,), device=dev)
                    hit_points = hit_points[sel]
            baseline_losses['diffcd'] = diffcd_surface_to_point_loss(
                coord, sdf_m, g, surface_mask, hit_points, query_field,
                newton_steps=config.diffcd_newton_steps, scale=config.scale)
            cur_loss += config.weight_diffcd * baseline_losses['diffcd']

        consistency_loss = 0.
        if config.consistency_loss_on:
            consistency_loss = (1.0 - F.cosine_similarity(g[near_index, :], g_near)).mean()
            cur_loss += config.weight_c * consistency_loss

        normal_loss = 0.
        if config.normal_loss_on:
            g_direction = g / g.norm(2, dim=-1)
            normal_diff = g_direction - normal_label
            normal_loss = (normal_diff[surface_mask].abs()).norm(2, dim=1).mean()
            cur_loss += config.weight_n * normal_loss

        T4 = get_time()
        opt.zero_grad(set_to_none=True)
        cur_loss.backward()
        opt.step()
        T5 = get_time()

        # optional terms stay plain floats when disabled, so unwrap defensively
        def _val(x):
            return x.item() if torch.is_tensor(x) else x

        postfix = {"iter": iter, "loss": cur_loss.item()}
        if config.ekional_loss_on:
            postfix["eikonal"] = _val(eikonal_loss)
        if config.hessian_loss_on:
            postfix["hessian"] = _val(hessian_loss)
        if config.hessian_intrinsic_loss_on:
            postfix["hess_int"] = _val(hessian_intrinsic_loss)
        for name, value in baseline_losses.items():
            postfix[name] = _val(value)
        progress_bar.set_postfix(postfix)

        # Logging with wandb
        if config.wandb_vis_on:
            wandb_log_content = {
                'iter': iter,
                'loss/total_loss': cur_loss,
                'loss/eikonal_loss': eikonal_loss,
                'loss/normal_loss': normal_loss,
                'timing(s)/load': T1 - T0,
                'timing(s)/get_indices': T2 - T1,
                'timing(s)/inference': T3 - T2,
                'timing(s)/cal_loss': T4 - T3,
                'timing(s)/back_prop': T5 - T4,
                'timing(s)/total': T5 - T0,
                'para/sigma': sigma_size,
                'para/anneal': anneal_t,
                'mem/peak_gb': torch.cuda.max_memory_allocated() / 1024 ** 3,
                'mem/hessian_points': int(surface_mask.numel()) if require_hessian else 0,
            }

            for name, value in baseline_losses.items():
                wandb_log_content['loss/' + name] = value
            if config.hessian_intrinsic_loss_on:
                wandb_log_content['loss/hessian_intrinsic_loss'] = hessian_intrinsic_loss
            if config.hessian_loss_on:
                wandb_log_content['loss/hessian_loss'] = hessian_loss
                if online_mask is not None:
                    # should fall over training: smoothing a region lowers its
                    # ||S||, which drops it back below the frozen threshold
                    wandb_log_content['mask/frac_at_high_weight'] = online_mask.last_frac

            if config.ray_loss:
                wandb_log_content['loss/render_loss'] = dr_loss
            else:
                wandb_log_content['loss/sdf_loss'] = sdf_loss
                wandb_log_content['loss/consistency_loss'] = consistency_loss
                # sem_loss might be defined in an extended context
                if 'sem_loss' in locals():
                    wandb_log_content['loss/sem_loss'] = sem_loss

            wandb.log(wandb_log_content)

        # Checkpoint saving logic is assumed to continue after this
        if (((iter+1) % config.save_freq_iters) == 0 and iter > 0):
            checkpoint_name = 'model/model_iter_' + str(iter+1)
            # octree.clear_temp()
            save_checkpoint(octree, phys_eng, sem_mlp, opt, run_path, checkpoint_name, iter)
            save_decoder(phys_eng, sem_mlp, run_path, checkpoint_name) # save both the gro and sem decoders

        # reconstruction by marching cubes
        if (((iter+1) % config.vis_freq_iters) == 0 and iter > 0): 
            print("Begin mesh reconstruction from the implicit map")    
            if not config.time_conditioned:           
                mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + ".ply"
                map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + ".ply"
                if config.mc_with_octree: # default
                    cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                else:
                    cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                if config.o3d_vis_on:
                    cur_mesh.transform(dataset.begin_pose_inv)
                    vis.update_mesh(cur_mesh)
            else:
                vis.stop()
                for frame_id in tqdm(range(dataset.total_pc_count)):
                    if (frame_id < config.begin_frame or frame_id > config.end_frame or \
                        frame_id % 2 != 0):
                        continue
                    mesher.ts = frame_id
                    mesh_path = run_path + '/mesh/mesh_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
                    map_path = run_path + '/map/sdf_map_iter_' + str(iter+1) + '_ts_' + str(frame_id) + ".ply"
                    if config.mc_with_octree: # default
                        cur_mesh = mesher.recon_octree_mesh(config.mc_query_level, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)
                    else:
                        cur_mesh = mesher.recon_bbx_mesh(dataset.map_bbx, config.mc_res_m, mesh_path, map_path, config.save_map, config.semantic_on)

                    if config.o3d_vis_on:
                        cur_mesh.transform(dataset.begin_pose_inv)
                        vis.update_mesh(cur_mesh)

    if config.o3d_vis_on:
        vis.stop()


if __name__ == "__main__":
    run_mapping_batch()