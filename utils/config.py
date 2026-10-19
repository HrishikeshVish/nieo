import yaml
import os
import torch
from typing import List

class SHINEConfig:
    def __init__(self):

        # Default values

        # settings
        self.name: str = "dummy"  # experiment name

        self.output_root: str = ""  # output root folder
        self.pc_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)

        self.label_path: str = "" # input point-wise label path, for semantic shine mapping

        self.load_model: bool = False  # load the pre-trained model or not
        self.model_path: str = "/"  # pre-trained model path

        self.first_frame_ref: bool = True  # if false, we directly use the world
        # frame as the reference frame
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 0  # end at this frame
        self.every_frame: int = 1  # process every x frame

        self.seed: int = 42  # random seed for the experiments
        self.num_workers: int = 12 # number of worker for the dataloader
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type
        self.pc_count_gpu_limit: int = 500 # maximum used frame number to be stored in the gpu

        # just a ramdom number for the global shift of the input on z axis (used to avoid octree boundary marching cubes issues on synthetic datasets like MaiCity)
        self.global_shift_default: float = 0. # 0.17241

        # baseline
        # self.run_baseline = False
        # # select from vdb_fusion, voxblox_simple, voxblox_merged, voxblox_fast
        # self.baseline_method = "vdb_fusion"
        # self.voxel_size_m = 0.2
        # self.sdf_trunc_m = 3 * self.voxel_size_m

        # process
        self.min_range: float = 2.75 # filter too-close points (and 0 artifacts)
        self.pc_radius: float = 20.0  # keep only the point cloud inside the
        # block with such radius (unit: m)
        self.min_z: float = -10.0  # filter for z coordinates (unit: m)
        self.max_z: float = 30.0

        self.rand_downsample: bool = True # apply random or voxel downsampling to input original point clcoud
        self.vox_down_m: float = 0.03 # the voxel size if using voxel downsampling (unit: m)
        self.rand_down_r: float = 1.0 # the decimation ratio if using random downsampling (0-1)

        # deprecated
        self.filter_noise: bool = False  # use SOR to remove the noise or not
        self.sor_nn: int = 25  # SOR neighborhood size
        self.sor_std: float = 2.5  # SOR std threshold
        # deprecated
        self.estimate_normal: bool = False  # estimate surface normal or not
        self.normal_radius_m: float = 0.2  # supporting radius for estimating the normal
        self.normal_max_nn: int = 20 # supporting neighbor count for estimating the normal

        # semantic related
        self.semantic_on: bool = False # semantic shine mapping on [semantic]
        self.sem_class_count: int = 20 # semantic class count: 20 for semantic kitti
        self.sem_label_decimation: int = 1 # use only 1/${sem_label_decimation} of the available semantic labels for training (fitting)
        self.filter_moving_object: bool = False

        # frame-wise downsampling voxel size for the merged map point cloud (unit: m)
        self.map_vox_down_m: float = 0.05 # 0.2 

        # octree
        self.tree_level_world: int = 10  # the total octree level, allocated for the whole space
        self.tree_level_feat: int = 4  # the octree levels with optimizable feature grid
        # start from the leaf level
        self.leaf_vox_size: float = 0.5  # voxel size of the octree leaf nodes (unit: m)
        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.05  # grid feature initialization standard deviation
        self.poly_int_on: bool = True # use polynomial interpolation or linear interpolation
        self.octree_from_surface_samples: bool = True  # Use all the surface samples or just the exact measurements to build the octree. If True may lead to larger memory, but is more robust while the reconstruction.

        # sampler
        # spilt into 3 parts for sampling
        self.surface_sample_range_m: float = 0.5 # 
        self.surface_sample_n: int = 5
        self.free_sample_begin_ratio: float = 0.3
        # self.free_sample_end_ratio: float = 1.0 # deprecated
        self.free_sample_end_dist_m: float = 0.5 # maximum distance after the surface (unit: m)
        self.free_sample_n: int = 2
        self.clearance_dist_m: float = 0.3
        self.clearance_sample_n: int = 0

        # space carving sampling related (deprecated)
        # self.carving_on = False
        # self.tree_level_carving = self.tree_level_world
        # self.carving_stop_depth_m = 0.5
        # self.carving_inte_thre_m = 0.1

        # incremental mapping
        self.continual_learning_reg: bool = True
        # regularization based
        self.lambda_forget: float = 1e5
        self.cal_importance_weight_down_rate: int = 2 # set it larger to save the consuming time
        
        # replay based
        self.window_replay_on: bool = True
        self.window_radius: float = 50.0 # unit: m

        # label
        self.occu_update_on: bool = False

        # decoder
        self.geo_mlp_level: int = 2
        self.geo_mlp_hidden_dim: int = 32
        self.geo_mlp_bias_on: bool = True
        # 'relu' (SHINE default), 'sine' (SIREN) or 'softplus'. Any Hessian
        # term needs C^2; relu is piecewise linear, so its own second
        # derivative is zero a.e. -- see model/decoder.py.
        self.mlp_activation: str = "relu"
        self.siren_omega0: float = 30.0

        self.sem_mlp_level: int = 2
        self.sem_mlp_hidden_dim: int = 32
        self.sem_mlp_bias_on: bool = True
        
        self.freeze_after_frame: int = 20  # For incremental mode only, if the decoder model is not loaded , it would be trained and freezed after such frame number

        # loss
        self.ray_loss: bool = False  # one loss on a whole ray (including depth estimation loss or the differentiable rendering loss)
        # the main loss type, select from the sample sdf loss ('sdf_bce', 'sdf_l1', 'sdf_l2') and the ray rendering loss ('dr', 'dr_neus')
        self.main_loss_type: str = 'sdf_bce'

        self.loss_reduction: str = 'mean' # select from 'mean' and 'sum' (for incremental mapping)
        
        self.sigma_sigmoid_m: float = 0.1
        self.sigma_scale_constant: float = 0.0 # scale factor adding to the constant sigma value (linear with the distance) [deprecated]
        self.logistic_gaussian_ratio: float = 0.55

        self.proj_correction_on: bool = False # conduct projective distance correction based on the sdf gradient or not [deprecated]
        
        self.predict_sdf: bool = False
        self.neus_loss_on: bool = False  # use the unbiased and occlusion-aware weights for differentiable rendering as introduced in NEUS
        self.loss_weight_on: bool = False  # if True, the weight would be given to the loss, if False, the weight would be used to change the sigmoid's shape
        self.behind_dropoff_on: bool = False  # behind surface drop off weight
        self.dropoff_min_sigma: float = 1.0
        self.dropoff_max_sigma: float = 5.0
        self.normal_loss_on: bool = False
        self.weight_n: float = 0.01
        self.ekional_loss_on: bool = False
        self.weight_e: float = 0.1

        # hessian energy (ours)
        self.hessian_loss_on: bool = False
        self.weight_h: float = 0.1
        # corrected discretization of the cited L1 surface-Hessian energy:
        # shape operator instead of the ambient hessian, and the measure
        # concentrated on the level set instead of the whole sampling band.
        # See utils.loss.intrinsic_hessian_loss. Separate flag so the original
        # ambient term stays available for ablation.
        self.hessian_intrinsic_loss_on: bool = False
        self.weight_hi: float = 0.1
        self.intrinsic_band_h_m: float = 0.003
        self.intrinsic_grad_floor: float = 0.2
        # area weighting for the surface-Hessian integral (their Eq. 18 sums
        # A_f ||H_f||, not a per-sample mean). Voxel size sets the scale at
        # which local sample density -- i.e. how often that patch was scanned --
        # is estimated and divided out.
        self.intrinsic_area_weight_on: bool = False
        self.intrinsic_area_voxel_m: float = 0.03
        self.intrinsic_area_count_floor: int = 8
        # squared Frobenius (L2 / E_H2) instead of the un-squared L1 form
        self.intrinsic_squared: bool = False
        # spatially varying weight_h: weight_h on ordinary surface,
        # weight_h_bump wherever a precomputed bump mask (utils/bump_mask.py)
        # fires. Empty path keeps a single global weight_h.
        self.hessian_mask_path: str = ""
        self.weight_h_bump: float = 3.0
        # Same idea for the intrinsic term. Kept separate because the two
        # terms differ by ~3 orders of magnitude in weight (weight_h 0.1 vs
        # weight_hi ~1e-3), so one shared 'bump weight' cannot serve both.
        self.weight_hi_bump: float = 0.003
        # ...or build the mask during the run itself from ||S||, no first pass.
        # Ignored when hessian_mask_path is set (the precomputed Monge-residual
        # mask is the stronger detector: AUC 0.904 vs 0.743 per point).
        self.hessian_mask_online: bool = False
        # 'monge' is the detector validated against hand-labelled artifact regions
        # (AUC 0.904 vs genuine edges); 'shape_op' is free but far weaker per
        # point (0.743).
        self.hessian_mask_score: str = "monge"
        self.hessian_mask_probe: int = 256      # points scored per scoring iteration
        self.hessian_mask_every: int = 4        # score every Nth iteration
        self.hessian_mask_sigmas: list = [0.05, 0.10]
        self.hessian_mask_voxel: float = 0.15   # the artifacts' own scale
        self.hessian_mask_momentum: float = 0.99
        self.hessian_mask_warmup: int = 1000
        self.hessian_mask_quantile: float = 0.8
        self.hessian_mask_soft: float = 0.15
        self.hessian_mask_adaptive: bool = False
        self.hessian_norm_scale: float = 7e-9
        # below this |grad f|, the eikonal constraint is already weakly enforced
        # (under-observed/low-confidence region); gate the hessian penalty down
        # there instead of applying it at full strength
        self.grad_gate_tau: float = 0.6

        # baseline regularizers ported from the compared prior work
        # [1] RegSDF, Zhang et al. CVPR 2022
        self.hessian_l1_loss_on: bool = False   # element-wise L1 norm of the hessian
        self.weight_hl1: float = 0.1
        self.minimal_surface_loss_on: bool = False  # regularized dirac on the sdf
        self.weight_ms: float = 0.1
        self.minimal_surface_eps_m: float = 0.1  # peak half-width in m (paper's 1/eps, rescaled)
        # Diagnostic-only, default False so it never changes the reported RegSDF
        # baseline (hessian_l1_loss_on/minimal_surface_loss_on stay on `all_mask`,
        # i.e. every ray sample, matching "we encourage the small second order
        # derivatives EVERYWHERE in space" -- their own paper's stated design).
        # When True, restricts BOTH of RegSDF's terms to surface_mask instead,
        # isolating exactly one variable (volumetric vs. surface-band support)
        # to test against our intrinsic term's band-Gaussian weighting, with
        # RegSDF's own ambient (non-normal-projected) Hessian operator and
        # weights otherwise left untouched.
        self.regsdf_surface_mask_on: bool = False
        # Optional add-on to [1] RegSDF: |K| = |kappa_1 kappa_2| developability
        # prior on the level set (utils/loss.py's gaussian_curvature_loss).
        # Default False so the reported RegSDF baseline is unchanged.
        self.gaussian_curvature_loss_on: bool = False
        self.weight_gk: float = 0.01
        # Run OUR two hessian terms (intrinsic tangential + GradHess normal) on
        # the full ray-sample set instead of surface_mask, i.e. on RegSDF's
        # domain. Default False keeps the published "ours" recipe unchanged.
        # NOTE this only changes the MASK; the terms' own distance weightings
        # (intrinsic_band_h_m's gaussian, grad_hessian_delta_per_m's beta) would
        # re-concentrate them at the surface anyway, so a volumetric config must
        # also widen/disable those.
        self.ours_volumetric_on: bool = False
        # [2] NeuralSSD, Xi et al. arXiv 2511.14283
        self.hessian_frob_loss_on: bool = False  # squared frobenius norm of the hessian
        self.weight_hf: float = 0.1
        self.normal_align_loss_on: bool = False  # squared gradient-to-normal residual
        self.weight_na: float = 0.1
        # [3] GradHess, Wang et al. NeurIPS 2023
        self.grad_hessian_align_loss_on: bool = False
        self.weight_gh: float = 0.1
        # paper Eq. 8's adaptive per-query weight beta_q = exp(-delta*|f(q)|),
        # concentrating the alignment penalty near the surface. Their delta=10
        # is in units where the shape spans 1; ours is per metre, so 0 disables
        # it and a sensible value is ~1/(surface band width). See
        # utils/loss.py's grad_hessian_align_loss.
        self.grad_hessian_delta_per_m: float = 0.0
        # [4] Neural-Singular-Hessian, Wang et al. TOG 2023
        self.singular_hessian_loss_on: bool = False
        self.weight_sh: float = 0.1
        self.relaxed_eikonal_loss_on: bool = False
        self.weight_re: float = 0.1
        self.relaxed_eikonal_sigma_min: float = 0.8
        # [5] DiGS, Ben-Shabat et al. CVPR 2022
        self.divergence_loss_on: bool = False
        self.weight_div: float = 0.1
        # [6] SIREN, Sitzmann et al. NeurIPS 2020
        self.siren_offsurface_loss_on: bool = False
        self.weight_so: float = 0.1
        self.siren_alpha: float = 100.0  # 1/m, penalizes |sdf| within ~1cm
        # [7] DiffCD, Haerenstam-Nielsen et al. ECCV 2024
        self.diffcd_loss_on: bool = False
        self.weight_diffcd: float = 0.1
        self.diffcd_newton_steps: int = 2
        self.diffcd_ref_count: int = 8192  # input points subsampled for the nn query
        # [8] Stochastic preconditioning, Ling et al. TOG 2025 (not a loss)
        self.stoch_precond_on: bool = False
        self.stoch_precond_sigma_m: float = 0.2  # initial blur std in m, annealed to 0

        # annealing applied to the regularizer weights, select from
        # 'none', 'digs' (50/25/25 phase split) and 'exp' (exponential decay to zero)
        self.anneal_schedule: str = 'none'
        self.anneal_final_ratio: float = 0.0

        # deprecated
        self.consistency_loss_on: bool = False
        self.weight_c: float = 1.0
        self.consistency_count: int = 1000
        self.consistency_range: float = 0.1 # the neighborhood points would be randomly select within the radius of xxx m
        
        self.history_weight: float = 1.0

        self.weight_s: float = 1.0  # weight for semantic classification loss

        # for dynamic reconstruction
        self.time_conditioned: bool = False

        # optimizer
        self.iters: int = 200
        self.opt_adam: bool = True  # use adam or sgd
        self.bs: int = 4096
        self.lr: float = 1e-3
        self.weight_decay: float = 0
        self.adam_eps: float = 1e-15
        self.lr_level_reduce_ratio: float = 1.0
        self.lr_iters_reduce_ratio: float = 0.1
        self.lr_decay_step: List = [10000, 50000, 100000]
        self.dropout: float = 0

        # eval
        self.wandb_vis_on: bool = False
        self.o3d_vis_on: bool = True # visualize the mesh in-the-fly using o3d visualzier or not [press space to pasue/resume]
        self.eval_on: bool = False
        self.eval_outlier_thre = 0.5  # unit:m
        self.eval_freq_iters: int = 100
        self.vis_freq_iters: int = 100
        self.save_freq_iters: int = 100
        self.mesh_freq_frame: int = 1  # do the reconstruction per x frames
        
        # marching cubes related
        self.mc_res_m: float = 0.1
        self.pad_voxel: int = 1
        self.mc_with_octree: bool = True # conducting marching cubes reconstruction within a certain level of the octree or within the axis-aligned bounding box of the whole map
        self.mc_query_level: int = 8
        self.mc_vis_level: int = 1 # masked the marching cubes for level higher than this
        self.mc_mask_on: bool = True # use mask for marching cubes to avoid the artifacts
        self.mc_local: bool = False # only do the marching cubes in a local bounding box or globally
        self.min_cluster_vertices: int = 50 # if a connected's vertices number is smaller than this value, it would get filtered
        
        self.infer_bs: int = 4096
        self.occ_binary_mc: bool = False
        self.grid_loss_vis_on: bool = False
        self.mesh_vis_on: bool = True
        self.save_map: bool = False # save the sdf map or not, the sdf would be saved in the intensity channel

        # initialization
        self.scale: float = 1.0
        self.world_size: float = 1.0

    def _opt(self, section, key, default, cast=None):
        # read an optional yaml key, falling back to the default set in __init__
        # so that config files written before the key existed keep working
        if key not in section:
            return default
        return cast(section[key]) if cast else section[key]

    def require_gradient(self):
        # does any enabled term need d(pred)/d(coord)?
        return self.normal_loss_on or self.ekional_loss_on or \
               self.proj_correction_on or self.consistency_loss_on or \
               self.normal_align_loss_on or self.relaxed_eikonal_loss_on or \
               self.diffcd_loss_on or self.require_hessian()

    def require_hessian(self):
        # does any enabled term need the second derivative?
        return self.hessian_loss_on or self.hessian_intrinsic_loss_on or \
               self.hessian_l1_loss_on or \
               self.hessian_frob_loss_on or self.grad_hessian_align_loss_on or \
               self.singular_hessian_loss_on or self.divergence_loss_on

    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # common
        self.name = config_args["setting"]["name"] 
        
        self.output_root = config_args["setting"]["output_root"]  
        self.pc_path = config_args["setting"]["pc_path"] 
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]

        # optional, when semantic shine mapping is on [semantic]
        if self.semantic_on:
            self.label_path =  config_args["setting"]["label_path"] 

        self.load_model = config_args["setting"]["load_model"]
        self.model_path = config_args["setting"]["model_path"]
        
        self.first_frame_ref = config_args["setting"]["first_frame_ref"]
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.every_frame = config_args["setting"]["every_frame"]
        self.pc_count_gpu_limit = self._opt(config_args["setting"], "pc_count_gpu_limit",
                                            self.pc_count_gpu_limit, int)
        # Shifts the reference frame along z so a scene
        # sitting far off the world origin can still fit a tight octree world --
        # lidar_dataset.py applies it to begin_pose_inv and the mesher undoes it
        # via global_transform, so saved meshes stay in the original frame.
        self.global_shift_default = self._opt(config_args["setting"], "global_shift_default",
                                              self.global_shift_default, float)

        self.device = config_args["setting"]["device"]
        self.gpu_id = config_args["setting"]["gpu_id"]
        # Bind gpu_id to the device. An externally set CUDA_VISIBLE_DEVICES
        # still wins, since in that case
        # the visible device is already renumbered to 0.
        if self.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            self.device = f"cuda:{self.gpu_id}"

        # process
        self.min_range = config_args["process"]["min_range_m"]
        self.pc_radius = config_args["process"]["pc_radius_m"]
        self.rand_downsample = config_args["process"]["rand_downsample"]
        self.vox_down_m = config_args["process"]["vox_down_m"]
        self.rand_down_r = config_args["process"]["rand_down_r"]
        self.min_z = config_args["process"]["min_z_m"]
        # re-enabled for the baselines that consume oriented normals ([1] RegSDF, [2] NeuralSSD)
        process_args = config_args["process"]
        self.estimate_normal = self._opt(process_args, "estimate_normal", self.estimate_normal)
        self.normal_radius_m = self._opt(process_args, "normal_radius_m", self.normal_radius_m, float)
        self.normal_max_nn = self._opt(process_args, "normal_max_nn", self.normal_max_nn, int)
        # self.filter_noise = config_args["process"]["filter_noise"]
        # self.semantic_on = config_args["process"]["semantic_on"]

        # sampler
        self.surface_sample_range_m = config_args["sampler"]["surface_sample_range_m"]
        self.surface_sample_n = config_args["sampler"]["surface_sample_n"]
        self.free_sample_begin_ratio = config_args["sampler"]["free_sample_begin_ratio"]
        self.free_sample_end_dist_m = config_args["sampler"]["free_sample_end_dist_m"]
        self.free_sample_n = config_args["sampler"]["free_sample_n"]
        # optional split
        # self.clearance_dist_m = config_args["sampler"]["clearance_dist_m"]
        # self.clearance_sample_n = config_args["sampler"]["clearance_sample_n"]

        # label
        # self.occu_update_on = config_args["label"]["occu_update_on"]
        # use bayersian update of the occupancy prob. as the new label

        # octree
        # the number of the total octree level (defining the world scale)
        self.tree_level_world = config_args["octree"]["tree_level_world"]
        # the number of the octree level used for storing feature grid
        self.tree_level_feat = config_args["octree"]["tree_level_feat"]
        # the size of the grid on octree's leaf level (unit: m)
        self.leaf_vox_size = config_args["octree"]["leaf_vox_size"]
        # feature vector's dimension
        self.feature_dim = config_args["octree"]["feature_dim"]
            
        # self.feature_std = config_args["octree"][
        #     "feature_std"
        # ]  # feature vector's initialization sigma (a zero mean, sigma standard deviation gaussian distribution)

        # use polynomial or linear interpolation of feature grids
        self.poly_int_on = config_args["octree"]["poly_int_on"]
        # build the octree from the surface samples or only the measurement points
        self.octree_from_surface_samples = config_args["octree"]["octree_from_surface_samples"]
            
        # decoder
        self.geo_mlp_level = config_args["decoder"]["mlp_level"] # number of the level of the mlp decoder
        self.geo_mlp_hidden_dim = config_args["decoder"]["mlp_hidden_dim"] # dimension of the mlp's hidden layer
        decoder_args = config_args["decoder"]
        self.mlp_activation = self._opt(decoder_args, "mlp_activation", self.mlp_activation, str)
        if self.mlp_activation not in ("relu", "sine", "softplus"):
            raise ValueError(f"decoder.mlp_activation must be relu/sine/softplus, got {self.mlp_activation}")
        self.siren_omega0 = self._opt(decoder_args, "siren_omega0", self.siren_omega0, float)
        # freeze the decoder after runing for x frames (used for incremental mapping to avoid forgeting)
        self.freeze_after_frame = config_args["decoder"]["freeze_after_frame"]

        # do the prediction conditioned on time (frame ID)
        # self.time_conditioned = config_args["decoder"][ 
        #     "time_conditioned"
        # ]

        # loss
        self.ray_loss = config_args["loss"]["ray_loss"]
        self.main_loss_type = config_args["loss"]["main_loss_type"]
        self.sigma_sigmoid_m = config_args["loss"]["sigma_sigmoid_m"]

        self.loss_weight_on = config_args["loss"]["loss_weight_on"]
        
        self.behind_dropoff_on = config_args["loss"][
            "behind_dropoff_on"
        ]  # apply "behind the surface" loss weight drop-off or not

        # self.normal_loss_on = config_args["loss"][
        #     "normal_loss_on"
        # ]  # use normal consistency loss [deprecated]
        # self.weight_n = float(config_args["loss"]["weight_n"])
        
        self.ekional_loss_on = config_args["loss"]["ekional_loss_on"] # use ekional loss (gradient norm = 1) or not
        self.weight_e = float(config_args["loss"]["weight_e"]) # weight for the ekional loss

        # optional keys, read defensively so that the configs written before
        # these existed still load
        loss_args = config_args["loss"]

        self.normal_loss_on = self._opt(loss_args, "normal_loss_on", self.normal_loss_on)
        self.weight_n = self._opt(loss_args, "weight_n", self.weight_n, float)

        # hessian energy (ours)
        self.hessian_loss_on = self._opt(loss_args, "hessian_loss_on", self.hessian_loss_on)
        self.weight_h = self._opt(loss_args, "weight_h", self.weight_h, float)
        self.hessian_intrinsic_loss_on = self._opt(loss_args, "hessian_intrinsic_loss_on",
                                                   self.hessian_intrinsic_loss_on)
        self.weight_hi = self._opt(loss_args, "weight_hi", self.weight_hi, float)
        self.intrinsic_band_h_m = self._opt(loss_args, "intrinsic_band_h_m",
                                            self.intrinsic_band_h_m, float)
        self.intrinsic_grad_floor = self._opt(loss_args, "intrinsic_grad_floor",
                                              self.intrinsic_grad_floor, float)
        self.intrinsic_area_weight_on = self._opt(loss_args, "intrinsic_area_weight_on",
                                                  self.intrinsic_area_weight_on)
        self.intrinsic_area_voxel_m = self._opt(loss_args, "intrinsic_area_voxel_m",
                                                self.intrinsic_area_voxel_m, float)
        self.intrinsic_area_count_floor = self._opt(loss_args, "intrinsic_area_count_floor",
                                                    self.intrinsic_area_count_floor, int)
        self.intrinsic_squared = self._opt(loss_args, "intrinsic_squared",
                                           self.intrinsic_squared)
        self.hessian_mask_path = self._opt(loss_args, "hessian_mask_path", self.hessian_mask_path, str)
        self.weight_h_bump = self._opt(loss_args, "weight_h_bump", self.weight_h_bump, float)
        self.weight_hi_bump = self._opt(loss_args, "weight_hi_bump", self.weight_hi_bump, float)
        self.hessian_mask_online = self._opt(loss_args, "hessian_mask_online", self.hessian_mask_online)
        self.hessian_mask_score = self._opt(loss_args, "hessian_mask_score", self.hessian_mask_score, str)
        self.hessian_mask_probe = self._opt(loss_args, "hessian_mask_probe", self.hessian_mask_probe, int)
        self.hessian_mask_every = self._opt(loss_args, "hessian_mask_every", self.hessian_mask_every, int)
        self.hessian_mask_sigmas = self._opt(loss_args, "hessian_mask_sigmas", self.hessian_mask_sigmas, list)
        self.hessian_mask_voxel = self._opt(loss_args, "hessian_mask_voxel", self.hessian_mask_voxel, float)
        self.hessian_mask_momentum = self._opt(loss_args, "hessian_mask_momentum", self.hessian_mask_momentum, float)
        self.hessian_mask_warmup = self._opt(loss_args, "hessian_mask_warmup", self.hessian_mask_warmup, int)
        self.hessian_mask_quantile = self._opt(loss_args, "hessian_mask_quantile", self.hessian_mask_quantile, float)
        self.hessian_mask_soft = self._opt(loss_args, "hessian_mask_soft", self.hessian_mask_soft, float)
        self.hessian_mask_adaptive = self._opt(loss_args, "hessian_mask_adaptive", self.hessian_mask_adaptive)
        self.hessian_norm_scale = self._opt(loss_args, "hessian_norm_scale", self.hessian_norm_scale, float)
        self.grad_gate_tau = self._opt(loss_args, "grad_gate_tau", self.grad_gate_tau, float)

        # [1] RegSDF
        self.hessian_l1_loss_on = self._opt(loss_args, "hessian_l1_loss_on", self.hessian_l1_loss_on)
        self.weight_hl1 = self._opt(loss_args, "weight_hl1", self.weight_hl1, float)
        self.minimal_surface_loss_on = self._opt(loss_args, "minimal_surface_loss_on", self.minimal_surface_loss_on)
        self.weight_ms = self._opt(loss_args, "weight_ms", self.weight_ms, float)
        self.minimal_surface_eps_m = self._opt(loss_args, "minimal_surface_eps_m", self.minimal_surface_eps_m, float)
        self.regsdf_surface_mask_on = self._opt(loss_args, "regsdf_surface_mask_on", self.regsdf_surface_mask_on)
        self.gaussian_curvature_loss_on = self._opt(loss_args, "gaussian_curvature_loss_on", self.gaussian_curvature_loss_on)
        self.weight_gk = self._opt(loss_args, "weight_gk", self.weight_gk, float)
        self.ours_volumetric_on = self._opt(loss_args, "ours_volumetric_on", self.ours_volumetric_on)

        # [2] NeuralSSD
        self.hessian_frob_loss_on = self._opt(loss_args, "hessian_frob_loss_on", self.hessian_frob_loss_on)
        self.weight_hf = self._opt(loss_args, "weight_hf", self.weight_hf, float)
        self.normal_align_loss_on = self._opt(loss_args, "normal_align_loss_on", self.normal_align_loss_on)
        self.weight_na = self._opt(loss_args, "weight_na", self.weight_na, float)

        # [3] GradHess
        self.grad_hessian_align_loss_on = self._opt(loss_args, "grad_hessian_align_loss_on", self.grad_hessian_align_loss_on)
        self.weight_gh = self._opt(loss_args, "weight_gh", self.weight_gh, float)
        self.grad_hessian_delta_per_m = self._opt(loss_args, "grad_hessian_delta_per_m",
                                                  self.grad_hessian_delta_per_m, float)

        # [4] Neural-Singular-Hessian
        self.singular_hessian_loss_on = self._opt(loss_args, "singular_hessian_loss_on", self.singular_hessian_loss_on)
        self.weight_sh = self._opt(loss_args, "weight_sh", self.weight_sh, float)
        self.relaxed_eikonal_loss_on = self._opt(loss_args, "relaxed_eikonal_loss_on", self.relaxed_eikonal_loss_on)
        self.weight_re = self._opt(loss_args, "weight_re", self.weight_re, float)
        self.relaxed_eikonal_sigma_min = self._opt(loss_args, "relaxed_eikonal_sigma_min", self.relaxed_eikonal_sigma_min, float)

        # [5] DiGS
        self.divergence_loss_on = self._opt(loss_args, "divergence_loss_on", self.divergence_loss_on)
        self.weight_div = self._opt(loss_args, "weight_div", self.weight_div, float)

        # [6] SIREN
        self.siren_offsurface_loss_on = self._opt(loss_args, "siren_offsurface_loss_on", self.siren_offsurface_loss_on)
        self.weight_so = self._opt(loss_args, "weight_so", self.weight_so, float)
        self.siren_alpha = self._opt(loss_args, "siren_alpha", self.siren_alpha, float)

        # [7] DiffCD
        self.diffcd_loss_on = self._opt(loss_args, "diffcd_loss_on", self.diffcd_loss_on)
        self.weight_diffcd = self._opt(loss_args, "weight_diffcd", self.weight_diffcd, float)
        self.diffcd_newton_steps = self._opt(loss_args, "diffcd_newton_steps", self.diffcd_newton_steps, int)
        self.diffcd_ref_count = self._opt(loss_args, "diffcd_ref_count", self.diffcd_ref_count, int)

        # [8] stochastic preconditioning
        self.stoch_precond_on = self._opt(loss_args, "stoch_precond_on", self.stoch_precond_on)
        self.stoch_precond_sigma_m = self._opt(loss_args, "stoch_precond_sigma_m", self.stoch_precond_sigma_m, float)

        self.anneal_schedule = self._opt(loss_args, "anneal_schedule", self.anneal_schedule)
        self.anneal_final_ratio = self._opt(loss_args, "anneal_final_ratio", self.anneal_final_ratio, float)

        # continual learning (incremental)
        # using the regularization based continuous learning or the rehersal based continuous learning
        self.continual_learning_reg = config_args["continual"]["continual_learning_reg"]
        # the forgeting lambda for regularization based continual learning
        self.lambda_forget = float(config_args["continual"]["lambda_forget"])

        # rehersal (replay) based method
        self.window_replay_on = config_args["continual"]["window_replay_on"]
        self.window_radius = config_args["continual"]["window_radius_m"]

        # self.history_sample_ratio = float(
        #     config_args["continuous"]["history_sample_ratio"]
        # )  # sample the history samples by a scale of the number of current samples
        # self.history_sample_res = config_args["continuous"][
        #     "history_sample_res"
        # ]  # the resolution of the kept history samples (unit: m)
        

        # optimizer
        self.iters = config_args["optimizer"]["iters"] # maximum iters (in our implementation, iters means iteration actually)
        self.bs = config_args["optimizer"]["batch_size"]
        # self.adam_eps = float(config_args["optimizer"]["adam_eps"])
        self.lr = float(config_args["optimizer"]["learning_rate"])
        # self.lr_level_reduce_ratio = config_args["optimizer"][
        #     "lr_level_reduce_ratio"
        # ]  # decay the learning rate for higher level of feature grids by such ratio
        # self.lr_iters_reduce_ratio = config_args["optimizer"][
        #     "lr_iters_reduce_ratio"
        # ]  # decay the learning rate after certain iterss by such ratio
        # Matters for resumed runs: the iteration
        # counter restarts at 0 on load_model, so the default [10000, ...]
        # re-fires a 10x decay partway through a continuation that a
        # from-scratch run would have run at constant lr. Pass an empty list to
        # disable decay entirely.
        self.lr_decay_step = self._opt(config_args["optimizer"], "lr_decay_step",
                                       self.lr_decay_step)
        self.lr_iters_reduce_ratio = self._opt(config_args["optimizer"],
                                               "lr_iters_reduce_ratio",
                                               self.lr_iters_reduce_ratio, float)
        self.weight_decay = float(config_args["optimizer"]["weight_decay"])  # coefficient for L2 regularization
        

        # vis and eval
        self.wandb_vis_on = config_args["eval"]["wandb_vis_on"] # use weight and bias to monitor the experiment or not
        self.o3d_vis_on = config_args["eval"]["o3d_vis_on"] # turn on the open3d visualizer to visualize the mapping progress or not
        self.vis_freq_iters = config_args["eval"]["vis_freq_iters"] # frequency for mesh reconstruction for batch mode (per x iters)
        self.save_freq_iters = config_args["eval"]["save_freq_iters"] # frequency for model saving for batch mode (per x iters)
        self.mesh_freq_frame = config_args["eval"]["mesh_freq_frame"] # frequency for mesh reconstruction for incremental mode (per x frame)
        self.mc_with_octree = config_args["eval"]["mc_with_octree"] # using octree to narrow down the region that needs the sdf query so as to boost the efficieny
        # if false, we query all the positions within the map bounding box
        self.mc_res_m = config_args["eval"]["mc_res_m"] # marching cubes grid sampling interval (unit: m)
        self.mc_vis_level = config_args["eval"]["mc_vis_level"]
        self.mc_local = config_args["eval"]["mc_local"]
        self.save_map = config_args["eval"]["save_map"]
        # self.mc_mask_on = config_args["eval"]["mc_mask_on"] # using masked marching cubes according to the octree or not, default true
        
        # tree level starting for reconstruction and visualization, the larger of this value, 
        # the larger holes would be filled (better completion), but at the same time more artifacts 
        # would appear at the boundary of the map
        # it's a trading-off of the compeltion and the artifacts

        # self.grid_loss_vis_on = config_args["eval"][
        #     "grid_loss_vis_on"
        # ]  # visualize the loss at each grid position or not [deprecated]
        # self.mesh_vis_on = config_args["eval"][
        #     "mesh_vis_on"
        # ]  # visualize the reconstructed mesh or not, if not, the mesh will still be exported and you can check it offline

        self.calculate_world_scale()
        self.infer_bs = self.bs * 16
        self.mc_query_level = self.tree_level_world - self.tree_level_feat + 1

        if self.window_radius <= 0:
            self.window_radius = self.pc_radius * 2.0
    
    # calculate the scale for compressing the world into a [-1,1] kaolin cube
    def calculate_world_scale(self):
        self.world_size = self.leaf_vox_size*(2**(self.tree_level_world-1)) 
        self.scale = 1.0 / self.world_size
