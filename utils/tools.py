from typing import List
import sys
import os
import multiprocessing
import getpass
import time
from pathlib import Path
from datetime import datetime
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.autograd import grad
import torch
import torch.nn as nn
import numpy as np
import wandb
import json
import open3d as o3d

from utils.config import SHINEConfig


# setup this run
def setup_experiment(config: SHINEConfig): 

    os.environ["NUMEXPR_MAX_THREADS"] = str(multiprocessing.cpu_count())
    # an exported CUDA_VISIBLE_DEVICES wins over the config's gpu_id
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.gpu_id))
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # begining timestamp
    run_name = config.name + "_" + ts  # modified to a name that is easier to index
        
    run_path = os.path.join(config.output_root, run_name)
    access = 0o755
    os.makedirs(run_path, access, exist_ok=True)
    assert os.access(run_path, os.W_OK)
    print(f"Start {run_path}")

    mesh_path = os.path.join(run_path, "mesh")
    map_path = os.path.join(run_path, "map")
    model_path = os.path.join(run_path, "model")
    os.makedirs(mesh_path, access, exist_ok=True)
    os.makedirs(map_path, access, exist_ok=True)
    os.makedirs(model_path, access, exist_ok=True)
    
    if config.wandb_vis_on:
        # set up wandb
        setup_wandb()
        wandb.init(project="SHINEMapping", config=vars(config), dir=run_path) # your own worksapce
        wandb.run.name = run_name         
    
    # set the random seed (for deterministic experiment results)
    o3d.utility.random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed) 

    return run_path


def setup_optimizer(config: SHINEConfig, octree_feat, mlp_geo_param, mlp_sem_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt


def setup_optimizer_traj(config: SHINEConfig, octree_feat, mlp_geo_param, mlp_sem_param, mlp_traj_param, sigma_size) -> Optimizer:
    lr_cur = config.lr
    opt_setting = []
    # weight_decay is for L2 regularization, only applied to MLP
    if mlp_geo_param is not None:
        mlp_geo_param_opt_dict = {'params': mlp_geo_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_geo_param_opt_dict)
    if config.semantic_on and mlp_sem_param is not None:
        mlp_sem_param_opt_dict = {'params': mlp_sem_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_sem_param_opt_dict)
    if mlp_traj_param is not None:
        mlp_traj_param_opt_dict = {'params': mlp_traj_param, 'lr': lr_cur, 'weight_decay': config.weight_decay} 
        opt_setting.append(mlp_traj_param_opt_dict)

    # feature octree
    for i in range(config.tree_level_feat):
        # try to also add L2 regularization on the feature octree (results not quite good)
        feat_opt_dict = {'params': octree_feat[config.tree_level_feat-i-1], 'lr': lr_cur} 
        lr_cur *= config.lr_level_reduce_ratio
        opt_setting.append(feat_opt_dict)
    # make sigma also learnable for differentiable rendering (but not for our method)
    if config.ray_loss:
        sigma_opt_dict = {'params': sigma_size, 'lr': config.lr}
        opt_setting.append(sigma_opt_dict)
    
    if config.opt_adam:
        opt = optim.Adam(opt_setting, betas=(0.9,0.99), eps = config.adam_eps) 
    else:
        opt = optim.SGD(opt_setting, momentum=0.9)
    
    return opt
    

# set up weight and bias
def setup_wandb():
    print("Weight & Bias logging option is on. Disable it by setting  wandb_vis_on: False  in the config file.")
    username = getpass.getuser()
    # print(username)
    wandb_key_path = username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):"
        )
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system('export WANDB_API_KEY=$(cat "' + wandb_key_path + '")')

def step_lr_decay(
    optimizer: Optimizer,
    learning_rate: float,
    iteration_number: int,
    steps: List,
    reduce: float = 1.0):

    if reduce > 1.0 or reduce <= 0.0:
        sys.exit(
            "The decay reta should be between 0 and 1."
        )

    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate


def num_model_weights(model: nn.Module) -> int:
    num_weights = int(
        sum(
            [
                np.prod(p.size())
                for p in filter(lambda p: p.requires_grad, model.parameters())
            ]
        )
    )
    return num_weights


def print_model_summary(model: nn.Module):
    for child in model.children():
        print(child)


def get_gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True,
        materialize_grads=True,
    )[0]
    return points_grad


def get_hessian(coord, gradient):
    """Hessian of the field whose gradient w.r.t. coord is `gradient`.

    Returns [N, 3, 3]. create_graph stays on so the hessian itself is
    differentiable, which every second-order regularizer needs.

    A locally affine field has a constant gradient that autograd detaches from
    the graph; the hessian is then exactly zero, so return it rather than
    letting the backward engine raise.
    """
    if not gradient.requires_grad:
        return torch.zeros(
            gradient.shape[0], 3, 3, device=gradient.device, dtype=gradient.dtype
        )
    rows = [get_gradient(coord, gradient[:, i]) for i in range(3)]
    return torch.stack(rows, dim=1)


def get_derivatives(coord, pred, sigma_sigmoid, scale, need_hessian=True):
    """Field value and derivatives in consistent physical units.

    `pred` is the decoder logit, not the sdf: sdf_bce_loss fits
    sigmoid(sdf_label / sigma_sigmoid), so sdf_scaled = pred * sigma_sigmoid.
    `coord` is in the scaled [-1, 1] cube, i.e. coord = xyz_metres * scale.
    Therefore

        sdf_m = pred * sigma_sigmoid / scale                 [m]
        g     = d(pred)/d(coord) * sigma_sigmoid             [-]   (norm -> 1)
        H_m   = d2(pred)/d(coord)2 * sigma_sigmoid * scale   [1/m]

    Returning metric quantities means a single loss weight transfers across
    datasets: sigma_sigmoid*scale differs by ~45x between the KITTI and the
    indoor RGB-D configs, which is what the raw logit derivatives would inherit.

    The hessian is scaled here rather than inside each loss so that
    determinants are taken on well-conditioned entries.
    """
    grad_pred = get_gradient(coord, pred)

    sdf_m = pred * sigma_sigmoid / scale
    g = grad_pred * sigma_sigmoid

    hessian_m = None
    if need_hessian:
        hessian_m = get_hessian(coord, grad_pred) * (sigma_sigmoid * scale)

    return sdf_m, g, hessian_m


def anneal_factor(iter, total_iters, schedule='none', final_ratio=0.0):
    """Multiplier applied to a regularizer weight over the course of training.

    'none' keeps the weight flat. 'digs' reproduces the DiGS 50/25/25 split
    (full weight for the first half, linear decay to `final_ratio` over the
    next quarter, then held there). 'exp' decays geometrically from 1 to
    `final_ratio` across the whole run, which is what Neural-Singular-Hessian's
    tau and stochastic preconditioning's sigma both want.
    """
    if schedule == 'none' or total_iters <= 0:
        return 1.0

    t = min(max(iter / total_iters, 0.0), 1.0)

    if schedule == 'digs':
        if t < 0.5:
            return 1.0
        if t < 0.75:
            return 1.0 + (final_ratio - 1.0) * (t - 0.5) / 0.25
        return final_ratio

    if schedule == 'exp':
        # geometric interpolation, floored so that final_ratio=0 still decays smoothly
        lo = max(final_ratio, 1e-6)
        return float(lo ** t)

    sys.exit("Unknown anneal_schedule '%s', use 'none', 'digs' or 'exp'" % schedule)


def perturb_coord(coord, sigma_scaled):
    """Stochastic preconditioning (Ling et al. 2025): blur the field by jittering
    query locations, reflecting rather than clamping at the domain boundary so the
    sample distribution stays uniform instead of piling up on the faces.
    """
    if sigma_scaled <= 0.0:
        return coord
    perturbed = coord + torch.randn_like(coord) * sigma_scaled
    # reflect back into [-1, 1]
    perturbed = torch.where(perturbed > 1.0, 2.0 - perturbed, perturbed)
    perturbed = torch.where(perturbed < -1.0, -2.0 - perturbed, perturbed)
    return perturbed.clamp(-1.0, 1.0)


def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True


def save_checkpoint(
    feature_octree, geo_decoder, sem_decoder, optimizer, run_path, checkpoint_name, iters
):
    torch.save(
        {
            "iters": iters,
            "feature_octree": feature_octree, # save the whole NN module (the hierachical features and the indexing structure)
            "geo_decoder": geo_decoder.state_dict(),
            "sem_decoder": sem_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        os.path.join(run_path, f"{checkpoint_name}.pth"),
    )
    print(f"save the model to {run_path}/{checkpoint_name}.pth")


def save_decoder(geo_decoder, sem_decoder, run_path, checkpoint_name):
    torch.save({"geo_decoder": geo_decoder.state_dict(), 
                "sem_decoder": sem_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_decoders.pth"),
    )

def save_geo_decoder(geo_decoder, run_path, checkpoint_name):
    torch.save({"geo_decoder": geo_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_geo_decoder.pth"),
    )

def save_sem_decoder(sem_decoder, run_path, checkpoint_name):
    torch.save({"sem_decoder": sem_decoder.state_dict()},
        os.path.join(run_path, f"{checkpoint_name}_sem_decoder.pth"),
    )

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UTF-8") as file:
        json.dump(content, file)
