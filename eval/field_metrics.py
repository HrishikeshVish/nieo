"""Field-smoothness metrics that need the trained decoder + octree, not just a
saved mesh -- Dirichlet and Hessian energy, as used by [4] Neural-Singular-
Hessian to compare smoothness-energy regularizers. Surface-distance metrics
(Chamfer/F-score) are largely insensitive to local smoothness; a metric
evaluated directly on the SDF is what a smoothness term should move first.

[4]'s formulas (Sec 5.5.2, both Monte-Carlo estimated here as a mean over
sampled query points rather than the literal volume integral -- consistent
with how utils/loss.py's hessian_frobenius_loss/hessian_l1_loss already
estimate the same quantities as training losses):
    E_D  = 1/2 integral_Omega ||grad f(x)||^2 dx      (Dirichlet energy)
    E_H2 = integral_Omega ||H f(x)||_F^2 dx            (Hessian energy, L2)
    E_H1 = integral_Omega ||H f(x)||_1 dx              (Hessian energy, L1)

Dirichlet energy is deliberately NOT one of the eight selectable regularizer
loss flags in utils/loss.py: pinning ||grad f|| -> 1 (the hard eikonal term
every recipe keeps on) and penalizing ||grad f||^2 -> 0 (what E_D as a loss
would do) directly fight each other -- [4] itself shows E_D produces "ghost
geometry" for exactly this reason. Here it is used purely as a post-hoc
smoothness metric on a trained field, never as something added to the loss.

Query points are sampled uniformly inside the trained octree's occupied
bounding box (from FeatureOctree.get_octree_nodes at config.mc_query_level --
the same level utils/mesher.py's recon_octree_mesh uses to bound marching
cubes), not the whole scene bbox, so the estimate isn't diluted by empty space
the field was never supervised in.

Usage:
    python eval/field_metrics.py \
        --checkpoint experiments/<run_dir>/model/model_iter_10000.pth \
        --method_name ours --dataset redwood
    (--config is auto-detected from the run's own copied *.yaml sibling next
    to model/ -- train.py copies it there at startup -- unless passed
    explicitly.)
"""
import argparse
import csv
import glob
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from utils.config import SHINEConfig
from utils.tools import get_derivatives
from model.feature_octree import FeatureOctree
from model.decoder import Decoder

CSV_COLUMNS = ['Dirichlet_energy (E_D)', 'Hessian_energy_L2 (E_H2)', 'Hessian_energy_L1 (E_H1)',
               'Eikonal_violation_MSE', 'Grad_norm_median',
               'Intrinsic_energy (||PHP||_F^2)', 'Mixed_energy (2||PHn||^2)',
               'Normal_energy ((n^T H n)^2)', 'Decomposition_residual',
               'Shape_op_energy (mean ||S||_F^2)', 'Shape_op_norm_median (1/m)',
               'Intrinsic_fraction', 'Valid_fraction',
               'Num_query_points', 'Query_level']


def _find_sibling_config(checkpoint_path):
    """checkpoint sits at <run_path>/model/xxx.pth; train.py
    copies the exact config used for that run to <run_path>/<original_name>.yaml
    at startup (shutil.copy2), so recover it from there rather than asking the
    caller to keep a config path in sync with a checkpoint by hand."""
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

    # weights_only=False: the checkpoint pickles a whole FeatureOctree object
    # (save_checkpoint stores it under "feature_octree"), which torch>=2.6's
    # default weights_only=True refuses to unpickle. These are our own runs'
    # checkpoints, so the usual untrusted-pickle concern doesn't apply.
    checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)

    octree = checkpoint["feature_octree"]
    phys_eng = Decoder(config, is_geo_encoder=True, is_time_conditioned=config.time_conditioned)
    phys_eng.load_state_dict(checkpoint["geo_decoder"])
    phys_eng.to(config.device)

    return config, octree, phys_eng


def sample_query_points(octree, config, num_points):
    nodes_coord_scaled = octree.get_octree_nodes(config.mc_query_level)
    min_bound = nodes_coord_scaled.min(axis=0)
    max_bound = nodes_coord_scaled.max(axis=0)
    span = torch.tensor(max_bound - min_bound, dtype=config.dtype, device=config.device)
    origin = torch.tensor(min_bound, dtype=config.dtype, device=config.device)
    return origin + torch.rand(num_points, 3, dtype=config.dtype, device=config.device) * span


def field_smoothness_metrics(config, octree, phys_eng, num_points=200000, batch_size=8192):
    sigma_sigmoid = config.logistic_gaussian_ratio * config.sigma_sigmoid_m * config.scale
    coord_all = sample_query_points(octree, config, num_points)

    dirichlet_terms, hessian_frob_terms, hessian_l1_terms = [], [], []
    eik_terms, intr_terms, mixed_terms, nn_terms, gnorm_terms = [], [], [], [], []
    shape_terms = []
    # Only points whose 8 interpolation corners are all real octree leaves carry
    # a meaningful field value. Sampling uniformly in the occupied BOUNDING BOX
    # puts most points in empty space, where query_feature returns the all-zero
    # trashbin row and the decoder emits a CONSTANT -- so |grad f| = 0 and H = 0
    # there by construction. Averaging those in does not merely add noise, it
    # drags every energy toward zero by whatever fraction of the box is empty,
    # which differs per scene. Same mask as utils/mesher.py uses.
    check_level = min(octree.featured_level_num, config.mc_vis_level) - 1
    n_valid_total, n_seen = 0, 0

    for head in range(0, num_points, batch_size):
        coord = coord_all[head:head + batch_size].clone().requires_grad_(True)
        feature = octree.query_feature(coord)
        valid = torch.all(octree.hierarchical_indices[check_level] >= 0, dim=1)
        pred = phys_eng(feature)
        _, g, hessian_m = get_derivatives(coord, pred, sigma_sigmoid, config.scale, need_hessian=True)

        n_seen += coord.shape[0]
        n_valid_total += int(valid.sum())
        if int(valid.sum()) == 0:
            continue
        g = g[valid]
        hessian_m = hessian_m[valid]

        dirichlet_terms.append((0.5 * g.pow(2).sum(dim=-1)).detach())
        hessian_frob_terms.append(hessian_m.pow(2).sum(dim=(-2, -1)).detach())
        hessian_l1_terms.append(hessian_m.abs().sum(dim=(-2, -1)).detach())

        # --- eikonal violation ------------------------------------------
        gn = g.norm(dim=-1)
        gnorm_terms.append(gn.detach())
        eik_terms.append((gn - 1.0).pow(2).detach())

        # --- orthogonal decomposition of the Hessian --------------------
        # ||H||_F^2 = ||PHP||_F^2 + 2||PHn||^2 + (n'Hn)^2, verified exactly
        # (max err 4e-14) on analytic SDFs. The tangential block is genuine
        # surface bending -- what our intrinsic term penalises -- while the
        # other two are normal-direction curvature, i.e. eikonal error, which
        # the GradHess alignment term handles. Reporting the split shows WHICH
        # part of the ambient Hessian energy a method actually reduced;
        # E_H2 alone cannot distinguish smoothing the surface from fixing the
        # metric.
        n = g / gn.clamp_min(1e-8).unsqueeze(-1)
        eye = torch.eye(3, device=g.device, dtype=g.dtype).expand(g.shape[0], 3, 3)
        P = eye - n.unsqueeze(-1) * n.unsqueeze(-2)
        PHP = torch.bmm(torch.bmm(P, hessian_m), P)
        Hn = torch.bmm(hessian_m, n.unsqueeze(-1)).squeeze(-1)
        PHn = torch.bmm(P, Hn.unsqueeze(-1)).squeeze(-1)
        nHn = (n * Hn).sum(dim=-1)
        intr_terms.append(PHP.pow(2).sum(dim=(-2, -1)).detach())
        mixed_terms.append((2.0 * PHn.pow(2).sum(dim=-1)).detach())
        nn_terms.append(nHn.pow(2).detach())

        # --- SCALE-INVARIANT curvature: the shape operator S = PHP/|grad f| ---
        # Every ambient energy above is scale-DEPENDENT: under f -> alpha*f,
        # E_H2 and E_D scale as alpha^2 and E_H1 as alpha. So a method can post
        # a low Hessian energy simply by letting |grad f| collapse, without its
        # level set being any smoother -- which is not a smoothing result, it is
        # an eikonal failure. S has units 1/length and is invariant to f ->
        # alpha*f (and to any radial reparameterisation), so it measures the
        # geometry of the level set itself. This is the only fair way to compare
        # the "smoothness" of two fields whose |grad f| differ.
        S = PHP / gn.clamp_min(1e-8).view(-1, 1, 1)
        s_fro = S.pow(2).sum(dim=(-2, -1)).sqrt()
        shape_terms.append(s_fro.detach())

    dirichlet_energy = torch.cat(dirichlet_terms).mean().item()
    hessian_energy_l2 = torch.cat(hessian_frob_terms).mean().item()
    hessian_energy_l1 = torch.cat(hessian_l1_terms).mean().item()
    intr = torch.cat(intr_terms).mean().item()
    mixed = torch.cat(mixed_terms).mean().item()
    nn = torch.cat(nn_terms).mean().item()
    gn_all = torch.cat(gnorm_terms)
    s_all = torch.cat(shape_terms)

    return {
        'Dirichlet_energy (E_D)': dirichlet_energy,
        'Hessian_energy_L2 (E_H2)': hessian_energy_l2,
        'Hessian_energy_L1 (E_H1)': hessian_energy_l1,
        'Eikonal_violation_MSE': torch.cat(eik_terms).mean().item(),
        'Grad_norm_median': gn_all.median().item(),
        'Intrinsic_energy (||PHP||_F^2)': intr,
        'Mixed_energy (2||PHn||^2)': mixed,
        'Normal_energy ((n^T H n)^2)': nn,
        # must equal E_H2 up to float error; a mismatch means the projector
        # P = I - nn^T is not idempotent (e.g. n is not unit length)
        'Decomposition_residual': abs(hessian_energy_l2 - (intr + mixed + nn)),
        'Shape_op_energy (mean ||S||_F^2)': s_all.pow(2).mean().item(),
        'Shape_op_norm_median (1/m)': s_all.median().item(),
        'Intrinsic_fraction': intr / max(hessian_energy_l2, 1e-30),
        'Valid_fraction': n_valid_total / max(n_seen, 1),
        'Num_query_points': num_points,
        'Query_level': config.mc_query_level,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a model/model_iter_*.pth from save_checkpoint")
    ap.add_argument("--config", default=None, help="defaults to the run's own copied *.yaml sibling")
    ap.add_argument("--dataset", required=True, help="label only, e.g. redwood")
    ap.add_argument("--method_name", required=True, help="e.g. ours, regsdf, hessian_w1p0")
    ap.add_argument("--num_points", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--output_dir", default=os.path.join(ROOT, "experiments/evaluation"))
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    config_path = args.config or _find_sibling_config(args.checkpoint)

    config, octree, phys_eng = load_model(config_path, args.checkpoint)
    metrics = field_smoothness_metrics(config, octree, phys_eng,
                                        num_points=args.num_points, batch_size=args.batch_size)
    print(f"{args.dataset} / {args.method_name}: {metrics}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.method_name}_field_eval.csv")
    with open(out_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(metrics)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
