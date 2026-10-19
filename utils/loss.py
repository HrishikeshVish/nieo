import math

import torch
import torch.nn.functional as F
import torch.nn as nn

# ---------------------------------------------------------------------------
# Our regularizer plus baseline regularizers from the compared prior work,
# refs [1]-[8] (see each function's docstring).
#
# Every function below takes quantities already converted to physical units by
# utils.tools.get_derivatives: sdf_m in metres, g dimensionless with unit norm
# on a true sdf, and hessian_m in 1/m. None of them re-derive anything, so the
# autograd graph is built once per iteration and shared.
#
# `mask` selects which samples the term applies to. surface_mask (weight > 0)
# is the near-surface band; ~surface_mask is the free-space set standing in for
# the off-surface domain the papers integrate over.
# ---------------------------------------------------------------------------

_EPS = 1e-8


def shape_operator(g, hessian_m, grad_floor=0.2):
    """S = P H P / |grad f| with P = I - n n^T: the level set's second
    fundamental form, expressed in ambient coordinates with a zero normal
    row/column. Eigenvalues are (kappa_1, kappa_2, 0), the principal curvatures.

    Subtlety: |grad f| is used TWICE and only one use may be clamped. `n` must stay a unit vector or P stops being a
    projector and leaks (1-c^2)^2 of n^T H n back into the tangential block,
    c = |grad f|/grad_floor -- exactly the normal-direction contamination the
    projection exists to remove, at exactly the under-constrained points where
    it does the most damage. Only the 1/|grad f| division is clamped.
    """
    raw_gn = g.norm(dim=-1, keepdim=True)
    n = g / raw_gn.clamp_min(_EPS)
    gn = raw_gn.clamp_min(grad_floor)
    eye = torch.eye(3, device=g.device, dtype=g.dtype).expand(g.shape[0], 3, 3)
    P = eye - n.unsqueeze(-1) * n.unsqueeze(-2)
    return torch.bmm(torch.bmm(P, hessian_m), P) / gn.unsqueeze(-1)


def principal_curvatures(S):
    """(kappa_1, kappa_2) of a shape operator, from its invariants.

    S is symmetric with eigenvalues (k1, k2, 0), so with T = tr(S) = k1+k2 and
    F2 = ||S||_F^2 = k1^2+k2^2:  k1*k2 = (T^2 - F2)/2  and
    (k1-k2)^2 = 2*F2 - T^2 >= 0, giving k = [T +- sqrt(2*F2 - T^2)]/2.
    The discriminant is clamped at 0 against round-off at umbilic points
    (k1 = k2), where it is analytically zero and the sqrt gradient is stiff.
    """
    T = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    F2 = S.pow(2).sum(dim=(-2, -1))
    disc = (2.0 * F2 - T.pow(2)).clamp_min(0.0)
    root = torch.sqrt(disc + _EPS)
    return 0.5 * (T + root), 0.5 * (T - root)


def gaussian_curvature_loss(g, hessian_m, mask, grad_floor=0.2):
    """OPTIONAL ADD-ON to [1] RegSDF: |K| = |kappa_1 * kappa_2|, the magnitude
    of Gaussian curvature of the level set -- a DEVELOPABILITY prior.

    RegSDF's own term is isotropic: ||H|| charges the same for every curvature
    direction, so a cylinder (kappa_2 = 0, exactly reconstructable, and most of
    what a man-made scene is made of -- wheels, panels, bed walls) costs it
    kappa_1^2, the same as noise of equal magnitude. This is why RegSDF trades
    recall for precision (e.g. on Tanks and Temples Truck): it erodes
    legitimately curved structure along with the noise.

    |K| is zero on exactly the developable surfaces (planes, cylinders, cones)
    and positive on spheres/saddles/noise, so it penalizes non-developability
    rather than curvature per se.

    Deliberately NOT the invariant shortcut on the ambient hessian. For a true
    sdf grad f is a zero-eigenvector of H, so (tr(H)^2 - ||H||_F^2)/2 would
    equal k1*k2 with no projector at all -- but only to the extent the eikonal
    constraint holds. With a third eigenvalue lambda_3 != 0 that expression is
    k1*k2 + lambda_3*(k1+k2), and the median |grad f| of trained fields is
    typically 0.6-0.75, nowhere near 1. The shortcut would report eikonal
    violation as curvature. Projecting first costs five lines and removes it.

    Signed K is deliberately avoided: by Gauss-Bonnet the integral of K over a
    closed surface is 2*pi*chi, a topological constant, so penalizing K rather
    than |K| would only push on topology.

    Distinct from [4] Neural-Singular-Hessian despite both involving a
    determinant: theirs is det of the AMBIENT 3x3, which is identically zero
    for any valid sdf (an eikonal-consistency prior). This is the determinant
    of the TANGENTIAL block, nonzero on spheres and saddles (a shape prior).
    """
    S = shape_operator(g[mask], hessian_m[mask], grad_floor)
    T = S.diagonal(dim1=-2, dim2=-1).sum(-1)
    F2 = S.pow(2).sum(dim=(-2, -1))
    K = 0.5 * (T.pow(2) - F2)
    return K.abs().mean()


def hessian_l1_loss(hessian_m, mask):
    """[1] RegSDF (Zhang et al., CVPR 2022): element-wise matrix 1-norm of the
    hessian. Note this is the entrywise sum of magnitudes, not the induced
    1-norm."""
    return hessian_m[mask].abs().sum(dim=(-2, -1)).mean()


def minimal_surface_loss(sdf_m, mask, eps_m=0.1):
    """[1] RegSDF minimal-surface term: a regularized Dirac on the sdf, which
    penalizes surface area and so compacts interpolated/extrapolated geometry.

    The paper uses delta_eps(z) = (eps/pi)/(eps^2 + z^2) with eps = 10 on a
    unit-normalized bounding box, i.e. a peak of half-width 1/eps = 0.1 scene
    extents. Taken literally on a 614 m world that is a 61 m band, so we
    parameterize by the half-width in metres instead.
    """
    eps = 1.0 / max(eps_m, _EPS)
    z = sdf_m[mask]
    return ((eps / math.pi) / (eps ** 2 + z ** 2)).mean()


def hessian_frobenius_loss(hessian_m, mask):
    """[2] NeuralSSD (Xi et al., 2025) E_H: squared Frobenius norm of the
    hessian. This is the closest prior L2 hessian energy to our own, reported
    as a baseline so the two can be compared under an identical backbone."""
    return hessian_m[mask].pow(2).sum(dim=(-2, -1)).mean()


def normal_alignment_loss(g, normal_label, mask):
    """[2] NeuralSSD E_N: squared residual between the field gradient and the
    (oriented) normal field. Requires config.estimate_normal."""
    if normal_label is None:
        raise ValueError(
            "normal_alignment_loss needs normals; set estimate_normal: True in "
            "the process section of the config"
        )
    return (g[mask] - normal_label[mask]).pow(2).sum(dim=-1).mean()


def grad_hessian_align_loss(g, hessian_m, mask, d_m=None, delta_per_m=0.0):
    """[3] GradHess (Wang et al., NeurIPS 2023): the gradient of a true sdf is
    an eigenvector of its hessian with eigenvalue zero, so ||H ghat||^2 -> 0.

    This is the gradient-weighted view of the Hessian, and it is a different
    operator from every other Hessian term in this file. Decompose H in the
    frame (n, t1, t2): the tangent-tangent block is the shape operator that
    `intrinsic_hessian_loss` penalizes -- genuine surface bending, so penalizing
    it trades detail for smoothness and contracts convex regions. The row and
    column this loss selects, H*ghat, is the *normal* direction, which for a
    true sdf is exactly zero (differentiate |grad f| = 1). Driving it down
    therefore removes eikonal inconsistency rather than curvature, which is why
    the paper reports feature preservation where smoothing energies blur, and
    why it should not produce the recall/shrinkage cost the tangential energies
    do.

    Paper Eq. 4-5, 7-8:
        L_align(q) = ||H f(q) ghat(q)||_2^2,  ghat = grad f / ||grad f||
        E_align    = sum_q beta_q L_align(q),  beta_q = exp(-delta |f(q)|)

    Args:
        d_m:         (N,) distance of each sample from the observed surface in
                     METRES, used for beta. Pass the sdf *label*, not the
                     prediction, for the same reason `intrinsic_hessian_loss`
                     does: beta computed on the prediction is a term the network
                     can cut by pushing |f| up at high-residual samples --
                     moving the surface -- instead of by aligning anything.
        delta_per_m: beta's decay rate, per metre. The paper's default of 10 is
                     stated in units where the shape is normalized to span 1, so
                     it is a *relative* width (~7% of the shape's extent) and
                     cannot be reused literally on a field in metres. Choose it
                     against the sampling band: with samples spanning +-15 mm,
                     delta = 200/m puts beta at 1.0 on the surface and 0.05 at
                     the band edge. 0 disables the weighting (uniform mean over
                     the mask).
    """
    g_masked = g[mask]
    ghat = g_masked / g_masked.norm(2, dim=-1, keepdim=True).clamp_min(_EPS)
    hg = torch.matmul(hessian_m[mask], ghat.unsqueeze(-1)).squeeze(-1)
    align = hg.pow(2).sum(dim=-1)

    if delta_per_m <= 0.0 or d_m is None:
        return align.mean()

    # normalized by sum(beta) so delta changes *where* the energy is measured
    # without also rescaling it -- otherwise raising delta looks like lowering
    # the loss and the weight would need retuning alongside it
    beta = torch.exp(-delta_per_m * d_m[mask].abs())
    return (beta * align).sum() / beta.sum().clamp_min(_EPS)


def singular_hessian_loss(hessian_m, mask):
    """[4] Neural-Singular-Hessian (Wang et al., TOG 2023): det(H) = 0 near the
    surface. Weaker than driving H to zero, so it suppresses spurious critical
    points without flattening genuine detail."""
    return torch.linalg.det(hessian_m[mask]).abs().mean()


def relaxed_eikonal_loss(g, mask, sigma_min=0.8):
    """[4] Neural-Singular-Hessian Eq. 12: only require the gradient not to
    vanish, rather than pinning it to exactly 1. Applied on-surface only."""
    return F.relu(-(g[mask].norm(2, dim=-1) - sigma_min)).mean()


def divergence_loss(hessian_m, mask):
    """[5] DiGS (Ben-Shabat et al., CVPR 2022): the divergence of the gradient
    field, i.e. the laplacian, i.e. the trace of the hessian."""
    laplacian = hessian_m[mask].diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return laplacian.abs().mean()


def siren_offsurface_loss(sdf_m, mask, alpha=100.0):
    """[6] SIREN (Sitzmann et al., NeurIPS 2020) L_D: penalize off-surface
    points that produce sdf values near zero.

    alpha is in 1/m so that the published value of 100 keeps its intended
    meaning ("penalize within about a centimetre"). The exponent is clamped
    because exp(-100) underflows float32.
    """
    return torch.exp(-(alpha * sdf_m[mask].abs()).clamp(max=30.0)).mean()


def diffcd_surface_to_point_loss(coord, sdf_m, g, mask, ref_points, query_fn,
                                 newton_steps=2, scale=1.0):
    """[7] DiffCD (Haerenstam-Nielsen et al., ECCV 2024): the surface-to-points
    half of the Chamfer distance, which the usual |f(x)| term omits.

    Project the masked samples onto the zero level set, find each projection's
    nearest input point, then measure that distance through a reparameterization
    of the surface point reproducing the paper's level-set derivative

        d x(theta) / d theta = -f_theta * g / ||g||^2                (Eq. 12-13)

    so no custom backward is needed.

    `query_fn(x) -> (sdf_m, g)` re-evaluates the field. The Newton steps run
    detached (a step is meaningless without refreshing f), and one final
    attached query at the projected point supplies the reparameterization --
    that last query is the only extra graph this term builds.

    coord/ref_points are in scaled cube units; the returned distance is metres.
    """
    # g is d(sdf_m)/d(x_metres), so a metric Newton step f/||g||^2 * g becomes
    # cube units after one factor of scale.
    with torch.no_grad():
        x_proj = coord[mask]
        f = sdf_m[mask].detach()
        g_cur = g[mask].detach()
        for _ in range(max(newton_steps, 1)):
            g_sq = g_cur.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS)
            x_proj = x_proj - (f.unsqueeze(-1) / g_sq) * g_cur * scale
            x_proj = x_proj.clamp(-1.0, 1.0)
            f, g_cur = query_fn(x_proj)
            f, g_cur = f.detach(), g_cur.detach()

        nn_idx = torch.cdist(x_proj, ref_points).argmin(dim=1)
        nearest = ref_points[nn_idx]

    # attached evaluation at the projected point drives the level-set derivative
    f_final, g_final = query_fn(x_proj)
    g_sq = g_final.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    x_reparam = x_proj - ((f_final - f_final.detach()).unsqueeze(-1) / g_sq) * g_final * scale

    return (x_reparam - nearest).norm(2, dim=-1).mean() / scale


def sdf_hessian_loss(g, hessian, surface_mask, hessian_norm_scale=7e-9, grad_gate_tau=0.6,
                     point_weight=None):
    """Ambient-Hessian form of our regularizer, kept for ablation (the default
    is `intrinsic_hessian_loss` below): penalize the (Frobenius-norm)
    curvature of the field near the surface, matching the L1 smoothness
    energy of [Rowe, Jacobson & Stein, "Sharpening and Sparsifying with
    Surface Hessians", SIGGRAPH Asia 2024], E_H(u) = sum_faces A_f ||H_f(u)||_F
    -- the un-squared Frobenius norm, aggregated linearly (mean, not
    mean-of-squares) across sampled points as a Monte-Carlo estimate of that
    integral. The un-squaring is what gives the sparsity-preserving behavior
    the paper documents: a linear per-point cost lets flat regions be driven
    to exactly zero curvature while tolerating a sparse set of genuine ridges,
    unlike a squared (L2) energy which penalizes ridges quadratically along
    with everything else.

    (The Frobenius and spectral norms differ only where curvature is spread
    across several principal directions, by up to ~sqrt(2) at an isotropic
    corner; at a clean single-direction edge they are identical.)

    grad_gate_tau: gates the penalty by local eikonal confidence, |grad f|.
    On trained Redwood fields, a large weight_h drags a
    broad, otherwise-flat surface area by several cm specifically where
    |grad f| is already far below 1 (eikonal weakly enforced, e.g.
    under-observed/grazing-angle lidar coverage) -- not because those points
    have more curvature to fix (matched control points with |grad f|~=1 had
    comparable or higher curvature and did not drift). A weak eikonal term
    pushes back less, so an ungated hessian penalty dominates there and drags
    the surface with it. The effect is a cliff, not a gradient (points with
    |grad f|<~0.5 saw ~5x the eventual drift of every other decile), so this
    ramps from 0 up to full weight over [0, grad_gate_tau] and holds at full
    weight above it, rather than gating smoothly across the whole [0,1] range
    (which would under-protect the actual problem points). This does cost
    some real detail -- about 11% of the top 5% highest-curvature (candidate
    real edge/corner) points also fall below grad_gate_tau=0.6 -- but the
    reverse check (points below tau average *lower* curvature than
    well-constrained points) suggests the tradeoff favors gating.

    point_weight: optional per-sample multiplier, already restricted to the
    surface_mask points, letting weight_h vary spatially. A single global
    weight_h has no setting that works: 0.1 leaves the noise artifacts
    untouched, 3.0 removes them but visibly over-smooths genuine geometry.
    With a bump mask (utils/bump_mask.py) this carries 3.0 on the artifacts
    and 0.1 elsewhere. Note this weights the *mean*, it does not renormalise by
    the weights: the term's overall magnitude is meant to rise when more of the
    batch is flagged, which is what makes it comparable to the scalar-weight
    runs it is being measured against.
    """
    hessian_norm = torch.linalg.norm(hessian[surface_mask], ord='fro', dim=(-2, -1))
    grad_norm = g[surface_mask].norm(2, dim=-1)
    gate = (grad_norm / grad_gate_tau).clamp(max=1.0)
    if point_weight is not None:
        gate = gate * point_weight
    return (hessian_norm * gate).mean() * hessian_norm_scale



def intrinsic_hessian_loss(g, hessian_m, d_m, surface_mask, band_h_m=0.003,
                           grad_floor=0.2, point_weight=None, area_weight=None,
                           squared=False):
    """Corrected discretization of the L1 surface-Hessian energy of
    [Rowe, Jacobson & Stein, "Sharpening and Sparsifying with Surface Hessians",
    SIGGRAPH Asia 2024], which `sdf_hessian_loss` above only approximates.

    That paper's energy is  E_H(u) = 1/2 * integral_Omega ||H u|| dx  (their
    Eq. 2), where Omega is a *surface*, and H u = grad grad u (their Eq. 3) is
    the *covariant* derivative of the gradient -- what they call "the four
    intrinsic Hessian entries", i.e. a 2x2 tangent-plane tensor. Two things
    differ in `sdf_hessian_loss`, both measured on a trained Ignatius field:

      1. OPERAND. It norms the ambient 3x3 Hessian of the SDF. Measured
         ||H n||/||H|| = 0.58 at every offset from -15mm to +15mm (0.579 on
         Redwood, so this is structural, not scene-specific): a majority of what
         it charges for is the normal row/column, which has no counterpart in
         the intrinsic 2x2 tensor. Since |grad f| runs 0.51-0.88 across that
         band rather than 1, that component is largely eikonal error, not
         surface bending. Two consequences: the residual acts as a total
         curvature flow that contracts convex regions (recall fell 0.89pp vs
         precision 0.30pp at weight_h=0.1 -- the surface shrinks), and the L1
         sparsity budget is spent allowing spikes in eikonal violation rather
         than at genuine creases, so the feature preservation the L1 form is
         chosen for never materializes.

         Fixed by penalizing the shape operator S = P H P / |grad f|,
         P = I - n n^T, which IS the level set's second fundamental form -- the
         intrinsic tensor, expressed in ambient coordinates with a zero normal
         row/column.

      2. SUPPORT. It averages over every sample with weight > 0, i.e. the whole
         +-surface_sample_range_m band, giving ~85% of the penalty to points
         that are not on the surface at all (measured: 6 of 7 equally-spaced
         shells are off-surface, each contributing ~15% of total ||H||). The
         paper integrates over the surface.

         Fixed by weighting each sample by a narrow Gaussian in its own sdf
         label, exp(-(d/band_h_m)^2), which concentrates the measure at d = 0
         and approximates integrating over the level set rather than the slab.
         Normalized by the weight sum, so band_h_m changes where the energy is
         measured without also rescaling it.

    Deliberately keeps the un-squared Frobenius norm and linear (mean)
    aggregation -- that part of `sdf_hessian_loss` already matches the paper,
    and is what makes the energy sparsifying rather than uniformly smoothing.

    Deliberately drops `grad_gate_tau`. That gate *reduces* the penalty where
    |grad f| is low, which is where the artifacts sit (bumps measured at
    |grad f| 0.796 vs 1.066 background), so it protects exactly what it should
    penalize; it exists to stop raw ||H|| dragging surfaces at low |grad f|,
    a failure the shape operator removes at its source. `grad_floor` clamps the
    |grad f| division instead, since S diverges as |grad f| -> 0.

    Args:
        g:            (N,3) field gradient, physical units
        hessian_m:    (N,3,3) metric hessian (1/m) -- NOT the raw-logit hessian
                      `sdf_hessian_loss` is scaled against
        d_m:          (N,) each sample's distance to the observed surface, in
                      metres. Pass the *label* (sdf_label / config.scale), not
                      the prediction: the weight must be fixed data. Weighting
                      by the predicted sdf would let the network cut this loss
                      by pushing high-curvature samples out of the band
                      (raising |f| there) instead of by reducing curvature,
                      which is a strictly easier way to lower the objective.
        surface_mask: near-surface band (weight > 0)
        band_h_m:     Gaussian half-width in metres concentrating the measure on
                      the surface. Should be well below surface_sample_range_m,
                      or this reduces to the slab average it is correcting.
        point_weight: optional per-sample multiplier (already restricted to
                      surface_mask), same role as in `sdf_hessian_loss`
        area_weight:  optional per-sample surface area (already restricted to
                      surface_mask). The cited energy is an AREA-weighted sum
                      over faces, sum_f A_f ||H_f(u)||_F; our samples arrive at
                      observation density instead, so without this the integral
                      is taken against the wrong measure and the L1 form's
                      "spikes allowed on measure-zero regions" behaviour keys
                      off how often a region was scanned rather than how much
                      area it covers.
    """
    g_s = g[surface_mask]
    H_s = hessian_m[surface_mask]
    d = d_m[surface_mask]

    # shape_operator keeps n unit-length and clamps only the 1/|grad f|
    # division, so P = I - n n^T stays a projector (see its docstring)
    S = shape_operator(g_s, H_s, grad_floor)
    s_norm = torch.linalg.norm(S, ord="fro", dim=(-2, -1))
    if squared:
        # E_H2-style: squared Frobenius, i.e. the L2 surface-Hessian energy.
        # Penalizes ridges quadratically, so it smooths uniformly instead of
        # sparsifying -- the control for whether the L1 form's feature
        # preservation is real. NOTE this changes the term's magnitude by
        # roughly ||S|| itself (order 1e2 here), so weights are not comparable
        # between the two: see intrinsic_squared in utils/config.py.
        s_norm = s_norm ** 2

    w = torch.exp(-(d / band_h_m) ** 2)
    if area_weight is not None:
        # sum_f A_f ||H_f|| (their Eq. 18) rather than a plain sample mean --
        # see LiDARDataset.build_area_weights for why sample density is not area
        w = w * area_weight
    if point_weight is not None:
        w = w * point_weight
    return (s_norm * w).sum() / w.sum().clamp_min(_EPS)


def sdf_diff_loss(pred, label, weight, scale, l2_loss=True):
    count = pred.shape[0]
    diff = pred - label
    diff_m = diff / scale # so it's still in m unit
    if l2_loss:
        loss = (weight * (diff_m**2)).sum() / count  # l2 loss
    else:
        loss = (weight * abs(diff_m)).sum() / count  # l1 loss
    return loss


def sdf_bce_loss(pred, label, sigma, weight, weighted=False, bce_reduction = "mean"):
    if weighted:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weight)
    else:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction)
    label_op = torch.sigmoid(label / sigma)  # occupancy prob
    loss = loss_bce(pred, label_op)
    return loss


def ray_estimation_loss(x, y, d_meas):  # for each ray
    # x as depth
    # y as sdf prediction
    # d_meas as measured depth

    # print(x.shape, y.shape, d_meas.shape)

    # regard each sample as a ray
    mat_A = torch.vstack((x, torch.ones_like(x))).transpose(0, 1)
    vec_b = y.view(-1, 1)

    # print(mat_A.shape, vec_b.shape)

    least_square_estimate = torch.linalg.lstsq(mat_A, vec_b).solution

    a = least_square_estimate[0]  # -> -1 (added in ekional loss term)
    b = least_square_estimate[1]

    # d_estimate = -b/a
    d_estimate = torch.clamp(-b / a, min=1.0, max=40.0)  # -> d

    # d_error = (d_estimate-d_meas)**2

    d_error = torch.abs(d_estimate - d_meas)

    # print(mat_A.shape, vec_b.shape, least_square_estimate.shape)
    # print(d_estimate.item(), d_meas.item(), d_error.item())

    return d_error


def ray_rendering_loss(x, y, d_meas):  # for each ray [should run in batch]
    # x as depth
    # y as occ.prob. prediction
    x = x.squeeze(1)
    sort_x, indices = torch.sort(x)
    sort_y = y[indices]

    w = torch.ones_like(y)
    for i in range(sort_x.shape[0]):
        w[i] = sort_y[i]
        for j in range(i):
            w[i] *= 1.0 - sort_y[j]

    d_render = (w * x).sum()

    d_error = torch.abs(d_render - d_meas)

    # print(x.shape, y.shape, d_meas.shape)
    # print(mat_A.shape, vec_b.shape, least_square_estimate.shape)
    # print(d_render.item(), d_meas.item(), d_error.item())

    return d_error


def batch_ray_rendering_loss(x, y, d_meas, neus_on=True):  # for all rays in a batch
    # x as depth [ray number * sample number]
    # y as prediction (the alpha in volume rendering) [ray number * sample number]
    # d_meas as measured depth [ray number]
    # w as the raywise weight [ray number]
    # neus_on determine if using the loss defined in NEUS

    # print(x.shape, y.shape, d_meas.shape, w.shape)

    sort_x, indices = torch.sort(x, 1)  # for each row
    sort_y = torch.gather(y, 1, indices)  # for each row

    if neus_on:
        neus_alpha = (sort_y[:, 1:] - sort_y[:, 0:-1]) / ( 1. - sort_y[:, 0:-1] + 1e-10)
        # avoid dividing by 0 (nan)
        # print(neus_alpha)
        alpha = torch.clamp(neus_alpha, min=0.0, max=1.0)
    else:
        alpha = sort_y

    one_minus_alpha = torch.ones_like(alpha) - alpha + 1e-10

    cum_mat = torch.cumprod(one_minus_alpha, 1)

    weights = cum_mat / one_minus_alpha * alpha

    weights_x = weights * sort_x[:, 0 : alpha.shape[1]]

    d_render = torch.sum(weights_x, 1)

    d_error = torch.abs(d_render - d_meas)

    # d_error = torch.abs(d_render - d_meas) * w # times ray-wise weight

    d_error_mean = torch.mean(d_error)

    return d_error_mean
