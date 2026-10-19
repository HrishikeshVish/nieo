"""Spatially varying weight for the Hessian regularizer.

A single global weight_h has no good setting: 0.1 leaves the noise artifacts in
place, 3.0 removes them but visibly over-smooths genuine geometry. Both classes
here make weight_h a function of position instead, so 3.0 lands on the artifacts
and 0.1 on everything else.

  BumpMask        a grid precomputed offline (an .npz with mask/origin/voxel_size)
                  from a finished first-pass field, scored with the Monge fit residual (AUC 0.904
                  against genuine sharp edges). Most accurate, but two-pass.

  OnlineBumpMask  built during the run itself from the shape operator, which
                  costs nothing because the hessian and gradient are already
                  computed for the loss. Single-pass, but a weaker per-point
                  discriminator (AUC 0.743) -- see that class for why voxel
                  accumulation is what makes it usable anyway.

Nearest-voxel lookup, not trilinear, in both: the fields vary smoothly at a much
coarser scale than their own resolution, so interpolation would add nothing but a
chance to get the axis order wrong.
"""
import numpy as np
import torch


def shape_operator_norm(g, hessian_m, eps=1e-6):
    """||S||_F where S = P H P / |grad f|, P = I - n n^T -- the curvature of the
    level set, as opposed to the raw hessian of the field.

    Measured on a trained Redwood field: ||S||/||H|| has median 0.646 and
    ||H n||/||H|| median 0.579, so about 35% of what a raw-hessian penalty
    charges for is normal-direction curvature and |grad f| variation (eikonal
    error), not surface bending. Raw ||H|| is also a near-useless bump
    discriminator here (separating power < 0.12 against the labelled regions)
    while ||S|| reaches 0.743, which is why the mask scores this and not ||H||.

    `hessian_m` must be the metric hessian (1/m), not the raw-logit one the
    ambient-L2 term (sdf_hessian_loss) is tuned against.
    """
    gn = g.norm(dim=-1, keepdim=True).clamp_min(eps)
    n = g / gn
    P = torch.eye(3, device=g.device, dtype=g.dtype).expand(g.shape[0], 3, 3) \
        - n.unsqueeze(-1) * n.unsqueeze(-2)
    S = torch.bmm(torch.bmm(P, hessian_m), P) / gn.unsqueeze(-1)
    return torch.linalg.norm(S, ord="fro", dim=(-2, -1))


class BumpMask:
    def __init__(self, path, device, dtype=torch.float32):
        z = np.load(path, allow_pickle=True)
        self.grid = torch.tensor(np.asarray(z["mask"], dtype=np.float32),
                                 device=device, dtype=dtype)
        self.origin = torch.tensor(z["origin"], device=device, dtype=dtype)
        self.voxel = float(z["voxel_size"])
        self.dims = torch.tensor(self.grid.shape, device=device, dtype=torch.long)
        self.path = path

    def __repr__(self):
        return (f"BumpMask({self.path}, dims={list(self.grid.shape)}, "
                f"voxel={self.voxel}m, {100*(self.grid>0.5).float().mean():.2f}% >0.5)")

    @torch.no_grad()
    def m(self, coord_scaled, scale):
        """m(x) in [0,1] for coordinates in the scaled [-1,1] cube.

        Points outside the grid get 0, i.e. the base weight -- the conservative
        direction, since the grid covers everything that was reconstructed.
        """
        xyz = coord_scaled.detach() / scale
        i = torch.floor((xyz - self.origin) / self.voxel).long()
        ok = ((i >= 0) & (i < self.dims)).all(dim=-1)
        i = i.clamp(min=torch.zeros_like(self.dims), max=self.dims - 1)
        v = self.grid[i[:, 0], i[:, 1], i[:, 2]]
        return torch.where(ok, v, torch.zeros_like(v))

    @torch.no_grad()
    def point_weight(self, coord_scaled, scale, w_base, w_bump):
        """Per-sample multiplier relative to w_base, so the caller can keep
        multiplying by config.weight_h exactly as before: the multiplier is 1
        outside the mask and w_bump/w_base inside it."""
        if w_base <= 0:
            raise ValueError("weight_h must be > 0 to use a bump mask")
        return 1.0 + (w_bump / w_base - 1.0) * self.m(coord_scaled, scale)


def monge_disc(n_rings=2, per_ring=8, device="cuda", dtype=torch.float32):
    """The 17-point stencil the offline detector was validated with."""
    uv = [(0.0, 0.0)]
    for r in range(1, n_rings + 1):
        for k in range(per_ring):
            th = 2 * np.pi * (k / per_ring + 0.5 * (r % 2) / per_ring)
            uv.append(((r / n_rings) * np.cos(th), (r / n_rings) * np.sin(th)))
    return torch.tensor(uv, device=device, dtype=dtype)


@torch.no_grad()
def _newton_project(c, query_fn, scale, steps=2):
    """Detached Newton projection onto the zero level set. `query_fn` returns the
    sdf in metres and the metric gradient, so the step is sdf*scale/|g| along -n."""
    for _ in range(steps):
        sdf, g = query_fn(c)
        gn = g.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        c = c - (sdf.detach().view(-1, 1) * scale / gn.detach()) * (g.detach() / gn.detach())
    return c.detach()


@torch.no_grad()
def monge_residual(x, query_fn, scale, sigmas_m, uv, proj_steps=2):
    """RMS residual of a quadratic fitted to the level set over discs of radius sigma.

    Validated against hand-labelled artifact regions: AUC 0.892 at 5cm and 0.904
    at 5cm+10cm against genuine sharp edges, 0.920 against correctly
    reconstructed surface. Medians were bumps 0.0045 m / edges 0.0019 / controls
    0.0017 / background 0.0023, so bumps sit about 2.4x above real creases -- the
    ordering a curvature magnitude does NOT give here, because mc_res_m=0.01 and
    leaf_vox_size=0.02 already round the real edges off (measured ||S||: edges
    27.3, background 30.3, bumps 45.9).

    Positions only -- no hessians at the samples. Cost is about 71 field
    evaluations per scored point at two scales, which is why the caller scores a
    small probe subset per iteration and lets the voxel accumulator do the rest.
    """
    x = _newton_project(x, query_fn, scale, proj_steps)
    _, g = query_fn(x)
    n = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    a = torch.zeros_like(n)
    a[:, 0] = 1.0
    fl = n[:, 0].abs() > 0.9
    a[fl] = 0.0
    a[fl, 1] = 1.0
    t1 = torch.cross(n, a, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t2 = torch.cross(n, t1, dim=-1)

    B, K = x.shape[0], uv.shape[0]
    eye6 = 1e-8 * torch.eye(6, device=x.device, dtype=x.dtype)
    total = torch.zeros(B, device=x.device, dtype=x.dtype)
    for s_m in sigmas_m:
        off = (s_m * scale) * (uv[:, 0].view(1, K, 1) * t1.unsqueeze(1)
                               + uv[:, 1].view(1, K, 1) * t2.unsqueeze(1))
        smp = _newton_project((x.unsqueeze(1) + off).reshape(-1, 3), query_fn,
                              scale, proj_steps).reshape(B, K, 3)
        rel = (smp - x.unsqueeze(1)) / scale
        u = (rel * t1.unsqueeze(1)).sum(-1) / s_m
        v = (rel * t2.unsqueeze(1)).sum(-1) / s_m
        h = (rel * n.unsqueeze(1)).sum(-1)
        A = torch.stack([u * u, u * v, v * v, u, v, torch.ones_like(u)], -1)
        AtA = torch.bmm(A.transpose(1, 2), A) + eye6
        cf = torch.linalg.solve(AtA, torch.bmm(A.transpose(1, 2), h.unsqueeze(-1)))
        total += ((h - torch.bmm(A, cf).squeeze(-1)) ** 2).mean(-1).sqrt()
    return total / len(sigmas_m)


class OnlineBumpMask:
    """Region weighting built during training, from quantities the loss already has.

    Per iteration it scores the batch's surface samples with ||S|| (free: the
    hessian and gradient are already computed), accumulates a per-voxel EMA of
    that score, and reads each sample's weight back out of the accumulator.

    WHY THE ACCUMULATOR IS THE WHOLE POINT

    Pointwise ||S|| is a weak bump discriminator -- AUC 0.743 against the
    labelled edges, versus 0.904 for the Monge residual. Used per sample it would
    mostly weight noise. But the artifacts are regional: the labelled bumps span
    3-32cm and 62% of the ripple energy sits at 5-30cm wavelengths, so averaging
    the score over a voxel of that size is averaging over the feature itself
    while the per-point estimation noise cancels. Spatial coherence has to be
    *inside* the score rather than applied to it afterwards -- thresholding noisy
    per-point values and clustering the result chains speckle into spurious
    room-sized "regions".

    The EMA also decouples the weight field's timescale from the optimiser's: at
    momentum 0.99 the score has a ~100-iteration memory, so the weight cannot
    chase a single gradient step. That matters because the loop is inherently
    self-limiting -- smoothing a region lowers its ||S||, which lowers its
    weight, which stops the smoothing. That is the desired behaviour (it stops
    when the job is done, unlike a fixed mask), but only if it is damped enough
    not to oscillate.

    THRESHOLDS ARE FROZEN AFTER WARM-UP by default. They are quantiles of the
    occupied-voxel score distribution, so if they kept updating, a fixed ~20% of
    the scene would carry the high weight forever no matter how smooth it got.
    Freezing turns them into an absolute (but scene-calibrated) level that the
    score field can then fall below as the artifacts are removed. Pass
    adaptive=True to keep them tracking.

    During warm-up the weight is uniform: the score is meaningless until the
    field has taken shape, and committing weight 3.0 to whatever looks rough at
    iteration 50 would smooth geometry that has not been reconstructed yet.
    """

    def __init__(self, octree, config, device, dtype, voxel=0.15, momentum=0.99,
                 warmup=1000, quantile=0.8, soft=0.15, refresh=50, adaptive=False,
                 score="monge", probe=256, every=4, sigmas=(0.05, 0.10)):
        nodes = octree.get_octree_nodes(config.mc_query_level) / config.scale
        lo = nodes.min(axis=0) - voxel
        hi = nodes.max(axis=0) + voxel
        dims = (np.ceil((hi - lo) / voxel).astype(int) + 1)
        self.origin = torch.tensor(lo, device=device, dtype=dtype)
        self.dims_t = torch.tensor(dims, device=device, dtype=torch.long)
        self.dims = tuple(int(d) for d in dims)
        self.n_vox = int(np.prod(dims))
        self.voxel = float(voxel)
        self.ema = torch.zeros(self.n_vox, device=device, dtype=dtype)
        self.seen = torch.zeros(self.n_vox, device=device, dtype=torch.bool)
        self.momentum = float(momentum)
        self.warmup = int(warmup)
        self.quantile = float(quantile)
        self.soft = float(soft)
        self.refresh = int(refresh)
        self.adaptive = bool(adaptive)
        self.score_mode = str(score)
        self.probe = int(probe)
        self.every = max(int(every), 1)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.uv = monge_disc(2, 8, device, dtype)
        self.lo_t = self.hi_t = None
        self.last_frac = 0.0
        self.last_score = 0.0

    def __repr__(self):
        s = (f"OnlineBumpMask(score={self.score_mode}, dims={list(self.dims)}, "
             f"voxel={self.voxel}m, momentum={self.momentum}, warmup={self.warmup}, "
             f"q={self.quantile}, adaptive={self.adaptive}")
        if self.score_mode == "monge":
            s += (f", sigmas={[f'{x*100:.0f}cm' for x in self.sigmas]}, "
                  f"probe={self.probe}/{self.every} iters")
        return s + ")"

    def _flat(self, coord_scaled, scale):
        xyz = coord_scaled.detach() / scale
        i = torch.floor((xyz - self.origin) / self.voxel).long()
        ok = ((i >= 0) & (i < self.dims_t)).all(dim=-1)
        i = i.clamp(min=torch.zeros_like(self.dims_t), max=self.dims_t - 1)
        flat = (i[:, 0] * self.dims[1] + i[:, 1]) * self.dims[2] + i[:, 2]
        return flat, ok

    @torch.no_grad()
    def update(self, coord_scaled, scale, score):
        """Accumulate this batch's scores into the voxel EMA."""
        flat, ok = self._flat(coord_scaled, scale)
        flat, score = flat[ok], score.detach()[ok]
        if flat.numel() == 0:
            return
        z = torch.zeros(self.n_vox, device=score.device, dtype=score.dtype)
        sums = z.index_add(0, flat, score)
        cnts = z.index_add(0, flat, torch.ones_like(score))
        touched = cnts > 0
        batch_mean = sums[touched] / cnts[touched]
        prev, was_seen = self.ema[touched], self.seen[touched]
        # first touch initialises rather than decaying towards zero from nothing
        self.ema[touched] = torch.where(
            was_seen, self.momentum * prev + (1 - self.momentum) * batch_mean, batch_mean)
        self.seen[touched] = True

    @torch.no_grad()
    def refresh_thresholds(self):
        v = self.ema[self.seen]
        if v.numel() < 32:
            return
        self.hi_t = torch.quantile(v.float(), self.quantile).to(v.dtype)
        self.lo_t = torch.quantile(v.float(), max(self.quantile - self.soft, 0.0)).to(v.dtype)

    @torch.no_grad()
    def m(self, coord_scaled, scale):
        if self.lo_t is None:
            return torch.zeros(coord_scaled.shape[0], device=coord_scaled.device,
                               dtype=coord_scaled.dtype)
        flat, ok = self._flat(coord_scaled, scale)
        s = self.ema[flat]
        denom = (self.hi_t - self.lo_t).clamp_min(1e-9)
        val = ((s - self.lo_t) / denom).clamp(0.0, 1.0)
        return torch.where(ok & self.seen[flat], val, torch.zeros_like(val))

    @torch.no_grad()
    def step(self, coord_scaled, scale, it, w_base, w_bump, query_fn=None, curv=None):
        """Score a probe subset, fold it into the field, then return a multiplier
        for the WHOLE batch.

        Scoring a subset while weighting everything is what makes the expensive
        detector affordable: the voxel EMA is the estimator, so each iteration
        only has to contribute a few hundred fresh samples to it. At the defaults
        (256 points every 4 iterations, ~71 field evaluations each) that is ~4.5k
        extra evaluations per iteration against a batch of 4096 that is already
        computing second derivatives.

        Returns None while warming up, leaving the caller on the plain scalar
        weight_h.
        """
        if it % self.every == 0 and coord_scaled.shape[0] > 0:
            k = min(self.probe, coord_scaled.shape[0]) if self.score_mode == "monge" \
                else coord_scaled.shape[0]
            if k < coord_scaled.shape[0]:
                sel = torch.randperm(coord_scaled.shape[0], device=coord_scaled.device)[:k]
                probe_x = coord_scaled[sel]
            else:
                sel, probe_x = None, coord_scaled
            if self.score_mode == "monge":
                if query_fn is None:
                    raise ValueError("score='monge' needs query_fn")
                s = monge_residual(probe_x, query_fn, scale, self.sigmas, self.uv)
            else:
                if curv is None:
                    raise ValueError("score='shape_op' needs curv")
                s = curv if sel is None else curv[sel]
            self.last_score = float(s.mean())
            self.update(probe_x, scale, s)
        if it < self.warmup:
            if self.refresh and it % self.refresh == 0:
                self.refresh_thresholds()
            return None
        if self.lo_t is None or (self.adaptive and self.refresh and it % self.refresh == 0):
            self.refresh_thresholds()
        if self.lo_t is None:
            return None
        if w_base <= 0:
            raise ValueError("weight_h must be > 0 to use a bump mask")
        mv = self.m(coord_scaled, scale)
        self.last_frac = float((mv > 0.5).float().mean())
        return 1.0 + (w_bump / w_base - 1.0) * mv
