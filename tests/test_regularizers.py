"""Analytic checks for the baseline regularizers in utils/loss.py.

Each term is evaluated on a closed-form field whose derivatives are known, at
two very different world scales (KITTI: 614 m world; indoor RGB-D: 41 m world).
The point of running both is that the metric values must agree: that is the
actual test of the unit conversion in utils.tools.get_derivatives, and it is
what lets one loss weight transfer across datasets.

Run:  python tests/test_regularizers.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loss import (
    divergence_loss,
    grad_hessian_align_loss,
    hessian_frobenius_loss,
    hessian_l1_loss,
    intrinsic_hessian_loss,
    minimal_surface_loss,
    relaxed_eikonal_loss,
    siren_offsurface_loss,
    singular_hessian_loss,
)
from utils.tools import get_derivatives

# two scale extremes: SHINE's KITTI config (0.3 m leaf) and an indoor RGB-D config (0.02 m leaf)
SCALES = {
    "kitti  (614 m world)": (1.0 / (0.3 * 2 ** 11), 0.55 * 0.1),
    "rgbd   ( 41 m world)": (1.0 / (0.02 * 2 ** 11), 0.55 * 0.02),
}

TOL = 2e-3


def _derivatives(sdf_metres_fn, points_m, scale, sigma_sigmoid_m):
    """Run a closed-form sdf through the same logit/scaling path as the model."""
    sigma_sigmoid = sigma_sigmoid_m * scale
    coord = (points_m * scale).clone().requires_grad_(True)

    # invert the pipeline's convention: the decoder emits a logit, and
    # sdf_scaled = pred * sigma_sigmoid, so pred = sdf_m * scale / sigma_sigmoid
    sdf_m_true = sdf_metres_fn(coord / scale)
    pred = sdf_m_true * scale / sigma_sigmoid

    return get_derivatives(coord, pred, sigma_sigmoid, scale, need_hessian=True)


def _report(name, values, expected):
    spread = max(values.values()) - min(values.values())
    ok_match = spread <= TOL * max(1.0, abs(expected))
    ok_value = all(abs(v - expected) <= TOL * max(1.0, abs(expected)) for v in values.values())
    status = "PASS" if (ok_match and ok_value) else "FAIL"
    detail = "  ".join("%s=%.6g" % (k.split()[0], v) for k, v in values.items())
    print("  [%s] %-26s expected=%-10.6g %s" % (status, name, expected, detail))
    return ok_match and ok_value


def test_plane():
    """SDF to the plane z = 0 is s = z. Gradient (0,0,1), hessian identically 0."""
    print("\nplane  s(x) = z    ->  grad=(0,0,1), H=0")
    torch.manual_seed(0)
    pts = torch.rand(4096, 3) * 20.0 - 10.0  # metres

    results = {k: {} for k in ("grad", "l1", "frob", "det", "div", "align")}
    for label, (scale, sigma_m) in SCALES.items():
        sdf_m, g, H = _derivatives(lambda p: p[:, 2], pts, scale, sigma_m)
        mask = torch.ones(pts.shape[0], dtype=torch.bool)

        results["grad"][label] = g.norm(2, dim=-1).mean().item()
        results["l1"][label] = hessian_l1_loss(H, mask).item()
        results["frob"][label] = hessian_frobenius_loss(H, mask).item()
        results["det"][label] = singular_hessian_loss(H, mask).item()
        results["div"][label] = divergence_loss(H, mask).item()
        results["align"][label] = grad_hessian_align_loss(g, H, mask).item()

    ok = True
    ok &= _report("eikonal |grad|", results["grad"], 1.0)
    ok &= _report("hessian_l1", results["l1"], 0.0)
    ok &= _report("hessian_frobenius", results["frob"], 0.0)
    ok &= _report("singular_hessian", results["det"], 0.0)
    ok &= _report("divergence", results["div"], 0.0)
    ok &= _report("grad_hessian_align", results["align"], 0.0)
    return ok


def test_sphere():
    """SDF to a sphere of radius r is s = |x| - r. At distance d from the centre
    the hessian has eigenvalues (0, 1/d, 1/d): trace = 2/d, det = 0, and the
    gradient spans the null space so ||H ghat||^2 = 0."""
    r, d = 3.0, 5.0
    print("\nsphere s(x) = |x| - %g, evaluated on |x| = %g  ->  trace(H)=%g, det(H)=0"
          % (r, d, 2.0 / d))
    torch.manual_seed(0)
    dirs = torch.randn(4096, 3)
    pts = dirs / dirs.norm(2, dim=-1, keepdim=True) * d

    results = {k: {} for k in ("grad", "trace", "det", "align", "frob")}
    for label, (scale, sigma_m) in SCALES.items():
        sdf_m, g, H = _derivatives(lambda p: p.norm(2, dim=-1) - r, pts, scale, sigma_m)
        mask = torch.ones(pts.shape[0], dtype=torch.bool)

        results["grad"][label] = g.norm(2, dim=-1).mean().item()
        results["trace"][label] = divergence_loss(H, mask).item()
        results["det"][label] = singular_hessian_loss(H, mask).item()
        results["align"][label] = grad_hessian_align_loss(g, H, mask).item()
        results["frob"][label] = hessian_frobenius_loss(H, mask).item()

    ok = True
    ok &= _report("eikonal |grad|", results["grad"], 1.0)
    ok &= _report("divergence (2/d)", results["trace"], 2.0 / d)
    ok &= _report("singular_hessian", results["det"], 0.0)
    ok &= _report("grad_hessian_align", results["align"], 0.0)
    ok &= _report("hessian_frobenius", results["frob"], 2.0 / d ** 2)
    return ok


def test_sdf_valued_terms():
    """Terms that read the sdf directly rather than its derivatives. These must
    also agree across scales, since sdf_m is metric."""
    print("\nsdf-valued terms on s(x) = z, |z| ~ U(0, 2) m")
    torch.manual_seed(0)
    pts = torch.zeros(4096, 3)
    pts[:, 2] = torch.rand(4096) * 2.0

    results = {k: {} for k in ("siren", "minsurf", "relaxed")}
    for label, (scale, sigma_m) in SCALES.items():
        sdf_m, g, H = _derivatives(lambda p: p[:, 2], pts, scale, sigma_m)
        mask = torch.ones(pts.shape[0], dtype=torch.bool)

        results["siren"][label] = siren_offsurface_loss(sdf_m, mask, alpha=100.0).item()
        results["minsurf"][label] = minimal_surface_loss(sdf_m, mask, eps_m=0.1).item()
        results["relaxed"][label] = relaxed_eikonal_loss(g, mask, sigma_min=0.8).item()

    # E[exp(-100 z)] for z ~ U(0,2) is (1 - e^-200)/200
    ok = True
    ok &= _report("siren_offsurface", results["siren"], 1.0 / 200.0)
    # gradient is exactly unit, so the relaxed eikonal hinge is inactive
    ok &= _report("relaxed_eikonal", results["relaxed"], 0.0)
    # minimal surface has no tidy closed form; assert only cross-scale agreement
    spread = max(results["minsurf"].values()) - min(results["minsurf"].values())
    rel = spread / max(results["minsurf"].values())
    status = "PASS" if rel <= 1e-3 else "FAIL"
    print("  [%s] %-26s cross-scale agreement, rel spread=%.3g" % (status, "minimal_surface", rel))
    ok &= rel <= 1e-3
    return ok


def test_intrinsic_hessian():
    """intrinsic_hessian_loss must return the shape operator of the level set,
    ||S||_F = sqrt(k1^2 + k2^2), and nothing else.

    Case 1 is the honest sdf on a sphere's own surface: k1 = k2 = 1/r.

    Case 2 is the regression guard for grad_floor. Take a radial
    reparameterisation f = h(|x| - r) with h(t) = a*t + b*t^2, so
    |grad f| = h'(d) can be pushed below grad_floor while H picks up a normal
    block h''(d) n n^T that a true shape operator must reject. The tangential
    block is h'(d)/|x| * P, so with the clamp active the expected value is
    (h'/max(h', floor)) * sqrt(2)/|x| -- and critically it does NOT depend on
    h''. Clamping a single norm that also normalises n would leave
    n short by c = h'/floor, leak (1-c^2)^2 * h'' back into the tangential
    block, and read 0.1807 here instead of 0.1414.
    """
    r, dist = 3.0, 5.0
    a, b, floor = 0.02, 0.02, 0.2
    d = dist - r
    hp = a + 2 * b * d  # h'(d), the gradient magnitude at the sample shell
    print("\nintrinsic (shape-operator) hessian energy")
    torch.manual_seed(0)
    dirs = torch.randn(4096, 3)
    dirs = dirs / dirs.norm(2, dim=-1, keepdim=True)

    results = {k: {} for k in ("surf", "clamped")}
    for label, (scale, sigma_m) in SCALES.items():
        mask = torch.ones(dirs.shape[0], dtype=torch.bool)
        zero = torch.zeros(dirs.shape[0])

        # case 1: true sdf, sampled on the surface itself
        _, g, H = _derivatives(lambda p: p.norm(2, dim=-1) - r, dirs * r, scale, sigma_m)
        results["surf"][label] = intrinsic_hessian_loss(
            g, H, zero, mask, band_h_m=0.003, grad_floor=floor).item()

        # case 2: |grad f| driven under grad_floor, with a nonzero normal block
        _, g, H = _derivatives(
            lambda p, a=a, b=b: a * (p.norm(2, dim=-1) - r) + b * (p.norm(2, dim=-1) - r) ** 2,
            dirs * dist, scale, sigma_m)
        results["clamped"][label] = intrinsic_hessian_loss(
            g, H, zero, mask, band_h_m=0.003, grad_floor=floor).item()

    ok = True
    ok &= _report("||S||_F on sphere (sqrt2/r)", results["surf"], 2.0 ** 0.5 / r)
    ok &= _report("||S||_F under grad_floor", results["clamped"],
                  (hp / max(hp, floor)) * 2.0 ** 0.5 / dist)
    return ok


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)  # isolate unit errors from float32 noise
    passed = (test_plane() & test_sphere() & test_sdf_valued_terms()
              & test_intrinsic_hessian())
    print("\n%s" % ("ALL PASS" if passed else "FAILURES PRESENT"))
    sys.exit(0 if passed else 1)
