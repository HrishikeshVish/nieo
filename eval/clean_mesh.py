"""Remove disconnected floating components from a reconstructed mesh before
evaluation.

Octree-based volumetric SDF reconstruction (unlike a purely coordinate-MLP
neural field) marching-cubes every occupied leaf independently, so a few
mis-labelled voxels far from the true surface -- background clutter that
survived the training-time crop, or an isolated noisy region of the field --
can marching-cube into a small watertight shell with no connection to the
main surface at all. These floaters are evaluation noise, not reconstruction
quality: they were never part of "the object" a baseline paper's post-hoc
crop is checking for, and scoring them (mainly against precision, since nearby
GT recall is unaffected either way) charges the method for geometry nobody
claims represents the object.

Standard practice in the compared baseline papers (and in NeRF/SDF eval
scripts generally) is to keep only the largest
connected component of the predicted mesh before computing Chamfer/F-score.
Applied identically to every method being compared, this is not a quality
improvement for any one of them -- it removes noise both share.
"""
import argparse
import os

import numpy as np
import open3d as o3d


def clean_mesh(mesh, keep_top_k=1, min_component_frac=0.0, verbose=True):
    """Keep only the largest `keep_top_k` connected component(s) of `mesh`,
    or any component covering at least `min_component_frac` of all triangles
    (whichever keeps more -- the two knobs are independent floors, not a
    combined one). Returns the cleaned mesh and a stats dict.
    """
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    total_tri = cluster_n_triangles.sum()

    order = np.argsort(-cluster_n_triangles)
    keep_labels = set(order[:keep_top_k].tolist())
    if min_component_frac > 0.0:
        frac_ok = np.where(cluster_n_triangles / total_tri >= min_component_frac)[0]
        keep_labels |= set(frac_ok.tolist())

    keep_mask = np.isin(triangle_clusters, list(keep_labels))
    kept_tri = cluster_n_triangles[list(keep_labels)].sum()

    cleaned = o3d.geometry.TriangleMesh(mesh)
    cleaned.remove_triangles_by_mask(~keep_mask)
    cleaned.remove_unreferenced_vertices()

    stats = dict(
        n_components=len(cluster_n_triangles),
        n_components_kept=len(keep_labels),
        triangles_before=int(total_tri),
        triangles_after=int(kept_tri),
        fraction_kept=float(kept_tri / total_tri),
    )
    if verbose:
        print(f"  {stats['n_components']} components -> kept {stats['n_components_kept']}, "
              f"{stats['triangles_before']} -> {stats['triangles_after']} triangles "
              f"({100*stats['fraction_kept']:.3f}% kept)")
    return cleaned, stats


def strip_sharp(mesh, s_norm, pct, verbose=True):
    """Chop off sharp ridges/spikes that stick out of the surface, then keep
    only the largest connected component of what survives.

    Connectivity alone cannot remove these: on a Tanks-and-Temples Truck
    mesh, after largest-component filtering the result is a single solid blob
    (13 non-manifold edges, 15 triangles -- negligible), so every remaining
    artifact is ATTACHED to the main body. Cutting the high-curvature vertices
    first severs the thin necks those spikes hang from, and the re-cluster then
    drops whatever is left dangling.

    No dilation step: re-adding the removed vertices adjacent to the core would
    put the spikes straight back, which is the opposite of the intent here.
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(s_norm) != len(verts):
        sys.exit(f"energy has {len(s_norm)} entries but mesh has {len(verts)} vertices")
    thr = np.percentile(s_norm[np.isfinite(s_norm)], pct)
    low = s_norm <= thr
    keep = low[tris].all(axis=1)
    kept = tris[keep]
    used = np.unique(kept)
    remap = -np.ones(verts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    out = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts[used]),
        o3d.utility.Vector3iVector(remap[kept]))
    if verbose:
        print(f"  strip_sharp p{pct}: |S|_F > {thr:.1f} 1/m cut -> "
              f"{len(tris):,} -> {len(kept):,} triangles "
              f"({100*(1-len(kept)/max(len(tris),1)):.2f}% removed)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--curvature_pct", type=float, default=None,
                     help="also chop vertices whose shape-operator norm ||S||_F is "
                          "above this percentile (e.g. 90), to remove sharp ridges "
                          "sticking out of the surface. Reads the cached "
                          "<mesh>.energy.npz (per-vertex array 's_norm').")
    ap.add_argument("--energy_npz", default=None,
                     help="override the auto-derived <mesh>.energy.npz path")
    ap.add_argument("--keep_top_k", type=int, default=1,
                     help="keep the K largest connected components (default: 1, "
                          "matches the largest-component-only protocol used by "
                          "the baseline papers)")
    ap.add_argument("--min_component_frac", type=float, default=0.0,
                     help="also keep any component covering at least this "
                          "fraction of all triangles, in addition to the top-K "
                          "(e.g. 0.01 to keep a real but separately-tessellated "
                          "part like a mirror). 0 disables this.")
    args = ap.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.pred_mesh)
    print(f"{args.pred_mesh}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    if args.curvature_pct is not None:
        npz = args.energy_npz or (os.path.splitext(args.pred_mesh)[0] + ".energy.npz")
        if not os.path.exists(npz):
            sys.exit(f"no cached energy at {npz}; pass --energy_npz")
        s_norm = np.load(npz)["s_norm"]
        mesh = strip_sharp(mesh, s_norm, args.curvature_pct)

    cleaned, stats = clean_mesh(mesh, args.keep_top_k, args.min_component_frac)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    o3d.io.write_triangle_mesh(args.out, cleaned)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
