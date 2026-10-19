"""Converts one Tanks-and-Temples Training scene (image set + T&T's own published
camera trajectory) into this repo's per-frame KITTI-style point-cloud format, via
COLMAP dense multi-view stereo ("synthetic LiDAR": per-frame back-projected
depth maps, each with a known sensor origin, as the ray sampler requires).

Deliberately does NOT run our own structure-from-motion. T&T's Training scenes
publish a camera trajectory (<Scene>_COLMAP_SfM.log, TrajectoryIO format -- the
same format Redwood's GT log uses, parsed here with rgbd_to_kitti_format.py's
load_poses(with_head=True)) that is already metric-scale and already the exact
frame T&T's own <Scene>_trans.txt alignment matrix was calibrated against.
Re-deriving poses via our own SfM would give an arbitrary, non-metric gauge that
no longer matches that published alignment -- so instead we feed T&T's known
poses directly into COLMAP as a "reconstruction from known camera poses"
(https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses).
That FAQ's workflow still runs feature extraction/matching + point_triangulator
before the dense step -- image_undistorter derives each image's stereo-matching
neighbor list from *point visibility*, so without any triangulated points every
image gets zero neighbors and patch_match_stereo silently produces nothing.
point_triangulator
keeps the input poses fixed and only adds 3D points from real 2D feature
matches, so this doesn't reintroduce independent-SfM gauge drift -- see
run_native_cli()/run_pycolmap() for the full step sequence.

Intrinsics: T&T's evaluation-data bundle does not ship a standalone
intrinsics file, only the extrinsic trajectory log, so the focal length is
resolved from --fx, a shipped cameras.txt, or the images' EXIF before falling
back to T&T's generic 0.7*W guess -- see load_or_default_intrinsics(). Check
the result with --validate_against_gt (on by default) and
dataset/estimate_tandt_focal.py rather than trusting it blind.

Requires scripts/tanksandtemples/setup_mvs_env.sh to have been run first (writes
scripts/tanksandtemples/.mvs_backend so this script knows whether to call
pycolmap's dense-MVS API directly or shell out to a native `colmap` CLI).

Usage (small subset first, as a smoke test):
    python dataset/tandt_to_kitti_format.py \
        --scene ignatius --data_root data/tanksandtemples --limit_frames 30

Then, once that passes validation, the full scene:
    python dataset/tandt_to_kitti_format.py \
        --scene ignatius --data_root data/tanksandtemples
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import open3d as o3d
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rgbd_to_kitti_format import load_poses, write_poses_kitti_format, alphanum_key  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MVS_BACKEND_MARKER = os.path.join(ROOT, "scripts", "tanksandtemples", ".mvs_backend")

SCENE_DISPLAY_NAME = {
    "ignatius": "Ignatius",
    "truck": "Truck",
    "barn": "Barn",
    "caterpillar": "Caterpillar",
    "courthouse": "Courthouse",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# backend selection
# --------------------------------------------------------------------------

def read_mvs_backend():
    if not os.path.exists(MVS_BACKEND_MARKER):
        sys.exit(f"{MVS_BACKEND_MARKER} not found -- run "
                 "scripts/tanksandtemples/setup_mvs_env.sh on a GPU node first")
    content = open(MVS_BACKEND_MARKER).read().strip()
    if content == "pycolmap":
        return "pycolmap", None
    if content.startswith("native_colmap:"):
        return "native_colmap", content.split(":", 1)[1]
    sys.exit(f"unrecognized backend marker in {MVS_BACKEND_MARKER}: {content!r}")


# --------------------------------------------------------------------------
# inputs: trajectory, alignment, intrinsics, images
# --------------------------------------------------------------------------

def load_transform(path):
    vals = np.loadtxt(path).reshape(-1)
    if vals.size != 16:
        sys.exit(f"expected a 4x4 (16-value) transform in {path}, got {vals.size} values")
    return vals.reshape(4, 4)


def decompose_similarity(trans, tol=1e-6):
    """Split T&T's <Scene>_trans.txt into a uniform scale and a rigid transform.

    That matrix is a *similarity*, not a rigid transform: it carries the scale
    that takes COLMAP's arbitrary SfM gauge to GT metres (Ignatius: 0.8829).
    Left-multiplying it onto each pose and stopping there would leave
    poses.txt holding scaled rotation blocks, so the
    per-frame points stay in SfM units and every distance in the training config
    silently means `value / scale` metres instead of `value` metres.

    Folding the scale into the *points* instead keeps the composition identical
    while making poses rigid:
        p_world = (s*Rt)(Rc p + tc) + tt = (Rt Rc)(s p) + (s*Rt*tc + tt)
    i.e. the rotation block loses the scale, the translation block is unchanged,
    and the camera-local points are premultiplied by s. Downstream, every metre
    in a YAML file is then a real metre.
    """
    R = trans[:3, :3]
    row_norms = np.linalg.norm(R, axis=1)
    scale = float(row_norms.mean())
    if row_norms.std() > tol * max(scale, 1.0):
        sys.exit(f"trans.txt rotation block is not a uniform scaling "
                 f"(row norms {row_norms}) -- cannot decompose into scale * rigid")
    rigid = np.eye(4)
    rigid[:3, :3] = R / scale
    rigid[:3, 3] = trans[:3, 3]
    orth_err = np.abs(rigid[:3, :3] @ rigid[:3, :3].T - np.eye(3)).max()
    if orth_err > 1e-6:
        sys.exit(f"trans.txt rotation block is not orthonormal after removing "
                 f"scale {scale} (max error {orth_err}) -- not a similarity transform")
    return scale, rigid


# --------------------------------------------------------------------------
# T&T's published crop volume (<Scene>.json)
# --------------------------------------------------------------------------

class CropVolume:
    """T&T's <Scene>.json SelectionPolygonVolume: a polygon in the two axes
    orthogonal to `orthogonal_axis`, extruded between axis_min and axis_max.

    This is the volume T&T's own evaluation toolkit crops to before scoring, so
    it is the region a reconstruction is actually graded on -- for Ignatius a
    ~1.9 x 1.8 m footprint around the statue, versus the ~43 x 47 m of scene the
    unfiltered dense-MVS output spans. Restricting the conversion to it (plus a
    margin, so the field is still supported across the scored region rather than
    ending exactly at its boundary) is what keeps the octree and the marching
    cubes grid proportional to the object instead of to the whole courtyard.

    The margin is a true Minkowski dilation by a disc -- inside the polygon, or
    within `margin_m` of one of its edges -- not a scaling about the centroid,
    which would widen a long thin polygon far more than a compact one.
    """

    AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

    def __init__(self, json_path, margin_m=0.0):
        import json
        with open(json_path) as f:
            spec = json.load(f)
        if spec.get("class_name") != "SelectionPolygonVolume":
            sys.exit(f"{json_path} is not a SelectionPolygonVolume "
                     f"(got {spec.get('class_name')!r})")
        self.axis = self.AXIS_INDEX[spec["orthogonal_axis"].upper()]
        self.axis_min = float(spec["axis_min"]) - margin_m
        self.axis_max = float(spec["axis_max"]) + margin_m
        self.margin_m = margin_m
        # the polygon's own coordinates live in the two non-orthogonal axes;
        # open3d writes the orthogonal one as a zero column, so drop it.
        poly = np.asarray(spec["bounding_polygon"], dtype=np.float64)
        self.plane_axes = [i for i in range(3) if i != self.axis]
        self.polygon = poly[:, self.plane_axes]

    def contains(self, points_world):
        """Boolean mask over an (N,3) array of world-frame points."""
        if len(points_world) == 0:
            return np.zeros(0, dtype=bool)
        a = points_world[:, self.axis]
        inside = (a >= self.axis_min) & (a <= self.axis_max)
        if not inside.any():
            return inside
        uv = points_world[:, self.plane_axes]
        inside[inside] = self._in_polygon_2d(uv[inside])
        return inside

    def _in_polygon_2d(self, uv):
        poly = self.polygon
        n = len(poly)
        u, v = uv[:, 0], uv[:, 1]

        # ray casting (crossing number) for strict containment
        inside = np.zeros(len(uv), dtype=bool)
        for i in range(n):
            u0, v0 = poly[i]
            u1, v1 = poly[(i + 1) % n]
            straddles = (v0 > v) != (v1 > v)
            with np.errstate(divide="ignore", invalid="ignore"):
                u_cross = u0 + (v - v0) * (u1 - u0) / (v1 - v0)
            inside ^= straddles & (u < u_cross)

        if self.margin_m <= 0:
            return inside

        # ...then dilate: also keep anything within margin_m of an edge
        near = np.zeros(len(uv), dtype=bool)
        remaining = ~inside
        if remaining.any():
            pu, pv = u[remaining], v[remaining]
            best = np.full(pu.shape, np.inf)
            for i in range(n):
                u0, v0 = poly[i]
                u1, v1 = poly[(i + 1) % n]
                du, dv = u1 - u0, v1 - v0
                seg_len2 = du * du + dv * dv
                if seg_len2 == 0:
                    d = np.hypot(pu - u0, pv - v0)
                else:
                    t = np.clip(((pu - u0) * du + (pv - v0) * dv) / seg_len2, 0.0, 1.0)
                    d = np.hypot(pu - (u0 + t * du), pv - (v0 + t * dv))
                best = np.minimum(best, d)
            near[remaining] = best <= self.margin_m
        return inside | near

    def describe(self):
        lo = self.polygon.min(axis=0)
        hi = self.polygon.max(axis=0)
        names = "XYZ"
        return (f"crop volume: {names[self.plane_axes[0]]} "
                f"[{lo[0]:.2f},{hi[0]:.2f}] x {names[self.plane_axes[1]]} "
                f"[{lo[1]:.2f},{hi[1]:.2f}] polygon ({len(self.polygon)} verts), "
                f"extruded along {names[self.axis]} "
                f"[{self.axis_min:.2f},{self.axis_max:.2f}], margin {self.margin_m} m")


def list_images(images_dir):
    names = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(IMAGE_EXTS)],
                    key=alphanum_key)
    if not names:
        sys.exit(f"no images found in {images_dir}")
    return names


def read_image_size(path):
    im = o3d.io.read_image(path)
    arr = np.asarray(im)
    return arr.shape[1], arr.shape[0]  # W, H


FULL_FRAME_SENSOR_WIDTH_MM = 36.0


def focal_from_exif(image_path, W):
    """fx in pixels from the image's own EXIF, or None.

    T&T's Training image sets are video frames from a Sony A7S II (verified on
    all five scenes: same body, same 21mm lens, 1920x1080). EXIF carries
    FocalLengthIn35mmFilm, which is by definition referred to a 36mm-wide
    frame, so fx = f_35mm * W / 36 regardless of the real sensor size --
    no per-body sensor table needed.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(image_path) as im:
            exif = im.getexif()
            ifd = exif.get_ifd(0x8769)  # ExifOffset
            f35 = ifd.get(41989)        # FocalLengthIn35mmFilm
            if not f35:
                return None
            return float(f35) * W / FULL_FRAME_SENSOR_WIDTH_MM
    except Exception:
        return None


def load_or_default_intrinsics(gt_dir, display_name, W, H, sample_image=None,
                                fx_override=None):
    """Resolve intrinsics, preferring hard evidence over T&T's generic guess.

    Order: explicit --fx > a shipped cameras.txt > the images' own EXIF >
    T&T's documented fx = 0.7*W approximation.

    That approximation is LAST for a reason. It corresponds to a 71-degree
    horizontal field of view, but T&T's Training footage was shot at 21mm on a
    full-frame body -- 81 degrees, fx = 1120 not 1344, a 20% error on every
    scene. Because the same (wrong) camera model is fed to COLMAP for dense
    stereo, that error does not merely misplace the back-projected points, it
    corrupts the depth maps themselves: patch-match matches along epipolar
    geometry that does not describe the real images. Measured on Ignatius with
    fx=0.7*W, input points sat a median 79.5mm from the GT surface (1.8% within
    T&T's 3mm tau) and frames six apart disagreed with each other by 120mm,
    while adjacent frames agreed to 7mm -- the signature of a wrong camera
    model rather than depth noise. Independently confirmed by projecting the GT
    cloud into the real images: the silhouette matches at fx~1160 and spills
    well past the statue at 1344.
    """
    if fx_override:
        cx, cy = W / 2.0, H / 2.0
        print(f"using --fx override: fx=fy={fx_override:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        return dict(fx=fx_override, fy=fx_override, cx=cx, cy=cy)

    for candidate in (f"{display_name}_cameras.txt", f"{display_name}_intrinsics.txt",
                      "cameras.txt"):
        path = os.path.join(gt_dir, candidate)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("#") or not line.strip():
                        continue
                    parts = line.split()
                    # COLMAP cameras.txt: CAMERA_ID MODEL WIDTH HEIGHT PARAMS...
                    params = [float(x) for x in parts[4:]]
                    if len(params) >= 4:
                        print(f"using shipped intrinsics from {path}: {params[:4]}")
                        return dict(fx=params[0], fy=params[1], cx=params[2], cy=params[3])

    cx, cy = W / 2.0, H / 2.0
    if sample_image:
        fx = focal_from_exif(sample_image, W)
        if fx:
            print(f"using intrinsics from EXIF ({os.path.basename(sample_image)}): "
                  f"fx=fy={fx:.1f}, cx={cx:.1f}, cy={cy:.1f} "
                  f"(T&T's generic 0.7*W would have been {0.7*W:.0f}, "
                  f"{100*(0.7*W/fx-1):+.0f}%)")
            return dict(fx=fx, fy=fx, cx=cx, cy=cy)

    fx = fy = 0.7 * W
    print(f"WARNING: no shipped intrinsics file and no usable EXIF -- falling back to "
          f"T&T's generic approximation fx=fy=0.7*W={fx:.1f}, cx={cx:.1f}, cy={cy:.1f}.\n"
          f"         This was measurably wrong (~20%) for every T&T Training scene "
          f"checked; if the reconstruction comes out noisy and detail-free, suspect "
          f"this first and pass --fx explicitly.")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy)


# --------------------------------------------------------------------------
# COLMAP "known poses" model construction
# https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses
# --------------------------------------------------------------------------

def build_known_pose_model(model_dir, image_names, poses_cam_to_world, intrinsics, W, H):
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {intrinsics['fx']} {intrinsics['fy']} "
                f"{intrinsics['cx']} {intrinsics['cy']}\n")

    with open(os.path.join(model_dir, "images.txt"), "w") as f:
        f.write("# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "# (blank second line per image -- 2D points get filled in later by\n"
                "# point_triangulator, not written here)\n")
        for i, (name, T_wc) in enumerate(zip(image_names, poses_cam_to_world), start=1):
            # T&T's .log poses are camera-to-world; COLMAP's images.txt wants the
            # inverse (world-to-camera) as a quaternion (scalar-first) + translation
            T_cw = np.linalg.inv(T_wc)
            q = Quaternion(matrix=T_cw[:3, :3])  # pyquaternion is scalar-first: w,x,y,z
            t = T_cw[:3, 3]
            f.write(f"{i} {q.w} {q.x} {q.y} {q.z} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")

    # left empty here -- point_triangulator (run before image_undistorter, see
    # run_native_cli()/run_pycolmap()) populates this with real triangulated
    # points from 2D feature matches, keeping the poses above exactly fixed.
    # Needed because image_undistorter builds each image's stereo-matching
    # neighbor list from point *visibility* -- an empty points3D.txt gives it
    # nothing to work with, and every image silently gets zero neighbors
    # ("Ignoring reference image ..., because it has no source images").
    open(os.path.join(model_dir, "points3D.txt"), "w").close()


# --------------------------------------------------------------------------
# running COLMAP (native CLI or pycolmap), both backends producing the same
# on-disk dense workspace layout
# --------------------------------------------------------------------------

def _clean_stale_outputs(paths):
    """image_undistorter's internal file-copy throws (doesn't overwrite) if its
    destination already exists, and a stale colmap.db from a previous attempt
    would silently skip re-matching. Every output this pipeline writes is
    wiped before each attempt so reruns can't collide with leftovers."""
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.exists(p):
            os.remove(p)


def sync_model_ids_to_db(model_dir, db_path):
    """Renumber the known-pose model's IMAGE_IDs to match colmap.db's.

    feature_extractor is multithreaded and assigns image_id in *completion*
    order, not in the order images were listed, so the database's id->name
    mapping is not reproducible and does not necessarily agree with the sorted
    order build_known_pose_model() numbered by. point_triangulator pairs the
    model against the database by id, and aborts on any disagreement
    ("Check failed: existing_frame.DataIds() == frame.DataIds()").

    This is a race (ids of adjacent frames get swapped), so a small run can
    pass and a larger one fail. Taking the ids from the database rather than
    assuming they match makes it deterministic.

    Poses stay bound to their filenames throughout; only the integer id
    attached to each row changes.
    """
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        db_ids = {name: image_id
                  for image_id, name in conn.execute("SELECT image_id, name FROM images")}

    images_txt = os.path.join(model_dir, "images.txt")
    header, rows = [], {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith("#"):
                header.append(line)
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            rows[parts[-1]] = parts[1:-1]  # pose + camera_id, keyed by filename

    missing = sorted(set(rows) - set(db_ids))
    if missing:
        sys.exit(f"{len(missing)} model images are absent from {db_path} "
                 f"(e.g. {missing[:3]}) -- feature_extractor skipped them, likely "
                 f"unreadable files; cannot triangulate against a partial database")

    remapped = 0
    with open(images_txt, "w") as f:
        f.writelines(header)
        for name, rest in sorted(rows.items(), key=lambda kv: db_ids[kv[0]]):
            f.write(f"{db_ids[name]} {' '.join(rest)} {name}\n\n")
            remapped += 1
    print(f"synced {remapped} image ids in {images_txt} to {os.path.basename(db_path)}")


def _write_image_list(scene_dir, image_names):
    """feature_extractor otherwise ingests every file in images_dir, which is
    only the same set as the known-pose model when --limit_frames is unset.
    With it set, the database ends up holding more frames than the model, and
    COLMAP 4.x's point_triangulator aborts merging them
    ("Check failed: existing_frame.DataIds() == frame.DataIds()"). Constraining
    extraction to exactly the model's images keeps the two consistent."""
    path = os.path.join(scene_dir, "image_list.txt")
    with open(path, "w") as f:
        f.write("\n".join(image_names) + "\n")
    return path


def run_native_cli(colmap_bin, images_dir, model_dir, workspace_dir,
                    depth_min_m, depth_max_m, intrinsics, image_names):
    scene_dir = os.path.dirname(model_dir)
    bin_model_dir = model_dir + "_bin"
    triangulated_dir = model_dir + "_triangulated"
    db_path = os.path.join(scene_dir, "colmap.db")
    _clean_stale_outputs([bin_model_dir, triangulated_dir, workspace_dir, db_path])
    image_list_path = _write_image_list(scene_dir, image_names)

    # see build_known_pose_model()'s points3D.txt comment: image_undistorter
    # needs real point visibility to build a stereo-matching neighbor list per
    # image, so triangulate real 2D-feature-match points before it runs.
    # point_triangulator keeps the poses above fixed -- it only adds points.
    # feature_extractor defaults to SIMPLE_RADIAL with auto-estimated focal
    # length, which creates a *second*, conflicting camera_id=1 in colmap.db
    # next to our known-pose model's PINHOLE camera_id=1 -- point_triangulator
    # then hard-fails merging the two ("existing_camera.model_id ==
    # camera.model_id" check, kPinhole vs. kSimpleRadial). Force the same
    # model/params/single-camera setup instead of
    # letting it guess.
    cam_params = f"{intrinsics['fx']},{intrinsics['fy']},{intrinsics['cx']},{intrinsics['cy']}"
    subprocess.run([colmap_bin, "feature_extractor",
                     "--database_path", db_path, "--image_path", images_dir,
                     "--image_list_path", image_list_path,
                     "--ImageReader.camera_model", "PINHOLE",
                     "--ImageReader.single_camera", "1",
                     "--ImageReader.camera_params", cam_params], check=True)

    # must happen after feature_extractor (which is what creates the database's
    # image ids) and before the model is converted/triangulated against it
    sync_model_ids_to_db(model_dir, db_path)
    os.makedirs(bin_model_dir, exist_ok=True)
    subprocess.run([colmap_bin, "model_converter",
                     "--input_path", model_dir, "--output_path", bin_model_dir,
                     "--output_type", "BIN"], check=True)

    # sequential_matcher assumes file-order adjacency ~= spatial adjacency --
    # true here, since T&T's image sets are extracted frames from one
    # continuous walk-around video (camera translation varies smoothly
    # between consecutive frame indices in every scene's .log).
    # Much cheaper than exhaustive_matcher's O(n^2) pairs on 1000+-image scenes.
    subprocess.run([colmap_bin, "sequential_matcher",
                     "--database_path", db_path], check=True)

    os.makedirs(triangulated_dir, exist_ok=True)
    subprocess.run([colmap_bin, "point_triangulator",
                     "--database_path", db_path, "--image_path", images_dir,
                     "--input_path", bin_model_dir, "--output_path", triangulated_dir], check=True)

    subprocess.run([colmap_bin, "image_undistorter",
                     "--image_path", images_dir, "--input_path", triangulated_dir,
                     "--output_path", workspace_dir, "--output_type", "COLMAP"], check=True)

    subprocess.run([colmap_bin, "patch_match_stereo",
                     "--workspace_path", workspace_dir,
                     "--PatchMatchStereo.depth_min", str(depth_min_m),
                     "--PatchMatchStereo.depth_max", str(depth_max_m),
                     "--PatchMatchStereo.geom_consistency", "true"], check=True)


def run_pycolmap(images_dir, model_dir, workspace_dir, depth_min_m, depth_max_m,
                  intrinsics, image_names):
    import pycolmap

    scene_dir = os.path.dirname(model_dir)
    bin_model_dir = model_dir + "_bin"
    triangulated_dir = model_dir + "_triangulated"
    db_path = os.path.join(scene_dir, "colmap.db")
    _clean_stale_outputs([bin_model_dir, triangulated_dir, workspace_dir, db_path])

    os.makedirs(bin_model_dir, exist_ok=True)
    reconstruction = pycolmap.Reconstruction(model_dir)
    reconstruction.write(bin_model_dir)

    try:
        # mirrors run_native_cli()'s step sequence -- see its comments for why
        # feature matching + triangulation has to happen before undistortion,
        # and why the camera model/params must match model_dir's PINHOLE
        # camera exactly (avoids a camera_id=1 model-type conflict in
        # point_triangulator's merge of the database against model_dir)
        cam_params = [intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]]
        # image_list keeps the database's image set identical to the known-pose
        # model's -- see _write_image_list() for the failure when it doesn't.
        pycolmap.extract_features(
            database_path=db_path, image_path=images_dir,
            camera_model="PINHOLE", camera_params=cam_params,
            image_list=list(image_names),
            reader_options=pycolmap.ImageReaderOptions(single_camera=True))
        pycolmap.match_sequential(database_path=db_path)
        os.makedirs(triangulated_dir, exist_ok=True)
        pycolmap.triangulate_points(reconstruction, db_path, images_dir, triangulated_dir)

        pycolmap.undistort_images(output_path=workspace_dir, input_path=triangulated_dir,
                                   image_path=images_dir)
        pycolmap.patch_match_stereo(workspace_dir,
                                     options=pycolmap.PatchMatchOptions(
                                         depth_min=depth_min_m, depth_max=depth_max_m,
                                         geom_consistency=True))
    except AttributeError as e:
        sys.exit(f"pycolmap's API doesn't match what this script expects ({e}). "
                 f"Inspect the installed version's API (extract_features / "
                 f"match_sequential / triangulate_points / undistort_images / "
                 f"patch_match_stereo) and adjust run_pycolmap(), or force the "
                 f"native CLI backend by editing {MVS_BACKEND_MARKER}.")


# --------------------------------------------------------------------------
# reading COLMAP's dense depth maps and back-projecting to per-frame points
# --------------------------------------------------------------------------

def read_colmap_depth_map(path):
    """COLMAP's dense depth-map binary format: a '&'-delimited text header
    (width&height&channels&) followed by a raw float32 array in Fortran order.
    Matches scripts/python/read_write_dense.py in the COLMAP source."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fid:
        header = b""
        amp_count = 0
        while amp_count < 3:
            byte = fid.read(1)
            if byte == b"":
                return None
            header += byte
            if byte == b"&":
                amp_count += 1
        width, height, channels = (int(x) for x in header.decode().rstrip("&").split("&"))
        array = np.fromfile(fid, np.float32)
    if array.size != width * height * channels:
        return None
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def backproject_depth_to_points(depth, intrinsics, max_depth_m, min_depth_m=0.05):
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(depth) & (depth > min_depth_m) & (depth < max_depth_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    d = depth[valid].astype(np.float64)
    u = us[valid].astype(np.float64)
    v = vs[valid].astype(np.float64)
    x = (u - intrinsics["cx"]) * d / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * d / intrinsics["fy"]
    return np.stack([x, y, d], axis=1)  # camera-local frame, +Z forward (OpenCV convention)


# --------------------------------------------------------------------------
# post-hoc sanity check against T&T's own GT point cloud
# --------------------------------------------------------------------------

def validate_against_gt(gt_ply_path, points_local_list, poses_gt_frame, sample_frames=5,
                         sample_points_per_frame=2000):
    gt = o3d.io.read_point_cloud(gt_ply_path)
    if len(gt.points) == 0:
        print(f"WARNING: {gt_ply_path} has no points, skipping validation")
        return
    gt_down = gt.voxel_down_sample(0.02)
    gt_tree = o3d.geometry.KDTreeFlann(gt_down)

    idxs = np.linspace(0, len(points_local_list) - 1, num=min(sample_frames, len(points_local_list)),
                        dtype=int)
    all_dists = []
    for i in idxs:
        pts = points_local_list[i]
        if len(pts) == 0:
            continue
        sub = pts[np.random.choice(len(pts), size=min(sample_points_per_frame, len(pts)),
                                    replace=False)]
        homog = np.concatenate([sub, np.ones((len(sub), 1))], axis=1)
        world = (poses_gt_frame[i] @ homog.T).T[:, :3]
        for p in world:
            _, idx1, dist2 = gt_tree.search_knn_vector_3d(p, 1)
            all_dists.append(np.sqrt(dist2[0]))

    all_dists = np.array(all_dists)
    if len(all_dists) == 0:
        print("WARNING: no points available for validation")
        return
    print(f"validation vs {gt_ply_path}: mean_dist={all_dists.mean():.3f}m "
          f"median_dist={np.median(all_dists):.3f}m "
          f"frac_within_20cm={np.mean(all_dists < 0.20):.1%} "
          f"(sampled {len(all_dists)} pts from {len(idxs)} frames)")
    if all_dists.mean() > 1.0:
        print("mean distance is > 1m -- alignment is probably wrong. Common culprits: "
              "trans.txt applied on the wrong side (try np.linalg.inv(trans) @ pose), "
              "or the .log poses being world-to-camera instead of camera-to-world "
              "for this particular scene's log.")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENE_DISPLAY_NAME))
    ap.add_argument("--data_root", default="data/tanksandtemples",
                     help="root written by dataset/download_tanks_and_temples.py")
    ap.add_argument("--output_root", default="",
                     help="default: <data_root>/<scene>_kitti_format")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="cap the number of views by UNIFORM subsampling across "
                         "the whole sequence (preserves coverage, reduces "
                         "density). Prefer this over --limit_frames for any run "
                         "whose output you intend to use.")
    ap.add_argument("--limit_frames", type=int, default=0,
                     help="process only the first N frames (for the small-subset "
                          "validation pass before committing to the full scene)")
    ap.add_argument("--max_depth_m", type=float, default=50.0)
    ap.add_argument("--depth_min_m", type=float, default=0.1,
                     help="COLMAP patch-match search range lower bound -- tighten to "
                          "the scene's actual extent (see the <Scene>.json crop file) "
                          "for faster/more accurate dense stereo")
    ap.add_argument("--depth_max_m", type=float, default=50.0)
    ap.add_argument("--validate_against_gt", action="store_true", default=True)
    ap.add_argument("--no_validate_against_gt", dest="validate_against_gt", action="store_false")
    ap.add_argument("--fx", type=float, default=0.0,
                     help="override the focal length in pixels (fx=fy), bypassing "
                          "EXIF and T&T's generic 0.7*W guess. The camera model is "
                          "used for dense stereo, not just back-projection, so this "
                          "has to be right BEFORE the MVS run -- see "
                          "load_or_default_intrinsics()")
    ap.add_argument("--reuse_mvs", action="store_true",
                     help="skip feature extraction/triangulation/dense stereo and "
                          "re-derive the point clouds from the depth maps already in "
                          "<scene>/mvs_workspace. Use this to re-run the cheap "
                          "back-projection/crop stage with different settings without "
                          "paying for dense MVS again (it dominates the runtime).")
    ap.add_argument("--crop_to_gt_volume", action="store_true", default=True,
                     help="restrict output to T&T's own <Scene>.json crop volume "
                          "(the region their evaluation toolkit scores), plus "
                          "--crop_margin_m. Without this the dense-MVS output spans "
                          "the whole scene -- for Ignatius ~43x47x13 m of courtyard "
                          "around a 2.5 m statue, which inflates the octree and makes "
                          "the marching cubes grid unallocatable at a fine mc_res_m.")
    ap.add_argument("--no_crop_to_gt_volume", dest="crop_to_gt_volume", action="store_false")
    ap.add_argument("--crop_margin_m", type=float, default=0.5,
                     help="dilate the crop volume by this much, so the SDF is still "
                          "supervised across the whole scored region instead of ending "
                          "exactly at its boundary")
    args = ap.parse_args()

    display_name = SCENE_DISPLAY_NAME[args.scene]
    scene_dir = os.path.join(args.data_root, args.scene)
    images_dir = os.path.join(scene_dir, "images")
    gt_dir = os.path.join(scene_dir, "gt")
    output_root = args.output_root or os.path.join(args.data_root, f"{args.scene}_kitti_format")

    log_path = os.path.join(gt_dir, f"{display_name}_COLMAP_SfM.log")
    trans_path = os.path.join(gt_dir, f"{display_name}_trans.txt")
    gt_ply_path = os.path.join(gt_dir, f"{display_name}.ply")
    for p in (log_path, trans_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p} -- run dataset/download_tanks_and_temples.py first")

    image_names = list_images(images_dir)
    poses_cam_to_world = load_poses(log_path, with_head=True)
    if len(poses_cam_to_world) != len(image_names):
        sys.exit(f"{len(poses_cam_to_world)} poses in {log_path} but {len(image_names)} "
                 f"images in {images_dir} -- these should correspond 1:1 by T&T's own "
                 f"construction; something is off (wrong scene? partial download?)")

    if args.limit_frames:
        # first-N, for cheap smoke tests only -- it truncates the trajectory, so
        # on a walkaround capture it reconstructs one side of the object and
        # nothing else. Use --max_frames for a real reduced-cost run.
        image_names = image_names[:args.limit_frames]
        poses_cam_to_world = poses_cam_to_world[:args.limit_frames]

    if args.max_frames and len(image_names) > args.max_frames:
        # UNIFORM subsample across the whole sequence, so coverage is preserved
        # and only view density drops. What this costs is the stereo baseline
        # between neighbouring selected views, which is what patch-match needs
        # to stay small relative to the camera-to-scene distance. Measured on
        # the T&T trajectories (metric, after folding in trans.txt's scale):
        # barn 410->250 gives baseline/distance 0.042 and courthouse 1106->250
        # gives 0.037, both at or below truck's native 0.048 -- and truck
        # converts cleanly at full density. So triangulation geometry stays in
        # a regime known to work here; the real cost is redundancy for the
        # geometric-consistency filter (filter_min_num_consistent: 2), i.e.
        # expect completeness loss on thin/occluded structure rather than
        # depth error. Re-measure with the same calculation before trusting
        # this on a scene with a different capture pattern.
        idx = np.linspace(0, len(image_names) - 1, args.max_frames).round().astype(int)
        idx = np.unique(idx)
        image_names = [image_names[i] for i in idx]
        poses_cam_to_world = [poses_cam_to_world[i] for i in idx]
        print(f"[max_frames] uniformly subsampled to {len(image_names)} views "
              f"(stride ~{(len(idx) and (idx[-1] - idx[0]) / max(1, len(idx) - 1)):.2f})")

    trans = load_transform(trans_path)
    # fold trans's uniform scale into the points rather than the poses, so
    # poses.txt is rigid and every distance downstream is in true metres
    sfm_to_m, _ = decompose_similarity(trans)
    # p_world = (s*Rt)(Rc p + tc) + tt = (Rt Rc)(s p) + (s*Rt*tc + tt): the
    # translation block of trans @ pose is already correct, only the rotation
    # block carries the scale, and the points take it up instead.
    poses_gt_frame = [trans @ P for P in poses_cam_to_world]
    for P in poses_gt_frame:
        P[:3, :3] /= sfm_to_m
    print(f"trans.txt scale (SfM unit -> m): {sfm_to_m:.6f} "
          f"-- folded into the points, poses.txt is now rigid")

    crop = None
    if args.crop_to_gt_volume:
        crop_json = os.path.join(gt_dir, f"{display_name}.json")
        if not os.path.exists(crop_json):
            sys.exit(f"missing {crop_json} (needed for --crop_to_gt_volume; pass "
                     f"--no_crop_to_gt_volume to convert the whole scene instead)")
        crop = CropVolume(crop_json, margin_m=args.crop_margin_m)
        print(crop.describe())

    W, H = read_image_size(os.path.join(images_dir, image_names[0]))
    intrinsics = load_or_default_intrinsics(
        gt_dir, display_name, W, H,
        sample_image=os.path.join(images_dir, image_names[0]),
        fx_override=args.fx)

    backend, backend_arg = read_mvs_backend()
    print(f"MVS backend: {backend}" + (f" ({backend_arg})" if backend_arg else ""))

    workspace_dir = os.path.join(scene_dir, "mvs_workspace")
    model_dir = os.path.join(scene_dir, "known_pose_model")

    if args.reuse_mvs:
        depth_dir = os.path.join(workspace_dir, "stereo", "depth_maps")
        if not os.path.isdir(depth_dir):
            sys.exit(f"--reuse_mvs given but {depth_dir} does not exist -- "
                     "run without it once to produce the depth maps")
        n_depth = len([f for f in os.listdir(depth_dir) if f.endswith(".geometric.bin")])
        print(f"--reuse_mvs: skipping dense stereo, using {n_depth} existing "
              f"geometric depth maps in {depth_dir}")
    else:
        build_known_pose_model(model_dir, image_names, poses_cam_to_world, intrinsics, W, H)
        print(f"running dense MVS on {len(image_names)} images (this needs a GPU)...")
        if backend == "native_colmap":
            run_native_cli(backend_arg, images_dir, model_dir, workspace_dir,
                            args.depth_min_m, args.depth_max_m, intrinsics,
                            image_names)
        else:
            run_pycolmap(images_dir, model_dir, workspace_dir,
                          args.depth_min_m, args.depth_max_m, intrinsics,
                          image_names)

    ply_out_dir = os.path.join(output_root, "tandt_ply")
    # wipe first: frames are renumbered contiguously from whatever survives, and
    # the loader pairs scans to poses purely by index. A re-run that keeps fewer
    # frames than the last one would otherwise leave stale .ply files behind and
    # silently pair them against the wrong poses.
    if os.path.isdir(ply_out_dir):
        shutil.rmtree(ply_out_dir)
    os.makedirs(ply_out_dir, exist_ok=True)

    points_local_list = []
    kept_poses = []
    frame_count = 0
    n_before = n_after = 0
    for name, pose in zip(image_names, poses_gt_frame):
        depth_path = os.path.join(workspace_dir, "stereo", "depth_maps", f"{name}.geometric.bin")
        depth = read_colmap_depth_map(depth_path)
        if depth is None:
            print(f"  [skip] no depth map for {name}")
            continue
        pts = backproject_depth_to_points(depth, intrinsics, args.max_depth_m)
        if len(pts) == 0:
            print(f"  [skip] no valid depth for {name}")
            continue

        # SfM units -> metres, so poses.txt stays rigid (see decompose_similarity)
        pts = pts * sfm_to_m
        n_before += len(pts)

        if crop is not None:
            world = pts @ pose[:3, :3].T + pose[:3, 3]
            pts = pts[crop.contains(world)]
            if len(pts) == 0:
                print(f"  [skip] no points inside the crop volume for {name}")
                continue
        n_after += len(pts)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(os.path.join(ply_out_dir, f"{frame_count:06d}.ply"), pcd)

        points_local_list.append(pts)
        kept_poses.append(pose)
        frame_count += 1

    if frame_count == 0:
        sys.exit("every frame was skipped -- dense MVS produced no usable depth maps"
                 + (" that fall inside the crop volume (is --crop_margin_m too tight, "
                    "or the alignment wrong?)" if crop is not None else ""))
    if crop is not None:
        print(f"crop volume kept {n_after}/{n_before} points "
              f"({n_after/max(n_before,1):.2%}) across {frame_count} frames")

    write_poses_kitti_format(kept_poses, os.path.join(output_root, "poses.txt"))
    print(f"wrote {frame_count}/{len(image_names)} frames to {ply_out_dir} "
          f"and {os.path.join(output_root, 'poses.txt')}")

    if args.validate_against_gt:
        if os.path.exists(gt_ply_path):
            validate_against_gt(gt_ply_path, points_local_list, kept_poses)
        else:
            print(f"WARNING: {gt_ply_path} not found, skipping validation")


if __name__ == "__main__":
    main()
