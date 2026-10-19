"""Downloads Tanks-and-Temples Training-scene data for the NIEO
COLMAP-derived synthetic-LiDAR pipeline (see dataset/tandt_to_kitti_format.py).

Per requested scene, fetches into data/tanksandtemples/<scene>/:
  - images/                     the official pre-sampled image set (a .zip, headless
                                 download via gdown; file IDs from the official
                                 downloader, github.com/isl-org/TanksAndTemples/
                                 python_toolbox/download_t2_dataset.py).
  - <Scene>.ply                 ground-truth laser-scan point cloud
  - <Scene>_trans.txt           alignment transform: T&T's own COLMAP_SfM trajectory -> GT frame
  - <Scene>.json                crop-box definition (the official benchmark's scene-extent bbox)
  - <Scene>_COLMAP_SfM.log      T&T's own published camera trajectory (TrajectoryIO .log format,
                                 same format Redwood's GT log uses -- parseable by
                                 dataset/rgbd_to_kitti_format.py's load_poses(with_head=True))
  - <Scene>_COLMAP.ply          T&T's own reference COLMAP reconstruction (not required by our
                                 pipeline, kept for sanity-checking against)
  - <Scene>_mapping_reference.txt

The last five files all live inside ONE Google Drive folder shared across every Training
scene (https://drive.google.com/open?id=1UoKPiUUsKa0AVHFOrnMRhc5hFngjkE-t) -- there is no
per-scene file ID for them, unlike the image-set zips. Folder downloads need `gdown`
(installed on demand into the current Python environment), which lists a public Drive
folder's contents via a method plain `requests` can't replicate. If gdown can't complete it
(rate limiting, folder too large, Drive changing its folder-listing behavior), this script
prints the manual browser-download fallback and moves on.

Deliberately does NOT fetch the raw 4K video for any scene: <Scene>_COLMAP_SfM.log's poses
were computed against the *image set*, not the video, so using the image set keeps
image-index <-> pose-index alignment exact without re-deriving frame correspondence.

Usage:
    python dataset/download_tanks_and_temples.py \
        --scenes ignatius,truck,barn,caterpillar,courthouse \
        --output_root data/tanksandtemples
"""
import argparse
import os
import subprocess
import sys
import zipfile

# scene key (as used on the CLI / throughout this pipeline) -> Google Drive file ID for
# that scene's pre-sampled Training-set image zip, from the official downloader's
# id_download_dict (github.com/isl-org/TanksAndTemples/python_toolbox/download_t2_dataset.py)
IMAGE_SET_FILE_IDS = {
    "ignatius": "0B-ePgl6HF260d0l0ZDNSZ3ZxREk",
    "truck": "0B-ePgl6HF260NEw3OGN4ckF0dnM",
    "barn": "0B-ePgl6HF260NzQySklGdXZyQzA",
    "caterpillar": "0B-ePgl6HF260b2JNbnZYYjczU2s",
    "courthouse": "0B-ePgl6HF260bHRNZTJnU1pWMVE",
}

# scene key -> display name used in T&T's own file naming (<Scene>.ply, <Scene>_trans.txt, ...)
SCENE_DISPLAY_NAME = {
    "ignatius": "Ignatius",
    "truck": "Truck",
    "barn": "Barn",
    "caterpillar": "Caterpillar",
    "courthouse": "Courthouse",
}

GT_BUNDLE_FOLDER_ID = "1UoKPiUUsKa0AVHFOrnMRhc5hFngjkE-t"
GT_BUNDLE_URL = f"https://drive.google.com/drive/folders/{GT_BUNDLE_FOLDER_ID}"

GT_FILE_SUFFIXES = [
    ".ply", "_trans.txt", ".json", "_COLMAP_SfM.log", "_COLMAP.ply", "_mapping_reference.txt",
]

PIP_CMD = [sys.executable, "-m", "pip"]


def download_file_from_google_drive(file_id, destination):
    """Headless download of a single public Google Drive file, via gdown.

    The confirm-token workaround used by the official (unmaintained) T&T
    downloader no longer works against Google's current "can't scan for
    viruses" interstitial page; gdown is maintained to track those changes."""
    import gdown
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    gdown.download(id=file_id, output=destination, quiet=False)


def fetch_image_set(scene, scene_dir):
    zip_path = os.path.join(scene_dir, "images.zip")
    images_dir = os.path.join(scene_dir, "images")
    if os.path.isdir(images_dir) and os.listdir(images_dir):
        print(f"[{scene}] images/ already populated, skipping download")
        return True

    print(f"[{scene}] downloading image set...")
    download_file_from_google_drive(IMAGE_SET_FILE_IDS[scene], zip_path)

    if not os.path.exists(zip_path) or not zipfile.is_zipfile(zip_path):
        print(f"[{scene}] ERROR: downloaded file is not a valid zip -- Google likely served "
              f"an HTML interstitial instead of the file, and gdown couldn't get past it either.")
        print(f"  Manual fallback: open https://drive.google.com/open?id="
              f"{IMAGE_SET_FILE_IDS[scene]} in a browser, download by hand, "
              f"and unzip into {images_dir}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

    os.makedirs(images_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(images_dir)
    os.remove(zip_path)

    # the zip may extract into a nested <Scene>/ subfolder rather than flat -- flatten it
    # so images_dir always directly contains the image files
    nested = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    if len(nested) == 1 and not any(f.lower().endswith((".jpg", ".png", ".jpeg"))
                                     for f in os.listdir(images_dir) if
                                     os.path.isfile(os.path.join(images_dir, f))):
        nested_dir = os.path.join(images_dir, nested[0])
        for f in os.listdir(nested_dir):
            os.rename(os.path.join(nested_dir, f), os.path.join(images_dir, f))
        os.rmdir(nested_dir)

    n_images = len([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"[{scene}] extracted {n_images} images into {images_dir}")
    return True


def ensure_gdown(auto_install):
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        pass
    if not auto_install:
        return False
    print("gdown not found in the current Python env, installing...")
    result = subprocess.run(PIP_CMD + ["install", "-q", "gdown"])
    if result.returncode != 0:
        return False
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_gt_bundle(bundle_dir, auto_install_gdown):
    """Downloads the single Drive folder shared by every Training scene's GT/pose/
    alignment/crop files into bundle_dir. Only attempted once regardless of how many
    scenes were requested -- the folder isn't scene-specific."""
    have_any = os.path.isdir(bundle_dir) and any(
        f.endswith((".ply", ".txt", ".json", ".log"))
        for _, _, files in os.walk(bundle_dir) for f in files)
    if have_any:
        print(f"GT bundle dir {bundle_dir} already has files, skipping folder download")
        return True

    if not ensure_gdown(auto_install_gdown):
        print("gdown unavailable -- cannot auto-download the GT/pose/alignment/crop bundle.")
        _print_manual_gt_instructions(bundle_dir)
        return False

    import gdown
    print(f"downloading GT bundle folder ({GT_BUNDLE_URL}) via gdown...")
    os.makedirs(bundle_dir, exist_ok=True)
    try:
        # use_cookies=True (gdown's default) lets it pick up ~/.cache/gdown/cookies.txt
        # if one exists -- relevant because T&T's own README says this folder needs a
        # Google account to list, unlike the per-scene image-set files above, so an
        # anonymous/headless request is expected to come back empty, not just flaky
        gdown.download_folder(url=GT_BUNDLE_URL, output=bundle_dir, quiet=False)
    except Exception as e:
        print(f"gdown folder download failed: {e}")
        _print_manual_gt_instructions(bundle_dir)
        return False

    if not any(f.endswith((".ply", ".txt", ".json", ".log"))
               for _, _, files in os.walk(bundle_dir) for f in files):
        print("gdown reported success but no expected files landed. Expected, per T&T's "
              "own README: this folder needs a Google account to list, so an anonymous "
              "request naturally sees zero files rather than failing loudly.")
        _print_manual_gt_instructions(bundle_dir)
        return False
    return True


def _print_manual_gt_instructions(bundle_dir):
    print(
        "Manual fallback:\n"
        f"  1. Open {GT_BUNDLE_URL} in a browser (Google account required for folder\n"
        "     download, per T&T's own README).\n"
        "  2. Download the whole folder (or just the files for the scenes you need --\n"
        "     <Scene>.ply, <Scene>_trans.txt, <Scene>.json, <Scene>_COLMAP_SfM.log,\n"
        "     <Scene>_COLMAP.ply, <Scene>_mapping_reference.txt).\n"
        f"  3. Place the files directly in {bundle_dir} (flat, no per-scene subfolder).\n"
        "  Alternative to a fully manual download: export your browser's Google cookies\n"
        "  to ~/.cache/gdown/cookies.txt (Netscape format, e.g. via a 'Get cookies.txt'\n"
        "  browser extension) while logged into an account that can see the folder, then\n"
        "  rerun this script -- gdown will use them automatically."
    )


def link_scene_gt_files(scene, scene_dir, bundle_dir):
    """Copies this scene's 6 files out of the shared bundle dir into its own scene
    folder, so config/data paths per scene don't need to know about the shared bundle."""
    display_name = SCENE_DISPLAY_NAME[scene]
    gt_dir = os.path.join(scene_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    missing = []
    for suffix in GT_FILE_SUFFIXES:
        fname = f"{display_name}{suffix}"
        src = os.path.join(bundle_dir, fname)
        # gdown.download_folder sometimes nests one level under the folder's own name
        if not os.path.exists(src):
            for root, _, files in os.walk(bundle_dir):
                if fname in files:
                    src = os.path.join(root, fname)
                    break
        dst = os.path.join(gt_dir, fname)
        if os.path.exists(src):
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        else:
            missing.append(fname)

    if missing:
        print(f"[{scene}] WARNING: missing from GT bundle: {missing}")
        return False
    print(f"[{scene}] GT files linked into {gt_dir}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True,
                     help="comma-separated scene keys, e.g. ignatius,truck,barn,caterpillar,courthouse")
    ap.add_argument("--output_root", default="data/tanksandtemples")
    ap.add_argument("--skip_gt_bundle", action="store_true",
                     help="only fetch image sets, skip the shared GT/pose/alignment/crop bundle")
    ap.add_argument("--no_auto_install_gdown", action="store_true",
                     help="don't pip install gdown automatically if missing")
    args = ap.parse_args()

    scenes = [s.strip().lower() for s in args.scenes.split(",") if s.strip()]
    unknown = [s for s in scenes if s not in IMAGE_SET_FILE_IDS]
    if unknown:
        sys.exit(f"unknown scene(s) {unknown}; known scenes: {sorted(IMAGE_SET_FILE_IDS)}")

    # both the per-scene image-set files and the shared GT bundle folder go through
    # gdown now (see download_file_from_google_drive's docstring for why) -- check
    # once upfront rather than letting each of the 5 scenes hit the same ImportError
    if not ensure_gdown(not args.no_auto_install_gdown):
        sys.exit("gdown is required (for both image-set and GT-bundle downloads) but "
                 "isn't available and couldn't be installed. Install it manually into "
                 f"the current Python env ({' '.join(PIP_CMD)} install gdown) and rerun.")

    os.makedirs(args.output_root, exist_ok=True)

    for scene in scenes:
        scene_dir = os.path.join(args.output_root, scene)
        os.makedirs(scene_dir, exist_ok=True)
        fetch_image_set(scene, scene_dir)

    if not args.skip_gt_bundle:
        bundle_dir = os.path.join(args.output_root, "gt_bundle")
        if fetch_gt_bundle(bundle_dir, not args.no_auto_install_gdown):
            for scene in scenes:
                link_scene_gt_files(scene, os.path.join(args.output_root, scene), bundle_dir)

    print("Done.")


if __name__ == "__main__":
    main()
