"""Build train / val / test reconstruction datasets from raw fastMRI files.

Each split comes from its OWN fastMRI source (no subject-splitting):
    train <- the existing *_multicoil_train folder
    val   <- *_multicoil_val   (auto-extracted from the .tar.xz if needed)
    test  <- *_multicoil_test  (auto-extracted from the .tar.xz if needed)

Output: one HDF5 file per slice, with tissue folders at the data root::

    /mnt/d/research/MRRecon/{knee,brain}/{train,val,test}/{volume}_{slice:03d}.h5
        kspace   (C, H, W) complex64   -- multi-coil k-space
        sens_map (C, H, W) complex64   -- ESPIRiT sensitivity maps (BART ecalib)
        rss      (h, w)   float32      -- fastMRI reconstruction_rss (if present)

Notes
-----
* ``--slices_per_vol 0`` (default) keeps ALL slices per volume; ``N>0`` keeps the
  central N slices.
* The fastMRI *test* set is the challenge set: its volumes usually have NO
  ``reconstruction_rss`` (no ground truth). Those slices are still written (kspace
  + ESPIRiT sens from the ACS), but they carry no ``rss``, so SSIM/PSNR/NMSE can
  not be computed against them -- only zero-filled SENSE metrics / inference.
  The build prints, per split, how many volumes had ground truth.
* Sources that are not present yet (still downloading) are skipped with a note;
  re-run when the download / extraction finishes.

Usage:
    python MakeDataset.py --tissue both                 # all slices, all splits
    python MakeDataset.py --tissue knee --slices_per_vol 10 --save_png
"""

import os
import sys
import glob
import json
import shutil
import argparse
import subprocess

import h5py as h5
import numpy as np
import tqdm

# --------------------------------------------------------------------------- #
# config / paths
# --------------------------------------------------------------------------- #
BART_PATH = "/home/sonwonjun/research/MRRecon/Paper/bart"
RAW_ROOT = "/mnt/d/research/MRRecon"
SAVE_ROOT = "/mnt/d/research/MRRecon"          # tissue folders live directly here

# Candidate source directory names per (tissue, split). The first that contains
# .h5 files is used; otherwise the matching ``{name}.tar.xz`` is extracted.
SOURCES = {
    "knee": {
        "train": ["knee_multicoil_train", "knee_multicoil_train_batch"],
        "val":   ["knee_multicoil_val", "multicoil_val"],
        "test":  ["knee_multicoil_test", "multicoil_test"],
    },
    "brain": {
        "train": ["brain_multicoil_train", "brain_multicoil_train_batch"],
        "val":   ["brain_multicoil_val", "multicoil_val"],
        "test":  ["brain_multicoil_test", "multicoil_test"],
    },
}

# keep a consistent (C, H, W) per tissue; None matches any value on that axis.
KEEP_SHAPE = {"knee": (15, 640, 368), "brain": (None, 640, 320)}

# --------------------------------------------------------------------------- #
# BART ESPIRiT
# --------------------------------------------------------------------------- #
os.environ["TOOLBOX_PATH"] = BART_PATH
os.environ["PATH"] = BART_PATH + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, os.path.join(BART_PATH, "python"))
assert shutil.which("bart"), f"bart not found; check BART_PATH={BART_PATH}"
from bart import bart  # noqa: E402


def espirit_sens_maps(kspace_slice):
    """ESPIRiT sensitivity maps for one slice. (C,H,W) complex -> (C,H,W) complex."""
    ks = kspace_slice.transpose(1, 2, 0)[None, ...]        # (1, H, W, C)
    smap = bart(1, "ecalib -d0 -m1 -r24", ks)              # (1, H, W, C, 1)
    return smap.transpose(3, 1, 2, 0).squeeze(-1)          # (C, H, W)


# --------------------------------------------------------------------------- #
# source resolution (extract tar.xz on demand)
# --------------------------------------------------------------------------- #
def _h5_in(d):
    return sorted(glob.glob(os.path.join(d, "*.h5"))) if os.path.isdir(d) else []


def resolve_source(tissue, split):
    """Return the list of raw .h5 volumes for (tissue, split), extracting the
    matching .tar.xz if the directory is not present yet. Returns [] if the
    source is unavailable (e.g. still downloading)."""
    for name in SOURCES[tissue][split]:
        files = _h5_in(os.path.join(RAW_ROOT, name))
        if files:
            return files, name
    # try extracting a candidate tar.xz
    for name in SOURCES[tissue][split]:
        tar = os.path.join(RAW_ROOT, name + ".tar.xz")
        if os.path.exists(tar):
            print(f"  extracting {os.path.basename(tar)} (this can take a while) ...")
            subprocess.run(["tar", "-xJf", tar, "-C", RAW_ROOT], check=True)
            # the tar may create {name}/ or multicoil_{split}/ -- search candidates
            for cand in SOURCES[tissue][split] + [f"multicoil_{split}"]:
                files = _h5_in(os.path.join(RAW_ROOT, cand))
                if files:
                    return files, cand
    return [], None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def central_indices(n_slices, k):
    """All indices if k<=0 or k>=n, else the central k slices."""
    if k <= 0 or k >= n_slices:
        return list(range(n_slices))
    start = (n_slices - k) // 2
    return list(range(start, start + k))


def shape_matches(want, shape):
    return all(w is None or w == s for w, s in zip(want, shape))


def save_preview_png(path, rss_slice):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = np.abs(rss_slice)
        plt.imsave(path, img, cmap="gray", vmax=0.6 * img.max())
    except Exception as e:
        print(f"  (png preview skipped: {e})")


def save_slice(out_dir, volume, sidx, kspace_slice, rss_slice, save_png):
    sens = espirit_sens_maps(kspace_slice).astype(np.complex64)
    with h5.File(os.path.join(out_dir, f"{volume}_{sidx:03d}.h5"), "w") as f:
        f.create_dataset("kspace", data=kspace_slice.astype(np.complex64))
        f.create_dataset("sens_map", data=sens)
        if rss_slice is not None:
            f.create_dataset("rss", data=rss_slice.astype(np.float32))
    if save_png and rss_slice is not None:
        png_dir = os.path.join(out_dir, "preview")
        os.makedirs(png_dir, exist_ok=True)
        save_preview_png(os.path.join(png_dir, f"{volume}_{sidx:03d}.png"), rss_slice)


# --------------------------------------------------------------------------- #
# build one (tissue, split)
# --------------------------------------------------------------------------- #
def build_split(tissue, split, slices_per_vol, save_png, limit=0):
    files, src = resolve_source(tissue, split)
    out_dir = os.path.join(SAVE_ROOT, tissue, split)
    if not files:
        print(f"  [{tissue}/{split}] source not available yet -- skipped "
              f"(looked for {SOURCES[tissue][split]} + .tar.xz). Re-run after download.")
        return None

    want = KEEP_SHAPE[tissue]
    vols = []
    for fp in files:
        try:
            with h5.File(fp, "r") as f:
                if "kspace" in f and shape_matches(want, tuple(f["kspace"].shape[1:])):
                    vols.append(fp)
        except Exception:
            continue
    if limit:
        vols = vols[:limit]

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  [{tissue}/{split}] source '{src}': {len(vols)}/{len(files)} volumes "
          f"match shape {want}")
    count, with_gt, used = 0, 0, []
    for fp in tqdm.tqdm(vols, desc=f"{tissue}/{split}"):
        volume = os.path.splitext(os.path.basename(fp))[0]
        with h5.File(fp, "r") as f:
            kspace = f["kspace"][:]
            rss = f["reconstruction_rss"][:] if "reconstruction_rss" in f else None
        if rss is not None:
            with_gt += 1
        for s in central_indices(kspace.shape[0], slices_per_vol):
            save_slice(out_dir, volume, s, kspace[s],
                       rss[s] if rss is not None else None, save_png)
            count += 1
        used.append(volume)

    note = "" if with_gt == len(vols) else f"  (WARNING: {len(vols) - with_gt} volumes have NO reconstruction_rss / GT)"
    print(f"  -> {out_dir}: {count} slices from {len(used)} volumes; "
          f"{with_gt}/{len(vols)} with GT{note}")
    return {"split": split, "source": src, "slices": count,
            "volumes": len(used), "volumes_with_gt": with_gt}


def main():
    ap = argparse.ArgumentParser(description="Build fastMRI train/val/test datasets")
    ap.add_argument("--tissue", default="both", choices=["knee", "brain", "both"])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--slices_per_vol", type=int, default=0,
                    help="0 = ALL slices per volume (default); N>0 = central N slices")
    ap.add_argument("--save_png", action="store_true",
                    help="also write a magnitude PNG preview per slice")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N volumes per split (debug)")
    args = ap.parse_args()

    tissues = ["knee", "brain"] if args.tissue == "both" else [args.tissue]
    for tissue in tissues:
        print(f"\n=== {tissue} -> {os.path.join(SAVE_ROOT, tissue)}/ "
              f"(slices_per_vol={args.slices_per_vol or 'ALL'}) ===")
        manifest = {"tissue": tissue, "slices_per_vol": args.slices_per_vol,
                    "shape": list(KEEP_SHAPE[tissue]), "splits": {}}
        for split in args.splits:
            info = build_split(tissue, split, args.slices_per_vol, args.save_png, args.limit)
            if info:
                manifest["splits"][split] = info
        if manifest["splits"]:
            os.makedirs(os.path.join(SAVE_ROOT, tissue), exist_ok=True)
            with open(os.path.join(SAVE_ROOT, tissue, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
    print("\nDone. Datasets under", SAVE_ROOT)


if __name__ == "__main__":
    main()
