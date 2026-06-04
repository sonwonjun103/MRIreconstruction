"""Build train / val / test reconstruction datasets from raw fastMRI files.

For each tissue (knee, brain) this creates subject-disjoint splits and writes one
HDF5 file per slice under::

    /mnt/d/research/MRRecon/data/{tissue}/{train,val,test}/{subject}_{slice:03d}.h5

Each slice .h5 holds:
    kspace   (C, H, W) complex64   -- multi-coil k-space
    sens_map (C, H, W) complex64   -- ESPIRiT sensitivity maps (BART ecalib)
    rss      (h, w)   float32      -- fastMRI reconstruction_rss (reference/preview)

Design (per the spec):
  * splits per tissue: train=500, val=100, test=200 slices (the zero-shot set == test)
  * subjects are disjoint across splits (no leakage)
  * per subject we take the most CENTRAL slices (middle of the volume / head)
  * sensitivity maps via ESPIRiT (BART `ecalib -d0 -m1 -r24`)

Why HDF5 (not PNG): k-space and sensitivity maps are complex-valued, multi-coil
data; PNG would discard phase / coils. A magnitude PNG preview can optionally be
written alongside (``--save_png``) purely for visual inspection.

Usage:
    python MakeDataset.py --tissue both --slices_per_subject 16 --save_png
"""

import os
import sys
import glob
import shutil
import argparse

import h5py as h5
import numpy as np
import tqdm

# --------------------------------------------------------------------------- #
# config / paths
# --------------------------------------------------------------------------- #
BART_PATH = "/home/sonwonjun/research/MRRecon/Paper/bart"
RAW_ROOT = "/mnt/d/research/MRRecon"
SAVE_ROOT = "/mnt/d/research/MRRecon/data"

RAW_DIR = {"knee": "knee_multicoil_train", "brain": "brain_multicoil_train"}
# keep only the dominant geometry per tissue so every split has consistent shape
KEEP_SHAPE = {"knee": (15, 640, 368), "brain": (16, 640, 320)}  # (C, H, W)
SPLIT_COUNTS = {"train": 500, "val": 100, "test": 200}

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
# helpers
# --------------------------------------------------------------------------- #
def central_slice_indices(n_slices, k):
    """Indices of the ``k`` most central slices (middle of the volume / head)."""
    mid = n_slices // 2
    start = max(mid - k // 2, 0)
    end = min(start + k, n_slices)
    return list(range(start, end))


def valid_subjects(tissue):
    """Raw files whose k-space matches the kept geometry for this tissue."""
    pattern = os.path.join(RAW_ROOT, RAW_DIR[tissue], "*.h5")
    want = KEEP_SHAPE[tissue]
    files = []
    for fp in sorted(glob.glob(pattern)):
        try:
            with h5.File(fp, "r") as f:
                if "kspace" not in f:
                    continue
                if tuple(f["kspace"].shape[1:]) == want:
                    files.append(fp)
        except Exception:
            continue
    return files


def save_preview_png(path, rss_slice):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = np.abs(rss_slice)
        plt.imsave(path, img, cmap="gray", vmax=0.6 * img.max())
    except Exception as e:
        print(f"  (png preview skipped: {e})")


def save_slice(tissue, split, subject, sidx, kspace_slice, rss_slice, save_png):
    out_dir = os.path.join(SAVE_ROOT, tissue, split)
    os.makedirs(out_dir, exist_ok=True)
    sens = espirit_sens_maps(kspace_slice).astype(np.complex64)

    h5_path = os.path.join(out_dir, f"{subject}_{sidx:03d}.h5")
    with h5.File(h5_path, "w") as f:
        f.create_dataset("kspace", data=kspace_slice.astype(np.complex64))
        f.create_dataset("sens_map", data=sens)
        if rss_slice is not None:
            f.create_dataset("rss", data=rss_slice.astype(np.float32))

    if save_png and rss_slice is not None:
        png_dir = os.path.join(out_dir, "preview")
        os.makedirs(png_dir, exist_ok=True)
        save_preview_png(os.path.join(png_dir, f"{subject}_{sidx:03d}.png"), rss_slice)


# --------------------------------------------------------------------------- #
# build one tissue
# --------------------------------------------------------------------------- #
def build_tissue(tissue, slices_per_subject, seed, save_png):
    subjects = valid_subjects(tissue)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    print(f"\n=== {tissue}: {len(subjects)} usable subjects "
          f"(shape {KEEP_SHAPE[tissue]}), need "
          f"{sum(SPLIT_COUNTS.values())} slices ===")

    subj_iter = iter(subjects)
    manifest = {}

    for split in ("train", "val", "test"):
        target = SPLIT_COUNTS[split]
        collected, used_subjects = 0, []
        pbar = tqdm.tqdm(total=target, desc=f"{tissue}/{split}")
        while collected < target:
            try:
                fp = next(subj_iter)
            except StopIteration:
                print(f"  WARNING: ran out of subjects for {tissue}/{split} "
                      f"({collected}/{target})")
                break
            subject = os.path.splitext(os.path.basename(fp))[0]
            with h5.File(fp, "r") as f:
                kspace = f["kspace"][:]
                rss = f["reconstruction_rss"][:] if "reconstruction_rss" in f else None
            used_subjects.append(subject)
            for s in central_slice_indices(kspace.shape[0], slices_per_subject):
                if collected >= target:
                    break
                save_slice(tissue, split, subject, s, kspace[s],
                           rss[s] if rss is not None else None, save_png)
                collected += 1
                pbar.update(1)
        pbar.close()
        manifest[split] = {"slices": collected, "subjects": used_subjects}
        print(f"  {tissue}/{split}: {collected} slices from "
              f"{len(used_subjects)} subjects")

    # write a manifest so splits / subject membership are reproducible & auditable
    import json
    with open(os.path.join(SAVE_ROOT, tissue, "manifest.json"), "w") as f:
        json.dump({"tissue": tissue, "shape": list(KEEP_SHAPE[tissue]),
                   "slices_per_subject": slices_per_subject, "seed": seed,
                   "splits": manifest}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Build fastMRI train/val/test datasets")
    ap.add_argument("--tissue", default="both", choices=["knee", "brain", "both"])
    ap.add_argument("--slices_per_subject", type=int, default=16,
                    help="number of central slices taken from each subject")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save_png", action="store_true",
                    help="also write a magnitude PNG preview per slice")
    args = ap.parse_args()

    tissues = ["knee", "brain"] if args.tissue == "both" else [args.tissue]
    for tissue in tissues:
        build_tissue(tissue, args.slices_per_subject, args.seed, args.save_png)
    print("\nDone. Datasets under", SAVE_ROOT)


if __name__ == "__main__":
    main()
