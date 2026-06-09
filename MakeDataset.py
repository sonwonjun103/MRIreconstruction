"""Build train / val / test reconstruction datasets from raw fastMRI files.

Split mapping (the fastMRI *test* set has no ground truth, so it is not used for
metrics; the GT-bearing fastMRI *val* set is split into our val + test):
    our train <- ALL of fastMRI *_multicoil_train  (maximise training data)
    our val   <- a volume-disjoint --val_holdout (default 0.1) of fastMRI *val*
                 -> model selection during training
    our test  <- the remaining ~90% of fastMRI *val*  (has reconstruction_rss/GT)
                 -> final metrics
    (fastMRI *_multicoil_test is ignored -- no GT; build it separately only for
     inference if ever needed.)
our val and test are disjoint subject subsets of the fastMRI val set; train is a
different fastMRI source, so there is no leakage anywhere.

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
# build a given list of volumes into our {tissue}/{out_split}
# --------------------------------------------------------------------------- #
def filter_shape(files, tissue):
    want = KEEP_SHAPE[tissue]
    vols = []
    for fp in files:
        try:
            with h5.File(fp, "r") as f:
                if "kspace" in f and shape_matches(want, tuple(f["kspace"].shape[1:])):
                    vols.append(fp)
        except Exception:
            continue
    return vols


def build_volumes(tissue, out_split, vols, slices_per_vol, save_png):
    out_dir = os.path.join(SAVE_ROOT, tissue, out_split)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    count, with_gt, used = 0, 0, []
    for fp in tqdm.tqdm(vols, desc=f"{tissue}/{out_split}"):
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

    note = "" if with_gt == len(vols) else \
        f"  (WARNING: {len(vols) - with_gt} volumes have NO reconstruction_rss / GT)"
    print(f"  -> {out_dir}: {count} slices from {len(used)} volumes; "
          f"{with_gt}/{len(vols)} with GT{note}")
    return {"slices": count, "volumes": len(used), "volumes_with_gt": with_gt}


def main():
    ap = argparse.ArgumentParser(description="Build fastMRI train/val/test datasets")
    ap.add_argument("--tissue", default="both", choices=["knee", "brain", "both"])
    ap.add_argument("--slices_per_vol", type=int, default=0,
                    help="0 = ALL slices per volume (default); N>0 = central N slices")
    ap.add_argument("--val_holdout", type=float, default=0.1,
                    help="fraction of the fastMRI VAL set held out (volume-disjoint) as "
                         "our validation set; the rest becomes our test set")
    ap.add_argument("--seed", type=int, default=1234, help="val/test holdout shuffle seed")
    ap.add_argument("--save_png", action="store_true",
                    help="also write a magnitude PNG preview per slice")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N volumes per built split (debug)")
    args = ap.parse_args()

    tissues = ["knee", "brain"] if args.tissue == "both" else [args.tissue]
    for tissue in tissues:
        print(f"\n=== {tissue} -> {os.path.join(SAVE_ROOT, tissue)}/ "
              f"(slices_per_vol={args.slices_per_vol or 'ALL'}, val_holdout={args.val_holdout}) ===")
        manifest = {"tissue": tissue, "slices_per_vol": args.slices_per_vol,
                    "val_holdout": args.val_holdout, "seed": args.seed,
                    "shape": list(KEEP_SHAPE[tissue]),
                    "mapping": "train <- ALL fastMRI train; val/test <- fastMRI val "
                               "(val_holdout split, both GT)",
                    "splits": {}}

        # our train  <-  ALL of fastMRI train
        train_files, src = resolve_source(tissue, "train")
        if train_files:
            train_vols = filter_shape(train_files, tissue)
            if args.limit:
                train_vols = train_vols[:args.limit]
            print(f"  [train src '{src}'] {len(train_vols)} volumes (all -> train)")
            manifest["splits"]["train"] = build_volumes(tissue, "train", train_vols,
                                                        args.slices_per_vol, args.save_png)
        else:
            print(f"  [train] fastMRI train source not found for {tissue} -- skipped")

        # our val + test  <-  fastMRI val, volume-disjoint holdout (both have GT)
        val_files, src = resolve_source(tissue, "val")
        if val_files:
            vols = filter_shape(val_files, tissue)
            np.random.default_rng(args.seed).shuffle(vols)
            n_val = max(1, int(round(len(vols) * args.val_holdout))) if len(vols) > 1 else 0
            our_val, our_test = vols[:n_val], vols[n_val:]
            if args.limit:
                our_val, our_test = our_val[:max(1, args.limit // 5)], our_test[:args.limit]
            print(f"  [val src '{src}' = fastMRI val] {len(vols)} vols -> "
                  f"val {len(our_val)} / test {len(our_test)}")
            manifest["splits"]["val"] = build_volumes(tissue, "val", our_val,
                                                     args.slices_per_vol, args.save_png)
            manifest["splits"]["test"] = build_volumes(tissue, "test", our_test,
                                                      args.slices_per_vol, args.save_png)
        else:
            print(f"  [val/test] fastMRI val source not available yet for {tissue} "
                  f"(still downloading?) -- skipped; re-run after download")

        if manifest["splits"]:
            os.makedirs(os.path.join(SAVE_ROOT, tissue), exist_ok=True)
            with open(os.path.join(SAVE_ROOT, tissue, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
    print("\nDone. Datasets under", SAVE_ROOT)


if __name__ == "__main__":
    main()
