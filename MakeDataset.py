"""Build train / val / test reconstruction datasets from raw fastMRI files.

Two dataset variants (choose with ``--mode``):
  * central : the 3 central slices per subject  -> {data_root}/{tissue}/
  * full    : ALL slices per subject            -> {data_root}/{tissue}_full/

One HDF5 file per slice::

    {data_root}/{tissue[_full]}/{train,val,test}/{subject}_{slice:03d}.h5
        kspace   (C, H, W) complex64   -- multi-coil k-space
        sens_map (C, H, W) complex64   -- ESPIRiT sensitivity maps (BART ecalib)
        rss      (h, w)   float32      -- fastMRI reconstruction_rss (reference/preview)

Splitting is **subject-level** (no leakage between adjacent slices):
  * knee  : all subjects split  val 10% / test 10% / train 80%
  * brain : grouped by MRI modality (file_brain_<MODALITY>_...), each modality
            split val 10% / test 10% (stratified). Files keep the modality in the
            filename, so training can filter by --modality.

The **same --seed gives the same subject split** for central and full, so the two
datasets are directly comparable.

Sensitivity maps via ESPIRiT (BART `ecalib -d0 -m1 -r24`). HDF5 (not PNG) because
k-space / sens maps are complex multi-coil; `--save_png` adds a magnitude preview.

Usage:
    python MakeDataset.py --tissue both --mode both --save_png
"""

import os
import sys
import glob
import json
import shutil
import argparse
from collections import defaultdict

import h5py as h5
import numpy as np
import tqdm

# --------------------------------------------------------------------------- #
# config / paths
# --------------------------------------------------------------------------- #
BART_PATH = "/home/sonwonjun/research/MRRecon/Paper/bart"
RAW_ROOT = "/mnt/d/research/MRRecon"
SAVE_ROOT = "/mnt/d/research/MRRecon/data"

# candidate raw subdirs per tissue (first one that contains .h5 files is used)
RAW_DIRS = {"knee": ["knee_multicoil_train", "knee_multicoil_train_batch"],
            "brain": ["brain_multicoil_train", "brain_multicoil_train_batch"]}
# keep a consistent (H, W) per tissue; None matches any value on that axis.
#   knee  -> exactly (15, 640, 368)
#   brain -> any coil count with (640, 320)  (brain coils vary 4/16/20; keep
#            batch_size=1 for brain SSDU/zeroshot since C differs per subject)
KEEP_SHAPE = {"knee": (15, 640, 368), "brain": (None, 640, 320)}  # (C, H, W)

VAL_FRAC, TEST_FRAC = 0.10, 0.10        # subject-level split fractions
N_CENTRAL = 3                           # central mode: center-1, center, center+1

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
def central3_indices(n_slices):
    """center-1, center, center+1 (clamped to valid range)."""
    mid = n_slices // 2
    return [i for i in (mid - 1, mid, mid + 1) if 0 <= i < n_slices]


def shape_matches(want, shape):
    return all(w is None or w == s for w, s in zip(want, shape))


def modality_of(path):
    """brain modality token: file_brain_<MODALITY>_... -> <MODALITY>."""
    parts = os.path.basename(path).split("_")
    return parts[2] if len(parts) > 2 else "UNKNOWN"


def raw_files(tissue):
    """First candidate raw subdir that actually contains .h5 files."""
    for sub in RAW_DIRS[tissue]:
        cand = sorted(glob.glob(os.path.join(RAW_ROOT, sub, "*.h5")))
        if cand:
            return cand
    return []


def valid_subjects(tissue):
    """Raw files whose k-space matches the kept geometry for this tissue."""
    want = KEEP_SHAPE[tissue]
    files = []
    for fp in raw_files(tissue):
        try:
            with h5.File(fp, "r") as f:
                if "kspace" in f and shape_matches(want, tuple(f["kspace"].shape[1:])):
                    files.append(fp)
        except Exception:
            continue
    return files


def split_subjects(subjects, seed):
    """Subject-level split -> dict(split -> [filepaths]). val/test = 10% each."""
    subs = list(subjects)
    np.random.default_rng(seed).shuffle(subs)
    n = len(subs)
    n_test = max(1, int(round(n * TEST_FRAC))) if n > 2 else 0
    n_val = max(1, int(round(n * VAL_FRAC))) if n > 2 else 0
    return {"test": subs[:n_test],
            "val": subs[n_test:n_test + n_val],
            "train": subs[n_test + n_val:]}


def clear_split_dirs(tissue_dir):
    for split in ("train", "val", "test"):
        d = os.path.join(SAVE_ROOT, tissue_dir, split)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def save_preview_png(path, rss_slice):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = np.abs(rss_slice)
        plt.imsave(path, img, cmap="gray", vmax=0.6 * img.max())
    except Exception as e:
        print(f"  (png preview skipped: {e})")


def save_slice(tissue_dir, split, subject, sidx, kspace_slice, rss_slice, save_png):
    out_dir = os.path.join(SAVE_ROOT, tissue_dir, split)
    sens = espirit_sens_maps(kspace_slice).astype(np.complex64)
    with h5.File(os.path.join(out_dir, f"{subject}_{sidx:03d}.h5"), "w") as f:
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
def build_tissue(tissue, seed, save_png, full=False):
    tissue_dir = f"{tissue}_full" if full else tissue
    mode = "full (all slices/subject)" if full else f"central ({N_CENTRAL} slices/subject)"
    subjects = valid_subjects(tissue)
    print(f"\n=== {tissue} [{mode}] -> {tissue_dir}/ : {len(subjects)} usable subjects "
          f"(shape {KEEP_SHAPE[tissue]}) ===")

    # group subjects: brain by modality, knee as one group
    if tissue == "brain":
        groups = defaultdict(list)
        for fp in subjects:
            groups[modality_of(fp)].append(fp)
    else:
        groups = {"all": subjects}

    # subject-level split per group, then merge (brain => modality-stratified)
    split_files = {"train": [], "val": [], "test": []}
    group_summary = {}
    for gname, gsubs in sorted(groups.items()):
        sp = split_subjects(gsubs, seed)
        for s in ("train", "val", "test"):
            split_files[s].extend(sp[s])
        group_summary[gname] = {s: len(sp[s]) for s in ("train", "val", "test")}
        print(f"  modality {gname:<10} subjects: "
              f"train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")

    clear_split_dirs(tissue_dir)

    manifest = {"tissue": tissue, "mode": "full" if full else "central",
                "seed": seed, "n_central": (None if full else N_CENTRAL),
                "shape": list(KEEP_SHAPE[tissue]),
                "val_frac": VAL_FRAC, "test_frac": TEST_FRAC,
                "modality_subjects": group_summary, "splits": {}}

    for split, subs in split_files.items():
        count, used = 0, []
        for fp in tqdm.tqdm(subs, desc=f"{tissue_dir}/{split}"):
            subject = os.path.splitext(os.path.basename(fp))[0]
            with h5.File(fp, "r") as f:
                kspace = f["kspace"][:]
                rss = f["reconstruction_rss"][:] if "reconstruction_rss" in f else None
            indices = range(kspace.shape[0]) if full else central3_indices(kspace.shape[0])
            for s in indices:
                save_slice(tissue_dir, split, subject, s, kspace[s],
                           rss[s] if rss is not None else None, save_png)
                count += 1
            used.append(subject)
        manifest["splits"][split] = {"slices": count, "subjects": used}
        print(f"  {tissue_dir}/{split}: {count} slices from {len(used)} subjects")

    os.makedirs(os.path.join(SAVE_ROOT, tissue_dir), exist_ok=True)
    with open(os.path.join(SAVE_ROOT, tissue_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Build fastMRI train/val/test datasets")
    ap.add_argument("--tissue", default="both", choices=["knee", "brain", "both"])
    ap.add_argument("--mode", default="central", choices=["central", "full", "both"],
                    help="central = 3 central slices/subject ({tissue}); "
                         "full = ALL slices/subject ({tissue}_full); both = build both")
    ap.add_argument("--seed", type=int, default=1234,
                    help="same seed => central and full share the SAME subject split")
    ap.add_argument("--save_png", action="store_true",
                    help="also write a magnitude PNG preview per slice")
    args = ap.parse_args()

    tissues = ["knee", "brain"] if args.tissue == "both" else [args.tissue]
    modes = ["central", "full"] if args.mode == "both" else [args.mode]
    for tissue in tissues:
        for mode in modes:
            build_tissue(tissue, args.seed, args.save_png, full=(mode == "full"))
    print("\nDone. Datasets under", SAVE_ROOT)


if __name__ == "__main__":
    main()
