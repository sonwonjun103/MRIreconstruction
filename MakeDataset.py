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

# same centered-orthonormal FFT convention as the rest of the toolkit
from mrrecon.data.transforms import ifft2c_np, fft2c_np, center_crop_2d, rss_np

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
        "train": ["knee_multicoil_train", "knee_multicoil_train_batch",
                  "knee_multicoil_train_batch_0/multicoil_train"],
        "val":   ["knee_multicoil_val", "multicoil_val"],
        "test":  ["knee_multicoil_test", "multicoil_test"],
    },
    "brain": {
        "train": ["brain_multicoil_train", "brain_multicoil_train_batch"],
        "val":   ["brain_multicoil_val", "multicoil_val"],
        "test":  ["brain_multicoil_test", "multicoil_test"],
    },
}

# The official fastMRI *test* (challenge) set: prospectively undersampled k-space +
# its real sampling mask, NO reconstruction_rss. Built into a separate
# ``test_challenge`` split for qualitative/inference only (no GT metrics). The .h5
# usually live one level down (e.g. knee_multicoil_test/multicoil_test/).
CHALLENGE_SOURCES = {
    "knee":  ["knee_multicoil_test/multicoil_test", "knee_multicoil_test", "multicoil_test"],
    "brain": ["brain_multicoil_test/multicoil_test", "brain_multicoil_test", "multicoil_test"],
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


def espirit_sens_maps(kspace_slice, calib=24):
    """ESPIRiT sensitivity maps for one slice. (C,H,W) complex -> (C,H,W) complex.

    ``calib`` is the BART ecalib calibration-region size (-r). For prospectively
    undersampled data it must not exceed the fully-sampled ACS width, else the
    calibration region would include aliased (undersampled) lines."""
    ks = kspace_slice.transpose(1, 2, 0)[None, ...]        # (1, H, W, C)
    smap = bart(1, f"ecalib -d0 -m1 -r{calib}", ks)        # (1, H, W, C, 1)
    return smap.transpose(3, 1, 2, 0).squeeze(-1)          # (C, H, W)


def _ifft1c(x, axis):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis),
                                       axis=axis, norm="ortho"), axes=axis)


def _fft1c(x, axis):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis),
                                      axis=axis, norm="ortho"), axes=axis)


def crop_readout_slice(kspace_slice, n):
    """Remove readout (axis -2) oversampling ONLY: 1-D IFFT along readout,
    center-crop to n, 1-D FFT back. (C,H,W) -> (C,n,W).

    The readout axis is fully sampled even for the prospectively undersampled test
    set, so this is alias-free; the phase axis (the undersampled axis) is left
    completely untouched, so its 1-D sampling mask stays exactly valid."""
    img = _ifft1c(kspace_slice, axis=-2)                   # (C,H,W) readout image
    h = img.shape[-2]
    top = max(0, (h - n) // 2)
    img = img[..., top:top + n, :]
    return _fft1c(img, axis=-2).astype(np.complex64)       # (C,n,W)


def acs_calib_width(mask, lo=6, hi=24):
    """Width (clamped to [lo,hi]) of the central fully-sampled ACS block of a 1-D
    phase mask -- a safe BART ecalib -r for prospectively undersampled data."""
    if mask is None:
        return hi
    m = np.asarray(mask).astype(bool).ravel()
    c = len(m) // 2
    l = c
    while l > 0 and m[l - 1]:
        l -= 1
    r = c
    while r < len(m) - 1 and m[r + 1]:
        r += 1
    return int(max(lo, min(hi, r - l + 1)))


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


def crop_kspace_slice(kspace_slice, n, readout_mode="crop"):
    """Crop a (C,H,W) k-space slice to (C,n,n).

    readout_mode:
      'crop'    : image-domain center-crop both axes (IFFT -> crop -> FFT).
                  Matches the fastMRI 320x320 GT exactly (RSS SSIM 1.0). [default]
      'evenodd' : remove readout (axis -2) oversampling by keeping every other
                  k-space line (640 -> 320), then image-crop the phase axis to n.
                  NOTE: introduces a readout half-pixel shift + residual aliasing,
                  so it does NOT match the official reconstruction_rss (SSIM ~0.85);
                  the stored GT is therefore derived from this processed k-space."""
    if readout_mode == "evenodd":
        k = kspace_slice[:, 0::2, :]                       # readout 640 -> 320 (even lines)
        coil = center_crop_2d(ifft2c_np(k), n)             # phase axis -> n (image-domain)
        return fft2c_np(coil).astype(np.complex64)
    coil = center_crop_2d(ifft2c_np(kspace_slice), n)      # 'crop': image-domain both axes
    return fft2c_np(coil).astype(np.complex64)


def save_preview_png(path, rss_slice):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = np.abs(rss_slice)
        plt.imsave(path, img, cmap="gray", vmax=0.6 * img.max())
    except Exception as e:
        print(f"  (png preview skipped: {e})")


def save_slice(out_dir, volume, sidx, kspace_slice, rss_slice, save_png, crop=0,
               readout_mode="crop"):
    full_slice = kspace_slice                              # keep full k-space for sens estimation
    if crop > 0:
        kspace_slice = crop_kspace_slice(kspace_slice, crop, readout_mode)   # (C,crop,crop)
        # 'crop' matches the official 320 rss (reuse it). 'evenodd' is shifted, so
        # always derive a self-consistent GT from the processed k-space.
        if (readout_mode == "evenodd" or rss_slice is None
                or tuple(rss_slice.shape) != (crop, crop)):
            rss_slice = rss_np(kspace_slice).astype(np.float32)
    # ESPIRiT on the FULL k-space (object has the readout-oversampling margin, so it does
    # not touch the FOV edge) then crop the maps -> avoids the FOV-edge artifact (a dark
    # band in the SENSE combine) that 'crop -> ecalib' produces. For 'evenodd' (decimated
    # readout) or no crop, estimate directly on the processed k-space.
    if crop > 0 and readout_mode == "crop":
        sens = center_crop_2d(espirit_sens_maps(full_slice), crop).astype(np.complex64)
    else:
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
def filter_shape(files, tissue, crop=0):
    """Volumes usable for this tissue. Without crop: exact KEEP_SHAPE. With crop:
    any volume large enough to center-crop to (crop,crop) (knee keeps 15 coils),
    so differing phase-FOV widths are all included."""
    want = KEEP_SHAPE[tissue]
    vols = []
    for fp in files:
        try:
            with h5.File(fp, "r") as f:
                if "kspace" not in f:
                    continue
                C, H, W = f["kspace"].shape[1:]            # per-volume (S,C,H,W)
        except Exception:
            continue
        if crop > 0:
            ok = (H >= crop and W >= crop) and (want[0] is None or C == want[0])
        else:
            ok = shape_matches(want, (C, H, W))
        if ok:
            vols.append(fp)
    return vols


def build_volumes(tissue, out_split, vols, slices_per_vol, save_png, crop=0,
                  readout_mode="crop"):
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
                       rss[s] if rss is not None else None, save_png, crop=crop,
                       readout_mode=readout_mode)
            count += 1
        used.append(volume)

    note = "" if with_gt == len(vols) else \
        f"  (WARNING: {len(vols) - with_gt} volumes have NO reconstruction_rss / GT)"
    print(f"  -> {out_dir}: {count} slices from {len(used)} volumes; "
          f"{with_gt}/{len(vols)} with GT{note}")
    return {"slices": count, "volumes": len(used), "volumes_with_gt": with_gt}


def resolve_challenge_source(tissue):
    """First CHALLENGE_SOURCES candidate (possibly nested) that holds .h5 files."""
    for name in CHALLENGE_SOURCES[tissue]:
        files = _h5_in(os.path.join(RAW_ROOT, name))
        if files:
            return files, name
    return [], None


def build_challenge_volumes(tissue, vols, slices_per_vol, save_png, readout=320):
    """Build the prospectively-undersampled official test set into
    {tissue}/test_challenge. Stores kspace (readout-cropped, phase native) + ESPIRiT
    sens + the real 1-D sampling mask; NO rss (the challenge set has no GT)."""
    out_dir = os.path.join(SAVE_ROOT, tissue, "test_challenge")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    count, used, accels = 0, [], []
    for fp in tqdm.tqdm(vols, desc=f"{tissue}/test_challenge"):
        volume = os.path.splitext(os.path.basename(fp))[0]
        with h5.File(fp, "r") as f:
            kspace = f["kspace"][:]                          # (S,C,640,W) undersampled
            mask = f["mask"][:] if "mask" in f else None
        calib = acs_calib_width(mask)
        if mask is not None:
            accels.append(len(mask) / max(1, int(np.asarray(mask).sum())))
        for s in central_indices(kspace.shape[0], slices_per_vol):
            k = crop_readout_slice(kspace[s], readout)       # (C,readout,W)
            # sens on FULL readout (object has oversampling margin) then crop readout ->
            # avoids the FOV-edge artifact (same fix as save_slice).
            H0 = kspace.shape[2]
            top = max(0, (H0 - readout) // 2)
            sens = espirit_sens_maps(kspace[s], calib)[:, top:top + readout, :].astype(np.complex64)
            with h5.File(os.path.join(out_dir, f"{volume}_{s:03d}.h5"), "w") as f:
                f.create_dataset("kspace", data=k)
                f.create_dataset("sens_map", data=sens)
                if mask is not None:
                    f.create_dataset("mask", data=np.asarray(mask).astype(bool))
            if save_png:
                png_dir = os.path.join(out_dir, "preview")
                os.makedirs(png_dir, exist_ok=True)
                save_preview_png(os.path.join(png_dir, f"{volume}_{s:03d}.png"),
                                 rss_np(k))                   # zero-filled RSS preview
            count += 1
        used.append(volume)

    acc = f", mean accel {np.mean(accels):.1f}x" if accels else ""
    print(f"  -> {out_dir}: {count} slices from {len(used)} volumes; "
          f"0/{len(vols)} with GT (prospectively undersampled{acc})")
    return {"slices": count, "volumes": len(used), "volumes_with_gt": 0,
            "prospectively_undersampled": True,
            "mean_accel": round(float(np.mean(accels)), 2) if accels else None}


def main():
    ap = argparse.ArgumentParser(description="Build fastMRI train/val/test datasets")
    ap.add_argument("--tissue", default="both", choices=["knee", "brain", "both"])
    ap.add_argument("--slices_per_vol", type=int, default=0,
                    help="0 = ALL slices per volume (default); N>0 = central N slices")
    ap.add_argument("--crop", type=int, default=0,
                    help="image-domain center-crop each slice to NxN at build time "
                         "(e.g. 320). Makes all volumes the same size so differing "
                         "phase-FOV widths are ALL usable; train with crop_size 0 after.")
    ap.add_argument("--readout_mode", default="crop", choices=["crop", "evenodd"],
                    help="how --crop removes readout oversampling: 'crop' (image-domain, "
                         "matches official GT, SSIM 1.0) [default] or 'evenodd' (keep every "
                         "other readout k-space line; GT then derived from it, SSIM ~0.85)")
    ap.add_argument("--val_holdout", type=float, default=0.1,
                    help="fraction of the fastMRI VAL set held out (volume-disjoint) as "
                         "our validation set")
    ap.add_argument("--test_holdout", type=float, default=0.0,
                    help="fraction of the fastMRI VAL set used as our test set (volume-"
                         "disjoint from val). 0 (default) = use ALL the remaining volumes "
                         "after val; e.g. 0.1 = a 10%% test set (the rest is unused).")
    ap.add_argument("--seed", type=int, default=1234, help="val/test holdout shuffle seed")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "valtest", "challenge"],
                    choices=["train", "valtest", "challenge"],
                    help="which split groups to (re)build: 'train' (our train <- fastMRI "
                         "train), 'valtest' (our val+test <- fastMRI val holdout, both GT), "
                         "'challenge' (our test_challenge <- official fastMRI test, no GT). "
                         "Groups not listed are left untouched on disk.")
    ap.add_argument("--save_png", action="store_true",
                    help="also write a magnitude PNG preview per slice")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N volumes per built split (debug)")
    args = ap.parse_args()

    tissues = ["knee", "brain"] if args.tissue == "both" else [args.tissue]
    for tissue in tissues:
        print(f"\n=== {tissue} -> {os.path.join(SAVE_ROOT, tissue)}/ "
              f"(slices_per_vol={args.slices_per_vol or 'ALL'}, val_holdout={args.val_holdout}) ===")
        # preserve manifest entries for split groups we are NOT rebuilding this run
        man_path = os.path.join(SAVE_ROOT, tissue, "manifest.json")
        prev_splits = {}
        if os.path.exists(man_path):
            try:
                prev_splits = json.load(open(man_path)).get("splits", {})
            except Exception:
                prev_splits = {}
        manifest = {"tissue": tissue, "slices_per_vol": args.slices_per_vol,
                    "crop": args.crop, "readout_mode": args.readout_mode,
                    "val_holdout": args.val_holdout, "test_holdout": args.test_holdout,
                    "seed": args.seed,
                    "shape": ([args.crop, args.crop] if args.crop else list(KEEP_SHAPE[tissue])),
                    "mapping": "train <- ALL fastMRI train; val/test <- fastMRI val "
                               "(val_holdout split, both GT); test_challenge <- official "
                               "fastMRI test (prospectively undersampled, NO GT)",
                    "splits": dict(prev_splits)}

        # our train  <-  ALL of fastMRI train
        if "train" in args.splits:
            train_files, src = resolve_source(tissue, "train")
            if train_files:
                train_vols = filter_shape(train_files, tissue, args.crop)
                if args.limit:
                    train_vols = train_vols[:args.limit]
                print(f"  [train src '{src}'] {len(train_vols)} volumes (all -> train)")
                manifest["splits"]["train"] = build_volumes(tissue, "train", train_vols,
                                                            args.slices_per_vol, args.save_png,
                                                            args.crop, args.readout_mode)
            else:
                print(f"  [train] fastMRI train source not found for {tissue} -- skipped "
                      f"(existing {tissue}/train left untouched)")

        # our val + test  <-  fastMRI val, volume-disjoint holdout (both have GT)
        if "valtest" in args.splits:
            val_files, src = resolve_source(tissue, "val")
            if val_files:
                vols = filter_shape(val_files, tissue, args.crop)
                np.random.default_rng(args.seed).shuffle(vols)
                n_val = max(1, int(round(len(vols) * args.val_holdout))) if len(vols) > 1 else 0
                if args.test_holdout > 0:
                    n_test = max(1, int(round(len(vols) * args.test_holdout)))
                    our_val, our_test = vols[:n_val], vols[n_val:n_val + n_test]
                else:
                    our_val, our_test = vols[:n_val], vols[n_val:]
                if args.limit:
                    our_val, our_test = our_val[:max(1, args.limit // 5)], our_test[:args.limit]
                print(f"  [val src '{src}' = fastMRI val] {len(vols)} vols -> "
                      f"val {len(our_val)} / test {len(our_test)}")
                manifest["splits"]["val"] = build_volumes(tissue, "val", our_val,
                                                         args.slices_per_vol, args.save_png,
                                                         args.crop, args.readout_mode)
                manifest["splits"]["test"] = build_volumes(tissue, "test", our_test,
                                                          args.slices_per_vol, args.save_png,
                                                          args.crop, args.readout_mode)
            else:
                print(f"  [val/test] fastMRI val source not available yet for {tissue} "
                      f"(still downloading?) -- skipped; re-run after download")

        # our test_challenge  <-  official fastMRI test set (prospectively undersampled, NO GT)
        if "challenge" in args.splits:
            ch_files, src = resolve_challenge_source(tissue)
            if ch_files:
                ch_vols = filter_shape(ch_files, tissue, args.crop or 320)
                if args.limit:
                    ch_vols = ch_vols[:args.limit]
                print(f"  [test_challenge src '{src}' = official fastMRI test] "
                      f"{len(ch_vols)} volumes (NO GT)")
                manifest["splits"]["test_challenge"] = build_challenge_volumes(
                    tissue, ch_vols, args.slices_per_vol, args.save_png,
                    readout=(args.crop or 320))
            else:
                print(f"  [test_challenge] official fastMRI test source not found for "
                      f"{tissue} -- skipped")

        if manifest["splits"]:
            os.makedirs(os.path.join(SAVE_ROOT, tissue), exist_ok=True)
            with open(os.path.join(SAVE_ROOT, tissue, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
    print("\nDone. Datasets under", SAVE_ROOT)


if __name__ == "__main__":
    main()
