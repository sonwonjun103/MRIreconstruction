"""Load the per-slice datasets built by ``MakeDataset.py``.

Layout (one HDF5 file per slice)::

    {data_root}/{tissue}/{split}/{subject}_{slice:03d}.h5
        kspace   (C, H, W) complex64
        sens_map (C, H, W) complex64
        rss      (h, w)    float32   [optional reference/preview]

The split is the *folder* (train / val / test); the zero-shot set is the test
split. Slices are loaded lazily (one file at a time) by the datasets, so the
multi-GB volumes never sit in RAM at once.
"""

from __future__ import annotations

import os
import glob

import h5py as h5
import numpy as np


def split_dir(data_root: str, tissue: str, split: str, full: bool = False) -> str:
    """Directory for a split. ``full=True`` selects the full-subject dataset
    ({tissue}_full); otherwise the central-slice dataset ({tissue})."""
    sub = f"{tissue}_full" if full else tissue
    return os.path.join(data_root, sub, split)


def list_slice_files(data_root: str, tissue: str, split: str = "train",
                     max_slices: int = -1, modality: str = "", full: bool = False):
    """Sorted list of per-slice .h5 paths for one split.

    ``modality`` (brain only) restricts to files of one MRI modality, matched on
    the ``_<MODALITY>_`` token in the filename (e.g. ``file_brain_AXT1POST_...``).
    The underscores make it exact: 'AXT1' does not match 'AXT1POST'/'AXT1PRE'.
    ``max_slices > 0`` keeps only the first N files (quick smoke tests).
    """
    d = split_dir(data_root, tissue, split, full)
    files = sorted(glob.glob(os.path.join(d, "*.h5")))
    if not files:
        mode = "--full_subject" if full else ""
        raise FileNotFoundError(
            f"no slice files in {d}\n"
            f"Build the dataset first:  python MakeDataset.py --tissue {tissue} {mode}".strip())
    if modality:
        files = [f for f in files if f"_{modality}_" in os.path.basename(f)]
        if not files:
            avail = available_modalities(data_root, tissue, split, full)
            raise FileNotFoundError(
                f"no '{modality}' slices in {d}. available: {avail}")
    if max_slices and max_slices > 0:
        files = files[:max_slices]
    return files


def available_modalities(data_root: str, tissue: str, split: str = "train", full: bool = False):
    """Sorted set of modality tokens present in a split (brain naming)."""
    d = split_dir(data_root, tissue, split, full)
    mods = set()
    for f in glob.glob(os.path.join(d, "*.h5")):
        parts = os.path.basename(f).split("_")
        if len(parts) > 2 and parts[1] == "brain":
            mods.add(parts[2])
    return sorted(mods)


def read_slice(path: str):
    """Read one slice file -> (kspace (C,H,W) complex, sens (C,H,W) complex,
    rss (h,w) float or None)."""
    with h5.File(path, "r") as f:
        kspace = f["kspace"][:]
        sens = f["sens_map"][:]
        rss = f["rss"][:] if "rss" in f else None
    return (np.ascontiguousarray(kspace), np.ascontiguousarray(sens),
            None if rss is None else np.ascontiguousarray(rss))


def peek_shape(data_root: str, tissue: str, split: str = "train", full: bool = False):
    """(C, H, W) of the first slice file in a split, without loading the array."""
    files = list_slice_files(data_root, tissue, split, full=full)
    with h5.File(files[0], "r") as f:
        c, h, w = f["kspace"].shape
    return int(c), int(h), int(w)
