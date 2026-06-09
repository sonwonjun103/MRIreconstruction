"""Datasets for the three reconstruction methods.

All three share one acquisition pipeline:
    normalise k-space -> build acquisition mask Omega -> SENSE-combine.

Coil layout throughout is (C, H, W). Masks broadcast over the coil axis.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset 

from .transforms import sense_combine_np, c2r_np
from .masks import undersampling_mask, SSDUMaskSplitter
from .loaders import read_slice


def _normalize(kspace: np.ndarray) -> np.ndarray:
    """Scale a multi-coil k-space slice so max|k| == 1."""
    scale = np.max(np.abs(kspace))
    return kspace if scale == 0 else kspace / scale


class SupervisedDataset(Dataset):
    """Zero-filled SENSE image -> fully-sampled SENSE image (both 2-channel).

    Takes a list of per-slice .h5 file paths (one split) and loads lazily.
    Returns dict with:
        x_in   : (2,H,W) float32  -- zero-filled SENSE recon (network input)
        target : (2,H,W) float32  -- fully-sampled SENSE recon (label)
        omega  : (1,H,W) float32  -- acquisition mask
    """

    def __init__(self, cfg, files, train=True):
        self.cfg = cfg
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kspace, sens, _ = read_slice(self.files[idx], crop_size=self.cfg.crop_size)
        kspace = _normalize(kspace)                    # (C,H,W) 
        H, W = kspace.shape[1:]

        # a fresh random mask each epoch when training, deterministic otherwise
        seed = None if self.train else (idx + 1)
        rng = np.random.default_rng(seed)
        omega = undersampling_mask((H, W), self.cfg.acc_rate, self.cfg.acs_lines,
                                   self.cfg.mask_type, rng=rng, vds_power=self.cfg.vds_power)

        target = sense_combine_np(kspace, sens)                 # (H,W) complex
        x_in = sense_combine_np(kspace * omega[None], sens)     # (H,W) complex

        return {
            "x_in": torch.from_numpy(c2r_np(x_in).astype(np.float32)),
            "target": torch.from_numpy(c2r_np(target).astype(np.float32)),
            "omega": torch.from_numpy(omega).unsqueeze(0),
        }


class DiffusionDataset(Dataset):
    """Fully-sampled SENSE images for training the unconditional diffusion prior.

    Returns ``x0`` (2,H,W) -- the coil-combined image scaled so its maximum
    magnitude is 1 (real/imag roughly in [-1, 1]), which is the data range DDPM
    expects. No masks/labels: the prior only learns p(x).
    """

    def __init__(self, cfg, files, train=True):
        self.cfg = cfg
        self.files = files
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kspace, sens, _ = read_slice(self.files[idx], crop_size=self.cfg.crop_size)
        kspace = _normalize(kspace)
        img = sense_combine_np(kspace, sens)                    # (H,W) complex
        scale = np.max(np.abs(img))
        if scale > 0:
            img = img / scale
        return {"x0": torch.from_numpy(c2r_np(img).astype(np.float32))}


class SSDUDataset(Dataset):
    """Self-supervised SSDU samples (split across the whole dataset).

    Returns dict with:
        x_in       : (2,H,W) float32   -- zero-filled SENSE recon from Theta
        ref_kspace : (C,H,W,2) float32 -- target k-space at the loss locations
        sens_maps  : (C,H,W) complex64
        trn_mask   : (1,H,W) float32   -- Theta (data-consistency set)
        loss_mask  : (1,H,W) float32   -- Lambda (loss set)
    """

    def __init__(self, cfg, files, train=True):
        self.cfg = cfg
        self.files = files
        self.train = train
        self.splitter = SSDUMaskSplitter(rho=cfg.rho)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        kspace, sens, _ = read_slice(self.files[idx], crop_size=self.cfg.crop_size)
        kspace = _normalize(kspace)                    # (C,H,W)
        H, W = kspace.shape[1:]

        seed = None if self.train else (idx + 1)
        rng = np.random.default_rng(seed)
        omega = undersampling_mask((H, W), self.cfg.acc_rate, self.cfg.acs_lines,
                                   self.cfg.mask_type, rng=rng, vds_power=self.cfg.vds_power)

        trn_mask, loss_mask = self.splitter.split(
            kspace.transpose(1, 2, 0), omega, method=self.cfg.divide_method, rng=rng)

        sub_kspace = kspace * trn_mask[None]
        ref_kspace = kspace * loss_mask[None]
        x_in = sense_combine_np(sub_kspace, sens)              # (H,W) complex

        return {
            "x_in": torch.from_numpy(c2r_np(x_in).astype(np.float32)),
            "ref_kspace": torch.from_numpy(
                np.stack([ref_kspace.real, ref_kspace.imag], -1).astype(np.float32)),
            "sens_maps": torch.from_numpy(sens),
            "trn_mask": torch.from_numpy(trn_mask).unsqueeze(0),
            "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0),
        }


class ZeroShotDataset(Dataset):
    """ZS-SSL: many (Theta, Lambda) realisations of a *single* scan.

    A fixed validation split ``Gamma`` is first carved out of ``Omega`` (used
    for early stopping). Every ``__getitem__`` then re-splits the remaining
    points ``Omega \\ Gamma`` into a fresh (Theta, Lambda) pair so each step
    sees a new realisation. ``__len__`` == ``cfg.zs_num_splits``.

    Use :meth:`val_sample` for the held-out validation example and
    :meth:`full_sample` for final inference on the entire acquisition.
    """

    def __init__(self, cfg, kspace_slice, sens_slice, omega):
        self.cfg = cfg
        self.kspace = _normalize(kspace_slice)         # (C,H,W)
        self.sens = sens_slice                         # (C,H,W)
        self.omega = omega                             # (H,W)
        self.splitter = SSDUMaskSplitter(rho=cfg.rho)

        # carve out a fixed validation set Gamma from Omega
        val_splitter = SSDUMaskSplitter(rho=cfg.zs_val_rho)
        rng = np.random.default_rng(cfg.seed)
        self.train_mask, self.val_mask = val_splitter.split(
            self.kspace.transpose(1, 2, 0), omega,
            method=cfg.divide_method, rng=rng)

    def __len__(self):
        return max(1, self.cfg.zs_num_splits)

    def _make(self, trn_mask, loss_mask):
        sub = self.kspace * trn_mask[None]
        ref = self.kspace * loss_mask[None]
        x_in = sense_combine_np(sub, self.sens)
        return {
            "x_in": torch.from_numpy(c2r_np(x_in).astype(np.float32)),
            "ref_kspace": torch.from_numpy(
                np.stack([ref.real, ref.imag], -1).astype(np.float32)),
            "sens_maps": torch.from_numpy(self.sens),
            "trn_mask": torch.from_numpy(trn_mask).unsqueeze(0),
            "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0),
        }

    def __getitem__(self, idx):
        # re-split the training portion (Omega \ Gamma) on every access
        rng = np.random.default_rng()
        trn_mask, loss_mask = self.splitter.split(
            self.kspace.transpose(1, 2, 0), self.train_mask,
            method=self.cfg.divide_method, rng=rng)
        return self._make(trn_mask, loss_mask)

    def val_sample(self):
        """DC on (Omega \\ Gamma), loss measured on Gamma."""
        return self._make(self.train_mask, self.val_mask)

    def full_sample(self):
        """Final inference: DC on the full acquisition Omega."""
        return self._make(self.omega, self.omega)
