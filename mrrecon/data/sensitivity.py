"""Estimate coil sensitivity maps from multi-coil k-space via ESPIRiT.

Backends, same signature ``(C, H, W) complex k-space -> (C, H, W) complex maps``:

* ``espirit_bart``  -- wraps BART ``ecalib`` (Uecker et al.). **Recommended.**
  This is the exact command used to build the aggregated h5 files
  (``ecalib -d0 -m1 -r24``), so it reproduces the stored maps bit-for-bit
  (verified: SSIM 1.0 on the coil-combined image). Requires a BART install.
* ``espirit_sigpy`` -- ``sigpy.mri.app.EspiritCalib``, a well-tested reference
  ESPIRiT. Accurate; requires ``pip install sigpy``.
* ``espirit_numpy`` -- a dependency-free NumPy ESPIRiT (calibration -> Hankel SVD
  -> per-pixel eigen-decomposition). **Approximate** -- it follows the ESPIRiT
  recipe but is not normalised to match BART (coil-combined SSIM ~0.75 in tests),
  so prefer ``bart``/``sigpy`` when available. Phase-referenced to coil 0.

``estimate_sensitivity`` dispatches by ``method``; ``sens_maps_volume`` runs a
whole (S, C, H, W) volume slice-by-slice.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from .transforms import ifft2c_np, fft2c_np

# default BART location (the python wrapper lives under <toolbox>/python)
DEFAULT_BART_PATH = "/home/sonwonjun/research/MRRecon/Paper/bart"


# --------------------------------------------------------------------------- #
# BART backend
# --------------------------------------------------------------------------- #
_BART = None


def _load_bart(toolbox_path: str):
    global _BART
    if _BART is not None:
        return _BART
    os.environ["TOOLBOX_PATH"] = toolbox_path
    os.environ["PATH"] = toolbox_path + os.pathsep + os.environ.get("PATH", "")
    pydir = os.path.join(toolbox_path, "python")
    if pydir not in sys.path:
        sys.path.insert(0, pydir)
    from bart import bart  # noqa: E402
    _BART = bart
    return _BART


def espirit_bart(kspace, num_maps=1, calib_radius=24, toolbox_path=DEFAULT_BART_PATH):
    """ESPIRiT via BART ``ecalib``. kspace (C,H,W) -> maps (C,H,W) [num_maps==1].

    Matches ``MakeDataset.py``: ``ecalib -d0 -m{num_maps} -r{calib_radius}`` on
    k-space laid out (1, H, W, C).
    """
    bart = _load_bart(toolbox_path)
    ks = kspace.transpose(1, 2, 0)[None, ...]                      # (1,H,W,C)
    cmd = f"ecalib -d0 -m{num_maps} -r{calib_radius}"
    smap = bart(1, cmd, ks)                                        # (1,H,W,C,[maps])
    smap = smap.transpose(3, 1, 2, 0).squeeze(-1)                  # (C,H,W) for m=1
    return smap.astype(np.complex64)


# --------------------------------------------------------------------------- #
# sigpy backend (reference ESPIRiT)
# --------------------------------------------------------------------------- #
def espirit_sigpy(kspace, calib_width=24, crop=0.95, thresh=0.02):
    """ESPIRiT via sigpy's EspiritCalib. kspace (C,H,W) -> maps (C,H,W)."""
    try:
        from sigpy.mri.app import EspiritCalib
    except Exception as e:  # pragma: no cover
        raise ImportError("sigpy not installed; `pip install sigpy` or use "
                          "method='bart'") from e
    maps = EspiritCalib(kspace, calib_width=calib_width, thresh=thresh,
                        crop=crop, show_pbar=False).run()
    return np.asarray(maps).astype(np.complex64)


# --------------------------------------------------------------------------- #
# NumPy ESPIRiT backend (approximate)
# --------------------------------------------------------------------------- #
def _calibration_region(kspace, calib_size):
    C, H, W = kspace.shape
    r = min(calib_size, H, W)
    y0, x0 = H // 2 - r // 2, W // 2 - r // 2
    return kspace[:, y0:y0 + r, x0:x0 + r]                         # (C,r,r)


def espirit_numpy(kspace, calib_size=24, kernel=6, sv_thresh=0.02,
                  eig_thresh=0.5):
    """Self-contained ESPIRiT (single map). kspace (C,H,W) -> maps (C,H,W).

    Parameters
    ----------
    calib_size : ACS square side used for calibration.
    kernel     : k-space kernel side (k x k).
    sv_thresh  : keep calibration singular vectors with sigma >= sv_thresh*sigma_max.
    eig_thresh : zero pixels whose top eigenvalue < eig_thresh * max-eigenvalue
                 (relative, in [0,1]); masks the no-signal background.
    """
    C, H, W = kspace.shape
    k = kernel
    cal = _calibration_region(kspace, calib_size)                  # (C,r,r)
    r = cal.shape[-1]

    # 1) block-Hankel calibration matrix: all k x k windows, all coils
    P = r - k + 1
    A = np.empty((P * P, C * k * k), dtype=np.complex64)
    row = 0
    for y in range(P):
        for x in range(P):
            A[row] = cal[:, y:y + k, x:x + k].reshape(-1)
            row += 1

    # 2) SVD -> signal subspace kernels
    _, S, Vh = np.linalg.svd(A, full_matrices=False)
    n = int(np.sum(S >= sv_thresh * S[0]))
    V = Vh[:n].conj().T                                            # (C*k*k, n)
    kernels = V.reshape(C, k, k, n)

    # 3) kernels -> image domain (centered, conj-flipped for convolution)
    ks = np.flip(np.flip(kernels, axis=1), axis=2).conj()         # (C,k,k,n)
    padded = np.zeros((C, H, W, n), dtype=np.complex64)
    y0, x0 = H // 2 - k // 2, W // 2 - k // 2
    padded[:, y0:y0 + k, x0:x0 + k, :] = ks
    gimg = ifft2c_np(padded, axes=(1, 2))                         # (C,H,W,n)

    # 4) per-pixel eigen-decomposition of G G^H  (vectorised over pixels).
    #    The top eigenvector (in coil space) is the sensitivity; eigh returns it
    #    unit-norm, so the map direction is independent of any global scaling.
    G = gimg.reshape(C, H * W, n).transpose(1, 0, 2)              # (HW, C, n)
    GGH = np.matmul(G, np.conj(np.transpose(G, (0, 2, 1))))       # (HW, C, C)
    eigvals, eigvecs = np.linalg.eigh(GGH)                         # ascending
    top_val = eigvals[:, -1]                                       # (HW,)
    sens = eigvecs[:, :, -1]                                       # (HW, C)

    # 5) phase-reference to coil 0; mask no-signal pixels by eigenvalue *relative
    #    to its maximum* (eig_thresh in [0,1]) -- robust to the overall scale.
    phase = np.exp(-1j * np.angle(sens[:, :1] + 1e-12))
    sens = sens * phase
    rel = top_val / (top_val.max() + 1e-12)
    sens = sens * (rel[:, None] >= eig_thresh)
    maps = sens.T.reshape(C, H, W)
    return maps.astype(np.complex64)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def estimate_sensitivity(kspace, method="bart", **kwargs):
    """Estimate (C,H,W) sensitivity maps from (C,H,W) k-space.

    method: 'bart' (ESPIRiT via BART, matches the dataset) or 'numpy'
    (dependency-free ESPIRiT). Extra kwargs are forwarded to the backend.
    """
    if method == "bart":
        return espirit_bart(kspace, **kwargs)
    if method == "sigpy":
        return espirit_sigpy(kspace, **kwargs)
    if method == "numpy":
        return espirit_numpy(kspace, **kwargs)
    raise ValueError(f"unknown sensitivity method: {method}")


def sens_maps_volume(kspace_volume, method="bart", **kwargs):
    """Estimate maps for a whole volume: (S,C,H,W) -> (S,C,H,W)."""
    return np.stack([estimate_sensitivity(ks, method=method, **kwargs)
                     for ks in kspace_volume], axis=0)
