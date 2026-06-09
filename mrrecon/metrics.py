"""Reconstruction quality metrics: SSIM, PSNR, NMSE, NMAE.

All operate on real-valued magnitude images (2-D numpy arrays). Inputs are
squeezed and, for SSIM/PSNR, normalised by the reference's maximum magnitude so
the metrics are scale-invariant and comparable across slices.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def _prep(org, recon):
    org = np.abs(np.squeeze(org)).astype(np.float64)
    recon = np.abs(np.squeeze(recon)).astype(np.float64)
    return org, recon


def nmse(org, recon) -> float:
    """Normalised mean-squared error: ||recon - org||^2 / ||org||^2."""
    org, recon = _prep(org, recon)
    denom = np.sum(org ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((org - recon) ** 2) / denom)


def nmae(org, recon) -> float:
    """Normalised mean absolute error: ||recon - org||_1 / ||org||_1.

    The L1 analogue of NMSE -- scale-invariant, so it is comparable across
    slices and tissues regardless of the k-space normalisation."""
    org, recon = _prep(org, recon)
    denom = np.sum(np.abs(org))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(org - recon)) / denom)


def psnr(org, recon, eps: float = 1e-10) -> float:
    """Peak signal-to-noise ratio (dB), peak taken from the reference."""
    org, recon = _prep(org, recon)
    mse = np.mean((org - recon) ** 2)
    return float(20 * np.log10(org.max() / (np.sqrt(mse) + eps)))


def ssim(org, recon) -> float:
    """Structural similarity, normalised by the reference maximum."""
    org, recon = _prep(org, recon)
    scale = org.max()
    if scale == 0:
        return float("nan")
    org_n, recon_n = org / scale, recon / scale
    data_range = org_n.max() - org_n.min()
    return float(_ssim(recon_n, org_n, data_range=data_range,
                       gaussian_weights=True, use_sample_covariance=False))


def all_metrics(org, recon) -> dict:
    """Convenience: returns {'ssim', 'psnr', 'nmse', 'nmae'} for one image pair."""
    return {"ssim": ssim(org, recon), "psnr": psnr(org, recon),
            "nmse": nmse(org, recon), "nmae": nmae(org, recon)}


def match_scale(ref, x):
    """Least-squares global scalar fitting magnitude image ``x`` to ``ref``
    (minimises ||a*x - ref||). Removes the arbitrary intensity scale between a
    normalised-k-space recon and the RSS ground truth before scoring."""
    ref = np.abs(np.squeeze(ref)).astype(np.float64)
    x = np.abs(np.squeeze(x)).astype(np.float64)
    den = float(np.sum(x * x))
    a = float(np.sum(ref * x)) / den if den > 0 else 1.0
    return x * a


def rss_metrics(rss, recon, crop_fn=None) -> dict:
    """Metrics of ``recon`` against the RSS ground truth with scale-matching
    (and optional center-crop applied to both via ``crop_fn``)."""
    rss = np.abs(np.squeeze(rss)); recon = np.abs(np.squeeze(recon))
    if crop_fn is not None:
        rss, recon = crop_fn(rss), crop_fn(recon)
    return all_metrics(rss, match_scale(rss, recon))
