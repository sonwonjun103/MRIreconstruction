"""Centered orthonormal FFTs, SENSE operators, and complex<->real helpers.

Everything in this project uses the *same* centered, orthonormal (``norm="ortho"``)
2-D FFT convention so that the numpy pre-processing in the datasets and the
torch operators inside the unrolled network are mutually consistent. With an
orthonormal FFT, ``E^H E`` is a well-conditioned normal operator and the
conjugate-gradient data-consistency block behaves nicely.

Coil dimension convention (matches the aggregated h5 files):
    k-space / coil images : (..., C, H, W)
    sensitivity maps       : (C, H, W)
    coil-combined image    : (H, W)
"""

from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# numpy (used in Dataset pre-processing, on CPU)
# --------------------------------------------------------------------------- #
def fft2c_np(x: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def ifft2c_np(x: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes, norm="ortho"), axes=axes)


def sense_combine_np(kspace: np.ndarray, sens_maps: np.ndarray) -> np.ndarray:
    """E^H y : multi-coil k-space (C,H,W) -> coil-combined complex image (H,W)."""
    coil_imgs = ifft2c_np(kspace)                       # (C,H,W)
    return np.sum(np.conj(sens_maps) * coil_imgs, axis=0)


def sense_expand_np(image: np.ndarray, sens_maps: np.ndarray) -> np.ndarray:
    """E x : coil-combined image (H,W) -> multi-coil k-space (C,H,W)."""
    return fft2c_np(sens_maps * image[None, ...])


def rss_np(kspace: np.ndarray) -> np.ndarray:
    """Root-sum-of-squares magnitude image from multi-coil k-space (C,H,W)."""
    coil_imgs = ifft2c_np(kspace)
    return np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))


# --------------------------------------------------------------------------- #
# torch (used inside the network / data-consistency, on GPU, complex tensors)
# --------------------------------------------------------------------------- #
def fft2c(x: torch.Tensor, dims=(-2, -1)) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=dims)
    x = torch.fft.fft2(x, dim=dims, norm="ortho")
    return torch.fft.fftshift(x, dim=dims)


def ifft2c(x: torch.Tensor, dims=(-2, -1)) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=dims)
    x = torch.fft.ifft2(x, dim=dims, norm="ortho")
    return torch.fft.fftshift(x, dim=dims)


# ---- complex <-> real (2-channel) interleaving ---- #
def c2r_chw(x: torch.Tensor) -> torch.Tensor:
    """complex (H,W) -> real (2,H,W)."""
    return torch.stack([x.real, x.imag], dim=0)


def r2c_chw(x: torch.Tensor) -> torch.Tensor:
    """real (2,H,W) -> complex (H,W)."""
    return torch.complex(x[0], x[1])


def c2r_last(x: torch.Tensor) -> torch.Tensor:
    """complex (...) -> real (..., 2)."""
    return torch.stack([x.real, x.imag], dim=-1)


def r2c_last(x: torch.Tensor) -> torch.Tensor:
    """real (..., 2) -> complex (...)."""
    return torch.complex(x[..., 0], x[..., 1])


def c2r_np(x: np.ndarray, axis=0) -> np.ndarray:
    return np.stack([x.real, x.imag], axis=axis)


def r2c_np(x: np.ndarray, axis=0) -> np.ndarray:
    real, imag = np.take(x, 0, axis=axis), np.take(x, 1, axis=axis)
    return real + 1j * imag
