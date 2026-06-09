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


def center_crop_2d(arr: np.ndarray, size) -> np.ndarray:
    """Center-crop the last two axes to (size, size) (or (h,w) if size is a tuple)."""
    sh, sw = (size, size) if np.isscalar(size) else size
    h, w = arr.shape[-2:]
    sh, sw = min(sh, h), min(sw, w)
    top, left = (h - sh) // 2, (w - sw) // 2
    return arr[..., top:top + sh, left:left + sw]


def remove_oversampling(kspace: np.ndarray, sens: np.ndarray, size=320):
    """Crop a multi-coil slice to (size,size) in the IMAGE domain.

    IFFT -> center-crop coil images -> FFT back gives k-space whose recon matches
    the fastMRI ground truth: readout oversampling (e.g. 640->320) is removed and
    the phase FOV is trimmed (e.g. 368->320). Sensitivity maps (image-domain) are
    cropped the same way so SENSE stays consistent.

    kspace (C,H,W) complex, sens (C,H,W) complex -> both (C,size,size).
    """
    coil = center_crop_2d(ifft2c_np(kspace), size)        # (C,size,size) complex image
    kspace_c = fft2c_np(coil)                             # (C,size,size) k-space
    sens_c = center_crop_2d(sens, size)
    return (np.ascontiguousarray(kspace_c).astype(kspace.dtype),
            np.ascontiguousarray(sens_c).astype(sens.dtype))


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
