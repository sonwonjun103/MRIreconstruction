"""Reconstruction networks and a small factory for the unrolled models."""

from .unet import UNet
from .resnet import ResNetDenoiser
from .unrolled import UnrolledSSDU
from .mymodel import UnrolledUNet
from .mamba import UnrolledMamba
from .swin import SwinIR
from .sense import cg_sense_recon

__all__ = ["UNet", "ResNetDenoiser", "UnrolledSSDU", "UnrolledUNet",
           "UnrolledMamba", "SwinIR", "cg_sense_recon",
           "build_unrolled", "build_supervised"]


def build_supervised(cfg):
    """Supervised image->image network selected by ``cfg.arch`` (MONAI backbones).

    'unet' -> MONAI UNet, 'swin' -> MONAI SwinUNETR.
    """
    arch = getattr(cfg, "arch", "unet")
    from .monai_nets import build_monai_unet, build_monai_swinunetr
    if arch == "unet":
        return build_monai_unet(cfg)
    if arch == "swin":
        return build_monai_swinunetr(cfg)
    raise ValueError(f"unknown supervised arch: {arch}")


def build_unrolled(cfg):
    """Return the unrolled network selected by ``cfg.model``.

    Both share the ``forward(input_x, sens_maps, trn_mask, loss_mask)`` API, so
    they are interchangeable in the SSDU / zero-shot engines and the evaluator.
    """
    name = getattr(cfg, "model", "ssdu")
    if name == "ssdu":
        return UnrolledSSDU(cfg)
    if name == "mymodel":
        return UnrolledUNet(cfg)
    if name == "mamba":
        return UnrolledMamba(cfg)
    raise ValueError(f"unknown unrolled model: {name}")
