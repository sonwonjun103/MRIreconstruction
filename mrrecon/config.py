"""Shared configuration: a single dataclass plus an argparse builder.

Every CLI script builds its own ``argparse.ArgumentParser`` from the helpers
here so flags stay consistent across supervised / SSDU / zero-shot runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from typing import Optional


# Root that holds the per-slice datasets produced by MakeDataset.py:
#   {data_root}/{tissue}/{train,val,test}/{volume}_{slice}.h5
DEFAULT_DATA_ROOT = "/mnt/d/research/MRRecon"


@dataclass
class Config:
    # ---- environment ----
    device: str = "cuda"
    seed: int = 1234 
    out_dir: str = "./runs"
    run_name: str = "run"

    # ---- data ----
    data_root: str = DEFAULT_DATA_ROOT
    tissue: Optional[str] = None          # "knee" or "brain" -- required
    modality: str = ""                    # brain only: AXFLAIR/AXT1/AXT1POST/... ("" = all)
    n_coils: int = 15
    max_slices: int = -1                  # -1 = use all; >0 = subset (debug/speed)
    full_subject: bool = False            # True -> use {tissue}_full (all slices/subject);
                                          # False -> {tissue} (central slices only)
    crop_size: int = 320                  # crop kspace+sens to NxN in image domain to match
                                          # the RSS GT (no-op if data already that size). 0=off

    # ---- undersampling (the *acquisition* mask Omega) ----
    acc_rate: int = 4
    acs_lines: int = 24
    mask_type: str = "random"             # "random" | "gaussian1d" | "vds"
    vds_power: float = 3.0                # variable-density polynomial power (mask_type=vds)

    # ---- SSDU mask splitting (Omega -> train Theta / loss Lambda) ----
    divide_method: str = "Gaussian_selection"  # or "uniform_selection"
    rho: float = 0.4                      # fraction of Omega put in the loss set

    # ---- unrolled model (ssdu / zeroshot) ----
    model: str = "ssdu"                   # "ssdu" (ResNet) | "mymodel" (U-Net) | "mamba"
    res_blocks: int = 15
    cg_iter: int = 10
    mu: float = 0.05
    nb_unroll_blocks: int = 10

    # ---- mymodel: hierarchical Mamba-regularised unrolled net (ZS-MambaRecon) ----
    mymodel_chans: int = 32               # base feature width C of the denoiser
    mymodel_pools: int = 3                # U-shape depth (number of 2x downsamplings)
    mymodel_ssm_blocks: int = 2           # RSSB (Mamba) blocks per coarse stage
    mymodel_mamba_levels: int = 1         # how many coarsest scales use Mamba (>=1)
    mymodel_dstate: int = 16              # SSM state dimension N
    mymodel_expand: int = 1               # SS2D inner expansion (1 = lean, zero-shot friendly)

    # ---- DCCNN (Deep Cascade): soft-DC cascades + pluggable backbone, RSS output ----
    use_dc: bool = False                  # supervised: wrap --arch in a DCCNN (data consistency)
    dc_cascades: int = 8                  # number of unrolled DC cascades
    cnn: str = "unet"                     # DCCNN backbone: "unet" | "swin" | "mamba"
    # ---- official E2E-VarNet (facebookresearch/fastMRI, learned SME; verbatim package) ----
    varnet_official: bool = False         # internal: set True by the 'varnet' method
    varnet_cascades: int = 12             # official cascades (official default)
    varnet_sens_chans: int = 8            # SME U-Net channels (official default)
    varnet_sens_pools: int = 4            # SME U-Net pools (official default)
    varnet_unet_chans: int = 18           # cascade NormUnet channels (official default)
    varnet_unet_pools: int = 4            # cascade NormUnet pools (official default)

    # ---- mamba: Mamba-ViT-regularised unrolled net ----
    mamba_dim: int = 128                  # token embedding dim
    mamba_depth: int = 4                  # number of Mamba-ViT blocks
    mamba_patch: int = 16                 # patch size (larger -> shorter sequence -> faster)
    mamba_dstate: int = 16                # SSM state dim
    mamba_expand: int = 2                 # inner expansion factor

    # ---- diffusion model (zero-shot reconstruction) ----
    diff_dim: int = 64                    # base channels of the noise-predictor U-Net
    diff_timesteps: int = 1000            # diffusion steps T
    diff_schedule: str = "cosine"         # "cosine" | "linear"
    diff_sampling_steps: int = 100        # DDIM steps at reconstruction
    diff_dc_lam: float = 1.0              # prior-proximity weight in DC (smaller -> trust measurements more)
    diff_dc_iter: int = 5                 # CG iterations per DC step

    # ---- classical CG-SENSE (no learning) ----
    sense_lam: float = 1e-2               # Tikhonov regularisation weight
    sense_cg_iter: int = 30

    # ---- supervised network ----
    arch: str = "unet"                    # "unet" | "swin" (Transformer)
    sup_target: str = "rss"               # "rss" (1-ch magnitude, fastMRI-standard) |
                                          # "sense" (2-ch complex SENSE image)
    loss: str = "l1"                      # "l1" | "l2" | "ssim" | "l1ssim"
    ssim_weight: float = 1.0              # weight of SSIM term in "l1ssim"
    unet_chans: int = 32
    unet_pools: int = 4
    unet_drop: float = 0.0
    # Swin (Transformer) supervised model
    swin_dim: int = 48
    swin_depths: int = 4                  # number of RSTB stages
    swin_blocks: int = 4                  # Swin blocks per stage
    swin_heads: int = 6
    swin_window: int = 8

    # ---- optimisation ----
    batch_size: int = 1
    epochs: int = 50
    lr: float = 1e-3
    num_workers: int = 4

    # ---- zero-shot specific ----
    zs_subject: str = ""                  # h5 file stem to fit, e.g. "file1001942"
    zs_slice: int = -1                    # -1 = centre slice
    zs_num_splits: int = 25               # (Theta, Lambda) realisations per epoch (ZS-SSL num_reps)
    zs_val_rho: float = 0.2               # fraction of Omega held out for validation
    zs_patience: int = 25                 # early-stopping patience (epochs)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# argparse plumbing
# --------------------------------------------------------------------------- #
def _add_common(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("environment")
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--out_dir", default="./runs")
    g.add_argument("--run_name", default="run")

    g = parser.add_argument_group("data")
    g.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    g.add_argument("--tissue", required=True, choices=["knee", "brain"],
                   help="tissue type; selects the {tissue} dataset folder")
    g.add_argument("--modality", default="",
                   help="brain only: restrict to one MRI modality "
                        "(AXFLAIR/AXT1/AXT1POST/AXT1PRE/AXT2). Empty = all modalities.")
    g.add_argument("--n_coils", type=int, default=15)
    g.add_argument("--max_slices", type=int, default=-1,
                   help="-1 uses all slices; a positive value subsets for speed")
    g.add_argument("--full_subject", action="store_true",
                   help="(deprecated, ignored) there is now a single dataset per "
                        "tissue at {data_root}/{tissue}/{split}")
    g.add_argument("--crop_size", type=int, default=320,
                   help="crop k-space+sens to NxN in image domain to match the RSS GT "
                        "(default 320). No-op if the data is already NxN (e.g. built "
                        "with MakeDataset --crop). Use 0 to disable entirely.")
    g.add_argument("--mode", default="central", choices=["central", "full"],
                   help="(deprecated, ignored) single dataset per tissue now")

    g = parser.add_argument_group("undersampling")
    g.add_argument("--acc_rate", type=int, default=4)
    g.add_argument("--acs_lines", type=int, default=24)
    g.add_argument("--mask_type", default="random",
                   choices=["random", "gaussian1d", "vds"],
                   help="undersampling pattern: random (uniform), gaussian1d/vds "
                        "(variable density, center-weighted)")
    g.add_argument("--vds_power", type=float, default=3.0,
                   help="variable-density polynomial power for --mask_type vds")

    g = parser.add_argument_group("optim")
    g.add_argument("--batch_size", type=int, default=1)
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--num_workers", type=int, default=4)


def _add_unrolled(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("unrolled")
    g.add_argument("--model", default="ssdu",
                   choices=["ssdu", "mymodel", "unetunroll", "mamba"],
                   help="regulariser: 'ssdu' (ResNet), 'mymodel' (proposed hierarchical "
                        "Mamba), 'unetunroll' (legacy U-Net), or 'mamba' (Mamba-ViT baseline)")
    g.add_argument("--res_blocks", type=int, default=15)
    g.add_argument("--cg_iter", type=int, default=10)
    g.add_argument("--mu", type=float, default=0.05)
    g.add_argument("--nb_unroll_blocks", type=int, default=10)
    g.add_argument("--mymodel_chans", type=int, default=32, help="base feature width C")
    g.add_argument("--mymodel_pools", type=int, default=3, help="U-shape depth")
    g.add_argument("--mymodel_ssm_blocks", type=int, default=2,
                   help="RSSB (Mamba) blocks per coarse stage")
    g.add_argument("--mymodel_mamba_levels", type=int, default=1,
                   help="number of coarsest scales that use Mamba (>=1; 1 = bottleneck only)")
    g.add_argument("--mymodel_dstate", type=int, default=16, help="SSM state dim N")
    g.add_argument("--mymodel_expand", type=int, default=1, help="SS2D inner expansion")
    # --- DCCNN (Deep Cascade): soft-DC cascades + pluggable backbone, fixed ESPIRiT ---
    g.add_argument("--dc_cascades", type=int, default=8,
                   help="number of DCCNN cascades (--method dccnn)")
    g.add_argument("--cnn", default="unet", choices=["unet", "swin", "mamba"],
                   help="DCCNN backbone: 'unet', 'swin', or 'mamba'")
    # --- official E2E-VarNet (facebookresearch/fastMRI, learned SME) (--method varnet) ---
    g.add_argument("--varnet_cascades", type=int, default=12,
                   help="official E2E-VarNet cascades (official default 12)")
    g.add_argument("--varnet_sens_chans", type=int, default=8)
    g.add_argument("--varnet_sens_pools", type=int, default=4)
    g.add_argument("--varnet_unet_chans", type=int, default=18)
    g.add_argument("--varnet_unet_pools", type=int, default=4)
    g.add_argument("--mamba_dim", type=int, default=128)
    g.add_argument("--mamba_depth", type=int, default=4)
    g.add_argument("--mamba_patch", type=int, default=16)
    g.add_argument("--mamba_dstate", type=int, default=16)
    g.add_argument("--mamba_expand", type=int, default=2)
    g.add_argument("--divide_method", default="Gaussian_selection",
                   choices=["Gaussian_selection", "uniform_selection"])
    g.add_argument("--rho", type=float, default=0.4)


def _add_unet(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("supervised net")
    g.add_argument("--arch", default="unet", choices=["unet", "swin"],
                   help="supervised backbone (MONAI): 'unet' (UNet) or 'swin' (SwinUNETR)")
    g.add_argument("--sup_target", default="rss", choices=["rss", "sense"],
                   help="supervised target: 'rss' (1-ch magnitude, fastMRI-standard) "
                        "or 'sense' (2-ch complex SENSE image)")
    g.add_argument("--use_dc", action="store_true",
                   help="wrap --arch (unet/swin) in a VarNet so it gets multi-coil "
                        "data consistency (DC). Off = plain image-to-image (no DC). "
                        "With --use_dc the output is always RSS.")
    g.add_argument("--loss", default="l1", choices=["l1", "l2", "ssim", "l1ssim"],
                   help="supervised loss (ssim/l1ssim use 1-SSIM on magnitude)")
    g.add_argument("--ssim_weight", type=float, default=1.0,
                   help="weight of the SSIM term when --loss l1ssim")
    g.add_argument("--unet_chans", type=int, default=32, help="MONAI UNet base channels")
    g.add_argument("--unet_pools", type=int, default=4, help="MONAI UNet depth (levels)")
    g.add_argument("--unet_drop", type=float, default=0.0)
    g.add_argument("--swin_dim", type=int, default=48,
                   help="SwinUNETR feature_size (multiple of 12)")
    g.add_argument("--swin_depths", type=int, default=4, help="(unused for MONAI SwinUNETR)")
    g.add_argument("--swin_blocks", type=int, default=4, help="(unused for MONAI SwinUNETR)")
    g.add_argument("--swin_heads", type=int, default=6, help="(unused for MONAI SwinUNETR)")
    g.add_argument("--swin_window", type=int, default=8, help="(unused for MONAI SwinUNETR)")


def _add_diffusion(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("diffusion")
    g.add_argument("--diff_dim", type=int, default=64)
    g.add_argument("--diff_timesteps", type=int, default=1000)
    g.add_argument("--diff_schedule", default="cosine", choices=["cosine", "linear"])
    g.add_argument("--diff_sampling_steps", type=int, default=100)
    g.add_argument("--diff_dc_lam", type=float, default=1.0)
    g.add_argument("--diff_dc_iter", type=int, default=5)


def _add_zeroshot(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("zeroshot")
    g.add_argument("--zs_subject", default="",
                   help="h5 stem to fit (in {tissue}_multicoil_train/). Empty -> first test slice.")
    g.add_argument("--zs_slice", type=int, default=-1)
    g.add_argument("--zs_num_splits", type=int, default=25,
                   help="(Theta,Lambda) realisations per epoch = official ZS-SSL num_reps (25)")
    g.add_argument("--zs_val_rho", type=float, default=0.2)
    g.add_argument("--zs_patience", type=int, default=25)


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config`, keeping only fields it knows about."""
    valid = Config().to_dict().keys()
    kwargs = {k: v for k, v in vars(args).items() if k in valid}
    cfg = Config(**kwargs)
    # '--mode full' and '--full_subject' both select the full-subject dataset
    if getattr(args, "mode", "central") == "full":
        cfg.full_subject = True
    return cfg


def _add_split(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])


def _add_eval(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True,
                        choices=["sense", "supervised", "dccnn", "varnet", "ssdu", "zeroshot", "diffusion"],
                        help="'sense' is the classical CG-SENSE baseline (no checkpoint)")
    parser.add_argument("--ckpt", default=None,
                        help="path to model .pt (not needed for --method sense)")
    parser.add_argument("--sense_lam", type=float, default=1e-2)
    parser.add_argument("--sense_cg_iter", type=int, default=30)
    parser.add_argument("--save_figs", action="store_true")


def configure_parser(parser: argparse.ArgumentParser, method: str) -> argparse.ArgumentParser:
    """Add the flag groups for ``method`` to an existing parser (also used by
    ``main.py`` for its subcommands). Methods: ``sense`` | ``supervised`` |
    ``ssdu`` | ``zeroshot`` | ``eval``."""
    _add_common(parser)
    if method == "sense":
        parser.add_argument("--sense_lam", type=float, default=1e-2)
        parser.add_argument("--sense_cg_iter", type=int, default=30)
        _add_split(parser)
        parser.add_argument("--save_figs", action="store_true")
    elif method == "supervised":
        _add_unet(parser)
        parser.add_argument("--dc_cascades", type=int, default=8,
                            help="cascades when --use_dc (wraps --arch in a DCCNN)")
    elif method in ("dccnn", "varnet"):
        # dccnn: our DC cascade + pluggable backbone (--cnn). varnet: official E2E-VarNet.
        _add_unet(parser)                # unet_chans/pools (unet/swin backbone) + loss
        _add_unrolled(parser)            # --cnn/--dc_cascades + mymodel_* + varnet_* flags
        _add_split(parser)
    elif method == "ssdu":
        _add_unrolled(parser)
    elif method == "sscu":
        _add_unrolled(parser)            # stub: same flags as ssdu for now
    elif method == "zeroshot":
        _add_unrolled(parser)
        _add_zeroshot(parser)
        _add_split(parser)
    elif method == "zeroshot-infer":
        _add_unrolled(parser)
        _add_zeroshot(parser)
        _add_split(parser)
        parser.add_argument("--ckpt", required=True,
                            help="zero-shot checkpoint to reconstruct from (e.g. runs/<run>/best.pt)")
    elif method == "diffusion":
        _add_diffusion(parser)
    elif method == "profile":
        _add_unet(parser)
        _add_unrolled(parser)
        _add_zeroshot(parser)
        _add_diffusion(parser)
        parser.add_argument("--profile_methods", nargs="+", default=None,
                            help="subset to profile, e.g. supervised ssdu mymodel mamba diffusion")
        parser.add_argument("--profile_iters", type=int, default=5,
                            help="timed train steps per model (after warm-up)")
    elif method == "eval":
        _add_unet(parser)
        _add_unrolled(parser)
        _add_diffusion(parser)
        _add_eval(parser)
        _add_split(parser)
    else:
        raise ValueError(f"unknown method: {method}")
    return parser


def build_parser(method: str) -> argparse.ArgumentParser:
    """Create a standalone parser for one method (used by the per-method scripts)."""
    return configure_parser(argparse.ArgumentParser(description=f"mrrecon {method}"), method)
