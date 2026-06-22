"""``python -m mrrecon.supervised`` -- train a supervised reconstruction net.

    python -m mrrecon.supervised --net unet   --tissue knee --run_name sup_unet
    python -m mrrecon.supervised --net swin   --tissue knee --run_name sup_swin
    python -m mrrecon.supervised --net dccnn  --tissue knee --backbone unet --dc_cascades 8
    python -m mrrecon.supervised --net varnet --tissue knee --varnet_cascades 12

``--net`` is the single selector; only supervised-relevant flags are shown by -h.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
warnings.filterwarnings("ignore")

from mrrecon.config import _add_common, _add_unet, _add_unrolled, config_from_args
from mrrecon.core.cli import launch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mrrecon.supervised",
        description="Supervised MRI reconstruction (RSS/SENSE-target training).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--net", default="unet",
                   choices=["unet", "swin", "dccnn", "varnet"],
                   help="unet/swin = image-to-image (no DC); dccnn = Deep Cascade "
                        "(DC + --backbone); varnet = official E2E-VarNet")
    p.add_argument("--backbone", default="unet", choices=["unet", "swin", "mamba"],
                   help="backbone CNN inside --net dccnn")
    _add_common(p)
    _add_unet(p)            # sup_target / loss / unet_chans|pools / swin_dim ...
    _add_unrolled(p)        # dc_cascades + varnet_* knobs (used by dccnn / varnet)
    return p


def main() -> None:
    args = build_parser().parse_args()
    net = args.net

    # derive the legacy arch/use_dc/backbone fields from the single --net selector
    if net in ("unet", "swin"):
        args.arch = net
        args.use_dc = False
    else:                                   # dccnn / varnet are DC unrolled nets
        args.cnn = args.backbone
    cfg = config_from_args(args)

    if net in ("unet", "swin"):
        from mrrecon.supervised.trainer import SupervisedTrainer
        run = SupervisedTrainer(cfg).train
    elif net == "dccnn":
        cfg.varnet_official = False
        from mrrecon.supervised.dc import VarNetTrainer
        run = VarNetTrainer(cfg).train
    else:                                   # varnet
        cfg.varnet_official = True
        from mrrecon.supervised.dc import VarNetTrainer
        run = VarNetTrainer(cfg).train

    launch(cfg, f"supervised:{net}", run)


if __name__ == "__main__":
    main() 
