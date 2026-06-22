"""``python -m mrrecon.zero_shot`` -- zero-shot reconstruction (single-scan / prior).

    python -m mrrecon.zero_shot --algo zsssl --tissue knee --split test --model mymodel
    python -m mrrecon.zero_shot --algo zsssl --tissue knee --split test --ckpt runs/zs1/acc4/best.pt
    python -m mrrecon.zero_shot --algo diffusion --tissue knee --epochs 100   # train the prior

``--algo zsssl`` fits one scan (train, or reconstruct from --ckpt). ``--algo
diffusion`` trains the diffusion prior used for DC-guided zero-shot sampling.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
warnings.filterwarnings("ignore")

from mrrecon.config import (_add_common, _add_unrolled, _add_zeroshot,
                            _add_diffusion, _add_split, config_from_args)
from mrrecon.core.cli import launch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mrrecon.zero_shot",
        description="Zero-shot MRI reconstruction (ZS-SSL single-scan / diffusion prior).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algo", default="zsssl", choices=["zsssl", "diffusion"],
                   help="zsssl = single-scan ZS-SSL; diffusion = train the prior")
    p.add_argument("--ckpt", default=None,
                   help="(zsssl) reconstruct from this checkpoint instead of fitting")
    _add_common(p)
    _add_split(p)
    _add_unrolled(p)
    _add_zeroshot(p)
    _add_diffusion(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    if args.algo == "zsssl":
        from mrrecon.zero_shot.zsssl import ZeroShotTrainer
        trainer = ZeroShotTrainer(cfg, split=args.split)
        run = (lambda: trainer.infer(args.ckpt)) if args.ckpt else trainer.train
        tag = "zero_shot:zsssl" + ("-infer" if args.ckpt else "")
    else:
        from mrrecon.zero_shot.diffusion import DiffusionTrainer
        run = DiffusionTrainer(cfg).train
        tag = "zero_shot:diffusion"

    launch(cfg, tag, run)


if __name__ == "__main__":
    main()
