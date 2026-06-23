"""``python -m mrrecon.eval`` -- evaluate a checkpoint (or classical SENSE).

    python -m mrrecon.eval --method ssdu  --run runs/ssdu1 --tissue knee --split test
    python -m mrrecon.eval --method varnet --ckpt runs/vn/acc4/best.pt --tissue knee
    python -m mrrecon.eval --method sense --tissue knee --split test --save_figs

``--run <dir>`` is a convenience: the checkpoint is taken as
``<dir>/acc<acc_rate>/best.pt`` (override with an explicit ``--ckpt``).
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
warnings.filterwarnings("ignore")

from mrrecon.config import (_add_common, _add_unet, _add_unrolled,
                            _add_diffusion, _add_eval, _add_split, config_from_args)
from mrrecon.core.cli import launch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mrrecon.eval",
        description="Evaluate a reconstruction checkpoint (RSS SSIM/PSNR/NMSE/NMAE).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run", default=None,
                   help="run dir; checkpoint = <run>/acc<acc_rate>/best.pt")
    p.add_argument("--reference", default="cgsense", choices=["cgsense", "zerofilled"],
                   help="(--split test_challenge only) SENSE-based reference for metrics: "
                        "cgsense (CG-SENSE de-aliased; recommended) or zerofilled "
                        "(aliased SENSE coil-combination). NOT a ground truth either way.")
    _add_common(p)
    _add_eval(p)            # --method (required) / --ckpt / sense_* / --save_figs
    _add_split(p)
    _add_unet(p)
    _add_unrolled(p)
    _add_diffusion(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    ckpt = args.ckpt
    if args.run:                            # --run convenience
        if ckpt is None:                    # -> checkpoint = <run>/acc<acc>/best.pt
            ckpt = os.path.join(args.run, f"acc{cfg.acc_rate}", "best.pt")
        if args.run_name == "run":          # -> write eval results into the model's run dir
            run = os.path.normpath(args.run)
            cfg.run_name = os.path.basename(run)
            cfg.out_dir = os.path.dirname(run) or "./runs"
    if args.method != "sense" and not ckpt:
        raise SystemExit("eval: --method %s needs --ckpt or --run" % args.method)

    if args.split == "test_challenge":      # prospectively undersampled, no GT
        from mrrecon.eval.challenge import ChallengeInference
        ci = ChallengeInference(cfg, method=args.method, ckpt=ckpt,
                                split=args.split, save_figs=args.save_figs,
                                reference=args.reference)
        launch(cfg, f"challenge:{args.method}", ci.run)
    else:
        from mrrecon.eval.evaluator import Evaluator
        ev = Evaluator(cfg, method=args.method, ckpt=ckpt,
                       split=args.split, save_figs=args.save_figs)
        launch(cfg, f"eval:{args.method}", ev.evaluate)


if __name__ == "__main__":
    main()
