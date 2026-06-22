"""``python -m mrrecon.self_supervised`` -- train a self-supervised net (no GT).

    python -m mrrecon.self_supervised --algo ssdu --tissue knee --run_name ssdu1
    python -m mrrecon.self_supervised --algo ssdu --tissue knee --model mymodel  # Mamba regulariser

Trains across the whole train split using Theta/Lambda mask splitting; validates
on val (RSS, monitoring only). ``--model`` picks the unrolled regulariser network.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../..")
warnings.filterwarnings("ignore")

from mrrecon.config import _add_common, _add_unrolled, config_from_args
from mrrecon.core.cli import launch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m mrrecon.self_supervised",
        description="Self-supervised MRI reconstruction (SSDU; no ground truth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algo", default="ssdu", choices=["ssdu", "sscu"],
                   help="self-supervised algorithm (sscu is a stub)")
    _add_common(p)
    _add_unrolled(p)        # --model regulariser + nb_unroll_blocks/res_blocks/cg_iter/mu/rho
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_args(args)

    if args.algo == "ssdu":
        from mrrecon.self_supervised.ssdu import SSDUTrainer
        run = SSDUTrainer(cfg).train
    else:
        from mrrecon.self_supervised.sscu import SSCUTrainer
        run = SSCUTrainer(cfg).train

    launch(cfg, f"self_supervised:{args.algo}", run)


if __name__ == "__main__":
    main()
