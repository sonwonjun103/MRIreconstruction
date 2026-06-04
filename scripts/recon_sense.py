#!/usr/bin/env python
"""Classical CG-SENSE reconstruction baseline (no training, no checkpoint).

Example:
    python scripts/recon_sense.py --tissue knee --split test \
        --sense_lam 1e-2 --sense_cg_iter 30 --run_name sense_knee --save_figs
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.engine.evaluator import Evaluator


def main():
    # reuse the eval parser but fix method to sense
    parser = build_parser("eval")
    args = parser.parse_args(["--method", "sense", *sys.argv[1:]])
    cfg = config_from_args(args)
    Evaluator(cfg, method="sense", ckpt=None,
              split=args.split, save_figs=args.save_figs).evaluate()


if __name__ == "__main__":
    main()
