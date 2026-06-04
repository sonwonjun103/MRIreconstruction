#!/usr/bin/env python
"""Evaluate a trained checkpoint and report SSIM / PSNR / NMSE.

Example:
    python scripts/evaluate.py --method ssdu --tissue knee --split test \
        --ckpt runs/ssdu_knee/best.pt --run_name ssdu_knee_eval --save_figs
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.engine.evaluator import Evaluator


def main():
    args = build_parser("eval").parse_args()
    cfg = config_from_args(args)
    Evaluator(cfg, method=args.method, ckpt=args.ckpt,
              split=args.split, save_figs=args.save_figs).evaluate()


if __name__ == "__main__":
    main()
