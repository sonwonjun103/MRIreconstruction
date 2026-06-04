#!/usr/bin/env python
"""Report per-model parameters, FLOPs, and training-time estimates.

Example:
    python scripts/profile_models.py --tissue knee \
        --profile_methods supervised ssdu mymodel mamba diffusion

Pass the model hyper-flags you intend to train with (e.g. --nb_unroll_blocks,
--cg_iter, --mamba_depth, --diff_dim) so the numbers match your real runs.
"""
import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mrrecon.config import build_parser, config_from_args
from mrrecon.profiling import profile_methods


def main():
    args = build_parser("profile").parse_args()
    cfg = config_from_args(args)
    profile_methods(cfg, methods=args.profile_methods, time_iters=args.profile_iters)


if __name__ == "__main__":
    main()
