#!/usr/bin/env python
"""Render training curves from a run's history.json -> <run_dir>/curves.png.

Useful to (re)generate the plot for an in-progress or finished run.

Example:
    python scripts/plot_curves.py runs/unet_knee
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrrecon.core.common import save_curves


def main():
    if len(sys.argv) != 2:
        print("usage: python scripts/plot_curves.py <run_dir>")
        sys.exit(1)
    run_dir = sys.argv[1]
    history = json.load(open(os.path.join(run_dir, "history.json")))
    out = os.path.join(run_dir, "curves.png")
    save_curves(history, out)
    print(f"saved {out}  ({len(history)} epochs, keys: "
          f"{[k for k in history[0] if k != 'epoch']})")


if __name__ == "__main__":
    main()
