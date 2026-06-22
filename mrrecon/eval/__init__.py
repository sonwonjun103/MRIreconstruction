"""Evaluation -- score any trained checkpoint (or classical SENSE) on a split.

Metrics are RSS-referenced SSIM / PSNR / NMSE / NMAE (falls back to SENSE when a
split has no ground-truth RSS).

Run:  python -m mrrecon.eval --method ssdu --run runs/ssdu1 --tissue knee --split test
      python -m mrrecon.eval --method sense --tissue knee --split test   # no checkpoint
"""
