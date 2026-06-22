"""Zero-shot reconstruction -- fit a single scan at inference time (no training set).

Algorithms, selected with ``--algo``:
    zsssl     -- ZS-SSL (Yaman et al.): the unrolled net is trained on ONE scan,
                 Omega split into train/loss/val sets, early-stopped per scan.
    diffusion -- train a diffusion prior, then reconstruct via DC-guided sampling.

Run:  python -m mrrecon.zero_shot --algo zsssl --tissue knee --split test
"""
