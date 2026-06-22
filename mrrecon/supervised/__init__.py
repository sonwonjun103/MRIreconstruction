"""Supervised reconstruction (trained against RSS / SENSE ground truth).

Networks, selected with ``--net``:
    unet  / swin   -- plain image-to-image MONAI backbone (no data consistency)
    dccnn          -- our Deep Cascade: soft-DC cascades + pluggable backbone
    varnet         -- official fastMRI E2E-VarNet (learned sensitivity maps)

Run:  python -m mrrecon.supervised --net unet --tissue knee --run_name sup1
"""
