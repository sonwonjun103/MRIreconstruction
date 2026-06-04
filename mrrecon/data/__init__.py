"""Data utilities: FFT/SENSE transforms, undersampling masks, loaders, datasets.

Sensitivity-map estimation (ESPIRiT) lives in ``sensitivity.py``:
    from mrrecon.data.sensitivity import estimate_sensitivity
    maps = estimate_sensitivity(kspace_slice, method="bart")   # (C,H,W) -> (C,H,W)
"""
