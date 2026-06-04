"""mrrecon: zero-shot / supervised / self-supervised MRI reconstruction on fastMRI.

A small, reusable toolkit built around the existing aggregated multi-coil
fastMRI data (k-space + BART sensitivity maps). Three reconstruction methods
share a single SENSE-based data pipeline and a common metrics module:

    * supervised  -- plain U-Net (zero-filled SENSE image -> image)
    * ssdu        -- self-supervised unrolled net trained across many slices
    * zeroshot    -- ZS-SSL: the same unrolled net trained on a single scan

See ``README.md`` for usage.
"""

__version__ = "0.1.0"
