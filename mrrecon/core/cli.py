"""Shared CLI launcher used by every method package's ``__main__``.

Each ``python -m mrrecon.<category>`` builds a slim, category-specific parser
(only the flags that category needs) and hands the resolved Config plus a
zero-arg runner to :func:`launch`, which sets up run-directory logging and the
standard banner -- exactly like the old ``main.py`` did, but per method folder.
"""

from __future__ import annotations


def launch(cfg, command: str, runner) -> None:
    """Tee stdout/stderr to runs/<run_name>/log.txt, print the banner, run."""
    from .common import start_file_logging
    log_path = start_file_logging(cfg)
    mod = f" modality={cfg.modality}" if cfg.modality else ""
    print(f">>> mrrecon {command} | tissue={cfg.tissue}{mod} acc={cfg.acc_rate} "
          f"run={cfg.run_name} | log -> {log_path}")
    try:
        runner()
    except Exception:
        import traceback
        traceback.print_exc()              # also captured in the log file
        raise
