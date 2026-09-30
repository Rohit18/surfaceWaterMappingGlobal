"""Locations of the inputs and outputs of the 10 m evaluation, read from environment variables.

The check runs behind manuscript v11 hard-coded these locations on NERSC Perlmutter. The repository copies read
them from the environment instead; README.md in this folder lists each variable and what it must contain.
"""
from __future__ import annotations

import os
from pathlib import Path


def env_root(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Set the environment variable {name} (see scripts/eval_10m/README.md).")
    return Path(value)
