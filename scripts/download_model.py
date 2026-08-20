#!/usr/bin/env python3
"""Download and verify the released S1+AEF model bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "rohitm9/surfaceWaterGlobal"
FILES = {
    "models/s1aef_bottleneck_resnet34_best.pth":
        "3b89c5b768be8d8b5a00260693ca5d2a051ec63969234702903677b9e326138f",
    "band_stats.npz":
        "35c28cac98892702999847e721d7e63721056a754a0c1b016a25b641ba32fbaa",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/s1aef_resnet34"))
    parser.add_argument("--revision", default="d57438286a45fd54205c470664f85c7ebb701013")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected in FILES.items():
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=args.revision,
                local_dir=output_dir,
            )
        )
        actual = sha256(downloaded)
        if actual != expected:
            raise RuntimeError(
                "Checksum mismatch for {}: expected {}, got {}".format(downloaded, expected, actual)
            )
        print("verified {} {}".format(actual, downloaded))


if __name__ == "__main__":
    main()
