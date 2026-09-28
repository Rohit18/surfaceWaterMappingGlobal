#!/usr/bin/env python3
"""Download and verify a released model bundle from Hugging Face.

The default revision is the tag `v2-labelclass`: the models trained on the Dynamic World `label` class that the
paper reports. The earlier mixed-label release stays available at `--revision v1-mixed-labels --no-verify`.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "rohitm9/surfaceWaterGlobal"
DEFAULT_REVISION = "v2-labelclass"
VARIANTS = {
    # S1 + acquisition-year AEF (k=16, seed 42): the paper's primary model.
    "fused_current": {
        "models/s1aef_bottleneck_resnet34_best.pth":
            "dd0252e2d08c693e3f88949a0a34a846f91dc7ab1668d91ba7ac0ddb688b06ec",
        "band_stats.npz":
            "35c28cac98892702999847e721d7e63721056a754a0c1b016a25b641ba32fbaa",
    },
    # Matched S1-only control (k=0, seed 42).
    "s1_only": {
        "variants/s1_only_k0_seed42/models/s1dw_resnet34_best.pth":
            "a5eab4f07d68256d5a7922a8943efb1b0a14cbc0a513315aff737426a57b6e68",
        "variants/s1_only_k0_seed42/band_stats.npz":
            "5e8aa167a4447887ca49fd1dac257e4df6911136277bf72e81ab9ef74b67c54d",
    },
    # S1 + previous-year AEF (k=16, seed 42), trained with t-1 embeddings.
    "fused_previous_year": {
        "variants/s1_aef_previous_year_k16_seed42/models/s1aef_bottleneck_resnet34_best.pth":
            "5bd312409fcf86e0a49258c37aa95a53867a5a6b2cef4f73d2e527a3d6fb2717",
        "variants/s1_aef_previous_year_k16_seed42/band_stats.npz":
            "fd15d29aefbab433ca3e02939822ef611ee4e213c871f26ed825e5ba6c2f7f6f",
    },
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
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="fused_current")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip checksum verification; required for revisions other than the default tag")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected in VARIANTS[args.variant].items():
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                revision=args.revision,
                local_dir=output_dir,
            )
        )
        if args.no_verify:
            print("downloaded {}".format(downloaded))
            continue
        actual = sha256(downloaded)
        if actual != expected:
            raise RuntimeError(
                "Checksum mismatch for {}: expected {}, got {}".format(downloaded, expected, actual)
            )
        print("verified {} {}".format(actual, downloaded))


if __name__ == "__main__":
    main()
