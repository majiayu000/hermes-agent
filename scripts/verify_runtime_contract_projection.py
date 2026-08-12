#!/usr/bin/env python3
"""Verify a downloaded canonical Runtime artifact against Hermes projection."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SCHEMA_NAMES = (
    "error.schema.json",
    "manifest.schema.json",
    "run-request.schema.json",
    "runtime-event.schema.json",
    "tool-request.schema.json",
    "tool-result.schema.json",
)
LOCAL_ROOT = Path(__file__).resolve().parents[1] / "gateway" / "contracts" / "runtime"


def bundle_digest(schema_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in sorted(SCHEMA_NAMES):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((schema_dir / name).read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def verify_projection(canonical_root: Path) -> str:
    canonical_schema_dir = canonical_root / "v1"
    local_schema_dir = LOCAL_ROOT / "v1"
    for name in SCHEMA_NAMES:
        canonical = (canonical_schema_dir / name).read_bytes()
        projected = (local_schema_dir / name).read_bytes()
        if canonical != projected:
            raise ValueError(f"Runtime contract projection differs for {name}")
    for name in ("compatibility.yaml", "CHANGELOG.md"):
        if (canonical_root / name).read_bytes() != (LOCAL_ROOT / name).read_bytes():
            raise ValueError(f"Runtime contract projection differs for {name}")
    canonical_digest = bundle_digest(canonical_schema_dir)
    projected_digest = bundle_digest(local_schema_dir)
    if canonical_digest != projected_digest:
        raise ValueError("Runtime contract bundle digest differs")
    return canonical_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        required=True,
        type=Path,
        help="Extracted canonical contracts/runtime directory",
    )
    args = parser.parse_args()
    try:
        digest = verify_projection(args.canonical_root.resolve())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
