#!/usr/bin/env python3
"""Verify a downloaded canonical Runtime artifact against Hermes projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gateway.runtime_contract_models import (
    decode_runtime_error,
    decode_runtime_event,
    decode_runtime_manifest,
    decode_runtime_run_request,
    decode_runtime_tool_request,
    decode_runtime_tool_result,
)


SCHEMA_NAMES = (
    "error.schema.json",
    "manifest.schema.json",
    "run-request.schema.json",
    "runtime-event.schema.json",
    "tool-request.schema.json",
    "tool-result.schema.json",
)
LOCAL_ROOT = Path(__file__).resolve().parents[1] / "gateway" / "contracts" / "runtime"
FIXTURE_DECODERS = {
    "error.json": decode_runtime_error,
    "manifest.json": decode_runtime_manifest,
    "run-request.json": decode_runtime_run_request,
    "runtime-event.json": decode_runtime_event,
    "tool-request.json": decode_runtime_tool_request,
    "tool-result.json": decode_runtime_tool_result,
}


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


def verify_golden_examples(canonical_root: Path) -> None:
    examples = canonical_root / "v1" / "examples"
    for fixture_name, decoder in FIXTURE_DECODERS.items():
        valid_path = examples / "valid" / fixture_name
        invalid_path = examples / "invalid" / fixture_name
        try:
            decoder(json.loads(valid_path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ValueError(f"valid Runtime fixture rejected: {fixture_name}") from exc
        try:
            decoder(json.loads(invalid_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        raise ValueError(f"invalid Runtime fixture accepted: {fixture_name}")


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
        verify_golden_examples(args.canonical_root.resolve())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
