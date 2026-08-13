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
    encode_runtime_event,
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


def verify_producer_run_requests(input_directory: Path) -> None:
    paths = sorted(input_directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no producer run requests in {input_directory}")
    for path in paths:
        try:
            decode_runtime_run_request(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ValueError(f"producer run request rejected: {path.name}") from exc


def emit_runtime_events(output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    error = {
        "code": "provider_timeout",
        "message": "The provider timed out.",
        "retryable": True,
    }
    events = {
        "run-started.json": {
            "run_id": "run_matrix",
            "type": "run_started",
            "payload": {
                "runtime": "hermes",
                "system_context_version": "matrix/v1",
                "system_context_mode": "replace",
                "system_context_digest": "sha256:" + "a" * 64,
            },
        },
        "heartbeat.json": {"run_id": "run_matrix", "type": "heartbeat", "payload": {}},
        "text-delta.json": {
            "run_id": "run_matrix",
            "type": "text_delta",
            "payload": {"delta": "hello"},
        },
        "control-request.json": {
            "run_id": "run_matrix",
            "type": "runtime_control_request",
            "payload": {
                "request_id": "control_matrix",
                "kind": "model_contract.get",
                "model": "text/test",
            },
        },
        "tool-request.json": {
            "run_id": "run_matrix",
            "type": "tool_request",
            "payload": {
                "call_id": "call_matrix",
                "name": "platform.test",
                "arguments": {},
                "skills": [],
            },
        },
        "activity-started.json": {
            "run_id": "run_matrix",
            "type": "activity_started",
            "payload": {"call_id": "call_local", "name": "web_search", "arguments": {}},
        },
        "activity-completed.json": {
            "run_id": "run_matrix",
            "type": "activity_completed",
            "payload": {"call_id": "call_local", "name": "web_search", "status": "completed"},
        },
        "usage.json": {
            "run_id": "run_matrix",
            "type": "usage",
            "payload": {"input_tokens": 1, "output_tokens": 1},
        },
        "completed.json": {
            "run_id": "run_matrix",
            "type": "completed",
            "payload": {"finish_reason": "stop", "text": "done"},
        },
        "error.json": {"run_id": "run_matrix", "type": "error", "payload": error},
    }
    for name, event in events.items():
        (output_directory / name).write_bytes(encode_runtime_event(event))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        required=True,
        type=Path,
        help="Extracted canonical contracts/runtime directory",
    )
    parser.add_argument(
        "--producer-run-requests",
        type=Path,
        help="directory of real Orchestrator adapter request bytes",
    )
    parser.add_argument(
        "--emit-runtime-events",
        type=Path,
        help="write bytes from Hermes' production event encoder",
    )
    args = parser.parse_args()
    try:
        digest = verify_projection(args.canonical_root.resolve())
        verify_golden_examples(args.canonical_root.resolve())
        if args.producer_run_requests:
            verify_producer_run_requests(args.producer_run_requests.resolve())
        if args.emit_runtime_events:
            emit_runtime_events(args.emit_runtime_events.resolve())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
