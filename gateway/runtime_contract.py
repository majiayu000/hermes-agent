"""Stable private contract shared by the Hermes Runtime HTTP surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gateway.runtime_contract_models import (
    decode_runtime_error,
    decode_runtime_manifest,
)

RUNTIME_PROTOCOL_VERSION = "2"
RUNTIME_DRIVER_FRAME_TYPES = (
    "run_started",
    "heartbeat",
    "text_delta",
    "runtime_control_request",
    "tool_request",
    "activity_started",
    "activity_completed",
    "usage",
    "completed",
    "error",
)
RUNTIME_CAPABILITIES = (
    "delegated_tools",
    "interrupt",
    "model_contract_control",
    "session_db_resume",
    "session_db_rebootstrap/v1",
    "retry_current_turn/v1",
    "system_context.replace",
    "llm_egress",
    "vision_llm_egress",
)

_RUNTIME_CONTRACT_ROOT = Path(__file__).resolve().parent / "contracts" / "runtime"
_RUNTIME_CONTRACT_SCHEMA_DIR = _RUNTIME_CONTRACT_ROOT / "v1"
_RUNTIME_CONTRACT_SCHEMA_NAMES = (
    "error.schema.json",
    "manifest.schema.json",
    "run-request.schema.json",
    "runtime-event.schema.json",
    "tool-request.schema.json",
    "tool-result.schema.json",
)
_RUN_REQUEST_SCHEMA_PATH = _RUNTIME_CONTRACT_SCHEMA_DIR / "run-request.schema.json"
_RUN_REQUEST_SCHEMA_BYTES = _RUN_REQUEST_SCHEMA_PATH.read_bytes()
_RUN_REQUEST_SCHEMA = json.loads(_RUN_REQUEST_SCHEMA_BYTES)
_RUN_REQUEST_METADATA = _RUN_REQUEST_SCHEMA["x-ultra-contract"]


def _runtime_contract_bundle_digest() -> str:
    digest = hashlib.sha256()
    for name in sorted(_RUNTIME_CONTRACT_SCHEMA_NAMES):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((_RUNTIME_CONTRACT_SCHEMA_DIR / name).read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


RUNTIME_CONTRACT_MAJOR = int(_RUN_REQUEST_METADATA["major"])
RUNTIME_CONTRACT_MINOR = int(_RUN_REQUEST_METADATA["minor"])
RUNTIME_RUN_REQUEST_SCHEMA_DIGEST = (
    "sha256:" + hashlib.sha256(_RUN_REQUEST_SCHEMA_BYTES).hexdigest()
)
RUNTIME_CONTRACT_SCHEMA_DIGEST = _runtime_contract_bundle_digest()
RUNTIME_RUN_REQUEST_FIELDS = frozenset(_RUN_REQUEST_SCHEMA["properties"])
RUNTIME_RUN_INTENTS = tuple(_RUN_REQUEST_SCHEMA["properties"]["intent"]["enum"])
RUNTIME_MANIFEST_FEATURES = tuple(_RUN_REQUEST_METADATA["features"])
_RUNTIME_EVENT_SCHEMA = json.loads(
    (_RUNTIME_CONTRACT_SCHEMA_DIR / "runtime-event.schema.json").read_bytes()
)
if (
    tuple(_RUNTIME_EVENT_SCHEMA["properties"]["type"]["enum"])
    != RUNTIME_DRIVER_FRAME_TYPES
):
    raise RuntimeError("Runtime event schema and frame vocabulary disagree")

_SAFE_ERROR_MESSAGES = {
    "content_policy_blocked": "The request could not be completed because of a content policy.",
    "insufficient_credits": "The current Account does not have enough credit for this request.",
    "model_incompatible": "The selected model could not accept the generated result.",
    "provider_empty_stream": "The creation service returned no output.",
    "provider_timeout": "The creation service timed out.",
    "provider_unavailable": "The creation service is temporarily unavailable.",
    "runtime_unavailable": "The creation service is temporarily unavailable.",
}
_RETRYABLE_ERROR_CODES = frozenset({
    "provider_empty_stream",
    "provider_timeout",
    "provider_unavailable",
    "runtime_unavailable",
})


def runtime_health_contract() -> dict[str, Any]:
    return {
        "runtime_protocol_version": RUNTIME_PROTOCOL_VERSION,
        "runtime_frame_types": list(RUNTIME_DRIVER_FRAME_TYPES),
        "runtime_capabilities": list(RUNTIME_CAPABILITIES),
    }


def runtime_manifest_contract(
    *,
    runtime_build: str,
    max_request_bytes: int,
    max_tool_result_bytes: int,
) -> dict[str, Any]:
    """Return the non-sensitive Runtime compatibility advertisement."""
    manifest = {
        "runtime": "hermes",
        "runtime_build": runtime_build,
        "contract": {
            "major": RUNTIME_CONTRACT_MAJOR,
            "min_minor": RUNTIME_CONTRACT_MINOR,
            "max_minor": RUNTIME_CONTRACT_MINOR,
            "schema_digests": [RUNTIME_CONTRACT_SCHEMA_DIGEST],
        },
        "intents": list(RUNTIME_RUN_INTENTS),
        "features": list(RUNTIME_MANIFEST_FEATURES),
        "limits": {
            "max_request_bytes": max_request_bytes,
            "max_tool_result_bytes": max_tool_result_bytes,
        },
    }
    return decode_runtime_manifest(manifest).model_dump(mode="python")


def runtime_error_envelope(code: str, *, support_id: str) -> dict[str, Any]:
    normalized = _normalize_error_code(code)
    envelope = {
        "code": normalized,
        "message": _SAFE_ERROR_MESSAGES.get(
            normalized,
            "The request could not be completed.",
        ),
        "retryable": normalized in _RETRYABLE_ERROR_CODES,
        "reason": normalized,
        "source": "runtime",
        "support_id": support_id,
    }
    return decode_runtime_error(envelope).model_dump(mode="python", exclude_none=True)


def _normalize_error_code(code: str) -> str:
    if not isinstance(code, str) or not 1 <= len(code) <= 128:
        return "unexpected_error"
    if any(
        not (
            character.isascii()
            and (character.islower() or character.isdigit() or character in "._")
        )
        for character in code
    ):
        return "unexpected_error"
    return code
