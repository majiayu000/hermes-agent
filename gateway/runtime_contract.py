"""Stable private contract shared by the Hermes Runtime HTTP surface."""

from __future__ import annotations

from typing import Any

RUNTIME_PROTOCOL_VERSION = "1"
RUNTIME_DRIVER_FRAME_TYPES = (
    "run_started",
    "heartbeat",
    "text_delta",
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
    "session_db_resume",
    "retry_current_turn/v1",
    "system_context.replace",
    "llm_egress",
)

_SAFE_ERROR_MESSAGES = {
    "content_policy_blocked": "The request could not be completed because of a content policy.",
    "insufficient_credits": "The current Account does not have enough credit for this request.",
    "model_incompatible": "The selected model could not accept the generated result.",
    "provider_empty_stream": "The creation service returned no output.",
    "provider_timeout": "The creation service timed out.",
    "provider_unavailable": "The creation service is temporarily unavailable.",
    "runtime_unavailable": "The creation service is temporarily unavailable.",
}
_RETRYABLE_ERROR_CODES = frozenset(
    {
        "provider_empty_stream",
        "provider_timeout",
        "provider_unavailable",
        "runtime_unavailable",
    }
)


def runtime_health_contract() -> dict[str, Any]:
    return {
        "runtime_protocol_version": RUNTIME_PROTOCOL_VERSION,
        "runtime_frame_types": list(RUNTIME_DRIVER_FRAME_TYPES),
        "runtime_capabilities": list(RUNTIME_CAPABILITIES),
    }


def runtime_error_envelope(code: str, *, support_id: str) -> dict[str, Any]:
    normalized = _normalize_error_code(code)
    return {
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


def _normalize_error_code(code: str) -> str:
    if not isinstance(code, str) or not 1 <= len(code) <= 128:
        return "unexpected_error"
    if any(not (character.isascii() and (character.islower() or character.isdigit() or character in "._")) for character in code):
        return "unexpected_error"
    return code
