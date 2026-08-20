"""Helpers for projecting private Runtime tool results as activity events."""

from __future__ import annotations

import json
from typing import Any


def activity_failure_message(result: Any) -> str:
    """Return a bounded public failure message for a failed tool envelope."""
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return ""
    if not isinstance(parsed, dict):
        return ""
    error = parsed.get("error")
    failed = parsed.get("success") is False or (
        error is not None and parsed.get("success") is not True
    )
    if not failed:
        return ""
    message = error
    if isinstance(error, dict):
        message = error.get("message") or error.get("code")
    if not isinstance(message, str) or not message.strip():
        return "runtime activity failed"
    return message.strip()[:240]
