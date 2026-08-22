"""Run-scoped opaque media reference resolution for native analysis tools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse

from gateway.api_server_shared import MAX_RUNTIME_ATTACHMENT_BYTES


@dataclass
class PendingMediaReference:
    ready: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


def rewrite_known_media_references(
    args: dict[str, Any],
    field_names: tuple[str, ...],
    references: dict[str, str],
) -> dict[str, Any]:
    rewritten = dict(args)
    for field_name in field_names:
        if field_name not in rewritten:
            continue
        value = rewritten.get(field_name)
        if isinstance(value, str):
            rewritten[field_name] = references.get(value.strip(), value)
        elif isinstance(value, list):
            rewritten[field_name] = [
                references.get(item.strip(), item) if isinstance(item, str) else item
                for item in value
            ]
    return rewritten


def image_analysis_sources(args: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for field_name in ("image_url", "image_paths"):
        value = args.get(field_name)
        if isinstance(value, str):
            sources.append(value.strip())
        elif isinstance(value, list):
            sources.extend(item.strip() for item in value if isinstance(item, str))
    return sources


def image_analysis_scope_error() -> str:
    return media_reference_error(
        "image_analysis_scope_denied",
        "image_analyze may only read HTTPS images or run-bound asset_id/output_id references",
        False,
    )


def resolve_media_arguments(
    session: Any,
    args: dict[str, Any],
    field_names: tuple[str, ...],
    media_type: str,
    parent_call_id: str,
) -> tuple[dict[str, Any], str | None]:
    rewritten = dict(args)
    for field_name in field_names:
        if field_name not in rewritten:
            continue
        value = rewritten.get(field_name)
        values = value if isinstance(value, list) else [value]
        resolved_values: list[Any] = []
        for item in values:
            if not isinstance(item, str):
                resolved_values.append(item)
                continue
            reference_id = item.strip()
            if urlparse(reference_id).scheme.lower() == "http":
                return rewritten, media_reference_error(
                    "insecure_media_reference",
                    "media analysis URLs must use HTTPS",
                    False,
                )
            if not reference_id.startswith(("asset_", "output_")):
                resolved_values.append(item)
                continue
            path, error = _resolve_media_reference(
                session, reference_id, media_type, parent_call_id
            )
            if error is not None:
                return rewritten, error
            resolved_values.append(path)
        rewritten[field_name] = (
            resolved_values if isinstance(value, list) else resolved_values[0]
        )
    return rewritten, None


def _resolve_media_reference(
    session: Any,
    reference_id: str,
    media_type: str,
    parent_call_id: str,
) -> tuple[str, str | None]:
    references = (
        session.allowed_image_references
        if media_type == "image"
        else session.allowed_video_references
    )
    existing = references.get(reference_id)
    if existing:
        return existing, None
    request_id = "media_" + hashlib.sha256(
        (parent_call_id + "\x00" + media_type + "\x00" + reference_id).encode()
    ).hexdigest()
    pending = PendingMediaReference()
    with session.lock:
        if request_id in session.pending_controls:
            return "", media_reference_error(
                "media_reference_conflict",
                "duplicate active media reference resolution",
                False,
            )
        session.pending_controls[request_id] = pending
    session.emit("runtime_control_request", {
        "request_id": request_id,
        "kind": "media_reference.resolve",
        "reference_id": reference_id,
        "media_type": media_type,
    })
    wait_timeout = session.deadline_seconds or session.unbounded_tool_wait_seconds
    if not pending.ready.wait(wait_timeout) or session.interrupted.is_set():
        with session.lock:
            session.pending_controls.pop(request_id, None)
        return "", media_reference_error(
            "media_reference_unavailable",
            "media reference resolution did not complete before the Run deadline",
            True,
        )
    control_result = pending.result or {}
    if not control_result.get("ok"):
        error = control_result.get("error")
        if not isinstance(error, dict):
            error = {}
        return "", media_reference_error(
            str(error.get("code") or "media_reference_unavailable"),
            str(error.get("message") or "media reference is unavailable"),
            bool(error.get("retryable")),
        )
    try:
        parts = session.materialize_media_reference(control_result.get("result"))
    except (TypeError, ValueError, OSError):
        return "", media_reference_error(
            "invalid_media_reference_result",
            "resolved media did not satisfy the Runtime attachment contract",
            False,
        )
    path_key = "_runtime_image_path" if media_type == "image" else "_runtime_video_path"
    paths = [str(part[path_key]) for part in parts if isinstance(part, dict) and part.get(path_key)]
    if len(paths) != 1:
        return "", media_reference_error(
            "invalid_media_reference_result",
            "resolved media type does not match the requested analysis tool",
            False,
        )
    path = str(Path(paths[0]).resolve())
    with session.lock:
        all_paths = session.allowed_image_paths | session.allowed_video_paths
        total_bytes = sum(
            candidate.stat().st_size
            for raw_path in all_paths
            if (candidate := Path(raw_path)).is_file()
        )
        if path not in all_paths and (
            len(all_paths) >= 8
            or total_bytes + Path(path).stat().st_size > MAX_RUNTIME_ATTACHMENT_BYTES
        ):
            Path(path).unlink(missing_ok=True)
            return "", media_reference_error(
                "media_reference_budget_exceeded",
                "resolved media exceeds the Runtime attachment budget",
                False,
            )
        if media_type == "image":
            session.allowed_image_paths.add(path)
            session.allowed_image_references[reference_id] = path
        else:
            session.allowed_video_paths.add(path)
            session.allowed_video_references[reference_id] = path
    return path, None


def media_reference_error(code: str, message: str, retryable: bool) -> str:
    return json.dumps({
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "field": "media_reference",
            "allowed": ["asset_id", "output_id", "https_url"],
            "retryable": retryable,
            "provider_submission_started": False,
        },
    }, ensure_ascii=False, separators=(",", ":"))


def invoke_video_analyze(session: Any, args: dict[str, Any], next_call: Any) -> Any:
    """Stop only an unchanged terminal video-analysis retry."""
    with session.video_analyze_lock:
        signature_key = session._tool_signature_key("video_analyze", args)
        with session.lock:
            prior_code = session.native_non_retryable_failures.get(signature_key, "")
        if prior_code:
            message = (
                "Blocked unchanged video_analyze retry after non-retryable error "
                f"{prior_code}."
            )
            session._halt_tool_loop(
                "video_analyze", args, "repeated_non_retryable_tool_call", message, 2
            )
            return json.dumps({
                "error": {
                    "code": "repeated_non_retryable_tool_call",
                    "message": message,
                    "retryable": False,
                },
            }, ensure_ascii=False, separators=(",", ":"))
        requested_path = str(args.get("video_url") or "").strip()
        parsed = urlparse(requested_path)
        remote_url = parsed.scheme.lower() == "https"
        try:
            resolved_path = str(Path(requested_path).resolve(strict=True))
        except (OSError, RuntimeError):
            resolved_path = ""
        if not remote_url and resolved_path not in session.allowed_video_paths:
            result: Any = media_reference_error(
                "video_analysis_scope_denied",
                "video_analyze requires a run-bound asset_id or output_id",
                False,
            )
        else:
            result = next_call(args) if callable(next_call) else args
        code = _native_non_retryable_failure_code(result)
        if code:
            with session.lock:
                session.native_non_retryable_failures[signature_key] = code
        return result


def _native_non_retryable_failure_code(result: Any) -> str:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict) and error.get("retryable") is False:
        return str(error.get("code") or "native_tool_failed")
    if payload.get("success") is False and payload.get("retryable") is False:
        return str(payload.get("error_code") or "native_tool_failed")
    return ""
