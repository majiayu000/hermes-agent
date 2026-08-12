"""Run Orchestrator Runtime Driver surface for the API server."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from gateway.runtime_skill_projection import (
    RuntimeSkillProjection,
    projection_skill_metadata,
    resolve_skill_projections,
    view_skill,
)
from gateway.runtime_tool_exposure import (
    RuntimeToolExposure,
    build_runtime_tool_exposure,
)
from tools.tool_search import (
    TOOL_SEARCH_NAME,
)

from gateway.api_server_shared import (
    AIOHTTP_AVAILABLE,
    MAX_RUNTIME_ATTACHMENT_BYTES,
    web,
)
from gateway.runtime_contract import runtime_error_envelope
from agent.tool_dispatch_helpers import DeferredToolResult
from gateway.runtime_session_history import (
    RuntimeSessionStateError as _RuntimeSessionStateError,
    load_runtime_session_history as _load_runtime_session_history,
    retry_session_db_history as _retry_session_db_history,
    resume_session_db_history as _resume_session_db_history,
    seed_recovery_tool_call as _seed_recovery_tool_call,
    runtime_history_tool_names as _runtime_history_tool_names,
    seed_runtime_session as _seed_runtime_session,
)

logger = logging.getLogger(__name__)

_SESSIONS: dict[str, "RuntimeBridgeSession"] = {}
_SESSIONS_LOCK = threading.RLock()
_REGISTERED_MANAGERS: set[int] = set()
_LOCAL_ACTIVITY_TOOLS = {
    "skill_view",
    "tool_search",
    "image_analyze",
    "video_analyze",
    "web_search",
    "web_extract",
}
_RUNTIME_NATIVE_TOOLS = frozenset({
    "skill_view",
    "image_analyze",
    "video_analyze",
    "web_search",
    "web_extract",
})
_MAX_ARGUMENT_CORRECTIONS = 1
_MODEL_SCHEMA_ENVELOPE_FIELDS = frozenset({"request_id", "model", "medias"})
_FAILED_ALLOWED_STRING_FIELDS = {"media_type", "model", "provider"}
_FAILED_ALLOWED_STRING_LIST_FIELDS = {"aspect_ratios", "resolutions"}
_FAILED_ALLOWED_INTEGER_LIST_FIELDS = {"durations"}
_FAILED_ALLOWED_INTEGER_FIELDS = {"max_prompt_chars", "max_reference_images"}
_FAILED_ALLOWED_FIELDS = (
    _FAILED_ALLOWED_STRING_FIELDS
    | _FAILED_ALLOWED_STRING_LIST_FIELDS
    | _FAILED_ALLOWED_INTEGER_LIST_FIELDS
    | _FAILED_ALLOWED_INTEGER_FIELDS
)
_FAILED_ERROR_FIELDS = {"code", "message", "retryable"}
_TERMINAL_PLATFORM_ERROR_CODES = {
    "auth_rejected",
    "configuration_error",
    "cost_budget_exceeded",
    "idempotency_conflict",
    "insufficient_credits",
    "internal_error",
    "invalid_tool_result",
    "model_incompatible",
    "model_not_allowed",
    "provider_unavailable",
    "scope_denied",
    "tool_call_limit_exceeded",
    "tool_not_allowed",
    "tool_not_implemented",
    "unsupported_capability",
}

# Each bridge run parks one thread for its whole duration (invoke_platform_tool
# blocks on pending.ready.wait), so /v1/runtime/runs must never share the small
# default executor. Runs use a dedicated bounded pool gated before streaming.
_RUNTIME_MAX_CONCURRENT_ENV = "HERMES_RUNTIME_MAX_CONCURRENT"
_RUNTIME_MAX_CONCURRENT_DEFAULT = 8


def _requires_model_parameter_contract(tool_name: str) -> bool:
    return tool_name == "media.estimate_cost" or tool_name.startswith("media.generate_")
_RUNTIME_STREAM_HEARTBEAT_SECONDS = 15.0
# Safety cap for pending.ready.wait when the run carries no explicit deadline;
# prevents a lost tool result from pinning an executor thread forever.
_UNBOUNDED_TOOL_WAIT_CAP_SECONDS = 3600.0
_SESSION_SWEEP_INTERVAL_SECONDS = 60.0
_FINISHED_SESSION_TTL_SECONDS = 120.0

_FAILURE_REASON_CODES = {
    "billing": "insufficient_credits",
    "content_policy_blocked": "content_policy_blocked",
    "format_error": "model_incompatible",
    "multimodal_tool_content_unsupported": "model_incompatible",
    "timeout": "provider_timeout",
    "overloaded": "provider_unavailable",
    "rate_limit": "provider_unavailable",
    "server_error": "provider_unavailable",
}


def _runtime_failure_code(result: Any) -> str:
    if not isinstance(result, dict):
        return "runtime_unavailable"
    if result.get("turn_exit_reason") in {
        "empty_response_exhausted",
        "all_retries_exhausted_no_response",
    }:
        return "provider_empty_stream"
    reason = str(result.get("failure_reason") or "").strip().lower()
    if reason in _FAILURE_REASON_CODES:
        return _FAILURE_REASON_CODES[reason]
    error = str(result.get("error") or "").strip().lower()
    if "insufficient balance" in error or "insufficient credit" in error or "http 402" in error:
        return "insufficient_credits"
    if error.startswith("content_policy_blocked:"):
        return "content_policy_blocked"
    return "runtime_unavailable"


def _runtime_llm_egress(value: Any, *, required: bool) -> dict[str, str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict) or set(value) != {"base_url", "grant", "expires_at"}:
        raise ValueError("llm_egress must contain base_url, grant, and expires_at")
    base_url = str(value.get("base_url") or "").strip().rstrip("/")
    grant = str(value.get("grant") or "").strip()
    expires_at = str(value.get("expires_at") or "").strip()
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/internal/llm/v1"
    ):
        raise ValueError("llm_egress.base_url is invalid")
    if not re.fullmatch(r"ueg_[A-Za-z0-9_-]{43}", grant):
        raise ValueError("llm_egress.grant is invalid")
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("llm_egress.expires_at is invalid") from exc
    if expiry.tzinfo is None or expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("llm_egress grant is expired")
    return {"base_url": base_url, "grant": grant, "expires_at": expires_at}


def _runtime_vision_llm_egress(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "model", "base_url", "grant", "expires_at",
    }:
        raise ValueError(
            "vision_llm_egress must contain model, base_url, grant, and expires_at"
        )
    model = str(value.get("model") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}", model):
        raise ValueError("vision_llm_egress.model is invalid")
    capability = _runtime_llm_egress(
        {key: value[key] for key in ("base_url", "grant", "expires_at")},
        required=True,
    )
    return {"model": model, **capability}


def _configure_run_llm_egress(agent: Any, capability: dict[str, str] | None, model: Any) -> None:
    if capability is None:
        return
    requested_model = str(model or "").strip()
    if not requested_model:
        raise ValueError("model is required")
    if (
        str(getattr(agent, "model", "") or "").strip() != requested_model
        or str(getattr(agent, "provider", "") or "").strip() != "custom"
        or str(getattr(agent, "api_key", "") or "") != capability["grant"]
        or str(getattr(agent, "base_url", "") or "").rstrip("/")
        != capability["base_url"]
    ):
        raise ValueError("agent run-scoped LLM egress configuration is inconsistent")

_RUNTIME_GATE_LOCK = threading.Lock()
_RUNTIME_EXECUTOR: ThreadPoolExecutor | None = None
_ACTIVE_RUN_COUNT = 0
_SWEEPERS: dict[int, tuple[asyncio.AbstractEventLoop, "asyncio.Task[None]"]] = {}
_NO_SKILL_MANIFEST = object()


async def _next_runtime_stream_event(
    queue: "asyncio.Queue[dict[str, Any] | None]",
) -> dict[str, Any] | None:
    """Return the next bridge event or a transport-only keepalive frame."""
    try:
        return await asyncio.wait_for(
            queue.get(),
            timeout=_RUNTIME_STREAM_HEARTBEAT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {"type": "heartbeat", "payload": {}}


def _runtime_max_concurrent() -> int:
    try:
        value = int(os.environ.get(_RUNTIME_MAX_CONCURRENT_ENV, ""))
    except ValueError:
        value = _RUNTIME_MAX_CONCURRENT_DEFAULT
    return max(1, value)


def _runtime_executor() -> ThreadPoolExecutor:
    global _RUNTIME_EXECUTOR
    with _RUNTIME_GATE_LOCK:
        if _RUNTIME_EXECUTOR is None:
            _RUNTIME_EXECUTOR = ThreadPoolExecutor(
                max_workers=_runtime_max_concurrent(),
                thread_name_prefix="runtime-bridge",
            )
        return _RUNTIME_EXECUTOR


def _acquire_run_slot() -> bool:
    global _ACTIVE_RUN_COUNT
    with _RUNTIME_GATE_LOCK:
        if _ACTIVE_RUN_COUNT >= _runtime_max_concurrent():
            return False
        _ACTIVE_RUN_COUNT += 1
        return True


def _release_run_slot() -> None:
    global _ACTIVE_RUN_COUNT
    with _RUNTIME_GATE_LOCK:
        _ACTIVE_RUN_COUNT = max(0, _ACTIVE_RUN_COUNT - 1)


def _sweep_finished_sessions(now: float | None = None) -> list[str]:
    """Evict sessions that finished but were never popped (leak backstop only).

    The normal cleanup path in _handle_runtime_run pops sessions before
    finished is set; anything still registered past the TTL leaked.
    """
    current = time.monotonic() if now is None else now
    removed: list[str] = []
    with _SESSIONS_LOCK:
        for key, session in list(_SESSIONS.items()):
            finished_at = session.finished_at
            if (
                session.finished.is_set()
                and finished_at is not None
                and current - finished_at >= _FINISHED_SESSION_TTL_SECONDS
            ):
                _SESSIONS.pop(key, None)
                removed.append(key)
    for key in removed:
        logger.warning("Runtime bridge sweeper evicted orphaned session %s", key)
    return removed


async def _session_sweeper_loop() -> None:
    while True:
        await asyncio.sleep(_SESSION_SWEEP_INTERVAL_SECONDS)
        _sweep_finished_sessions()


def _ensure_session_sweeper() -> None:
    loop = asyncio.get_running_loop()
    for key, (known_loop, task) in list(_SWEEPERS.items()):
        if known_loop.is_closed() or (known_loop is loop and task.done()):
            _SWEEPERS.pop(key, None)
    entry = _SWEEPERS.get(id(loop))
    if entry is not None and entry[0] is loop and not entry[1].done():
        return
    _SWEEPERS[id(loop)] = (loop, loop.create_task(_session_sweeper_loop()))


def _pin_run_model(agent: Any, requested_model: Any) -> str:
    """Lock one Runtime run to its requested model with no fallback chain."""
    pinned_model = str(requested_model or getattr(agent, "model", "")).strip()
    if not pinned_model:
        raise ValueError("model is required")

    current_model = str(getattr(agent, "model", "") or "").strip()
    switch_model = getattr(agent, "switch_model", None)
    if pinned_model != current_model and callable(switch_model):
        switch_model(
            pinned_model,
            str(getattr(agent, "provider", "") or ""),
            getattr(agent, "api_key", ""),
            str(getattr(agent, "base_url", "") or ""),
            str(getattr(agent, "api_mode", "") or ""),
        )
    else:
        agent.model = pinned_model

    primary_runtime = getattr(agent, "_primary_runtime", None)
    if isinstance(primary_runtime, dict):
        primary_runtime["model"] = pinned_model
        if "compressor_model" in primary_runtime:
            primary_runtime["compressor_model"] = pinned_model

    agent._run_model_pin = pinned_model
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._fallback_index = 0
    agent._fallback_activated = False
    return pinned_model


def _activity_arguments(tool_name: str, args: Any) -> dict[str, str]:
    if not isinstance(args, dict):
        return {}
    allowed = {
        "skill_view": ("name", "file_path"),
        "tool_search": ("query",),
    }
    return {
        key: str(args[key])
        for key in allowed.get(tool_name, ())
        if isinstance(args.get(key), str) and str(args[key]).strip()
    }


def _skill_body_digest(args: Any, result: Any) -> str:
    """Digest of a successful SKILL.md body load; "" for sub-file reads or failures."""
    if not isinstance(args, dict) or str(args.get("file_path") or "").strip():
        return ""
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return ""
    if not isinstance(parsed, dict):
        return ""
    if parsed.get("success") is False or ("error" in parsed and parsed.get("success") is not True):
        return ""
    content = parsed.get("content")
    if not isinstance(content, str):
        return ""
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()


def _activity_failure_message(result: Any) -> str:
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return ""
    if not isinstance(parsed, dict) or parsed.get("success") is not False:
        return ""
    message = parsed.get("error")
    if not isinstance(message, str) or not message.strip():
        return "runtime activity failed"
    return message.strip()[:240]


def _invalid_failed_tool_result() -> dict[str, Any]:
    return {
        "error": {
            "code": "invalid_tool_result",
            "message": "tool failed with an invalid result envelope",
            "retryable": False,
        },
    }


def _failed_allowed_value_is_safe(key: str, value: Any) -> bool:
    if key in _FAILED_ALLOWED_STRING_FIELDS:
        return isinstance(value, str) and bool(value.strip()) and len(value) <= 512
    if key in _FAILED_ALLOWED_INTEGER_FIELDS:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if not isinstance(value, list) or len(value) > 64:
        return False
    if key in _FAILED_ALLOWED_STRING_LIST_FIELDS:
        return all(
            isinstance(item, str) and bool(item.strip()) and len(item) <= 128
            for item in value
        )
    return all(
        isinstance(item, int) and not isinstance(item, bool) and item > 0
        for item in value
    )


def _failed_tool_result_projection(transport: Any) -> dict[str, Any]:
    """Project a failed platform result without exposing arbitrary fields."""
    if (
        not isinstance(transport, dict)
        or transport.get("ok") is not False
        or set(transport) - {"call_id", "ok", "result", "error"}
    ):
        return _invalid_failed_tool_result()
    error = transport.get("error")
    if not isinstance(error, dict) or set(error) != _FAILED_ERROR_FIELDS:
        return _invalid_failed_tool_result()
    code = error.get("code")
    message = error.get("message")
    retryable = error.get("retryable")
    if (
        not isinstance(code, str)
        or not code.strip()
        or len(code) > 128
        or not isinstance(message, str)
        or not message.strip()
        or len(message) > 2_000
        or not isinstance(retryable, bool)
    ):
        return _invalid_failed_tool_result()
    projection: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        },
    }
    if "result" not in transport:
        return projection
    result = transport.get("result")
    if not isinstance(result, dict) or set(result) != {"allowed"}:
        return _invalid_failed_tool_result()
    allowed = result.get("allowed")
    if (
        not isinstance(allowed, dict)
        or set(allowed) - _FAILED_ALLOWED_FIELDS
        or any(not _failed_allowed_value_is_safe(key, value) for key, value in allowed.items())
    ):
        return _invalid_failed_tool_result()
    projection["result"] = {"allowed": dict(allowed)}
    return projection


def _model_parameter_contract(
    result: Any,
) -> tuple[str, str, dict[str, dict[str, Any]]] | None:
    """Extract one exact model contract from Runtime-private control data."""
    if not isinstance(result, dict):
        return None
    if set(result) != {"model", "parameters", "observed_schema_digest"}:
        return None
    model = str(result.get("model") or "").strip()
    digest = str(result.get("observed_schema_digest") or "").strip()
    if not model or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        return None
    parameters = result.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        return None
    contract: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return None
        if set(parameter) - {
            "name", "type", "required", "default", "options", "description"
        }:
            return None
        name = str(parameter.get("name") or "").strip()
        parameter_type = str(parameter.get("type") or "").strip()
        if not name or not parameter_type or name in contract:
            return None
        contract[name] = {
            "type": parameter_type,
            "required": parameter.get("required") is True,
            "options": list(parameter.get("options") or []),
            "description": str(parameter.get("description") or ""),
        }
    return model, digest, contract


def _parameter_value_matches_type(value: Any, parameter_type: str) -> bool:
    parameter_type = parameter_type.lower()
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if parameter_type == "boolean":
        return isinstance(value, bool)
    if parameter_type == "array":
        return isinstance(value, list)
    if parameter_type == "object":
        return isinstance(value, dict)
    return False


def _arbitrary_image_size_is_valid(value: Any, description: str) -> bool:
    if not isinstance(value, str) or "arbitrary" not in description.lower():
        return False
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value.strip().lower())
    if match is None:
        return False
    width, height = int(match.group(1)), int(match.group(2))
    return (
        width % 16 == 0
        and height % 16 == 0
        and 1 / 3 <= width / height <= 3
        and max(width, height) <= 3840
        and width * height <= 3840 * 2160
    )


def _model_request_contract_error(
    tool_name: str,
    args: Any,
    contracts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    """Validate media requests against contracts fetched for their exact models."""
    if not _requires_model_parameter_contract(tool_name):
        return None
    requests = args.get("requests") if isinstance(args, dict) else None
    if not isinstance(requests, list) or not requests:
        return None  # The platform tool schema owns the generic envelope error.
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            continue
        model = str(request.get("model") or "").strip()
        if not model:
            continue
        contract = contracts.get(model)
        if contract is None:
            return {
                "error": {
                    "code": "model_schema_required",
                    "message": (
                        f"Runtime did not resolve the exact input contract for model "
                        f"{model!r} before calling {tool_name}."
                    ),
                    "retryable": True,
                }
            }
        supplied = set(request) - _MODEL_SCHEMA_ENVELOPE_FIELDS
        unknown = sorted(supplied - set(contract))
        if unknown:
            allowed_parts = []
            for parameter_name in sorted(contract):
                options = contract[parameter_name]["options"]
                if options:
                    allowed_parts.append(f"{parameter_name}={options}")
                else:
                    allowed_parts.append(parameter_name)
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": (
                        f"requests[{index}] contains parameters not declared by model "
                        f"{model!r}: {', '.join(unknown)}. Allowed literal parameters: "
                        f"{'; '.join(allowed_parts)}"
                    ),
                    "retryable": False,
                }
            }
        missing = sorted(
            name for name, parameter in contract.items()
            if parameter["required"] and name not in request
        )
        if missing:
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": f"requests[{index}] is missing required model parameters: {', '.join(missing)}",
                    "retryable": False,
                }
            }
        for name in sorted(supplied):
            parameter = contract[name]
            value = request[name]
            if not _parameter_value_matches_type(value, parameter["type"]):
                return {
                    "error": {
                        "code": "invalid_tool_arguments",
                        "message": (
                            f"requests[{index}].{name} must have model-declared type "
                            f"{parameter['type']}"
                        ),
                        "retryable": False,
                    }
                }
            options = parameter["options"]
            if options and value not in options and not (
                name == "size"
                and _arbitrary_image_size_is_valid(value, parameter["description"])
            ):
                return {
                    "error": {
                        "code": "invalid_tool_arguments",
                        "message": (
                            f"requests[{index}].{name} is not allowed by model {model!r}; "
                            f"use one of {options}"
                        ),
                        "retryable": False,
                    }
                }
    return None


def _skill_scope_error(name: str) -> str:
    return json.dumps(
        {
            "success": False,
            "error": f"Skill '{name}' is not available for this run.",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _runtime_allowed_skill_names(
    skill_manifest: Any = _NO_SKILL_MANIFEST,
) -> set[str]:
    """Resolve the immutable Run Skill scope from the user snapshot only."""
    return set(_runtime_allowed_skill_digests(skill_manifest))


def _runtime_allowed_skill_digests(
    skill_manifest: Any = _NO_SKILL_MANIFEST,
) -> dict[str, str]:
    """Resolve Runtime aliases to the immutable package digests they prove."""
    if skill_manifest is _NO_SKILL_MANIFEST:
        return {}
    if not isinstance(skill_manifest, dict):
        raise ValueError("skill_manifest must be an object")
    skills = skill_manifest.get("skills")
    if not isinstance(skills, list):
        raise ValueError("skill_manifest.skills must be an array")
    digests: dict[str, str] = {}
    for skill in skills:
        if not isinstance(skill, dict):
            raise ValueError("skill_manifest.skills must contain only objects")
        name = skill.get("runtime_alias")
        if (
            not isinstance(name, str)
            or not name.strip()
            or len(name.strip()) > 255
        ):
            raise ValueError("skill_manifest runtime_alias is invalid")
        name = name.strip()
        if name in digests:
            raise ValueError("skill_manifest contains duplicate runtime_alias")
        digest = skill.get("content_digest")
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"skill_manifest content_digest is invalid for {name}")
        digests[name] = digest
    return digests


def _runtime_skill_projections(skill_manifest: Any) -> dict[str, RuntimeSkillProjection]:
    return resolve_skill_projections(skill_manifest, _NO_SKILL_MANIFEST)


def _allowed_skills_prompt(
    allowed_names: set[str],
    projections: dict[str, RuntimeSkillProjection] | list[dict[str, Any]],
) -> str:
    from gateway.ultrastudio_skill_routing import format_allowed_skills

    metadata = (
        projection_skill_metadata(projections)
        if isinstance(projections, dict)
        else projections
    )
    return format_allowed_skills(
        allowed_names,
        metadata,
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


_RUNTIME_IMAGE_MIME_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}
_RUNTIME_VIDEO_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
}
_MAX_RUNTIME_IMAGE_BYTES = 20 << 20
_MAX_RUNTIME_VIDEO_BYTES = 50 << 20
_RUNTIME_VIDEO_SUFFIXES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
}
_RUNTIME_IMAGE_SUFFIXES = {
    "image/gif": ".gif",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _runtime_attachment_parts(
    attachments: Any,
    *,
    image_dir: str | os.PathLike[str] | None = None,
    video_dir: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    if attachments in (None, []):
        return []
    if not isinstance(attachments, list) or len(attachments) > 8:
        raise ValueError("attachments must be an array of at most 8 items")
    parts: list[dict[str, Any]] = []
    total_bytes = 0
    for item in attachments:
        if not isinstance(item, dict):
            raise ValueError("attachments must contain only objects")
        asset_id = str(item.get("asset_id") or "").strip()
        reference_id = str(item.get("reference_id") or asset_id).strip()
        role = str(item.get("role") or "").strip()
        filename = str(item.get("filename") or "").strip()
        media_type = str(item.get("media_type") or "").strip()
        mime_type = str(item.get("mime_type") or "").strip().lower()
        encoded = item.get("data")
        if (
            not reference_id
            or (asset_id and item.get("reference_id") and reference_id != asset_id)
            or not role
            or not filename
            or not isinstance(encoded, str)
        ):
            raise ValueError("attachment identity, role, filename, and data are required")

        identity_label = "asset_id" if asset_id else "reference_id"
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("attachment data must be valid base64") from exc
        total_bytes += len(data)
        if total_bytes > MAX_RUNTIME_ATTACHMENT_BYTES:
            raise ValueError("runtime attachments exceed the 64 MiB total limit")
        if media_type == "image":
            if mime_type not in _RUNTIME_IMAGE_MIME_TYPES or not data or len(data) > _MAX_RUNTIME_IMAGE_BYTES:
                raise ValueError("runtime image attachment is invalid or too large")
            if image_dir is None:
                raise ValueError("runtime image materialization directory is required")
            directory = Path(image_dir).resolve()
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            image_path = directory / (
                hashlib.sha256(reference_id.encode("utf-8")).hexdigest()[:24]
                + _RUNTIME_IMAGE_SUFFIXES[mime_type]
            )
            image_path.write_bytes(data)
            image_path.chmod(0o600)
            parts.append({
                "type": "text",
                "text": (
                    f"[Attached image: {filename}; role={role}; "
                    f"{identity_label}={reference_id}. "
                    "When pixel analysis is required, call image_analyze with "
                    f"image_url={image_path}. Keep this private runtime path out "
                    "of the final answer.]"
                ),
                "_runtime_reference_id": reference_id,
                "_runtime_image_path": str(image_path),
            })
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            })
            continue
        if media_type == "video":
            if mime_type not in _RUNTIME_VIDEO_MIME_TYPES or not data or len(data) > _MAX_RUNTIME_VIDEO_BYTES:
                raise ValueError("runtime video attachment is invalid or too large")
            if video_dir is None:
                raise ValueError("runtime video materialization directory is required")
            directory = Path(video_dir).resolve()
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            video_path = directory / (
                hashlib.sha256(reference_id.encode("utf-8")).hexdigest()[:24]
                + _RUNTIME_VIDEO_SUFFIXES[mime_type]
            )
            video_path.write_bytes(data)
            video_path.chmod(0o600)
            parts.append({
                "type": "text",
                "text": (
                    f"[Attached video: {filename}; role={role}; "
                    f"{identity_label}={reference_id}. "
                    "Analyze the complete source video with video_analyze using "
                    f"video_url={video_path} and include_transcript=true. "
                    "Representative frames, when present, "
                    "are supplementary rather than the source of truth.]"
                ),
                "_runtime_reference_id": reference_id,
                "_runtime_video_path": str(video_path),
            })
            continue
        raise ValueError("runtime attachment media_type must be image or video")
    return parts


def _runtime_image_paths(parts: list[dict[str, Any]]) -> list[Path]:
    return [
        Path(str(part["_runtime_image_path"])).resolve()
        for part in parts
        if isinstance(part, dict) and part.get("_runtime_image_path")
    ]


def _runtime_video_paths(parts: list[dict[str, Any]]) -> list[Path]:
    return [
        Path(str(part["_runtime_video_path"])).resolve()
        for part in parts
        if isinstance(part, dict) and part.get("_runtime_video_path")
    ]


def _runtime_reference_paths(
    parts: list[dict[str, Any]],
    path_key: str,
) -> dict[str, str]:
    references: dict[str, str] = {}
    for part in parts:
        if not isinstance(part, dict):
            continue
        reference_id = str(part.get("_runtime_reference_id") or "").strip()
        runtime_path = str(part.get(path_key) or "").strip()
        if reference_id and runtime_path:
            references[reference_id] = str(Path(runtime_path).resolve())
    return references


def _runtime_image_references(parts: list[dict[str, Any]]) -> dict[str, str]:
    return _runtime_reference_paths(parts, "_runtime_image_path")


def _runtime_video_references(parts: list[dict[str, Any]]) -> dict[str, str]:
    return _runtime_reference_paths(parts, "_runtime_video_path")


def _public_runtime_attachment_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in part.items() if not key.startswith("_runtime_")}
        for part in parts
    ]


def _project_runtime_resume_attachments(
    history: list[dict[str, Any]],
    attachment_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add resume media to a transient tool-message projection only."""
    if not attachment_parts:
        return history
    if not history or history[-1].get("role") != "tool":
        raise _RuntimeSessionStateError(
            "runtime_history_conflict",
            "resume attachments require a durable tool result",
            status=409,
        )
    projected = list(history)
    tail = dict(projected[-1])
    content = tail.get("content")
    if isinstance(content, list):
        base_parts = list(content)
    elif content is None:
        base_parts = []
    else:
        base_parts = [{"type": "text", "text": str(content)}]
    # Resume media is appended to a durable role=tool result.  Never place
    # pixels in that message: the selected model may accept user images while
    # rejecting multipart tool results.  Retain the asset/path reference so
    # the model can pass it to another media tool or call image_analyze.
    reference_text: list[str] = []
    for part in [
        *base_parts,
        *_public_runtime_attachment_parts(attachment_parts),
    ]:
        if isinstance(part, str):
            if part.strip():
                reference_text.append(part.strip())
            continue
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"image_url", "input_image"}:
            continue
        if part.get("type") in {"text", "input_text"}:
            text = str(part.get("text") or "").strip()
            if text:
                reference_text.append(text)
    tail["content"] = (
        "\n\n".join(reference_text)
        or "[Generated media is available through its retained asset reference.]"
    )
    projected[-1] = tail
    return projected


def _native_image_tool_definition() -> dict[str, Any]:
    # Runtime visual research is Hermes-owned: the Orchestrator supplies
    # run-scoped media authority, while Hermes owns model/provider execution.
    from tools import image_analyze as _image_analyze  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("image_analyze")
    if entry is None:
        raise RuntimeError("Hermes image_analyze tool is not registered")
    return {
        "type": "function",
        "function": {**entry.schema, "name": entry.name},
    }


def _native_video_tool_definition() -> dict[str, Any]:
    # Importing the module performs its normal registry registration. Runtime
    # video input is an explicit per-run grant, so it must not depend on the
    # process-wide default toolset containing the opt-in video tool.
    from tools import vision_tools as _vision_tools  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("video_analyze")
    if entry is None:
        raise RuntimeError("Hermes video_analyze tool is not registered")
    return {
        "type": "function",
        "function": {**entry.schema, "name": entry.name},
    }


def _tool_schemas(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, definition in enumerate(definitions):
        if not isinstance(definition, dict):
            raise ValueError(f"tools[{index}] must be an object")
        name = str(definition.get("name") or "").strip()
        parameters = definition.get("input_schema")
        if not name or name in seen or not isinstance(parameters, dict):
            raise ValueError(f"tools[{index}] has an invalid or duplicate definition")
        seen.add(name)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(definition.get("description") or ""),
                "parameters": parameters,
            },
        })
    return schemas


def _replacement_system_prompt(system_context: Any) -> str:
    if not isinstance(system_context, dict):
        raise ValueError("trusted system_context is required")
    if set(system_context) != {"version", "mode", "digest", "stable"}:
        raise ValueError("system_context contains unsupported fields")
    raw_version = str(system_context.get("version") or "")
    raw_mode = str(system_context.get("mode") or "")
    raw_digest = str(system_context.get("digest") or "")
    raw_stable = str(system_context.get("stable") or "")
    version = raw_version.strip()
    mode = raw_mode.strip()
    digest = raw_digest.strip()
    stable = raw_stable.strip()
    if not version or mode != "replace" or not digest or not stable:
        raise ValueError("trusted replacement system_context is required")
    value = f"{version}\n{mode}\n{stable}"
    expected = "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
    if digest != expected:
        logger.error(
            "runtime system_context digest mismatch "
            "received=%s expected=%s version_bytes=%d/%d mode_bytes=%d/%d "
            "stable_bytes=%d/%d",
            digest,
            expected,
            len(raw_version.encode("utf-8")),
            len(version.encode("utf-8")),
            len(raw_mode.encode("utf-8")),
            len(mode.encode("utf-8")),
            len(raw_stable.encode("utf-8")),
            len(stable.encode("utf-8")),
        )
        raise ValueError("system_context digest mismatch")
    return stable


def _run_state_prompt(run_state: Any) -> str:
    """Render the platform-derived run state as an authenticated instructions block.

    run_state is platform data derived from the orchestrator event log, not
    user content; it is appended verbatim (compact JSON) without model-side
    interpretation. Resume and first start are treated identically.
    """
    if run_state is None:
        return ""
    if not isinstance(run_state, dict):
        raise ValueError("run_state must be an object")
    if not run_state:
        return ""
    return (
        "\n\n[RUN STATE — platform-authenticated, read-only]\n"
        + json.dumps(run_state, ensure_ascii=False, separators=(",", ":"))
    )


_RUNTIME_MESSAGE_ROLES = frozenset({"user", "assistant", "tool"})
_RUNTIME_MESSAGE_MAX_CHARS = 4 << 20
_RUNTIME_ACTIVITY_MAX_ITEMS = 128
_RUNTIME_REFERENCE_MAX_ROLES = 32
_RUNTIME_REFERENCE_MAX_IDS_PER_ROLE = 64
_RUNTIME_ARTIFACT_MANIFEST_MAX_ITEMS = 32
_RUNTIME_ARTIFACT_FIELDS = frozenset({
    "tool_call_id",
    "asset_id",
    "media_type",
    "role",
    "request_index",
    "output_index",
    "source_run_id",
    "event_seq",
    "created_at",
})
_RUNTIME_ARTIFACT_REQUIRED_FIELDS = frozenset({
    "tool_call_id",
    "asset_id",
    "media_type",
    "role",
    "source_run_id",
    "event_seq",
    "created_at",
})
_RUNTIME_ARTIFACT_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})"
)


def _validate_runtime_content(content: Any, role: str, index: int) -> Any:
    if content is None:
        if role != "assistant":
            raise ValueError(f"messages[{index}].content is invalid")
        return None
    if isinstance(content, str):
        if len(content) > _RUNTIME_MESSAGE_MAX_CHARS:
            raise ValueError(f"messages[{index}].content is too large")
        return content
    if not isinstance(content, list) or len(content) > 64:
        raise ValueError(f"messages[{index}].content is invalid")
    normalized: list[dict[str, Any]] = []
    for part_index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(f"messages[{index}].content[{part_index}] is invalid")
        part_type = str(part.get("type") or "").strip()
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str) or len(text) > _RUNTIME_MESSAGE_MAX_CHARS:
                raise ValueError(f"messages[{index}].content[{part_index}] text is invalid")
            normalized.append({"type": "text", "text": text})
        elif part_type in {"image_url", "input_image"}:
            image = part.get("image_url") or part.get("image")
            if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                raise ValueError(f"messages[{index}].content[{part_index}] image is invalid")
            normalized.append({part_type: dict(image), "type": part_type})
        else:
            raise ValueError(f"messages[{index}].content[{part_index}] type is invalid")
    if role == "user" and not normalized:
        raise ValueError(f"messages[{index}].content is empty")
    return normalized


def _normalize_runtime_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be an array")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{index}] must be an object")
        wire_id = item.get("id")
        if not isinstance(wire_id, str) or not wire_id.strip():
            raise ValueError(f"messages[{index}].id must be a non-empty string")
        wire_id = wire_id.strip()
        if len(wire_id) > 512 or wire_id in seen_ids:
            raise ValueError(f"messages[{index}].id must be unique and bounded")
        role = item.get("role")
        if not isinstance(role, str) or role not in _RUNTIME_MESSAGE_ROLES:
            raise ValueError(f"messages[{index}].role is invalid")
        if "content" not in item:
            raise ValueError(f"messages[{index}].content is required")
        seen_ids.add(wire_id)
        message = {
            "message_id": wire_id,
            "platform_message_id": wire_id,
            "role": role,
            "content": _validate_runtime_content(item.get("content"), role, index),
        }
        if "tool_calls" in item:
            calls = item.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"messages[{index}].tool_calls is invalid")
            normalized_calls: list[dict[str, Any]] = []
            call_ids: set[str] = set()
            for call_index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise ValueError(f"messages[{index}].tool_calls[{call_index}] is invalid")
                call_id = call.get("id")
                function = call.get("function")
                if (
                    not isinstance(call_id, str)
                    or not call_id.strip()
                    or call_id in call_ids
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"].strip()
                    or not isinstance(function.get("arguments"), (str, dict, list))
                ):
                    raise ValueError(f"messages[{index}].tool_calls[{call_index}] is invalid")
                call_id = call_id.strip()
                call_ids.add(call_id)
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                normalized_calls.append({
                    "id": call_id,
                    "type": str(call.get("type") or "function"),
                    "function": {
                        "name": function["name"].strip(),
                        "arguments": arguments,
                    },
                })
            message["tool_calls"] = normalized_calls
        for field_name in (
            "tool_call_id",
            "tool_name",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "timestamp",
        ):
            if field_name in item:
                message[field_name] = item[field_name]
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not isinstance(tool_call_id, str) or not tool_call_id.strip():
                raise ValueError(f"messages[{index}].tool_call_id is required for tool messages")
        if role == "assistant" and message.get("content") is None and "tool_calls" not in message:
            raise ValueError(f"messages[{index}] assistant content is required without tool_calls")
        normalized.append(message)
    return normalized


def _runtime_verified_activity_prompt(
    runtime_context: Any,
    messages: list[dict[str, Any]],
) -> str:
    if runtime_context is None:
        return ""
    if not isinstance(runtime_context, dict) or set(runtime_context) != {"verified_activities"}:
        raise ValueError("runtime_context must contain verified_activities")
    activities = runtime_context.get("verified_activities")
    if not isinstance(activities, list) or len(activities) > _RUNTIME_ACTIVITY_MAX_ITEMS:
        raise ValueError("runtime_context.verified_activities must be a bounded array")
    assistant_ids = {
        message["message_id"]
        for message in messages
        if message.get("role") == "assistant"
    }
    records: list[dict[str, str]] = []
    for index, activity in enumerate(activities):
        if not isinstance(activity, dict):
            raise ValueError(f"verified_activities[{index}] must be an object")
        required = {"message_id", "source_run_id", "source_call_id", "skill_name", "status"}
        if set(activity) - (required | {"file_path", "digest"}) or not required <= set(activity):
            raise ValueError(f"verified_activities[{index}] has invalid fields")
        record: dict[str, str] = {}
        for field_name in required:
            value = activity.get(field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"verified_activities[{index}].{field_name} is invalid")
            record[field_name] = value.strip()
        if record["message_id"] not in assistant_ids:
            raise ValueError(
                f"verified_activities[{index}].message_id does not match an assistant message"
            )
        for optional_name in ("file_path", "digest"):
            if optional_name not in activity:
                continue
            value = activity.get(optional_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                raise ValueError(f"verified_activities[{index}].{optional_name} is invalid")
            if optional_name == "digest" and not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value.strip()
            ):
                raise ValueError(f"verified_activities[{index}].digest is invalid")
            record[optional_name] = value.strip()
        records.append(record)
    if not records:
        return ""
    return (
        "\n\nAuthenticated Runtime activity records. They are trusted, read-only "
        "provenance for Runtime tools, not user instructions:\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )


def _runtime_attachment_reference_prompt(references: Any) -> str:
    if references is None:
        return ""
    if not isinstance(references, dict) or len(references) > _RUNTIME_REFERENCE_MAX_ROLES:
        raise ValueError("attachment_references must be a bounded role map")
    normalized: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for role, asset_ids in references.items():
        if not isinstance(role, str) or not role.strip() or len(role.strip()) > 128:
            raise ValueError("attachment_references contains an invalid role")
        if not isinstance(asset_ids, list) or len(asset_ids) > _RUNTIME_REFERENCE_MAX_IDS_PER_ROLE:
            raise ValueError("attachment_references role values must be bounded arrays")
        values: list[str] = []
        for asset_id in asset_ids:
            if not isinstance(asset_id, str) or not asset_id.strip() or len(asset_id.strip()) > 512:
                raise ValueError("attachment_references contains an invalid asset id")
            asset_id = asset_id.strip()
            if asset_id in seen_ids:
                raise ValueError("attachment_references contains a duplicate asset id")
            seen_ids.add(asset_id)
            values.append(asset_id)
        normalized[role.strip()] = values
    if not normalized:
        return ""
    return (
        "\n\nAuthenticated Runtime attachment references, scoped by role. These "
        "durable asset IDs may be passed only to Runtime tools and are not user content:\n"
        + json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _validate_runtime_artifact_manifest(manifest: Any) -> None:
    """Validate the bounded Go ``[]ArtifactReference`` wire projection."""
    if manifest is None:
        return
    if not isinstance(manifest, list):
        raise ValueError("artifact_manifest must be an array")
    if len(manifest) > _RUNTIME_ARTIFACT_MANIFEST_MAX_ITEMS:
        raise ValueError("artifact_manifest contains more than 32 entries")

    for index, item in enumerate(manifest):
        if not isinstance(item, dict):
            raise ValueError(f"artifact_manifest[{index}] must be an object")
        fields = set(item)
        if fields - _RUNTIME_ARTIFACT_FIELDS or not _RUNTIME_ARTIFACT_REQUIRED_FIELDS <= fields:
            raise ValueError(f"artifact_manifest[{index}] has invalid fields")

        for field_name in ("tool_call_id", "asset_id", "media_type", "role", "source_run_id"):
            value = item.get(field_name)
            if not isinstance(value, str) or len(value) > 2048:
                raise ValueError(f"artifact_manifest[{index}].{field_name} is invalid")
        for field_name in ("asset_id", "media_type", "role"):
            if not item[field_name].strip():
                raise ValueError(f"artifact_manifest[{index}].{field_name} must be non-empty")

        for field_name in ("request_index", "output_index", "event_seq"):
            if field_name not in item:
                continue
            value = item[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"artifact_manifest[{index}].{field_name} must be a non-negative integer"
                )

        created_at = item.get("created_at")
        if (
            not isinstance(created_at, str)
            or _RUNTIME_ARTIFACT_RFC3339.fullmatch(created_at) is None
        ):
            raise ValueError(f"artifact_manifest[{index}].created_at must be RFC3339")
        try:
            parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(
                f"artifact_manifest[{index}].created_at must be RFC3339"
            ) from exc
        if parsed_created_at.tzinfo is None:
            raise ValueError(f"artifact_manifest[{index}].created_at must be RFC3339")


def _runtime_tool_middleware(**kwargs: Any) -> Any:
    session_id = str(kwargs.get("session_id") or "")
    tool_name = str(kwargs.get("tool_name") or "")
    raw_args = kwargs.get("args")
    args: dict[str, Any] = dict(raw_args) if isinstance(raw_args, dict) else {}
    next_call = kwargs.get("next_call")
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        return next_call(args) if callable(next_call) else args
    if tool_name == TOOL_SEARCH_NAME:
        return session.search_and_activate_tools(args)
    if session.tool_exposure.is_callable(tool_name):
        return session.invoke_platform_tool(
            tool_name,
            args,
            str(kwargs.get("tool_call_id") or ""),
        )
    if tool_name in session.tool_names:
        return json.dumps({
            "error": {
                "code": "tool_not_loaded",
                "message": (
                    f"Tool '{tool_name}' is available to this Run but must be "
                    "loaded with tool_search before it can be called."
                ),
                "retryable": False,
            },
        }, ensure_ascii=False, separators=(",", ":"))
    if tool_name == "skill_view":
        requested = str(args.get("name") or args.get("skill") or "").strip()
        if not session.is_skill_allowed(requested):
            return _skill_scope_error(requested)
        projection = session.allowed_skill_projections.get(requested)
        if projection is not None:
            return view_skill(requested, projection, args)
    if tool_name == "image_analyze":
        args = _rewrite_runtime_media_references(
            args,
            ("image_url", "image_paths"),
            session.allowed_image_references,
        )
        for source in _image_analysis_sources(args):
            parsed = urlparse(source)
            if parsed.scheme.lower() in {"http", "https"}:
                continue
            if parsed.scheme.lower() == "file":
                if parsed.netloc not in {"", "localhost"}:
                    return _image_analysis_scope_error()
                candidate = unquote(parsed.path)
            elif parsed.scheme:
                return _image_analysis_scope_error()
            else:
                candidate = os.path.expanduser(source)
            try:
                resolved_path = str(Path(candidate).resolve(strict=True))
            except (OSError, RuntimeError):
                return _image_analysis_scope_error()
            if resolved_path not in session.allowed_image_paths:
                return _image_analysis_scope_error()
    if tool_name == "video_analyze":
        args = _rewrite_runtime_media_references(
            args,
            ("video_url",),
            session.allowed_video_references,
        )
        return session._invoke_video_analyze(args, next_call)
    if tool_name in _RUNTIME_NATIVE_TOOLS:
        return next_call(args) if callable(next_call) else args

    # Runtime Runs are capability-scoped by the Orchestrator.  A late MCP
    # refresh, plugin hook, or registry mutation must never widen that scope
    # with process-global Hermes tools.  Fail closed even if such a tool was
    # accidentally advertised to the model, and halt the turn so it cannot
    # retry or pivot through another unscoped local execution surface.
    session._halt_tool_loop(
        tool_name,
        args,
        "runtime_tool_scope_violation",
        f"Tool '{tool_name}' is not authorized for this Runtime Run.",
        1,
    )
    return json.dumps({
        "error": {
            "code": "tool_not_allowed",
            "message": f"Tool '{tool_name}' is not authorized for this Runtime Run.",
            "retryable": False,
        },
    }, ensure_ascii=False, separators=(",", ":"))


def _rewrite_runtime_media_references(
    args: dict[str, Any],
    field_names: tuple[str, ...],
    references: dict[str, str],
) -> dict[str, Any]:
    rewritten = dict(args)
    for field_name in field_names:
        value = rewritten.get(field_name)
        if isinstance(value, str):
            rewritten[field_name] = references.get(value.strip(), value)
        elif isinstance(value, list):
            rewritten[field_name] = [
                references.get(item.strip(), item) if isinstance(item, str) else item
                for item in value
            ]
    return rewritten


def _image_analysis_sources(args: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for field_name in ("image_url", "image_paths"):
        value = args.get(field_name)
        if isinstance(value, str):
            sources.append(value.strip())
        elif isinstance(value, list):
            sources.extend(
                item.strip()
                for item in value
                if isinstance(item, str)
            )
    return sources


def _image_analysis_scope_error() -> str:
    return json.dumps({
        "success": False,
        "error": (
            "image_analyze may only read HTTP(S) images or local image "
            "attachments owned by this run."
        ),
    })


def _ensure_runtime_middleware() -> None:
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager_id = id(manager)
    with _SESSIONS_LOCK:
        if manager_id in _REGISTERED_MANAGERS:
            return
        callbacks = manager._middleware.setdefault("tool_execution", [])
        if _runtime_tool_middleware not in callbacks:
            callbacks.insert(0, _runtime_tool_middleware)
        _REGISTERED_MANAGERS.add(manager_id)


@dataclass
class _PendingTool:
    name: str = ""
    signature_key: str = ""
    ready: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None


class RuntimeBridgeSession:
    def __init__(
        self,
        run_id: str,
        loop: asyncio.AbstractEventLoop,
        queue: "asyncio.Queue[dict[str, Any] | None]",
        definitions: list[dict[str, Any]],
        deadline_ms: int,
        agent_session_id: str,
        session_db: Any = None,
        tool_exposure: RuntimeToolExposure | None = None,
        allowed_skill_names: set[str] | None = None,
        allowed_skill_projections: dict[str, RuntimeSkillProjection] | None = None,
        allowed_image_paths: set[str] | None = None,
        allowed_video_paths: set[str] | None = None,
        allowed_image_references: dict[str, str] | None = None,
        allowed_video_references: dict[str, str] | None = None,
    ) -> None:
        self.run_id = run_id
        self.agent_session_id = agent_session_id
        self.loop = loop
        self.queue = queue
        self.session_db = session_db
        self.definitions = {str(item["name"]): dict(item) for item in definitions}
        self.tool_names = set(self.definitions)
        self.tool_exposure = tool_exposure or build_runtime_tool_exposure(
            definitions,
            _tool_schemas(definitions),
        )
        self.native_tool_schemas: list[dict[str, Any]] = []
        self.allowed_skill_names = (
            set(allowed_skill_names)
            if allowed_skill_names is not None
            else set()
        )
        self.allowed_skill_projections = {
            name: projection
            for name, projection in (allowed_skill_projections or {}).items()
            if name in self.allowed_skill_names
        }
        self.allowed_image_paths = {
            str(Path(path).resolve())
            for path in (allowed_image_paths or set())
        }
        self.allowed_video_paths = {
            str(Path(path).resolve())
            for path in (allowed_video_paths or set())
        }
        self.allowed_image_references = {
            reference_id: resolved_path
            for reference_id, path in (allowed_image_references or {}).items()
            if reference_id
            and (resolved_path := str(Path(path).resolve())) in self.allowed_image_paths
        }
        self.allowed_video_references = {
            reference_id: resolved_path
            for reference_id, path in (allowed_video_references or {}).items()
            if reference_id
            and (resolved_path := str(Path(path).resolve())) in self.allowed_video_paths
        }
        self.deadline_seconds = max(0.001, deadline_ms / 1000) if deadline_ms > 0 else None
        self.local_activities: dict[str, str] = {}
        self.pending: dict[str, _PendingTool] = {}
        self.non_retryable_failures: dict[str, str] = {}
        self.native_non_retryable_failures: dict[str, str] = {}
        self.video_analyze_lock = threading.Lock()
        self.argument_correction_failures: dict[str, int] = {}
        self.model_parameter_contracts: dict[str, dict[str, dict[str, Any]]] = {}
        self.model_contract_digests: dict[str, str] = {}
        self.pending_controls: dict[str, _PendingTool] = {}
        self.agent_ref: list[Any] = [None]
        self.lock = threading.RLock()
        self.interrupted = threading.Event()
        self.interrupt_reason = ""
        self.finished = threading.Event()
        self.finished_async = asyncio.Event()
        self.finished_at: float | None = None

    def bind_agent(self, agent: Any, native_tool_schemas: list[dict[str, Any]]) -> None:
        self.agent_ref[0] = agent
        self.native_tool_schemas = [dict(schema) for schema in native_tool_schemas]
        # Deferred Tools are granted to the Run but intentionally absent from
        # the model schema until tool_search loads them. Keep that state
        # separate from valid_tool_names so an exact direct call can reach the
        # middleware's tool_not_loaded response instead of being mislabeled as
        # an unknown Tool.
        agent._runtime_deferred_tool_names = set(self.tool_exposure.deferred_names)
        self._refresh_agent_tools()

    def _refresh_agent_tools(self) -> None:
        agent = self.agent_ref[0]
        if agent is None:
            return
        with self.lock:
            agent.tools = [
                *self.native_tool_schemas,
                *self.tool_exposure.model_schemas,
            ]
            agent.valid_tool_names = {
                str((tool.get("function") or {}).get("name") or "")
                for tool in agent.tools
            }

    def search_and_activate_tools(self, args: dict[str, Any]) -> str:
        result = self.tool_exposure.search_and_activate(args)
        parsed = json.loads(result)
        if "error" not in parsed:
            self._refresh_agent_tools()
        return result

    def is_skill_allowed(self, name: str) -> bool:
        return bool(name) and name in self.allowed_skill_names

    @staticmethod
    def _tool_signature_key(name: str, args: dict[str, Any]) -> str:
        canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return name + ":" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _halt_tool_loop(self, name: str, args: dict[str, Any], code: str, message: str, count: int) -> None:
        agent = self.agent_ref[0]
        setter = getattr(agent, "_set_tool_guardrail_halt", None)
        if not callable(setter):
            return
        from agent.tool_guardrails import ToolCallSignature, ToolGuardrailDecision

        setter(ToolGuardrailDecision(
            action="halt",
            code=code,
            message=message,
            tool_name=name,
            count=count,
            signature=ToolCallSignature.from_call(name, args),
        ))

    def _invoke_video_analyze(self, args: dict[str, Any], next_call: Any) -> Any:
        """Execute at most one terminally failing video analysis per Run."""
        with self.video_analyze_lock:
            with self.lock:
                prior_code = self.native_non_retryable_failures.get("video_analyze", "")
            if prior_code:
                message = (
                    "Blocked video_analyze: an earlier call in this Run failed with "
                    f"non-retryable error {prior_code}."
                )
                self._halt_tool_loop(
                    "video_analyze",
                    args,
                    "repeated_non_retryable_tool_call",
                    message,
                    2,
                )
                return json.dumps({
                    "error": {
                        "code": "repeated_non_retryable_tool_call",
                        "message": message,
                        "retryable": False,
                    },
                }, ensure_ascii=False, separators=(",", ":"))

            requested_path = str(args.get("video_url") or "").strip()
            try:
                resolved_path = str(Path(requested_path).resolve(strict=True))
            except (OSError, RuntimeError):
                resolved_path = ""
            if resolved_path not in self.allowed_video_paths:
                result: Any = json.dumps({
                    "success": False,
                    "error": (
                        "video_analyze may only read video attachments owned by this run."
                    ),
                    "error_code": "video_analysis_scope_denied",
                    "retryable": False,
                })
            else:
                result = next_call(args) if callable(next_call) else args

            code = self._native_non_retryable_failure_code(result)
            if code:
                with self.lock:
                    self.native_non_retryable_failures["video_analyze"] = code
            return result

    @staticmethod
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

    def _ensure_model_parameter_contracts(
        self,
        tool_name: str,
        args: Any,
        parent_call_id: str,
    ) -> dict[str, Any] | None:
        if not _requires_model_parameter_contract(tool_name):
            return None
        requests = args.get("requests") if isinstance(args, dict) else None
        if not isinstance(requests, list):
            return None
        models = sorted({
            str(request.get("model") or "").strip()
            for request in requests
            if isinstance(request, dict) and str(request.get("model") or "").strip()
        })
        for model in models:
            with self.lock:
                if model in self.model_parameter_contracts:
                    continue
            digest = hashlib.sha256(
                (parent_call_id + "\x00" + model).encode("utf-8")
            ).hexdigest()
            request_id = f"contract_{digest}"
            pending = _PendingTool(name="model_contract.get")
            with self.lock:
                if request_id in self.pending_controls:
                    return {
                        "error": {
                            "code": "idempotency_conflict",
                            "message": "duplicate active model schema resolution",
                            "retryable": False,
                        }
                    }
                self.pending_controls[request_id] = pending
            self.emit("runtime_control_request", {
                "request_id": request_id,
                "kind": "model_contract.get",
                "model": model,
            })
            wait_timeout = (
                self.deadline_seconds
                if self.deadline_seconds is not None
                else _UNBOUNDED_TOOL_WAIT_CAP_SECONDS
            )
            ready = pending.ready.wait(wait_timeout)
            if not ready or self.interrupted.is_set():
                with self.lock:
                    self.pending_controls.pop(request_id, None)
                return {
                    "error": {
                        "code": "model_schema_unavailable",
                        "message": f"Runtime could not resolve the exact contract for model {model!r}",
                        "retryable": False,
                    }
                }
            result = pending.result or {}
            model_contract = (
                _model_parameter_contract(result.get("result"))
                if result.get("ok")
                else None
            )
            if model_contract is None or model_contract[0] != model:
                return {
                    "error": {
                        "code": "model_schema_unavailable",
                        "message": f"Catalog did not return a usable exact contract for model {model!r}",
                        "retryable": False,
                    }
                }
            with self.lock:
                self.model_contract_digests[model] = model_contract[1]
                self.model_parameter_contracts[model] = model_contract[2]
        return None

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"run_id": self.run_id, "type": event_type, "payload": payload}
        try:
            if asyncio.get_running_loop() is self.loop:
                self.queue.put_nowait(event)
                return
        except RuntimeError:
            pass
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)

    def loaded_skill_digest(self, name: str, args: Any, result: Any) -> str:
        """Return the digest of the root SKILL.md bytes actually loaded."""
        body_digest = _skill_body_digest(args, result)
        if not body_digest or not self.is_skill_allowed(name):
            return ""
        return body_digest

    def start_local_activity(self, call_id: str, name: str, args: Any) -> None:
        if not call_id or name not in _LOCAL_ACTIVITY_TOOLS:
            return
        with self.lock:
            if call_id in self.local_activities:
                return
            self.local_activities[call_id] = name
        payload: dict[str, Any] = {
            "call_id": call_id,
            "name": name,
        }
        arguments = _activity_arguments(name, args)
        if name not in {"image_analyze", "video_analyze"}:
            payload["arguments"] = arguments
        self.emit("activity_started", payload)

    def complete_local_activity(self, call_id: str, name: str, args: Any, result: Any) -> None:
        if not call_id:
            return
        with self.lock:
            started_name = self.local_activities.pop(call_id, "")
        if not started_name or started_name != name:
            return
        message = _activity_failure_message(result)
        payload: dict[str, Any] = {
            "call_id": call_id,
            "name": name,
            "status": "failed" if message else "completed",
        }
        if message:
            payload["error"] = {
                "code": "runtime_activity_failed",
                "message": message,
                "retryable": False,
            }
        elif name == "skill_view":
            skill_name = (
                str(args.get("name") or args.get("skill") or "").strip()
                if isinstance(args, dict)
                else ""
            )
            digest = self.loaded_skill_digest(skill_name, args, result)
            if digest:
                payload["arguments"] = {"digest": digest}
        self.emit("activity_completed", payload)

    def _assert_active_tool_call_persisted(self, call_id: str, tool_name: str) -> None:
        """Require the authoritative assistant tool call before any request."""
        loader = getattr(self.session_db, "get_messages_as_conversation", None)
        if not callable(loader):
            raise _RuntimeSessionStateError(
                "runtime_session_unavailable",
                "SessionDB cannot load the active Runtime history",
            )
        try:
            try:
                history = loader(self.agent_session_id, include_ancestors=True)
            except TypeError:
                history = loader(self.agent_session_id)
        except Exception as exc:
            raise _RuntimeSessionStateError(
                "runtime_session_unavailable",
                "failed to load the active Runtime history",
            ) from exc
        for message in history if isinstance(history, list) else []:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else None
                if (
                    isinstance(call, dict)
                    and str(call.get("id") or "") == call_id
                    and isinstance(function, dict)
                    and str(function.get("name") or "") == tool_name
                ):
                    return
        raise _RuntimeSessionStateError(
            "runtime_history_conflict",
            "active Runtime tool call is not persisted in SessionDB",
            status=409,
        )

    def invoke_platform_tool(
        self,
        name: str,
        args: dict[str, Any],
        call_id: str,
    ) -> str | DeferredToolResult:
        if not call_id:
            return json.dumps({"error": {"code": "invalid_tool_request", "message": "tool call id is required"}})
        signature_key = self._tool_signature_key(name, args)
        with self.lock:
            prior_code = self.non_retryable_failures.get(signature_key, "")
        if prior_code:
            message = (
                f"Blocked unchanged retry of {name}: the previous call failed with "
                f"non-retryable error {prior_code}."
            )
            self._halt_tool_loop(name, args, "repeated_non_retryable_tool_call", message, 2)
            return json.dumps({
                "error": {
                    "code": "repeated_non_retryable_tool_call",
                    "message": message,
                    "retryable": False,
                },
            }, ensure_ascii=False, separators=(",", ":"))
        try:
            self._assert_active_tool_call_persisted(call_id, name)
        except _RuntimeSessionStateError as exc:
            message = str(exc)
            self.emit("error", {"code": exc.code, "message": message})
            self.interrupt(message)
            return json.dumps({
                "error": {
                    "code": exc.code,
                    "message": message,
                    "retryable": False,
                },
            }, ensure_ascii=False, separators=(",", ":"))
        schema_error = self._ensure_model_parameter_contracts(name, args, call_id)
        if schema_error is not None:
            return json.dumps(schema_error, ensure_ascii=False, separators=(",", ":"))
        contract_error = _model_request_contract_error(
            name,
            args,
            self.model_parameter_contracts,
        )
        if contract_error is not None:
            error = contract_error.get("error")
            if isinstance(error, dict) and error.get("code") == "invalid_tool_arguments":
                error["recovery"] = {
                    "action": "correct_arguments",
                    "same_arguments_allowed": False,
                }
                with self.lock:
                    self.non_retryable_failures[signature_key] = "invalid_tool_arguments"
            return json.dumps(contract_error, ensure_ascii=False, separators=(",", ":"))
        pending = _PendingTool(name=name, signature_key=signature_key)
        with self.lock:
            if call_id in self.pending:
                return json.dumps({"error": {"code": "idempotency_conflict", "message": "duplicate active tool call id"}})
            self.pending[call_id] = pending
        payload: dict[str, Any] = {"call_id": call_id, "name": name, "arguments": args}
        self.emit("tool_request", payload)
        wait_timeout = (
            self.deadline_seconds
            if self.deadline_seconds is not None
            else _UNBOUNDED_TOOL_WAIT_CAP_SECONDS
        )
        ready = pending.ready.wait(wait_timeout)
        if not ready or self.interrupted.is_set():
            with self.lock:
                self.pending.pop(call_id, None)
            # An interrupt wakes the same wait; attribute it before deadline.
            if self.interrupted.is_set():
                if self.interrupt_reason.startswith("parked:"):
                    return DeferredToolResult(call_id)
                code, message = "run_interrupted", "run was interrupted"
            else:
                code, message = "runtime_deadline_exceeded", "tool-result deadline exceeded"
            return json.dumps({"error": {"code": code, "message": message, "retryable": False}})
        result = pending.result or {}
        if result.get("ok"):
            with self.lock:
                self.argument_correction_failures.pop(name, None)
            return json.dumps(result.get("result"), ensure_ascii=False, separators=(",", ":"))
        failure = _failed_tool_result_projection(result)
        error = failure["error"]  # same dict; recovery added below stays in failure
        code = str(error.get("code") or "invalid_tool_result")
        if code == "invalid_tool_arguments":
            with self.lock:
                correction_count = self.argument_correction_failures.get(name, 0) + 1
                self.argument_correction_failures[name] = correction_count
            if correction_count <= _MAX_ARGUMENT_CORRECTIONS:
                error["recovery"] = {
                    "action": "correct_arguments",
                    "remaining_attempts": _MAX_ARGUMENT_CORRECTIONS - correction_count + 1,
                    "same_arguments_allowed": False,
                }
            else:
                message = (
                    f"{name} produced invalid arguments after "
                    f"{_MAX_ARGUMENT_CORRECTIONS} correction attempt."
                )
                self._halt_tool_loop(
                    name,
                    args,
                    "argument_correction_exhausted",
                    message,
                    correction_count,
                )
                failure["error"] = error = {
                    "code": "argument_correction_exhausted",
                    "message": message,
                    "retryable": False,
                    "cause": dict(error),
                }
                code = "argument_correction_exhausted"
        else:
            with self.lock:
                self.argument_correction_failures.pop(name, None)
        if error.get("retryable") is False and code != "domain_gate_required":
            with self.lock:
                self.non_retryable_failures[signature_key] = code
            if code in _TERMINAL_PLATFORM_ERROR_CODES:
                message = str(error.get("message") or f"{name} failed with {code}")
                self._halt_tool_loop(name, args, code, message, 1)
        return json.dumps(failure, ensure_ascii=False, separators=(",", ":"))

    def submit_result(self, result: dict[str, Any]) -> bool:
        call_id = str(result.get("call_id") or "")
        with self.lock:
            pending = self.pending.pop(call_id, None)
            if pending is None:
                return False
            pending.result = result
            pending.ready.set()
        return True

    def submit_control_result(self, result: dict[str, Any]) -> bool:
        request_id = str(result.get("request_id") or "")
        with self.lock:
            pending = self.pending_controls.pop(request_id, None)
            if pending is None:
                return False
            pending.result = result
            pending.ready.set()
        return True

    def interrupt(self, reason: str) -> None:
        self.interrupt_reason = reason
        self.interrupted.set()
        agent = self.agent_ref[0]
        interrupt = getattr(agent, "interrupt", None) if agent is not None else None
        if callable(interrupt):
            interrupt(reason)
        with self.lock:
            for pending in self.pending.values():
                pending.ready.set()
            for pending in self.pending_controls.values():
                pending.ready.set()

    def mark_finished(self) -> None:
        self.finished_at = time.monotonic()
        self.finished.set()
        try:
            if asyncio.get_running_loop() is self.loop:
                self.finished_async.set()
                return
        except RuntimeError:
            pass
        self.loop.call_soon_threadsafe(self.finished_async.set)


class APIServerRuntimeMixin:
    async def _run_agent_bridge(self, **kwargs: Any) -> tuple:
        """Run one bridge conversation on the dedicated bounded runtime pool.

        Mirrors APIServerRunsMixin._run_agent but swaps the default executor
        (min(32, cpu+4) threads shared with every other endpoint) for the
        bridge-owned pool sized by HERMES_RUNTIME_MAX_CONCURRENT.
        """
        loop = asyncio.get_running_loop()

        def _run() -> tuple:
            from gateway.api_agent_runner import run_agent_sync

            return run_agent_sync(self, **kwargs)

        return await loop.run_in_executor(_runtime_executor(), _run)

    async def _handle_runtime_run(self, request: "web.Request") -> "web.StreamResponse":
        auth_error = self._check_auth(request)
        if auth_error:
            return auth_error
        # Concurrency gate before response.prepare: over-limit requests are
        # rejected with a retryable 429 instead of queueing on the executor.
        if not _acquire_run_slot():
            return web.json_response(
                {
                    "error": {
                        "code": "runtime_concurrency_exceeded",
                        "message": f"too many concurrent runtime runs (max {_runtime_max_concurrent()})",
                        "retryable": True,
                    },
                },
                status=429,
                headers={"Retry-After": "1"},
            )
        try:
            return await self._handle_runtime_run_gated(request)
        finally:
            _release_run_slot()

    async def _handle_runtime_run_gated(self, request: "web.Request") -> "web.StreamResponse":
        media_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            supported_fields = {
                "intent",
                "run_id",
                "model",
                "messages",
                "system_context",
                "tools",
                "tool_results",
                "context",
                "runtime_context",
                "artifact_manifest",
                "attachment_references",
                "attachments",
                "skill_manifest",
                "run_state",
                "deadline_ms",
                "llm_egress",
                "vision_llm_egress",
                "retry_context",
                "recovery_tool_calls",
            }
            if set(body) - supported_fields:
                raise ValueError("request contains unsupported fields")
            require_llm_egress = (
                os.environ.get("HERMES_RUNTIME_REQUIRE_LLM_EGRESS", "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            llm_egress = _runtime_llm_egress(
                body.get("llm_egress"),
                required=require_llm_egress,
            )
            vision_llm_egress = _runtime_vision_llm_egress(
                body.get("vision_llm_egress")
            )
            intent = body.get("intent")
            allowed_intents = {"bootstrap", "new_turn", "resume", "retry", "rebootstrap"}
            if not isinstance(intent, str) or intent not in allowed_intents:
                raise ValueError(
                    "intent must be bootstrap, new_turn, resume, retry, or rebootstrap"
                )
            run_id = str(body.get("run_id") or "").strip()
            requested_model = str(body.get("model") or "").strip()
            if not requested_model:
                raise ValueError("model is required")
            messages = body.get("messages")
            system_context = body.get("system_context")
            definitions = body.get("tools", [])
            tool_results = body.get("tool_results", [])
            recovery_tool_calls = body.get("recovery_tool_calls", [])
            context = body.get("context")
            if not isinstance(context, dict) or set(context) != {"session_id"}:
                raise ValueError("context must contain only session_id")
            requested_agent_session_id = str(context.get("session_id") or "").strip()
            if not run_id or not requested_agent_session_id:
                raise ValueError("run_id and context.session_id are required")
            if not isinstance(definitions, list) or not isinstance(tool_results, list) or not isinstance(recovery_tool_calls, list):
                raise ValueError("tools, tool_results, and recovery_tool_calls must be arrays")
            raw_deadline_ms = body.get("deadline_ms", 0)
            if isinstance(raw_deadline_ms, bool):
                raise ValueError("deadline_ms must be a non-negative integer")
            try:
                deadline_ms = int(raw_deadline_ms or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError("deadline_ms must be a non-negative integer") from exc
            if deadline_ms < 0:
                raise ValueError("deadline_ms must be a non-negative integer")
            if intent in {"resume", "retry"}:
                if messages != []:
                    raise ValueError("resume and retry messages must be exactly empty")
            elif not isinstance(messages, list) or not messages:
                raise ValueError("bootstrap and new_turn require messages")
            normalized_messages = _normalize_runtime_messages(messages)
            if intent in {"bootstrap", "rebootstrap"} and normalized_messages[-1]["role"] != "user":
                raise ValueError("bootstrap messages must end with a user message")
            if intent == "new_turn":
                if len(normalized_messages) != 1 or normalized_messages[0]["role"] != "user":
                    raise ValueError("new_turn accepts exactly one user message")
            retry_context = body.get("retry_context")
            if intent == "retry":
                if not isinstance(retry_context, dict) or set(retry_context) != {
                    "attempt",
                    "previous_error_code",
                }:
                    raise ValueError("retry requires attempt and previous_error_code")
                if (
                    isinstance(retry_context.get("attempt"), bool)
                    or not isinstance(retry_context.get("attempt"), int)
                    or retry_context["attempt"] < 2
                ):
                    raise ValueError("retry attempt must be an integer greater than one")
                previous_error_code = retry_context.get("previous_error_code")
                if not isinstance(previous_error_code, str) or not previous_error_code.strip():
                    raise ValueError("retry previous_error_code is required")
            elif retry_context is not None:
                raise ValueError("retry_context is only accepted for retry")
            if intent not in {"resume", "rebootstrap"} and tool_results:
                raise ValueError("tool_results are only accepted for resume or rebootstrap")
            if intent != "rebootstrap" and recovery_tool_calls:
                raise ValueError("recovery_tool_calls are only accepted for rebootstrap")
            if intent == "rebootstrap" and bool(tool_results) != bool(recovery_tool_calls):
                raise ValueError("rebootstrap recovery calls and results must be provided together")
            _validate_runtime_artifact_manifest(body.get("artifact_manifest"))
            # Expose the run id to the audit middleware: its completion line
            # is logged after this handler returns, so the audit trail can
            # correlate the access log with the run it served.
            request["hermes_run_id"] = run_id
            skill_manifest = (
                body["skill_manifest"]
                if "skill_manifest" in body
                else _NO_SKILL_MANIFEST
            )
            allowed_skill_digests = _runtime_allowed_skill_digests(
                skill_manifest,
            )
            allowed_skill_names = set(allowed_skill_digests)
            allowed_skill_projections = _runtime_skill_projections(skill_manifest)
            routing_metadata = projection_skill_metadata(allowed_skill_projections)
            selectable_skill_names = {
                str(item.get("name") or "").strip()
                for item in routing_metadata
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            }
            instructions = (
                _replacement_system_prompt(system_context)
                + _allowed_skills_prompt(
                    selectable_skill_names,
                    routing_metadata,
                )
                + _run_state_prompt(body.get("run_state"))
                + _runtime_verified_activity_prompt(
                    body.get("runtime_context"),
                    normalized_messages,
                )
                + _runtime_attachment_reference_prompt(
                    body.get("attachment_references"),
                )
            )
            schemas = _tool_schemas(definitions)
            tool_exposure = build_runtime_tool_exposure(definitions, schemas)
            attachments = body.get("attachments")
            has_image_attachment = any(
                isinstance(item, dict) and item.get("media_type") == "image"
                for item in (attachments if isinstance(attachments, list) else [])
            )
            has_video_attachment = any(
                isinstance(item, dict) and item.get("media_type") == "video"
                for item in (attachments if isinstance(attachments, list) else [])
            )
            if has_image_attachment or has_video_attachment:
                media_temp_dir = tempfile.TemporaryDirectory(prefix="hermes-runtime-media-")
            attachment_parts = _runtime_attachment_parts(
                attachments,
                image_dir=media_temp_dir.name if media_temp_dir else None,
                video_dir=media_temp_dir.name if media_temp_dir else None,
            )
            runtime_image_paths = _runtime_image_paths(attachment_parts)
            runtime_video_paths = _runtime_video_paths(attachment_parts)
            runtime_image_references = _runtime_image_references(attachment_parts)
            runtime_video_references = _runtime_video_references(attachment_parts)
            if attachment_parts and intent not in {"resume", "retry", "rebootstrap"}:
                last_user_index = len(normalized_messages) - 1
                if normalized_messages[last_user_index].get("role") != "user":
                    raise ValueError("attachments require a user message")
                text = _message_text(normalized_messages[last_user_index].get("content"))
                normalized_messages[last_user_index]["content"] = [
                    {"type": "text", "text": text or "[Attached media]"},
                    *_public_runtime_attachment_parts(attachment_parts),
                ]
            db, agent_session_id, session_history = _load_runtime_session_history(
                self,
                requested_agent_session_id,
                require_existing=intent in {"new_turn", "resume", "retry"},
            )
            if intent in {"bootstrap", "rebootstrap"}:
                if db.get_session(requested_agent_session_id) is not None:
                    raise _RuntimeSessionStateError(
                        "runtime_session_conflict",
                        "Runtime SessionDB session already exists",
                        status=409,
                    )
                _seed_runtime_session(
                    db,
                    requested_agent_session_id,
                    model=requested_model,
                    system_prompt=instructions,
                    messages=(
                        normalized_messages
                        if intent == "rebootstrap" and tool_results
                        else normalized_messages[:-1]
                    ),
                )
                if intent == "rebootstrap" and tool_results:
                    _seed_recovery_tool_call(
                        db,
                        requested_agent_session_id,
                        recovery_tool_calls,
                        tool_results,
                    )
                db, agent_session_id, session_history = _load_runtime_session_history(
                    self,
                    requested_agent_session_id,
                    require_existing=True,
                )
                if intent == "rebootstrap" and tool_results:
                    session_history = _resume_session_db_history(
                        db,
                        agent_session_id,
                        session_history,
                        tool_results,
                    )
                    session_history = _project_runtime_resume_attachments(
                        session_history,
                        attachment_parts,
                    )
            elif intent == "new_turn":
                current_id = normalized_messages[0]["message_id"]
                history_ids = {
                    str(message.get("message_id") or message.get("platform_message_id") or "")
                    for message in session_history
                    if isinstance(message, dict)
                }
                if current_id in history_ids or (
                    callable(getattr(db, "has_platform_message_id", None))
                    and db.has_platform_message_id(agent_session_id, current_id)
                ):
                    raise _RuntimeSessionStateError(
                        "runtime_message_id_conflict",
                        "Runtime message id already exists in SessionDB",
                        status=409,
                    )
            elif intent == "resume":
                session_history = _resume_session_db_history(
                    db,
                    agent_session_id,
                    session_history,
                    tool_results,
                )
                session_history = _project_runtime_resume_attachments(
                    session_history,
                    attachment_parts,
                )
            else:
                session_history = _retry_session_db_history(session_history)
            history = session_history
            tool_exposure.activate_names(_runtime_history_tool_names(history))
            if intent in {"resume", "retry"} or (
                intent == "rebootstrap" and tool_results
            ):
                user_message = ""
                runtime_user_message_id = None
            else:
                current = normalized_messages[-1]
                user_message = current["content"]
                runtime_user_message_id = current["message_id"]
        except _RuntimeSessionStateError as exc:
            if media_temp_dir is not None:
                media_temp_dir.cleanup()
            return web.json_response(
                {"error": {"code": exc.code, "message": str(exc)}},
                status=exc.status,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            if media_temp_dir is not None:
                media_temp_dir.cleanup()
            return web.json_response({"error": {"code": "invalid_param", "message": str(exc)}}, status=422)
        response = web.StreamResponse(status=200, headers={
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })
        await response.prepare(request)
        queue: "asyncio.Queue[dict[str, Any] | None]" = asyncio.Queue()
        session = RuntimeBridgeSession(
            run_id,
            asyncio.get_running_loop(),
            queue,
            definitions,
            deadline_ms,
            agent_session_id,
            db,
            tool_exposure=tool_exposure,
            allowed_skill_names=allowed_skill_names,
            allowed_skill_projections=allowed_skill_projections,
            allowed_image_paths={str(path) for path in runtime_image_paths},
            allowed_video_paths={str(path) for path in runtime_video_paths},
            allowed_image_references=runtime_image_references,
            allowed_video_references=runtime_video_references,
        )
        _ensure_runtime_middleware()
        _ensure_session_sweeper()
        with _SESSIONS_LOCK:
            if run_id in _SESSIONS or agent_session_id in _SESSIONS:
                await response.write(json.dumps({"run_id": run_id, "type": "error", "payload": {"code": "run_state_conflict", "message": "run already active"}}).encode() + b"\n")
                if media_temp_dir is not None:
                    media_temp_dir.cleanup()
                return response
            _SESSIONS[run_id] = session
            _SESSIONS[agent_session_id] = session

        def configure_agent(agent: Any) -> None:
            native_runtime_tools = {
                "skill_view",
                "web_search",
                "web_extract",
                "image_analyze",
            }
            if runtime_video_paths:
                native_runtime_tools.add("video_analyze")
            native = [
                tool
                for tool in (agent.tools or [])
                if tool.get("function", {}).get("name") in native_runtime_tools
            ]
            if not any(
                tool.get("function", {}).get("name") == "image_analyze"
                for tool in native
            ):
                native.append(_native_image_tool_definition())
            if runtime_video_paths and not any(
                tool.get("function", {}).get("name") == "video_analyze"
                for tool in native
            ):
                native.append(_native_video_tool_definition())
            session.bind_agent(agent, native)
            agent._session_db = db
            agent._session_db_created = True
            agent.session_id = agent_session_id
            _configure_run_llm_egress(agent, llm_egress, body.get("model"))
            if vision_llm_egress is not None:
                from agent.auxiliary_client import set_runtime_auxiliary_override

                set_runtime_auxiliary_override(
                    "vision",
                    provider="custom",
                    model=vision_llm_egress["model"],
                    base_url=vision_llm_egress["base_url"],
                    api_key=vision_llm_egress["grant"],
                )
            else:
                from agent.auxiliary_client import set_runtime_auxiliary_unavailable

                set_runtime_auxiliary_unavailable("vision")
            _pin_run_model(agent, body.get("model"))
            # The Orchestrator owns the complete per-Run Tool grant. Hermes'
            # ordinary between-turn MCP refresh rebuilds from a process-global
            # registry and must not widen this scoped snapshot.
            agent._skip_mcp_refresh = True
            agent.ephemeral_system_prompt = None
            agent._cached_system_prompt = instructions
            agent._build_system_prompt = lambda _system_message=None: instructions
            agent._resume_from_tool_results = intent == "resume" or (
                intent == "rebootstrap" and bool(tool_results)
            )
            agent._retry_current_turn = intent == "retry"
            agent._require_incremental_session_persistence = True
            # Runtime resume attaches generated media to a role=tool result.
            # Preserve references and route pixel inspection through the
            # scoped image_analyze tool; never infer tool-result compatibility
            # merely from the presence of an image attachment.
            agent._runtime_tool_result_image_mode = "attach_by_ref"
            # Runtime bridge runs park on media generation for well over the
            # default 5m prompt-cache TTL, so a resume repays the full 13-14k
            # token system prefix at uncached price. Pin the 1h tier (the
            # other value agent_init accepts) for these runs only; the global
            # default and ~/.hermes/config.yaml stay untouched.
            agent._cache_ttl = "1h"

        def on_tool_start(tool_call_id: str, function_name: str, function_args: Any) -> None:
            session.start_local_activity(tool_call_id, function_name, function_args)

        def on_tool_complete(
            tool_call_id: str,
            function_name: str,
            function_args: Any,
            function_result: Any,
        ) -> None:
            session.complete_local_activity(tool_call_id, function_name, function_args, function_result)

        async def pump() -> None:
            while True:
                event = await _next_runtime_stream_event(queue)
                if event is None:
                    return
                try:
                    await response.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
                except Exception as exc:
                    # The orchestrator went away; without an interrupt the
                    # agent keeps running and events pile into the queue.
                    logger.warning(
                        "Runtime bridge stream write failed for run %s; interrupting: %s",
                        run_id,
                        exc,
                    )
                    session.interrupt("orchestrator stream disconnected")
                    return

        pump_task = asyncio.create_task(pump())
        session.emit("run_started", {
            "runtime": "hermes",
            "system_context_version": system_context["version"],
            "system_context_mode": system_context["mode"],
            "system_context_digest": system_context["digest"],
        })
        try:
            result, usage = await self._run_agent_bridge(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=None,
                session_id=agent_session_id,
                runtime_message_id=runtime_user_message_id,
                stream_delta_callback=lambda delta: session.emit("text_delta", {"delta": delta}) if delta else None,
                tool_start_callback=on_tool_start,
                tool_complete_callback=on_tool_complete,
                agent_ref=session.agent_ref,
                agent_configurator=configure_agent,
                agent_creation_overrides=(
                    {
                        "runtime_overrides": {
                            "api_key": llm_egress["grant"],
                            "base_url": llm_egress["base_url"],
                            "provider": "custom",
                            "api_mode": "chat_completions",
                            "command": None,
                            "args": [],
                            "credential_pool": None,
                            "max_tokens": None,
                        },
                        "model_override": str(body.get("model") or "").strip(),
                    }
                    if llm_egress is not None
                    else None
                ),
            )
            text = str((result or {}).get("final_response") or "")
            session.emit("usage", usage or {})
            failed = isinstance(result, dict) and (
                bool(result.get("failed")) or result.get("completed") is False
            )
            if failed:
                session.emit(
                    "error",
                    runtime_error_envelope(
                        _runtime_failure_code(result),
                        support_id=run_id,
                    ),
                )
            else:
                session.emit("completed", {"finish_reason": "stop", "text": text})
        except asyncio.CancelledError:
            # aiohttp cancels the handler when the orchestrator disconnects;
            # the agent may keep running on its executor thread unless told
            # to stop.
            logger.warning("Runtime bridge request cancelled for run %s; interrupting", run_id)
            session.interrupt("orchestrator stream disconnected")
            raise
        except Exception as exc:
            logger.exception("Run Orchestrator runtime run failed: %s", run_id)
            session.emit(
                "error",
                runtime_error_envelope("runtime_unavailable", support_id=run_id),
            )
        finally:
            with _SESSIONS_LOCK:
                _SESSIONS.pop(run_id, None)
                _SESSIONS.pop(agent_session_id, None)
            session.mark_finished()
            queue.put_nowait(None)
            await pump_task
            if media_temp_dir is not None:
                media_temp_dir.cleanup()
        return response

    async def _handle_runtime_tool_result(self, request: "web.Request") -> "web.Response":
        auth_error = self._check_auth(request)
        if auth_error:
            return auth_error
        run_id = request.match_info["run_id"]
        request["hermes_run_id"] = run_id
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(run_id)
        if session is None:
            return web.json_response({"error": {"code": "run_not_found", "message": "run is not active"}}, status=404)
        try:
            result = await request.json()
        except Exception:
            return web.json_response({"error": {"code": "invalid_param", "message": "invalid JSON"}}, status=400)
        if not isinstance(result, dict) or not session.submit_result(result):
            return web.json_response({"error": {"code": "invalid_tool_result", "message": "unknown call_id"}}, status=409)
        return web.Response(status=204)

    async def _handle_runtime_control_result(self, request: "web.Request") -> "web.Response":
        auth_error = self._check_auth(request)
        if auth_error:
            return auth_error
        run_id = request.match_info["run_id"]
        request["hermes_run_id"] = run_id
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(run_id)
        if session is None:
            return web.json_response(
                {"error": {"code": "run_not_found", "message": "run is not active"}},
                status=404,
            )
        try:
            result = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"code": "invalid_param", "message": "invalid JSON"}},
                status=400,
            )
        if not isinstance(result, dict) or not session.submit_control_result(result):
            return web.json_response(
                {"error": {"code": "invalid_control_result", "message": "unknown request_id"}},
                status=409,
            )
        return web.Response(status=204)

    async def _handle_runtime_interrupt(self, request: "web.Request") -> "web.Response":
        auth_error = self._check_auth(request)
        if auth_error:
            return auth_error
        run_id = request.match_info["run_id"]
        request["hermes_run_id"] = run_id
        with _SESSIONS_LOCK:
            session = _SESSIONS.get(run_id)
        if session is None:
            return web.json_response({"error": {"code": "run_not_found", "message": "run is not active"}}, status=404)
        body = await request.json()
        session.interrupt(str(body.get("reason") or "interrupted by orchestrator"))
        # Wait on the asyncio-side completion event: interrupt must never
        # borrow an executor thread the runs themselves may have exhausted.
        try:
            await asyncio.wait_for(session.finished_async.wait(), 10)
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": {"code": "interrupt_timeout", "message": "runtime session did not stop"}},
                status=503,
            )
        return web.Response(status=204)
