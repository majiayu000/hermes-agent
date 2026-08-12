"""SessionDB ownership and durable Runtime tool-result helpers."""

from __future__ import annotations

import json
from typing import Any


class RuntimeSessionStateError(RuntimeError):
    """SessionDB state is unavailable or conflicts with a Runtime request."""

    def __init__(self, code: str, message: str, status: int = 503) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def runtime_history_tool_names(history: list[dict[str, Any]]) -> set[str]:
    """Return every tool name that has appeared in authoritative history."""
    names: set[str] = set()
    for message in history:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") if isinstance(call, dict) else None
            name = str(function.get("name") or "") if isinstance(function, dict) else ""
            if name:
                names.add(name)
    return names


def _tool_result_content_equal(left: Any, right: Any) -> bool:
    try:
        return json.loads(str(left)) == json.loads(str(right))
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(left or "") == str(right or "")


def _project_tool_result(
    matching_call: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Project the supported durable result into an OpenAI tool message."""
    allowed_keys = {"tool_call_id", "status", "output", "output_ref", "error"}
    if set(result) - allowed_keys:
        raise ValueError("tool_result contains unsupported fields")
    call_id = str(result.get("tool_call_id") or "").strip()
    if not call_id:
        raise ValueError("tool_result.tool_call_id is required")
    status = result.get("status")
    if not isinstance(status, str):
        raise ValueError("tool_result.status is required")
    status = status.strip()
    if status == "succeeded":
        if ("output" in result) == ("output_ref" in result):
            raise ValueError("succeeded tool_result requires exactly one output field")
        if "output_ref" in result:
            output_ref = result.get("output_ref")
            if not isinstance(output_ref, str) or not output_ref.strip():
                raise ValueError("tool_result.output_ref must be a non-empty string")
            content: Any = {
                "status": "externalized",
                "output_ref": output_ref,
            }
        else:
            content = result.get("output")
    elif status == "failed":
        if set(result) != {"tool_call_id", "status", "error"}:
            raise ValueError("failed tool_result requires only an error")
        if not isinstance(result.get("error"), dict):
            raise ValueError("tool_result.error must be an object")
        content = {"error": result["error"]}
    else:
        raise ValueError("tool_result.status must be succeeded or failed")

    function = matching_call.get("function") or {}
    tool_name = str(function.get("name") or "").strip()
    if not tool_name:
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "assistant tool call has no tool name",
            status=409,
        )
    return {
        "role": "tool",
        "name": tool_name,
        "tool_call_id": call_id,
        "content": json.dumps(content, ensure_ascii=False, separators=(",", ":")),
    }


def resume_session_db_history(
    db: Any,
    session_id: str,
    history: list[dict[str, Any]],
    tool_results: Any,
) -> list[dict[str, Any]]:
    """Persist one real result at the unfinished SessionDB tool call."""
    if (
        not isinstance(tool_results, list)
        or len(tool_results) != 1
        or not isinstance(tool_results[0], dict)
    ):
        raise ValueError("exactly one tool_result is required for Runtime resume")
    result = tool_results[0]
    call_id = str(result.get("tool_call_id") or "").strip()
    if not call_id:
        raise ValueError("tool_result.tool_call_id is required")

    assistant_index = -1
    matching_call: dict[str, Any] | None = None
    for index in range(len(history) - 1, -1, -1):
        message = history[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        matches = [
            call
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and str(call.get("id") or "") == call_id
        ]
        if matches:
            assistant_index = index
            matching_call = matches[0]
            break
    if matching_call is None:
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "tool_result does not match an assistant tool call in SessionDB",
            status=409,
        )

    projected = _project_tool_result(matching_call, result)
    later = history[assistant_index + 1 :]
    prior_results = [
        message
        for message in later
        if isinstance(message, dict)
        and message.get("role") == "tool"
        and str(message.get("tool_call_id") or "") == call_id
    ]
    if prior_results:
        if (
            len(prior_results) == 1
            and _tool_result_content_equal(
                prior_results[0].get("content"),
                projected["content"],
            )
        ):
            return history
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "conflicting tool_result already exists in SessionDB",
            status=409,
        )
    if any(
        isinstance(message, dict) and message.get("role") != "tool"
        for message in later
    ):
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "SessionDB continued past the requested tool call",
            status=409,
        )

    sibling_ids = {
        str(call.get("id") or "")
        for call in (history[assistant_index].get("tool_calls") or [])
        if isinstance(call, dict) and str(call.get("id") or "") != call_id
    }
    completed_siblings = {
        str(message.get("tool_call_id") or "")
        for message in later
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    if not sibling_ids.issubset(completed_siblings):
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "SessionDB contains more than one unfinished tool call",
            status=409,
        )

    try:
        db.append_message(
            session_id=session_id,
            role="tool",
            content=projected["content"],
            tool_name=projected["name"],
            tool_call_id=call_id,
        )
    except Exception as exc:
        raise RuntimeSessionStateError(
            "runtime_session_unavailable",
            "failed to persist resumed tool result in SessionDB",
        ) from exc
    return [*history, projected]


def retry_session_db_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate that SessionDB can continue the current logical turn.

    Provider failures occur before a completed assistant message is committed,
    so an eligible retry resumes from the existing user/tool tail without
    inserting a synthetic user message.
    """
    if not history or not isinstance(history[-1], dict):
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "same-turn retry requires existing SessionDB history",
            status=409,
        )
    if history[-1].get("role") not in {"user", "tool"}:
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "same-turn retry history must end with user or tool state",
            status=409,
        )
    return list(history)


def rebootstrap_session_db_history(
    db: Any,
    session_id: str,
    *,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    recovery_tool_calls: Any,
    tool_results: Any,
    allowed_tool_names: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Rebuild one missing Runtime session from platform-durable truth.

    A normal rebootstrap seeds the canonical prefix and leaves the final user
    message for the agent loop. A waiting-tool rebootstrap additionally
    restores the exact assistant tool call before projecting its already
    durable result. It never invokes the tool.
    """
    if not isinstance(recovery_tool_calls, list):
        raise ValueError("recovery_tool_calls must be an array")
    if not isinstance(tool_results, list):
        raise ValueError("tool_results must be an array")
    recovering_tool_result = bool(recovery_tool_calls or tool_results)
    if recovering_tool_result and (
        len(recovery_tool_calls) != 1 or len(tool_results) != 1
    ):
        raise ValueError(
            "rebootstrap recovery requires exactly one tool call and result"
        )

    seed_messages = list(messages[:-1])
    if recovering_tool_result:
        call = recovery_tool_calls[0]
        if not isinstance(call, dict) or set(call) != {
            "tool_call_id",
            "tool_name",
            "args",
        }:
            raise ValueError("recovery_tool_calls contains an invalid call")
        call_id = call.get("tool_call_id")
        tool_name = call.get("tool_name")
        args = call.get("args")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or len(call_id.strip()) > 512
        ):
            raise ValueError("recovery tool_call_id is invalid")
        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
            or len(tool_name.strip()) > 255
            or tool_name.strip() not in allowed_tool_names
        ):
            raise ValueError("recovery tool_name is not in the Run toolset")
        if not isinstance(args, dict):
            raise ValueError("recovery tool args must be an object")
        result = tool_results[0]
        if (
            not isinstance(result, dict)
            or str(result.get("tool_call_id") or "").strip() != call_id.strip()
        ):
            raise ValueError("recovery tool call and result ids must match")
        seed_messages = [
            *messages,
            {
                "message_id": f"runtime-recovery:{call_id.strip()}",
                "platform_message_id": f"runtime-recovery:{call_id.strip()}",
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id.strip(),
                        "type": "function",
                        "function": {
                            "name": tool_name.strip(),
                            "arguments": json.dumps(
                                args,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                ],
            },
        ]

    seed_runtime_session(
        db,
        session_id,
        model=model,
        system_prompt=system_prompt,
        messages=seed_messages,
    )
    try:
        history = db.get_messages_as_conversation(
            session_id,
            include_ancestors=True,
        )
        if not isinstance(history, list) or any(
            not isinstance(item, dict) for item in history
        ):
            raise RuntimeSessionStateError(
                "runtime_history_conflict",
                "SessionDB returned invalid Runtime history",
                status=409,
            )
        if recovering_tool_result:
            history = resume_session_db_history(
                db,
                session_id,
                history,
                tool_results,
            )
        return history, recovering_tool_result
    except Exception as recovery_exc:
        cleanup_exc: Exception | None = None
        delete_session = getattr(db, "delete_session", None)
        if callable(delete_session):
            try:
                delete_session(session_id)
            except Exception as exc:
                cleanup_exc = exc
        if cleanup_exc is not None:
            raise ExceptionGroup(
                "Runtime rebootstrap and cleanup both failed",
                [recovery_exc, cleanup_exc],
            ) from recovery_exc
        raise


def seed_runtime_session(
    db: Any,
    session_id: str,
    *,
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> None:
    """Create a Runtime-owned session and seed its public message prefix."""
    try:
        db.create_session(
            session_id=session_id,
            source="api_server",
            model=model,
            system_prompt=system_prompt,
        )
        for message in messages:
            db.append_message(
                session_id=session_id,
                role=message["role"],
                content=message.get("content"),
                tool_name=message.get("tool_name"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
                finish_reason=message.get("finish_reason"),
                reasoning=message.get("reasoning") if message["role"] == "assistant" else None,
                reasoning_content=(
                    message.get("reasoning_content")
                    if message["role"] == "assistant"
                    else None
                ),
                reasoning_details=(
                    message.get("reasoning_details")
                    if message["role"] == "assistant"
                    else None
                ),
                codex_reasoning_items=(
                    message.get("codex_reasoning_items")
                    if message["role"] == "assistant"
                    else None
                ),
                codex_message_items=(
                    message.get("codex_message_items")
                    if message["role"] == "assistant"
                    else None
                ),
                platform_message_id=message["message_id"],
                timestamp=message.get("timestamp"),
            )
    except Exception as seed_exc:
        cleanup_exc: Exception | None = None
        try:
            delete_session = getattr(db, "delete_session", None)
            if callable(delete_session):
                delete_session(session_id)
        except Exception as exc:
            cleanup_exc = exc
        cause: BaseException = seed_exc
        if cleanup_exc is not None:
            cause = ExceptionGroup(
                "Runtime SessionDB seed and cleanup both failed",
                [seed_exc, cleanup_exc],
            )
        raise RuntimeSessionStateError(
            "runtime_session_unavailable",
            "failed to seed Runtime SessionDB history",
        ) from cause


def load_runtime_session_history(
    adapter: Any,
    requested_session_id: str,
    *,
    require_existing: bool,
) -> tuple[Any, str, list[dict[str, Any]]]:
    """Load the authoritative history for one Runtime session."""
    try:
        db = adapter._ensure_session_db()
    except Exception as exc:
        raise RuntimeSessionStateError(
            "runtime_session_unavailable",
            "SessionDB is unavailable",
        ) from exc
    if db is None:
        raise RuntimeSessionStateError(
            "runtime_session_unavailable",
            "SessionDB is unavailable",
        )
    try:
        session = db.get_session(requested_session_id)
        if session is None:
            if require_existing:
                raise RuntimeSessionStateError(
                    "runtime_session_not_found",
                    "runtime SessionDB history does not exist",
                    status=409,
                )
            return db, requested_session_id, []
        resolver = getattr(db, "resolve_resume_session_id", None)
        resolved_session_id = (
            resolver(requested_session_id)
            if callable(resolver)
            else requested_session_id
        )
        history = db.get_messages_as_conversation(
            resolved_session_id,
            include_ancestors=True,
        )
    except RuntimeSessionStateError:
        raise
    except Exception as exc:
        raise RuntimeSessionStateError(
            "runtime_session_unavailable",
            "failed to load Runtime SessionDB history",
        ) from exc
    if not isinstance(history, list) or any(not isinstance(item, dict) for item in history):
        raise RuntimeSessionStateError(
            "runtime_history_conflict",
            "SessionDB returned invalid Runtime history",
            status=409,
        )
    return db, str(resolved_session_id or requested_session_id), history


__all__ = [
    "RuntimeSessionStateError",
    "load_runtime_session_history",
    "rebootstrap_session_db_history",
    "retry_session_db_history",
    "resume_session_db_history",
    "runtime_history_tool_names",
    "seed_runtime_session",
]
