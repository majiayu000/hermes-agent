"""Opt-in Hermes Langfuse observer. See README.md for setup and configuration."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse, propagate_attributes
except Exception:  # pragma: no cover - fail-open when optional dep is missing
    Langfuse = None
    propagate_attributes = None


@dataclass
class TraceState:
    trace_id: str
    root_ctx: Any
    root_span: Any
    run_id: str = ""
    generations: Dict[str, Any] = field(default_factory=dict)
    tools: Dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: Dict[str, list] = field(default_factory=dict)
    turn_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: Dict[str, TraceState] = {}
# Hard cap on live trace state. Each turn keys _TRACE_STATE by a unique
# turn_id, and an entry is normally reclaimed by _finish_trace when a turn
# ends cleanly (final response has content and no tool calls). A turn that
# never reaches that state — interrupted, a tool-only final step, or empty
# final content — would otherwise linger forever, so over the cap we evict
# the least-recently-updated entries (ending their root span first). The cap
# is far above any realistic concurrent-live-turn working set; it exists only
# to bound the leak from non-finalizing turns, not to limit concurrency.
_MAX_TRACE_STATE = 256
_LANGFUSE_CLIENT = None

# Langfuse-issued keys always carry these prefixes (cloud or self-hosted —
# the prefix is baked into the server-side issuance flow, not a UI hint).
# Anything else (`placeholder`, `test-key`, `your-langfuse-key`, etc.) is a
# leftover template value and would cause the SDK to silently accept the
# credentials at construction time but drop every trace at flush time.
# See #23823 — the silent-failure bug this guard fixes.
_LANGFUSE_KEY_PREFIXES: Dict[str, str] = {
    "HERMES_LANGFUSE_PUBLIC_KEY": "pk-lf-",
    "HERMES_LANGFUSE_SECRET_KEY": "sk-lf-",
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(*names: str) -> bool:
    for name in names:
        value = _env(name).lower()
        if value:
            return value in {"1", "true", "yes", "on"}
    return False


def _debug_enabled() -> bool:
    return _env_bool("HERMES_LANGFUSE_DEBUG")


def _debug(message: str) -> None:
    if _debug_enabled():
        logger.info("Langfuse tracing: %s", message)


# Sentinel: "_get_langfuse() has tried and failed". Lets us short-circuit
# every subsequent hook call without re-checking env vars or re-attempting
# SDK init. Tests clear this by reloading the module via
# ``sys.modules.pop(...) + importlib.import_module(...)`` rather than via a
# dedicated reset function. Runtime callers cannot reset the cache; if an
# operator fixes a misconfigured credential they must restart the process.
_INIT_FAILED = object()


def _redact_key_preview(value: str) -> str:
    """Return a brief, log-safe preview of a credential value.

    Keeps enough characters to disambiguate common placeholders
    (``placeholder``, ``test-key``, ``your-key``) without echoing a
    real secret in full if an operator pasted one into the wrong env
    var.  Used only for the once-per-process placeholder-detection
    warning in :func:`_get_langfuse`.
    """
    if not value:
        return "<empty>"
    if len(value) <= 12:
        return repr(value)
    return repr(value[:6] + "...")


def _validate_langfuse_key(env_name: str, value: str) -> Optional[str]:
    """Return an error message if ``value`` is not a real Langfuse key.

    Returns ``None`` when the value matches the documented Langfuse
    prefix for ``env_name``, or when no prefix is registered for the
    name (in which case we trust the operator).  When validation
    fails the returned string is suitable for direct inclusion in a
    single log line — it names the env var and shows a safe preview.
    """
    expected = _LANGFUSE_KEY_PREFIXES.get(env_name, "")
    if not expected:
        return None
    if value.startswith(expected):
        return None
    return (
        f"{env_name}={_redact_key_preview(value)} "
        f"(expected {expected!r} prefix)"
    )


def _get_langfuse() -> Optional[Langfuse]:
    """Return a cached Langfuse client, or ``None`` if unavailable.

    Activation of this plugin is controlled by the Hermes plugin system —
    this function only handles the runtime-availability gate (SDK installed
    + credentials present). The result is cached: on the first call we try
    to construct a client, and every subsequent call returns that client
    (or fast-returns ``None`` if init failed).
    """
    global _LANGFUSE_CLIENT
    if _LANGFUSE_CLIENT is _INIT_FAILED:
        return None
    if _LANGFUSE_CLIENT is not None:
        return _LANGFUSE_CLIENT

    if Langfuse is None:
        _LANGFUSE_CLIENT = _INIT_FAILED
        return None

    public_key = _env("HERMES_LANGFUSE_PUBLIC_KEY") or _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("HERMES_LANGFUSE_SECRET_KEY") or _env("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        _LANGFUSE_CLIENT = _INIT_FAILED
        return None

    # Reject placeholder credentials with a one-shot warning so the
    # operator sees the misconfiguration instead of silently shipping a
    # broken observability stack (#23823).  The SDK does not validate
    # keys at construction time — it queues traces in memory and only
    # discovers the auth failure when the background flush thread tries
    # to post them, by which point the warning is buried under whatever
    # else the process is logging.  Catch it here, surface it once, and
    # short-circuit via the same _INIT_FAILED path as the empty case.
    placeholder_issues = [
        msg
        for msg in (
            _validate_langfuse_key("HERMES_LANGFUSE_PUBLIC_KEY", public_key),
            _validate_langfuse_key("HERMES_LANGFUSE_SECRET_KEY", secret_key),
        )
        if msg
    ]
    if placeholder_issues:
        logger.warning(
            "Langfuse plugin: credentials look like placeholders, traces will "
            "NOT be emitted (%s). Set real Langfuse keys (pk-lf-... / sk-lf-...) "
            "or unset HERMES_LANGFUSE_PUBLIC_KEY / HERMES_LANGFUSE_SECRET_KEY to "
            "silence this warning.",
            "; ".join(placeholder_issues),
        )
        _LANGFUSE_CLIENT = _INIT_FAILED
        return None

    base_url = _env("HERMES_LANGFUSE_BASE_URL") or _env("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    environment = _env("HERMES_LANGFUSE_ENV") or _env("LANGFUSE_ENV")
    release = _env("HERMES_LANGFUSE_RELEASE") or _env("LANGFUSE_RELEASE")
    sample_rate = _env("HERMES_LANGFUSE_SAMPLE_RATE")

    kwargs: Dict[str, Any] = {
        "public_key": public_key,
        "secret_key": secret_key,
        "base_url": base_url,
    }
    if environment:
        kwargs["environment"] = environment
    if release:
        kwargs["release"] = release
    if sample_rate:
        try:
            kwargs["sample_rate"] = float(sample_rate)
        except ValueError:
            logger.warning("Invalid HERMES_LANGFUSE_SAMPLE_RATE=%r", sample_rate)

    try:
        _LANGFUSE_CLIENT = Langfuse(**kwargs)
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("Could not initialize Langfuse client: %s", exc)
        _LANGFUSE_CLIENT = _INIT_FAILED
        return None

    return _LANGFUSE_CLIENT


def _scope_prefix(task_id: str, session_id: str) -> str:
    """The task/session/thread prefix shared by every trace-key shape."""
    if task_id:
        return f"task:{task_id}"
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _trace_key(
    task_id: str,
    session_id: str,
    *,
    turn_id: str = "",
    api_request_id: str = "",
) -> str:
    """Build a stable in-process trace scope key for one agent turn.

    Older Hermes paths only expose ``task_id``/``session_id``. Newer paths
    pass ``turn_id`` and ``api_request_id`` in LLM/tool hooks; when present,
    they must scope trace state so concurrent requests sharing one task/session
    never collide. ``turn_id`` is preferred over ``api_request_id`` so the
    turn-level ``post_llm_call`` hook (which carries ``turn_id`` but no
    ``api_request_id``) resolves to the same key as the request-level hooks.
    """
    if turn_id:
        return f"{_scope_prefix(task_id, session_id)}:turn:{turn_id}"
    if api_request_id:
        return f"{_scope_prefix(task_id, session_id)}:api:{api_request_id}"
    # Legacy shape: a bare ``task_id`` (NOT the ``task:`` prefix) when present,
    # otherwise the session/thread prefix. Kept distinct for backward
    # compatibility with keys minted before turn/request scoping existed.
    if task_id:
        return task_id
    return _scope_prefix(task_id, session_id)


from .payloads import (
    _is_base64_data_uri,
    _redact_data_uri,
    _truncate_text,
    _maybe_parse_json_string,
    _looks_like_read_file_payload,
    _parse_read_file_lines,
    _build_read_file_preview,
    _normalize_read_file_payload,
    _normalize_payload,
    _safe_value,
    _extract_last_user_message,
    _coerce_request_messages,
    _serialize_messages,
    _serialize_tool_calls,
    _serialize_assistant_message,
)


from .usage import _usage_and_cost, _summary_usage_and_cost


def _start_root_trace(task_key: str, *, task_id: str, session_id: str, platform: str, provider: str, model: str,
                      api_mode: str, messages: Any, client: Langfuse,
                      turn_id: str = "", api_request_id: str = "", run_id: str = "") -> TraceState:
    trace_id = client.create_trace_id(seed=f"run:{run_id}" if run_id else f"{session_id or 'sessionless'}::{task_key}")
    trace_input = _extract_last_user_message(messages)
    metadata = {
        "source": "hermes",
        "task_id": task_id,
        "turn_id": turn_id,
        "api_request_id": api_request_id,
        "platform": platform,
        "provider": provider,
        "model": model,
        "api_mode": api_mode,
    }
    if run_id:
        metadata["run_id"] = run_id

    # session_id must be passed in trace_context for Langfuse session grouping.
    trace_ctx: Dict[str, Any] = {"trace_id": trace_id}
    if session_id:
        trace_ctx["session_id"] = session_id

    if propagate_attributes is not None:
        try:
            with propagate_attributes(
                session_id=session_id or task_key,
                trace_name="Hermes turn",
                tags=["hermes", "langfuse"],
            ):
                root_ctx = client.start_as_current_observation(
                    trace_context=trace_ctx,
                    name="Hermes turn",
                    as_type="chain",
                    input=trace_input,
                    metadata=metadata,
                    end_on_exit=False,
                )
                root_span = root_ctx.__enter__()
        except Exception:
            root_ctx = client.start_as_current_observation(
                trace_context=trace_ctx,
                name="Hermes turn",
                as_type="chain",
                input=trace_input,
                metadata=metadata,
                end_on_exit=False,
            )
            root_span = root_ctx.__enter__()
    else:
        root_ctx = client.start_as_current_observation(
            trace_context=trace_ctx,
            name="Hermes turn",
            as_type="chain",
            input=trace_input,
            metadata=metadata,
            end_on_exit=False,
        )
        root_span = root_ctx.__enter__()

    try:
        root_span.set_trace_io(input=trace_input)
    except Exception:
        pass

    _debug(f"started trace {trace_id} for {task_key}")
    return TraceState(trace_id=trace_id, root_ctx=root_ctx, root_span=root_span, run_id=run_id)


def _start_child_observation(state: TraceState, *, client: Langfuse, name: str, as_type: str,
                             input_value: Any, metadata: Optional[dict] = None,
                             model: Optional[str] = None, model_parameters: Optional[dict] = None) -> Any:
    return state.root_span.start_observation(
        name=name,
        as_type=as_type,
        input=input_value,
        metadata={**(metadata or {}), **({"run_id": state.run_id} if state.run_id else {})},
        model=model,
        model_parameters=model_parameters,
    )


def _end_observation(observation: Any, *, output: Any = None, metadata: Optional[dict] = None,
                     usage_details: Optional[dict] = None, cost_details: Optional[dict] = None,
                     level: str | None = None, status_message: str | None = None) -> None:
    if observation is None:
        return
    try:
        update_kwargs: Dict[str, Any] = {}
        if level:
            update_kwargs["level"] = level
        if status_message:
            update_kwargs["status_message"] = status_message
        if output is not None:
            update_kwargs["output"] = output
        if metadata:
            update_kwargs["metadata"] = metadata
        if usage_details:
            update_kwargs["usage_details"] = usage_details
        if cost_details:
            update_kwargs["cost_details"] = cost_details
        if update_kwargs:
            observation.update(**update_kwargs)
        observation.end()
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"end observation failed: {exc}")


def _merge_trace_output(output: Any, state: TraceState) -> Any:
    if not state.turn_tool_calls:
        return output

    merged = dict(output) if isinstance(output, dict) else {"content": output}
    merged["tool_calls"] = list(state.turn_tool_calls)
    return merged


def _evict_stale_locked() -> None:
    """Drop least-recently-updated trace state to make room for a new entry.

    Caller MUST hold ``_STATE_LOCK`` and call this immediately before inserting
    one new entry. Bounds the leak from turns that never reach ``_finish_trace``
    (interrupted / tool-only final step / empty final content), whose unique
    per-turn key would otherwise linger forever. We evict down to
    ``_MAX_TRACE_STATE - 1`` so that the about-to-be-added entry leaves the dict
    at ``_MAX_TRACE_STATE`` — a true ceiling. The evicted entry's root span is
    ended so it is not left dangling on the Langfuse side.
    """
    over = len(_TRACE_STATE) - (_MAX_TRACE_STATE - 1)
    if over <= 0:
        return
    # Oldest-first by last_updated_at; evict just enough to make room.
    stale = sorted(_TRACE_STATE.items(), key=lambda kv: kv[1].last_updated_at)[:over]
    for key, state in stale:
        _TRACE_STATE.pop(key, None)
        try:
            state.root_span.end()
        except Exception as exc:  # pragma: no cover - fail-open
            _debug(f"evict stale trace failed: {exc}")


def _finish_trace(task_key: str, *, output: Any = None, metadata: dict | None = None,
                  level: str | None = None) -> None:
    client = _get_langfuse()
    if client is None:
        return

    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return

    try:
        for observation in state.generations.values():
            _end_observation(observation)
        for observation in state.tools.values():
            _end_observation(observation)
        for queue in state.pending_tools_by_name.values():
            for observation in queue:
                _end_observation(observation)
        final_output = _merge_trace_output(output, state)
        if final_output is not None:
            state.root_span.set_trace_io(output=final_output)
            state.root_span.update(output=final_output)
        _end_observation(state.root_span, metadata=metadata, level=level)
    except Exception as exc:  # pragma: no cover - fail-open
        _debug(f"finish trace failed: {exc}")
    finally:
        try:
            client.flush()
        except Exception:
            pass


def _assistant_has_tool_calls(message: Any) -> bool:
    return bool(getattr(message, "tool_calls", None))


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def on_pre_llm_call(*, task_id: str = "", session_id: str = "", platform: str = "", model: str = "",
                    provider: str = "", base_url: str = "", api_mode: str = "",
                    api_call_count: int = 0, messages: Any = None, turn_type: str = "user",
                    conversation_history: Any = None, user_message: Any = None,
                    turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    # Older Hermes branches used pre_llm_call for request-scoped tracing and
    # passed the actual API messages. Current Hermes also has a turn-scoped
    # pre_llm_call used for context injection; tracing that hook creates an
    # extra orphan/root trace before the real request trace. Only trace the
    # legacy request-shaped call here.
    if not isinstance(messages, list):
        return

    client = _get_langfuse()
    if client is None:
        return

    # messages is a list only for legacy Hermes branches that fired
    # pre_llm_call with API messages directly. Current Hermes fires
    # pre_llm_call for context injection (conversation_history/user_message,
    # no messages list) — tracing that would create orphan traces.
    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()


def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    turn_type: str = "user",
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    turn_id: str = "",
    api_request_id: str = "",
    run_id: str = "",
    **_: Any,
) -> None:
    client = _get_langfuse()
    if client is None:
        return

    input_messages = _coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            state = _start_root_trace(
                task_key,
                task_id=task_id,
                session_id=session_id,
                platform=platform,
                provider=provider,
                model=model,
                api_mode=api_mode,
                messages=input_messages,
                client=client,
                turn_id=turn_id,
                api_request_id=api_request_id,
                run_id=run_id,
            )
            _evict_stale_locked()
            _TRACE_STATE[task_key] = state
        state.last_updated_at = time.time()
        previous = state.generations.pop(req_key, None)
        if previous is not None:
            _end_observation(previous)
        state.generations[req_key] = _start_child_observation(
            state,
            client=client,
            name=f"LLM call {api_call_count}",
            as_type="generation",
            input_value=_serialize_messages(input_messages),
            metadata={
                "provider": provider,
                "platform": platform,
                "api_mode": api_mode,
                "base_url": base_url,
            },
            model=model,
            model_parameters={"api_mode": api_mode, "provider": provider},
        )


def on_post_llm_call(*, task_id: str = "", session_id: str = "", provider: str = "", base_url: str = "",
                     api_mode: str = "", model: str = "", api_call_count: int = 0,
                     assistant_message: Any = None, response: Any = None,
                     api_duration: float = 0.0, finish_reason: str = "",
                     usage: Any = None, assistant_content_chars: int = 0,
                     assistant_tool_call_count: int = 0, assistant_response: Any = None,
                     turn_id: str = "", api_request_id: str = "",
                     **_: Any) -> None:
    client = _get_langfuse()
    if client is None:
        return

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        generation = state.generations.pop(req_key, None) if state else None
    if state is None or generation is None:
        return

    # Handle both call patterns:
    # 1. post_api_request: passes usage (dict), assistant_content_chars, assistant_tool_call_count
    # 2. post_llm_call: passes assistant_message (object), response (object), assistant_response (str)
    if assistant_message is not None:
        output = _serialize_assistant_message(assistant_message)
    elif assistant_response is not None:
        # post_llm_call passes assistant_response as a plain string
        output = {"content": _safe_value(assistant_response), "reasoning": None, "tool_calls": []}
    else:
        # post_api_request path — reconstruct from summary kwargs
        output = {
            "content": f"[{assistant_content_chars} chars]" if assistant_content_chars else None,
            "reasoning": None,
            "tool_calls": [{"id": f"tc_{i}"} for i in range(assistant_tool_call_count)] if assistant_tool_call_count else [],
        }

    if output.get("tool_calls"):
        state.turn_tool_calls.extend(output["tool_calls"])

    # Extract usage: prefer a real response object that carries usage, else
    # fall back to the usage summary dict from post_api_request.
    #
    # post_api_request passes `response` as a SANITIZED dict (no ``.usage``
    # attribute) alongside a separate `usage` summary dict. Gating on
    # ``response is not None`` here took the response-object path on that dict,
    # where ``getattr(response, "usage", None)`` is always None — so usage and
    # cost were silently dropped for every gateway turn. Gate on a real
    # ``.usage`` attribute instead so the usage-dict fallback below is reached.
    if getattr(response, "usage", None) is not None:
        usage_details, cost_details = _usage_and_cost(
            response,
            provider=provider,
            api_mode=api_mode,
            model=model,
            base_url=base_url,
        )
    elif isinstance(usage, dict) and usage:
        usage_details, cost_details = _summary_usage_and_cost(usage, model=model, provider=provider, base_url=base_url)
    else:
        usage_details, cost_details = {}, {}

    tool_count = len(output.get("tool_calls", [])) or assistant_tool_call_count
    gen_metadata: Dict[str, Any] = {"tool_call_count": tool_count, "cost_source": "estimate"}
    if api_duration and api_duration > 0:
        gen_metadata["api_duration_s"] = round(api_duration, 3)
    if finish_reason:
        gen_metadata["finish_reason"] = finish_reason
    _end_observation(
        generation,
        output=output,
        usage_details=usage_details,
        cost_details=cost_details,
        metadata=gen_metadata,
    )

    has_tools = _assistant_has_tool_calls(assistant_message) if assistant_message else (assistant_tool_call_count > 0)
    has_content = bool(output.get("content"))
    if not has_tools and has_content:
        _finish_trace(task_key, output=output)


def on_pre_tool_call(*, tool_name: str = "", args: Any = None, task_id: str = "",
                     session_id: str = "", tool_call_id: str = "",
                     turn_id: str = "", api_request_id: str = "", **_: Any) -> None:
    client = _get_langfuse()
    if client is None:
        return

    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        observation = _start_child_observation(
            state,
            client=client,
            name=f"Tool: {tool_name}",
            as_type="tool",
            input_value=_safe_value(args),
            metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
        )
        if tool_call_id:
            state.tools[tool_call_id] = observation
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(observation)


def on_post_tool_call(*, tool_name: str = "", args: Any = None, result: Any = None,
                      task_id: str = "", session_id: str = "", tool_call_id: str = "",
                      turn_id: str = "", api_request_id: str = "", status: str = "ok",
                      error_type: str = "", error_message: str = "", **_: Any) -> None:
    task_key = _trace_key(
        task_id,
        session_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
    )
    observation = None

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        if tool_call_id:
            observation = state.tools.pop(tool_call_id, None)
        if observation is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                observation = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)

    if observation is None:
        return

    if isinstance(result, str):
        result_value = _maybe_parse_json_string(result)
    else:
        result_value = result
    result_value = _normalize_payload(result_value, tool_name=tool_name, args=args)
    safe_result_value = _safe_value(result_value, parse_json_strings=True)

    # Backfill so the generation's tool_call record carries the result alongside arguments.
    if tool_call_id:
        with _STATE_LOCK:
            state = _TRACE_STATE.get(task_key)
            if state is not None:
                for tool_call in reversed(state.turn_tool_calls):
                    if tool_call.get("id") == tool_call_id:
                        tool_call["output"] = safe_result_value
                        function_payload = tool_call.get("function")
                        if isinstance(function_payload, dict):
                            function_payload["output"] = safe_result_value
                        break

    _end_observation(
        observation,
        output=safe_result_value,
        metadata={"tool_name": tool_name, "args": _safe_value(args, parse_json_strings=True),
                  "status": status, "error_type": error_type},
        level="ERROR" if status == "error" else "WARNING" if status in {"blocked", "cancelled"} else None,
        status_message=_safe_value(error_message) if error_message else None,
    )


def on_api_request_error(*, task_id: str = "", session_id: str = "", turn_id: str = "",
                         api_request_id: str = "", api_call_count: int = 0,
                         error: dict | None = None, status_code: int | None = None,
                         retry_count: int = 0, retryable: bool = False,
                         api_duration: float = 0, **_: object) -> None:
    key = _trace_key(task_id, session_id, turn_id=turn_id, api_request_id=api_request_id)
    with _STATE_LOCK:
        state = _TRACE_STATE.get(key)
        generation = state.generations.pop(_request_key(api_call_count), None) if state else None
    _end_observation(
        generation, output=_safe_value(error), level="ERROR",
        metadata={"status_code": status_code, "retry_count": retry_count,
                  "retryable": retryable, "api_duration_s": api_duration},
    )


def on_session_end(*, task_id: str = "", session_id: str = "", turn_id: str = "",
                   completed: bool = False, interrupted: bool = False, **_: object) -> None:
    key = _trace_key(task_id, session_id, turn_id=turn_id)
    _finish_trace(
        key, metadata={"completed": completed, "interrupted": interrupted},
        level="WARNING" if interrupted else None if completed else "ERROR",
    )


def register(ctx) -> None:
    # Register for both hook name variants so the plugin works across
    # Hermes versions.  pre_api_request / post_api_request fire per API
    # call (preferred); pre_llm_call / post_llm_call fire once per turn.
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("on_session_end", on_session_end)
