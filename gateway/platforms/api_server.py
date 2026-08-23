"""OpenAI-compatible API server platform adapter facade.

The implementation is split across focused mixins under ``gateway.api_server_*``
to keep this import path stable without keeping a multi-thousand-line module.
"""

from __future__ import annotations

from gateway.api_server_shared import *
from gateway.api_server_core import APIServerCoreMixin
from gateway.api_server_sessions import APIServerSessionsMixin
from gateway.api_server_session_control import APIServerSessionControlMixin
from gateway.api_server_chat import APIServerChatMixin
from gateway.api_server_sse import APIServerSSEMixin
from gateway.api_server_responses import APIServerResponsesMixin
from gateway.api_server_jobs import APIServerJobsMixin
from gateway.api_server_runs import APIServerRunsMixin
from gateway.api_server_runtime import APIServerRuntimeMixin
from gateway.runtime_attempt_control import APIServerRuntimeControlMixin
from gateway.api_server_lifecycle import APIServerLifecycleMixin


def _approval_notify(
    approval_data: Dict[str, Any],
    *,
    loop: "asyncio.AbstractEventLoop",
    q: "asyncio.Queue[Optional[Dict[str, Any]]]",
    run_id: str,
    set_run_status,
) -> None:
    """Enqueue a redacted approval request for API/SSE clients."""
    event = dict(approval_data or {})
    if "command" in event:
        from gateway.run import _redact_approval_command

        command = _redact_approval_command(event.get("command"))
        event["command"] = command
    event.update({
        "event": "approval.request",
        "run_id": run_id,
        "timestamp": time.time(),
        "choices": ["once", "session", "always", "deny"],
    })
    set_run_status(
        run_id,
        "waiting_for_approval",
        last_event="approval.request",
    )
    try:
        loop.call_soon_threadsafe(q.put_nowait, event)
    except Exception:
        logger.debug("Failed to enqueue API approval notification", exc_info=True)


class APIServerAdapter(
    APIServerCoreMixin,
    APIServerSessionsMixin,
    APIServerSessionControlMixin,
    APIServerChatMixin,
    APIServerSSEMixin,
    APIServerResponsesMixin,
    APIServerJobsMixin,
    APIServerRunsMixin,
    APIServerRuntimeControlMixin,
    APIServerRuntimeMixin,
    APIServerLifecycleMixin,
    BasePlatformAdapter,
):
    """OpenAI-compatible HTTP API server adapter."""

    supports_async_delivery: bool = False


__all__ = [name for name in globals() if not name.startswith("__")]
