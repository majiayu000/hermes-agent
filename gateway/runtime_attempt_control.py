"""Typed lifecycle controls for Run Orchestrator Runtime attempts."""

from __future__ import annotations

import asyncio
from typing import Any

from gateway.api_server_shared import web
from gateway.api_server_runtime import _SESSIONS, _SESSIONS_LOCK

_SUSPEND_REASONS = frozenset({"tool_operation", "human_approval", "user_input"})
_CANCEL_CAUSES = frozenset({"user_requested", "deadline_exceeded"})
_ABORT_CAUSES = frozenset({"event_stream_idle_timeout"})


def _wake_pending(session: Any) -> None:
    with session.lock:
        for pending in session.pending.values():
            pending.ready.set()
        for pending in session.pending_controls.values():
            pending.ready.set()


def _install_controlled_emit(session: Any) -> None:
    if hasattr(session, "_runtime_control_original_emit"):
        return
    session._runtime_control_original_emit = session.emit

    def emit_with_control_state(event_type: str, payload: Any) -> None:
        original_emit = session._runtime_control_original_emit
        if getattr(session, "termination_kind", "") == "cancel" and event_type == "error":
            original_emit("completed", {"finish_reason": "canceled", "text": ""})
            return
        if getattr(session, "runtime_suspended", False) and event_type in {"completed", "error"}:
            return
        original_emit(event_type, payload)

    session.emit = emit_with_control_state


def suspend_attempt(session: Any, reason: str) -> None:
    """Park one attempt without invoking the Agent interrupt API."""
    _install_controlled_emit(session)
    session.runtime_suspended = True
    session.interrupt_reason = f"parked:{reason}"
    session.interrupted.set()
    agent = session.agent_ref[0]
    if agent is not None:
        agent._runtime_suspended = True
        agent.iteration_budget.exhaust()

    _wake_pending(session)


def terminate_attempt(session: Any, kind: str, cause: str) -> None:
    _install_controlled_emit(session)
    session.runtime_suspended = False
    session.termination_kind = kind
    session.termination_cause = cause
    agent = session.agent_ref[0]
    if agent is not None:
        agent._runtime_termination_kind = kind
        agent._runtime_termination_cause = cause

    session.interrupt(f"{kind}:{cause}")


class APIServerRuntimeControlMixin:
    async def _runtime_session_control(
        self,
        request: "web.Request",
        *,
        action: str,
        field: str,
        allowed_values: frozenset[str],
    ) -> "web.Response":
        auth_error = await self._authenticate_runtime_request(request)
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
            body = await request.json()
        except Exception:
            return web.json_response(
                {"error": {"code": "invalid_param", "message": "invalid JSON"}},
                status=400,
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"error": {"code": "invalid_param", "message": "request body must be an object"}},
                status=400,
            )
        value = str(body.get(field) or "")
        if value not in allowed_values:
            return web.json_response(
                {"error": {"code": "invalid_param", "message": f"invalid {action} {field}"}},
                status=400,
            )

        if action == "suspend":
            suspend_attempt(session, value)
        else:
            terminate_attempt(session, action, value)

        try:
            await asyncio.wait_for(session.finished_async.wait(), 10)
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": {"code": f"{action}_timeout", "message": "runtime session did not stop"}},
                status=503,
            )
        return web.Response(status=204)

    async def _handle_runtime_suspend(self, request: "web.Request") -> "web.Response":
        return await self._runtime_session_control(
            request, action="suspend", field="reason", allowed_values=_SUSPEND_REASONS
        )

    async def _handle_runtime_cancel(self, request: "web.Request") -> "web.Response":
        return await self._runtime_session_control(
            request, action="cancel", field="cause", allowed_values=_CANCEL_CAUSES
        )

    async def _handle_runtime_abort(self, request: "web.Request") -> "web.Response":
        return await self._runtime_session_control(
            request, action="abort", field="cause", allowed_values=_ABORT_CAUSES
        )
