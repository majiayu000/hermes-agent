import asyncio
import threading
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.iteration_budget import IterationBudget
from agent.tool_dispatch_helpers import DeferredToolResult
from gateway.api_server_lifecycle import APIServerLifecycleMixin
from gateway.api_server_runtime import RuntimeBridgeSession, _SESSIONS, _SESSIONS_LOCK
from gateway.runtime_attempt_control import (
    APIServerRuntimeControlMixin,
    suspend_attempt,
    terminate_attempt,
)


class _Pending:
    def __init__(self):
        self.ready = threading.Event()


class _Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.pending = {"call_1": _Pending()}
        self.pending_controls = {"control_1": _Pending()}
        self.interrupted = threading.Event()
        self.interrupt_reason = ""
        self.agent = SimpleNamespace(iteration_budget=IterationBudget(12))
        self.agent_ref = [self.agent]
        self.events = []
        self.interrupt_calls = []

    def emit(self, event_type, payload):
        self.events.append((event_type, payload))

    def interrupt(self, reason):
        self.interrupt_calls.append(reason)


class _CallDB:
    def get_messages_as_conversation(self, *_args, **_kwargs):
        return [{
            "role": "assistant",
            "tool_calls": [{
                "id": "call_media",
                "function": {"name": "media.generate_image", "arguments": "{}"},
            }],
        }]


class _RuntimeOnlyAdapter(APIServerRuntimeControlMixin, APIServerLifecycleMixin):
    name = "test"

    def __init__(self):
        self._app = web.Application()

    async def _handle_health(self, _request):
        return web.Response()

    async def _handle_runtime_manifest(self, _request):
        return web.Response()

    async def _handle_runtime_run(self, _request):
        return web.Response()

    async def _handle_runtime_tool_result(self, _request):
        return web.Response()

    async def _handle_runtime_control_result(self, _request):
        return web.Response()


class _ControlAdapter(APIServerRuntimeControlMixin):
    async def _authenticate_runtime_request(self, _request):
        return None


def test_suspend_wakes_delegated_tool_without_agent_interrupt():
    session = _Session()

    suspend_attempt(session, "tool_operation")

    assert session.interrupted.is_set()
    assert session.interrupt_reason == "parked:tool_operation"
    assert session.pending["call_1"].ready.is_set()
    assert session.pending_controls["control_1"].ready.is_set()
    assert session.agent.iteration_budget.remaining == 0
    assert session.agent._runtime_suspended is True
    assert session.interrupt_calls == []

    session.emit("activity", {"state": "parked"})
    session.emit("completed", {})
    session.emit("error", {})
    assert session.events == [("activity", {"state": "parked"})]


def test_runtime_only_routes_expose_typed_controls_without_interrupt(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
    adapter = _RuntimeOnlyAdapter()

    adapter._setup_routes()

    paths = {route.resource.canonical for route in adapter._app.router.routes()}
    assert "/v1/runtime/runs/{run_id}/suspend" in paths
    assert "/v1/runtime/runs/{run_id}/cancel" in paths
    assert "/v1/runtime/runs/{run_id}/abort" in paths
    assert "/v1/runtime/runs/{run_id}/interrupt" not in paths


@pytest.mark.asyncio
async def test_suspend_returns_deferred_result_through_real_runtime_bridge():
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "run_suspend",
        asyncio.get_running_loop(),
        queue,
        [{"name": "media.generate_image", "input_schema": {"type": "object"}}],
        10_000,
        "thread_suspend",
        _CallDB(),
    )
    interrupts = []
    session.agent_ref[0] = SimpleNamespace(
        iteration_budget=IterationBudget(12),
        interrupt=lambda reason: interrupts.append(reason),
    )
    call = asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool,
        "media.generate_image",
        {},
        "call_media",
    ))
    assert (await queue.get())["type"] == "tool_request"

    suspend_attempt(session, "tool_operation")

    assert await call == DeferredToolResult("call_media")
    assert interrupts == []


def test_cancel_keeps_explicit_cause_and_projects_canceled_completion():
    session = _Session()

    terminate_attempt(session, "cancel", "user_requested")

    assert session.agent._runtime_termination_kind == "cancel"
    assert session.agent._runtime_termination_cause == "user_requested"
    assert session.interrupt_calls == ["cancel:user_requested"]

    session.emit("error", {"code": "runtime_failed"})
    assert session.events == [("completed", {"finish_reason": "canceled", "text": ""})]


def test_abort_keeps_attempt_failure_distinct_from_user_cancel():
    session = _Session()

    terminate_attempt(session, "abort", "event_stream_idle_timeout")

    assert session.agent._runtime_termination_kind == "abort"
    assert session.interrupt_calls == ["abort:event_stream_idle_timeout"]
    session.emit("error", {"code": "runtime_failed"})
    assert session.events == [("error", {"code": "runtime_failed"})]


def test_cancel_during_suspend_still_emits_canceled_terminal_state():
    session = _Session()
    suspend_attempt(session, "tool_operation")

    terminate_attempt(session, "cancel", "user_requested")
    session.emit("error", {"code": "runtime_failed"})

    assert session.interrupt_calls == ["cancel:user_requested"]
    assert session.events == [("completed", {"finish_reason": "canceled", "text": ""})]


@pytest.mark.asyncio
async def test_non_object_control_body_fails_before_cancel_side_effect():
    session = _Session()
    with _SESSIONS_LOCK:
        _SESSIONS["run_invalid"] = session
    app = web.Application()
    adapter = _ControlAdapter()
    app.router.add_post("/v1/runtime/runs/{run_id}/cancel", adapter._handle_runtime_cancel)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.post("/v1/runtime/runs/run_invalid/cancel", json=[])
        assert response.status == 400
        assert session.interrupt_calls == []
    finally:
        await client.close()
        with _SESSIONS_LOCK:
            _SESSIONS.pop("run_invalid", None)
