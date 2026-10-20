from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.api_server_runtime as runtime_module
from gateway.api_server_runtime import APIServerRuntimeMixin, RuntimeBridgeSession
from gateway.runtime_session_history import resume_session_db_history
from hermes_state import SessionDB
from run_agent import AIAgent


async def _start_calls(db, call_ids):
    db.create_session("delivery_session", "api_server")
    db.append_message("delivery_session", role="assistant", content=None, tool_calls=[{
        "id": call_id, "type": "function",
        "function": {"name": "platform.test", "arguments": "{}"},
    } for call_id in call_ids])
    queue = asyncio.Queue()
    session = RuntimeBridgeSession(
        "delivery_run", asyncio.get_running_loop(), queue,
        [{"name": "platform.test", "input_schema": {"type": "object"}}],
        2000, "delivery_session", db,
    )
    invocations = {call_id: asyncio.create_task(asyncio.to_thread(
        session.invoke_platform_tool, "platform.test", {}, call_id,
    )) for call_id in call_ids}
    for _ in call_ids:
        event = await asyncio.wait_for(queue.get(), 2)
        assert event["type"] == "tool_request"
    return session, invocations


@pytest.mark.asyncio
async def test_delivery_is_durable_before_ack_and_agent_flush_is_idempotent(tmp_path):
    db_path = tmp_path / "delivery.db"
    db = SessionDB(db_path=db_path)
    session, invocations = await _start_calls(db, ["call_delivery"])
    result = {"call_id": "call_delivery", "ok": True, "result": {"asset_id": "asset_test"}}
    try:
        assert session.submit_result(result)
        # Another connection sees the commit before the Agent has flushed anything.
        reopened = SessionDB(db_path=db_path)
        try:
            history = reopened.get_messages_as_conversation("delivery_session")
            assert json.loads(history[-1]["content"]) == result["result"]
            assert len(history) == 2
        finally:
            reopened.close()
        content = await asyncio.wait_for(invocations["call_delivery"], 2)
        agent = AIAgent.__new__(AIAgent)
        agent._session_db = db
        agent._session_db_created = True
        agent.session_id = "delivery_session"
        agent._last_flushed_db_idx = 0
        agent._require_incremental_session_persistence = True
        agent._apply_persist_user_message_override = lambda messages: None
        tool_message = {"role": "tool", "tool_call_id": "call_delivery", "tool_name": "platform.test", "content": content}
        agent._flush_messages_to_session_db([history[0], tool_message], [history[0]])
        assert session.submit_result(dict(result))
        assert not session.submit_result({**result, "result": {"asset_id": "conflict"}})
        assert len(db.get_messages_as_conversation("delivery_session")) == 2
        assert db.get_session("delivery_session")["message_count"] == 2
        assert resume_session_db_history(db, "delivery_session", history, [result]) == history
    finally:
        await asyncio.gather(*invocations.values(), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_out_of_order_results_commit_without_waiting_for_siblings(tmp_path):
    db = SessionDB(db_path=tmp_path / "parallel.db")
    session, invocations = await _start_calls(db, ["first", "second"])
    try:
        assert session.submit_result({"call_id": "second", "ok": True, "result": "second result"})
        assert not invocations["first"].done()
        assert json.loads(await asyncio.wait_for(invocations["second"], 2)) == "second result"
        assert session.submit_result({"call_id": "first", "ok": True, "result": "first result"})
        await asyncio.wait_for(invocations["first"], 2)
        messages = db.get_messages_as_conversation("delivery_session")
        assert [item["tool_call_id"] for item in messages if item["role"] == "tool"] == ["second", "first"]
    finally:
        await asyncio.gather(*invocations.values(), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_http_failed_commit_does_not_ack_or_wake_and_redelivery_commits(tmp_path, monkeypatch):
    class Adapter(APIServerRuntimeMixin):
        def _check_auth(self, request):
            return None

    db = SessionDB(db_path=tmp_path / "failed-commit.db")
    session, invocations = await _start_calls(db, ["call_delivery"])
    original_append = db.append_message
    fail_commit = True

    def append(*args, **kwargs):
        if fail_commit:
            raise OSError("simulated unavailable database")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(db, "append_message", append)
    monkeypatch.setitem(runtime_module._SESSIONS, "delivery_run", session)
    app = web.Application()
    app.router.add_post("/v1/runtime/runs/{run_id}/tool-results", Adapter()._handle_runtime_tool_result)
    client = TestClient(TestServer(app))
    await client.start_server()
    result = {"call_id": "call_delivery", "ok": False,
              "error": {"code": "provider_rejected", "message": "original explanation"},
              "result": {"message": "field details", "submission_started": False}}
    try:
        endpoint = "/v1/runtime/runs/delivery_run/tool-results"
        failed = await client.post(endpoint, json=result)
        assert failed.status == 503
        assert not session.pending["call_delivery"].ready.is_set()
        assert not invocations["call_delivery"].done()
        assert len(db.get_messages_as_conversation("delivery_session")) == 1
        fail_commit = False
        accepted = await client.post(endpoint, json=result)
        assert accepted.status == 204
        assert len(db.get_messages_as_conversation("delivery_session")) == 2
        assert (await client.post(endpoint, json=result)).status == 204
        assert (await client.post(endpoint, json={**result, "result": "conflict"})).status == 409
        assert len(db.get_messages_as_conversation("delivery_session")) == 2
        await asyncio.wait_for(invocations["call_delivery"], 2)
    finally:
        await client.close()
        await asyncio.gather(*invocations.values(), return_exceptions=True)
        db.close()


@pytest.mark.asyncio
async def test_session_db_tool_identity_is_atomic_across_connections(tmp_path):
    path = tmp_path / "atomic-delivery.db"
    first = SessionDB(db_path=path)
    second = SessionDB(db_path=path)
    try:
        first.create_session("session", "api_server")
        rows = await asyncio.gather(*[
            asyncio.to_thread(db.append_message, "session", role="tool", tool_call_id="same_call", content=content)
            for db, content in [(first, '{"a":1,"b":2}'), (second, '{"b":2,"a":1}')]
        ])
        assert rows[0] == rows[1]
        assert first.get_session("session")["message_count"] == 1
        with pytest.raises(ValueError, match="conflicting result"):
            second.append_message("session", role="tool", tool_call_id="same_call", content="different result")
        assert first.get_session("session")["message_count"] == 1
    finally:
        first.close()
        second.close()
