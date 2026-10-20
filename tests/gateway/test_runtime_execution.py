import asyncio
import hashlib
import json
from types import SimpleNamespace
from threading import RLock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from hermes_state import SessionDB
from gateway.api_server_runtime import APIServerRuntimeMixin
from gateway.runtime_attempt_control import APIServerRuntimeControlMixin
from gateway.runtime_execution import ExecutionJournal, Execution, subscribe
from gateway.runtime_session_history import RuntimeSessionStateError
from gateway.runtime_egress import refresh_run_egress


def run_body():
    stable = "test instructions"
    return {
        "run_id": "run_f1",
        "model": "test-model",
        "intent": "bootstrap",
        "context": {"session_id": "session_f1"},
        "messages": [{"id": "message_f1", "role": "user", "content": "hello"}],
        "tools": [],
        "deadline_ms": 30000,
        "system_context": {
            "version": "test/v1",
            "mode": "replace",
            "stable": stable,
            "digest": "sha256:"
            + hashlib.sha256(("test/v1\nreplace\n" + stable).encode()).hexdigest(),
        },
    }


class Adapter(APIServerRuntimeControlMixin, APIServerRuntimeMixin):
    def __init__(self, db):
        self.db = db
        self.calls = 0
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    def _ensure_session_db(self):
        return self.db

    def _check_auth(self, request):
        return None

    async def _run_agent_bridge(self, **kwargs):
        self.calls += 1
        kwargs["stream_delta_callback"]("one")
        self.started.set()
        await self.release.wait()
        kwargs["stream_delta_callback"]("two")
        return {"final_response": "onetwo"}, {}


@pytest.mark.asyncio
async def test_disconnect_and_lost_start_response_replay_original_execution(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = Adapter(db)
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    app.router.add_get(
        "/v1/runtime/runs/{run_id}/events", adapter._handle_runtime_attach
    )
    app.router.add_post(
        "/v1/runtime/runs/{run_id}/attach", adapter._handle_runtime_attach
    )
    for suffix, handler in (
        ("tool-results", adapter._handle_runtime_tool_result),
        ("control-results", adapter._handle_runtime_control_result),
        ("cancel", adapter._handle_runtime_cancel),
        ("suspend", adapter._handle_runtime_suspend),
        ("abort", adapter._handle_runtime_abort),
        ("interrupt", adapter._handle_runtime_interrupt),
    ):
        app.router.add_post("/v1/runtime/runs/{run_id}/" + suffix, handler)
    async with TestClient(TestServer(app)) as client:
        original = await client.post(
            "/v1/runtime/runs?execution_id=exec_f1&lease_epoch=1", json=run_body()
        )
        assert original.status == 200
        await asyncio.wait_for(adapter.started.wait(), 2)
        first = json.loads(await original.content.readline())
        assert first["type"] == "run_started"
        original.close()
        execution = adapter._execution_registry()["exec_f1"]
        assert not execution.session.interrupted.is_set()
        # Same Start after a lost response is a subscription, never another Agent.
        replay = await client.post(
            "/v1/runtime/runs?execution_id=exec_f1&lease_epoch=1", json=run_body()
        )
        assert replay.status == 200
        assert adapter.calls == 1
        bad = run_body()
        bad["messages"][0]["content"] = "different"
        conflict = await client.post(
            "/v1/runtime/runs?execution_id=exec_f1&lease_epoch=1", json=bad
        )
        assert conflict.status == 409
        replay.close()
        replay = await client.post(
            "/v1/runtime/runs/run_f1/attach?execution_id=exec_f1&lease_epoch=2",
            json={"model": "test-model"},
        )
        assert replay.status == 200
        stale = await client.get(
            "/v1/runtime/runs/run_f1/events?execution_id=exec_f1&lease_epoch=1"
        )
        assert stale.status == 409
        for suffix in ("tool-results", "control-results", "cancel", "suspend", "abort", "interrupt"):
            stale_control = await client.post(
                f"/v1/runtime/runs/run_f1/{suffix}?execution_id=exec_f1&lease_epoch=1",
                json={},
            )
            assert stale_control.status == 409
        assert not execution.session.interrupted.is_set()
        for invalid in ([], {"model": "other"}, {"model": "test-model", "extra": True}):
            rejected = await client.post(
                "/v1/runtime/runs/run_f1/attach?execution_id=exec_f1&lease_epoch=2",
                json=invalid,
            )
            assert rejected.status == 422
        adapter.release.set()
        frames = [json.loads(line) for line in (await replay.text()).splitlines()]
        assert [f["payload"]["delta"] for f in frames if f["type"] == "text_delta"] == [
            "one",
            "two",
        ]
        assert frames[-1]["type"] == "completed"
        recovered = await client.get(
            "/v1/runtime/runs/run_f1/events?execution_id=exec_f1&lease_epoch=2"
        )
        assert recovered.status == 200
        assert [
            json.loads(line) for line in (await recovered.text()).splitlines()
        ] == frames
        denied = await client.get(
            "/v1/runtime/runs/another/events?execution_id=exec_f1&lease_epoch=1"
        )
        assert denied.status == 409
        assert adapter.calls == 1
    db.close()


def test_execution_journal_reopens_and_keeps_identity(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    journal = ExecutionJournal(db)
    assert journal.create("exec_f1", "run_f1", "digest")
    assert not journal.create("exec_f1", "run_f1", "digest")
    with pytest.raises(RuntimeSessionStateError):
        journal.create("exec_f1", "run_f1", "different")
    journal.append(
        "exec_f1",
        {"run_id": "run_f1", "type": "text_delta", "payload": {"delta": "kept"}},
    )
    db.close()
    db = SessionDB(db_path=path)
    journal = ExecutionJournal(db)
    assert journal.frames("exec_f1", 0)[0][0] == 1
    assert json.loads(journal.frames("exec_f1", 0)[0][1])["payload"]["delta"] == "kept"
    assert not journal.finished("exec_f1", "run_f1")
    with pytest.raises(RuntimeSessionStateError):
        journal.finished("exec_f1", "another")
    db.close()


@pytest.mark.asyncio
async def test_initial_journal_failure_does_not_start_agent_or_leak_slot(tmp_path, monkeypatch):
    import gateway.api_server_runtime as runtime

    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = Adapter(db)
    original_append = ExecutionJournal.append

    def fail_initial_frame(self, identity, event):
        if event["type"] == "run_started":
            raise OSError("test journal write failure")
        return original_append(self, identity, event)

    monkeypatch.setattr(ExecutionJournal, "append", fail_initial_frame)
    slots_before = runtime._ACTIVE_RUN_COUNT
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/runtime/runs?execution_id=exec_f1&lease_epoch=1", json=run_body()
        )
        frames = [json.loads(line) for line in (await response.text()).splitlines()]
        assert [frame["type"] for frame in frames] == ["error"]
        assert adapter.calls == 0
        assert adapter._execution_registry() == {}
        assert runtime._ACTIVE_RUN_COUNT == slots_before
        assert ExecutionJournal(db).finished("exec_f1", "run_f1")
    db.close()


@pytest.mark.asyncio
async def test_missing_live_execution_is_not_restarted(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    ExecutionJournal(db).create("exec_f1", "run_f1", "digest")
    adapter = Adapter(db)
    app = web.Application()
    app.router.add_get(
        "/v1/runtime/runs/{run_id}/events", adapter._handle_runtime_attach
    )
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            "/v1/runtime/runs/run_f1/events?execution_id=exec_f1&lease_epoch=1"
        )
        assert response.status == 409
        assert (await response.json())["error"]["code"] == "runtime_execution_not_found"
        assert adapter.calls == 0
    db.close()


def test_new_lease_fences_old_subscriber_and_control(tmp_path):
    from gateway.runtime_execution import controls_execution

    db = SessionDB(db_path=tmp_path / "state.db")
    journal = ExecutionJournal(db)
    journal.create("exec_fence", "run_fence", "digest")
    execution = Execution("exec_fence", "run_fence", journal)
    journal.claim_subscription("exec_fence", 10)
    assert controls_execution(
        SimpleNamespace(query={"execution_id": "exec_fence", "lease_epoch": "10"}),
        execution,
    )
    journal.claim_subscription("exec_fence", 20)
    with pytest.raises(RuntimeSessionStateError):
        journal.claim_subscription("exec_fence", 10)
    assert not controls_execution(
        SimpleNamespace(query={"execution_id": "exec_fence", "lease_epoch": "10"}),
        execution,
    )
    assert not controls_execution(
        SimpleNamespace(query={"execution_id": "wrong", "lease_epoch": "20"}), execution
    )
    assert not controls_execution(
        SimpleNamespace(query={"execution_id": "exec_fence", "lease_epoch": "bad"}),
        execution,
    )
    assert controls_execution(
        SimpleNamespace(query={"execution_id": "exec_fence", "lease_epoch": "20"}),
        execution,
    )
    db.close()


def test_attach_refreshes_only_original_model_route_without_restarting():
    from datetime import datetime, timezone, timedelta

    expiry = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    old = {
        "base_url": "http://localhost/internal/llm/v1",
        "grant": "ueg_" + "a" * 43,
        "expires_at": expiry,
    }
    new = {**old, "grant": "ueg_" + "b" * 43}
    client = SimpleNamespace(api_key=old["grant"])
    lock = RLock()
    agent = SimpleNamespace(
        _run_llm_egress=True,
        api_key=old["grant"],
        _client_kwargs={"api_key": old["grant"]},
        client=client,
        _openai_client_lock=lambda: lock,
    )
    session = SimpleNamespace(
        model="test-model",
        llm_egress=dict(old),
        vision_llm_egress={"model": "vision-test", **old},
        agent_ref=[agent],
        lock=lock,
    )
    vision_reference = session.vision_llm_egress
    refresh_run_egress(
        session,
        {
            "model": "test-model",
            "llm_egress": new,
            "vision_llm_egress": {"model": "vision-test", **new},
        },
    )
    assert agent.client is client
    assert (
        client.api_key
        == new["grant"]
        == agent.api_key
        == agent._client_kwargs["api_key"]
    )
    assert vision_reference["grant"] == new["grant"]
    for change in (
        {"model": "other"},
        {"llm_egress": {**new, "base_url": "http://elsewhere/internal/llm/v1"}},
        {"vision_llm_egress": None},
    ):
        with pytest.raises(ValueError):
            refresh_run_egress(
                session,
                {
                    "model": "test-model",
                    "llm_egress": new,
                    "vision_llm_egress": {"model": "vision-test", **new},
                    **change,
                },
            )
    assert client.api_key == new["grant"]
    agent._run_llm_egress = False
    with pytest.raises(ValueError, match="does not own run-scoped egress"):
        refresh_run_egress(
            session,
            {
                "model": "test-model",
                "llm_egress": new,
                "vision_llm_egress": {"model": "vision-test", **new},
            },
        )


@pytest.mark.asyncio
async def test_runtime_deadline_survives_disconnected_subscriber(tmp_path):
    import gateway.api_server_runtime as bridge

    class DeadlineAdapter(Adapter):
        async def _run_agent_bridge(self, **kwargs):
            self.calls += 1
            self.started.set()
            session = bridge._SESSIONS["run_f1"]
            while not session.interrupted.is_set():
                await asyncio.sleep(0.005)
            return {"failed": True, "completed": False}, {}

    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = DeadlineAdapter(db)
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    async with TestClient(TestServer(app)) as client:
        body = run_body()
        body["deadline_ms"] = 60
        response = await client.post(
            "/v1/runtime/runs?execution_id=exec_deadline&lease_epoch=1", json=body
        )
        assert response.status == 200
        await asyncio.wait_for(adapter.started.wait(), 1)
        session = bridge._SESSIONS["run_f1"]
        response.close()
        await asyncio.wait_for(session.finished_async.wait(), 1)
        assert session.interrupt_reason == "run deadline exceeded"
        assert adapter.calls == 1
    db.close()


@pytest.mark.asyncio
async def test_subscription_heartbeat_and_admission_validation(tmp_path, monkeypatch):
    import gateway.runtime_execution as module

    monkeypatch.setattr(module, "_HEARTBEAT_SECONDS", 0.01)
    db = SessionDB(db_path=tmp_path / "state.db")
    adapter = Adapter(db)
    app = web.Application()
    app.router.add_post("/v1/runtime/runs", adapter._handle_runtime_run)
    async with TestClient(TestServer(app)) as client:
        for query in (
            "",
            "?execution_id=exec_bad",
            "?execution_id=exec_bad&lease_epoch=-1",
            "?execution_id=exec_bad&lease_epoch=9223372036854775808",
        ):
            rejected = await client.post("/v1/runtime/runs" + query, json=run_body())
            assert rejected.status == 422
        assert adapter.calls == 0
        response = await client.post(
            "/v1/runtime/runs?execution_id=exec_heartbeat&lease_epoch=1",
            json=run_body(),
        )
        assert response.status == 200
        assert json.loads(await response.content.readline())["type"] == "run_started"
        assert json.loads(await response.content.readline())["type"] == "text_delta"
        heartbeat = json.loads(await asyncio.wait_for(response.content.readline(), 1))
        assert heartbeat["type"] == "heartbeat"
        adapter.release.set()
        frames = [json.loads(line) for line in (await response.text()).splitlines()]
        assert frames[-1]["type"] == "completed"
        assert not any(
            json.loads(frame)["type"] == "heartbeat"
            for _, frame in ExecutionJournal(db).frames("exec_heartbeat", 0)
        )
    db.close()
