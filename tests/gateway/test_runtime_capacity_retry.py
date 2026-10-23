"""HTTP admission regressions using real Runtime handlers and a temporary journal."""

import asyncio
import hashlib
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.api_server_runtime as runtime
from gateway.runtime_execution import ExecutionJournal
from hermes_state import SessionDB


class CapacityAdapter(runtime.APIServerRuntimeMixin):
    def __init__(self, db):
        self.db = db
        self.calls = 0
        self.release = asyncio.Event()

    def _ensure_session_db(self):
        return self.db

    def _check_auth(self, request):
        return None

    async def _run_agent_bridge(self, **kwargs):
        # Only the external Agent is replaced; admission, journal, session
        # bootstrap, streaming, and slot cleanup use the production path.
        self.calls += 1
        await self.release.wait()
        return {"final_response": "done"}, {}


class TestRuntimeCapacityRetry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.env = patch.dict(os.environ, {
            "HOME": str(self.home),
            "HERMES_HOME": str(self.home / ".hermes"),
            "HERMES_RUNTIME_MAX_CONCURRENT": "1",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.db = SessionDB(db_path=self.home / "state.db")
        self.adapter = CapacityAdapter(self.db)
        self.prefix = uuid.uuid4().hex
        self.slots_before = runtime._ACTIVE_RUN_COUNT
        self.assertEqual(self.slots_before, 0)
        app = web.Application()
        app.router.add_post("/v1/runtime/runs", self.adapter._handle_runtime_run)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.adapter.release.set()
        tasks = [execution.task for execution in self.adapter._execution_registry().values()
                 if execution.task is not None]
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks), 2)
        await self.client.close()
        with runtime._SESSIONS_LOCK:
            for key in list(runtime._SESSIONS):
                if key.startswith(self.prefix):
                    runtime._SESSIONS.pop(key)
        self.db.close()
        self.assertEqual(runtime._ACTIVE_RUN_COUNT, self.slots_before)

    def body(self, name):
        stable = "test instructions"
        return {
            "run_id": self.prefix + name,
            "model": "test-model",
            "intent": "bootstrap",
            "context": {"session_id": self.prefix + name + "_session"},
            "messages": [{"id": "message_1", "role": "user", "content": "hello"}],
            "tools": [],
            "deadline_ms": 30000,
            "system_context": {
                "version": "test/v1", "mode": "replace", "stable": stable,
                "digest": "sha256:" + hashlib.sha256(
                    ("test/v1\nreplace\n" + stable).encode()
                ).hexdigest(),
            },
        }

    async def post(self, name, *, body=None, epoch=1):
        return await self.client.post(
            f"/v1/runtime/runs?execution_id={self.prefix + name}&lease_epoch={epoch}",
            json=self.body(name) if body is None else body,
        )

    async def read_started(self, response):
        self.assertEqual(response.status, 200)
        event = json.loads(await asyncio.wait_for(response.content.readline(), 2))
        self.assertEqual(event["type"], "run_started")

    async def frames(self, response):
        return [json.loads(line) for line in
                (await asyncio.wait_for(response.text(), 2)).splitlines()]

    async def test_rejected_identity_retries_after_capacity_returns(self):
        first = await self.post("first")
        await self.read_started(first)
        rejected = await self.post("retry")
        self.assertEqual(rejected.status, 429)
        self.assertEqual(rejected.headers["Retry-After"], "1")
        self.assertEqual(await rejected.json(), {"error": {
            "code": "runtime_concurrency_exceeded", "retryable": True,
        }})
        self.adapter.release.set()
        self.assertEqual((await self.frames(first))[-1]["type"], "completed")
        retry = await self.post("retry")
        events = await self.frames(retry)
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[-1]["type"], "completed")
        self.assertEqual(self.adapter.calls, 2)
        replay = await self.post("retry")
        self.assertEqual(await self.frames(replay), events)
        self.assertEqual(self.adapter.calls, 2)

    async def test_repeated_overload_does_not_persist_an_execution(self):
        first = await self.post("first")
        await self.read_started(first)
        for _ in range(2):
            rejected = await self.post("retry")
            self.assertEqual(rejected.status, 429)
            self.assertTrue((await rejected.json())["error"]["retryable"])
        with self.db._lock:
            conn = self.db._conn
            assert conn is not None
            row = conn.execute(
                "SELECT execution_id FROM runtime_executions WHERE execution_id=?",
                (self.prefix + "retry",),
            ).fetchone()
        self.assertIsNone(row)
        self.assertEqual(ExecutionJournal(self.db).frames(self.prefix + "retry", 0), [])
        self.assertNotIn(self.prefix + "retry", self.adapter._execution_registry())
        self.assertEqual(self.adapter.calls, 1)

    async def test_running_identity_replays_and_conflicts_even_when_full(self):
        first = await self.post("first")
        await self.read_started(first)
        replay = await self.post("first", epoch=2)
        await self.read_started(replay)
        conflict_body = self.body("first")
        conflict_body["messages"][0]["content"] = "different"
        conflict = await self.post("first", body=conflict_body, epoch=2)
        self.assertEqual(conflict.status, 409)
        self.assertEqual((await conflict.json())["error"]["code"],
                         "runtime_execution_conflict")
        stale = await self.post("first", epoch=1)
        self.assertEqual(stale.status, 409)
        self.assertEqual(self.adapter.calls, 1)
        self.adapter.release.set()
        self.assertEqual((await self.frames(replay))[-1]["type"], "completed")
        first.close()

    async def test_terminal_execution_replays_without_admission(self):
        self.adapter.release.set()
        finished = await self.post("finished")
        expected = await self.frames(finished)
        self.adapter.release.clear()
        active = await self.post("active")
        await self.read_started(active)
        replay = await self.post("finished")
        self.assertEqual(await self.frames(replay), expected)
        changed_run = self.body("finished")
        changed_run["run_id"] = self.prefix + "other"
        self.assertEqual((await self.post("finished", body=changed_run)).status, 409)
        self.assertEqual(self.adapter.calls, 2)

    async def test_missing_live_execution_is_not_readmitted(self):
        body = self.body("missing")
        digest = hashlib.sha256(json.dumps(
            body, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        ExecutionJournal(self.db).create(self.prefix + "missing", body["run_id"], digest)
        with patch.object(runtime, "_acquire_run_slot") as acquire:
            response = await self.post("missing")
        self.assertEqual(response.status, 409)
        self.assertEqual((await response.json())["error"]["code"],
                         "runtime_execution_not_found")
        acquire.assert_not_called()
        self.assertEqual(self.adapter.calls, 0)

    async def test_create_failure_releases_reserved_capacity(self):
        with patch.object(ExecutionJournal, "create", side_effect=OSError("test write failure")):
            with self.assertLogs("aiohttp.server", level="ERROR"):
                response = await self.post("failure")
            self.assertEqual(response.status, 500)
        self.assertEqual(runtime._ACTIVE_RUN_COUNT, self.slots_before)
        self.assertEqual(self.adapter.calls, 0)
        self.adapter.release.set()
        recovered = await self.post("recovered")
        self.assertEqual((await self.frames(recovered))[-1]["type"], "completed")

    async def test_raced_existing_execution_releases_unused_slot(self):
        self.adapter.release.set()
        response = await self.post("finished")
        expected = await self.frames(response)
        # Simulate another writer committing after the read-only lookup.
        with patch.object(ExecutionJournal, "exists", return_value=False):
            replay = await self.post("finished")
            self.assertEqual(await self.frames(replay), expected)
        self.assertEqual(runtime._ACTIVE_RUN_COUNT, self.slots_before)
        self.assertEqual(self.adapter.calls, 1)

    async def test_raced_conflict_releases_unused_slot(self):
        self.adapter.release.set()
        await self.frames(await self.post("finished"))
        changed = self.body("finished")
        changed["model"] = "other-model"
        with patch.object(ExecutionJournal, "exists", return_value=False):
            conflict = await self.post("finished", body=changed)
        self.assertEqual(conflict.status, 409)
        self.assertEqual((await conflict.json())["error"]["code"],
                         "runtime_execution_conflict")
        self.assertEqual(runtime._ACTIVE_RUN_COUNT, self.slots_before)
        self.assertEqual(self.adapter.calls, 1)

    async def test_terminal_error_is_not_retried_after_journal_reopen(self):
        invalid = self.body("invalid")
        invalid["unsupported_field"] = True
        response = await self.post("invalid", body=invalid)
        self.assertEqual(response.status, 422)
        self.assertEqual(self.adapter.calls, 0)
        self.assertEqual(runtime._ACTIVE_RUN_COUNT, self.slots_before)
        self.db.close()
        self.db = SessionDB(db_path=self.home / "state.db")
        self.adapter.db = self.db
        with patch.object(runtime, "_acquire_run_slot") as acquire:
            replay = await self.post("invalid", body=invalid)
            events = await self.frames(replay)
        acquire.assert_not_called()
        self.assertEqual([event["type"] for event in events], ["error"])
        self.assertEqual(events[0]["payload"]["code"], "runtime_unavailable")
        self.assertEqual(self.adapter.calls, 0)

    async def test_rejected_start_can_retry_after_journal_reopen(self):
        first = await self.post("first")
        await self.read_started(first)
        self.assertEqual((await self.post("retry")).status, 429)
        self.adapter.release.set()
        await self.frames(first)
        self.db.close()
        self.db = SessionDB(db_path=self.home / "state.db")
        self.adapter.db = self.db
        events = await self.frames(await self.post("retry"))
        self.assertEqual(events[0]["type"], "run_started")
        self.assertEqual(events[-1]["type"], "completed")
        self.assertEqual(self.adapter.calls, 2)

    async def test_simultaneous_same_identity_starts_only_one_agent(self):
        first, second = await asyncio.gather(self.post("same"), self.post("same"))
        self.assertEqual(first.status, 200)
        self.assertEqual(second.status, 200)
        self.adapter.release.set()
        await self.frames(first)
        await self.frames(second)
        self.assertEqual(self.adapter.calls, 1)
