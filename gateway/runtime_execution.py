"""Durable execution journal; HTTP handlers only subscribe to a live execution."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass, field

from aiohttp import web
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_state import SessionDB
from gateway.runtime_contract_models import encode_runtime_event
from gateway.runtime_session_history import RuntimeSessionStateError


_HEARTBEAT_SECONDS = 15.0


class ExecutionJournal:
    def __init__(self, db: SessionDB) -> None:
        self.db = db

        def initialize(conn: sqlite3.Connection) -> None:
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_executions (
                execution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL,
                request_digest TEXT NOT NULL, finished INTEGER NOT NULL DEFAULT 0,
                lease_epoch INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_execution_frames (
                execution_id TEXT NOT NULL REFERENCES runtime_executions(execution_id) ON DELETE CASCADE,
                seq INTEGER NOT NULL, frame TEXT NOT NULL, PRIMARY KEY(execution_id, seq))""")

        db._execute_write(initialize)

    def create(self, execution_id: str, run_id: str, digest: str) -> bool:
        def write(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT run_id, request_digest FROM runtime_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
            if row:
                if tuple(row) != (run_id, digest):
                    raise RuntimeSessionStateError(
                        "runtime_execution_conflict",
                        "execution identity has different content",
                        409,
                    )
                return False
            conn.execute(
                "INSERT INTO runtime_executions(execution_id,run_id,request_digest) VALUES(?,?,?)",
                (execution_id, run_id, digest),
            )
            return True

        return self.db._execute_write(write)

    def finished(self, execution_id: str, run_id: str) -> bool:
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT finished FROM runtime_executions WHERE execution_id=? AND run_id=?",
                (execution_id, run_id),
            ).fetchone()
        if row is None:
            raise RuntimeSessionStateError(
                "runtime_execution_not_found", "original execution is unavailable", 409
            )
        return bool(row[0])

    def append(self, execution_id: str, event: dict) -> None:
        encoded = encode_runtime_event(event).decode()

        def write(conn: sqlite3.Connection) -> None:
            conn.execute(
                """INSERT INTO runtime_execution_frames(execution_id,seq,frame)
                SELECT ?, COALESCE(MAX(seq),0)+1, ? FROM runtime_execution_frames WHERE execution_id=?""",
                (execution_id, encoded, execution_id),
            )
            if event["type"] in {"completed", "error"}:
                conn.execute(
                    "UPDATE runtime_executions SET finished=1 WHERE execution_id=?",
                    (execution_id,),
                )

        self.db._execute_write(write)

    def claim_subscription(self, execution_id: str, epoch: int) -> None:
        def write(conn: sqlite3.Connection) -> None:
            result = conn.execute(
                "UPDATE runtime_executions SET lease_epoch=? WHERE execution_id=? AND lease_epoch<=?",
                (epoch, execution_id, epoch),
            )
            if result.rowcount != 1:
                raise RuntimeSessionStateError(
                    "runtime_execution_conflict",
                    "subscription belongs to a newer host lease",
                    409,
                )

        self.db._execute_write(write)

    def current_epoch(self, execution_id: str) -> int:
        with self.db._lock:
            row = self.db._conn.execute(
                "SELECT lease_epoch FROM runtime_executions WHERE execution_id=?",
                (execution_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def finish(self, execution_id: str) -> None:
        self.db._execute_write(
            lambda conn: conn.execute(
                "UPDATE runtime_executions SET finished=1 WHERE execution_id=?",
                (execution_id,),
            )
        )

    def frames(self, execution_id: str, after: int) -> list[tuple[int, str]]:
        with self.db._lock:
            rows = self.db._conn.execute(
                "SELECT seq,frame FROM runtime_execution_frames WHERE execution_id=? AND seq>? ORDER BY seq LIMIT 128",
                (execution_id, after),
            ).fetchall()
        return [(int(row[0]), str(row[1])) for row in rows]


@dataclass
class Execution:
    execution_id: str
    run_id: str
    journal: ExecutionJournal
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    session: object | None = None
    task: asyncio.Task | None = None
    subscriber: int = 0

    def emit(self, event: dict, loop: asyncio.AbstractEventLoop) -> None:
        # Persistence precedes visibility, including terminal frames.
        self.journal.append(self.execution_id, event)
        loop.call_soon_threadsafe(self.changed.set)


def execution_id(request: web.Request) -> str:
    value = request.query.get("execution_id", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("execution_id query parameter is required")
    return value


def lease_epoch(request: web.Request) -> int:
    value = request.query.get("lease_epoch", "")
    if not value.isascii() or not value.isdecimal() or len(value) > 19:
        raise ValueError("positive lease_epoch query parameter is required")
    epoch = int(value)
    if not 0 < epoch < 2**63:
        raise ValueError("lease_epoch is out of range")
    return epoch


def controls_execution(request: web.Request, execution: Execution) -> bool:
    try:
        return request.query.get(
            "execution_id"
        ) == execution.execution_id and lease_epoch(
            request
        ) == execution.journal.current_epoch(execution.execution_id)
    except ValueError:
        return False


async def subscribe(request: web.Request, execution: Execution) -> web.StreamResponse:
    execution.journal.claim_subscription(execution.execution_id, lease_epoch(request))
    execution.subscriber += 1
    subscriber = execution.subscriber
    execution.changed.set()
    response = web.StreamResponse(
        headers={
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-store",
            "X-Runtime-Execution-Id": execution.execution_id,
        }
    )
    await response.prepare(request)
    cursor = 0
    try:
        while subscriber == execution.subscriber:
            execution.changed.clear()
            rows = execution.journal.frames(execution.execution_id, cursor)
            for cursor, frame in rows:
                if subscriber != execution.subscriber:
                    return response
                decoded = json.loads(frame)
                if decoded["type"] == "runtime_control_request":
                    session = execution.session
                    if session is None:
                        continue
                    with session.lock:
                        pending = session.pending_controls.get(
                            decoded["payload"]["request_id"]
                        )
                        if pending is None or pending.result is not None:
                            continue
                await response.write(frame.encode() + b"\n")
            if rows:
                continue
            if execution.journal.finished(execution.execution_id, execution.run_id):
                return response
            try:
                await asyncio.wait_for(execution.changed.wait(), _HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                await response.write(
                    encode_runtime_event({
                        "run_id": execution.run_id,
                        "type": "heartbeat",
                        "payload": {},
                    })
                    + b"\n"
                )
    except (ConnectionError, asyncio.CancelledError):
        # An observer disappearing does not own the Agent's lifetime.
        pass
    return response


class RuntimeExecutionMixin:
    def _execution_registry(self) -> dict[str, Execution]:
        if not hasattr(self, "_runtime_executions"):
            self._runtime_executions = {}
        return self._runtime_executions

    async def _handle_runtime_attach(self, request: web.Request) -> web.StreamResponse:
        auth_error = await self._authenticate_runtime_request(request)
        if auth_error:
            return auth_error
        try:
            identity = execution_id(request)
            run_id = request.match_info["run_id"]
            journal = ExecutionJournal(self._ensure_session_db())
            finished = journal.finished(identity, run_id)
            execution = self._execution_registry().get(identity)
            if execution is None:
                if not finished:
                    raise RuntimeSessionStateError(
                        "runtime_execution_not_found",
                        "Runtime process no longer owns the original execution",
                        409,
                    )
                execution = Execution(identity, run_id, journal)
            if execution.run_id != run_id:
                raise RuntimeSessionStateError(
                    "runtime_execution_conflict",
                    "execution belongs to another run",
                    409,
                )
            epoch = lease_epoch(request)
            execution.journal.claim_subscription(identity, epoch)
            if request.method == "POST":
                body = await request.json()
                if not isinstance(body, dict) or set(body) - {
                    "model",
                    "llm_egress",
                    "vision_llm_egress",
                }:
                    raise ValueError("invalid attach request")
                if execution.session is not None and not finished:
                    execution.session.refresh_egress(body)
            return await subscribe(request, execution)
        except RuntimeSessionStateError as exc:
            return web.json_response(
                {"error": {"code": exc.code, "message": str(exc)}}, status=exc.status
            )
        except (ValueError, TypeError) as exc:
            return web.json_response(
                {"error": {"code": "invalid_param", "message": str(exc)}}, status=422
            )
