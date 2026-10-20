"""Durable user additions to a live Runtime execution, consumed by the Agent loop."""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from aiohttp import web
from gateway.runtime_contract_models import RuntimeUserInput
from gateway.runtime_execution import controls_execution, execution_id

if TYPE_CHECKING:
    from gateway.api_server_runtime import RuntimeBridgeSession


class RuntimeInputs:
    def __init__(self, session: RuntimeBridgeSession) -> None:
        self.session = session
        self.accepting = True
        session.session_db._execute_write(lambda conn: conn.execute("""
            CREATE TABLE IF NOT EXISTS runtime_user_inputs (
                run_id TEXT NOT NULL, message_id TEXT NOT NULL, payload TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0, ordinal INTEGER NOT NULL,
                PRIMARY KEY(run_id, message_id))
        """))

    def accept(self, body: RuntimeUserInput) -> None:
        session = self.session
        raw = body.model_dump_json(exclude_none=True)
        with session.lock:
            def save(conn: sqlite3.Connection) -> bool:
                prior = conn.execute(
                    "SELECT payload,applied FROM runtime_user_inputs WHERE run_id=? AND message_id=?",
                    (session.run_id, body.message_id),
                ).fetchone()
                if prior:
                    if json.loads(prior[0]) != json.loads(raw):
                        raise ValueError("message_id has different content")
                    return bool(prior[1])
                if not self.accepting or session.finished.is_set() or session.interrupted.is_set():
                    raise ValueError("original execution is no longer accepting input")
                conn.execute("""INSERT INTO runtime_user_inputs(run_id,message_id,payload,ordinal)
                    SELECT ?,?,?,COALESCE(MAX(ordinal),0)+1 FROM runtime_user_inputs WHERE run_id=?""",
                    (session.run_id, body.message_id, raw, session.run_id))
                return False
            applied = session.session_db._execute_write(save)
            # Re-announce the durable receipt on retries/re-attachment. Events
            # are idempotent projections; context consumption remains once.
            session.emit("input_received", body.model_dump(exclude_none=True))
            if applied:
                session.emit("input_applied", {"message_id": body.message_id})

    def consume(self, messages: list[dict], history: list[dict], finishing: bool = False) -> bool:
        """The Agent calls this only at a closed tool boundary or before finality.

        Check-for-input and closing acceptance share one lock: a message is either
        incorporated or explicitly rejected; it cannot disappear after a final answer.
        """
        session = self.session
        with session.lock:
            db = session.session_db
            with db._lock:
                rows = db._conn.execute(
                    "SELECT message_id,payload FROM runtime_user_inputs WHERE run_id=? AND applied=0 ORDER BY ordinal",
                    (session.run_id,),
                ).fetchall()
            if not rows:
                if finishing:
                    self.accepting = False
                return False
            for message_id, raw in rows:
                body = RuntimeUserInput.model_validate_json(raw)
                text = body.text
                if body.attachment_asset_ids:
                    text += "\n[User attached asset references]\n" + json.dumps(body.attachment_asset_ids, ensure_ascii=False)
                identity = f"input_{session.run_id}_{message_id}"
                if not any(item.get("message_id") == identity for item in messages):
                    messages.append({"role": "user", "content": text, "message_id": identity})
                agent = session.agent_ref[0]
                agent._session_messages = messages
                agent._persist_session(messages, history)
                db._execute_write(lambda conn: conn.execute(
                    "UPDATE runtime_user_inputs SET applied=1 WHERE run_id=? AND message_id=?",
                    (session.run_id, message_id),
                ))
                session.emit("input_applied", {"message_id": message_id})
            return True


class RuntimeInputMixin:
    async def _handle_runtime_input(self, request: web.Request) -> web.Response:
        auth_error = await self._authenticate_runtime_request(request)
        if auth_error:
            return auth_error
        try:
            identity = execution_id(request)
            execution = self._execution_registry().get(identity)
            if execution is None or execution.run_id != request.match_info["run_id"] or not controls_execution(request, execution):
                return web.json_response({"error": {"code": "runtime_execution_conflict", "message": "original execution is unavailable"}}, status=409)
            body = RuntimeUserInput.model_validate(await request.json())
            if execution.session is None:
                return web.json_response({"error": {"code": "runtime_execution_not_ready"}}, status=409)
            execution.session.inputs.accept(body)
            return web.Response(status=204)
        except (ValueError, TypeError) as exc:
            return web.json_response({"error": {"code": "runtime_input_rejected", "message": str(exc)}}, status=409)
