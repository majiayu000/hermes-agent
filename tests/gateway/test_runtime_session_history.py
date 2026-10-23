from __future__ import annotations

import json
import asyncio

import pytest
from gateway.api_server_runtime import RuntimeBridgeSession
from gateway.runtime_contract_models import decode_runtime_run_request

from gateway.runtime_session_history import resume_session_db_history
from hermes_state import SessionDB


def test_resume_preserves_partial_media_jobs_for_agent_recovery(tmp_path):
    db = SessionDB(db_path=tmp_path / "runtime-state.db")
    session_id = "thread_partial_media"
    try:
        db.create_session(session_id, "api_server")
        db.append_message(
            session_id,
            role="assistant",
            content=None,
            tool_calls=[{
                "id": "call_video",
                "type": "function",
                "function": {
                    "name": "media.generate_video",
                    "arguments": json.dumps({
                        "requests": [
                            {
                                "request_id": "video_v1",
                                "model": "video-model",
                                "prompt": "render",
                            },
                            {
                                "request_id": "video_v2",
                                "model": "video-model",
                                "prompt": "render",
                            },
                        ],
                    }),
                },
            }],
        )
        result = {
            "call_id": "call_video",
            "ok": True,
            "result": {
                "batch_status": "partial",
                "jobs": [
                    {
                        "request_id": "video_v1",
                        "status": "failed",
                        "error": {
                            "code": "content_policy_blocked",
                            "retryable": False,
                        },
                    },
                    {
                        "request_id": "video_v2",
                        "status": "succeeded",
                        "output_ids": ["output_video_v2"],
                    },
                ],
            },
        }

        resumed = resume_session_db_history(
            db,
            session_id,
            db.get_messages_as_conversation(session_id),
            [result],
        )

        projected = json.loads(resumed[-1]["content"])
        assert projected == result["result"]
        assert projected["jobs"][0]["error"]["code"] == "content_policy_blocked"
        assert projected["jobs"][1]["output_ids"] == ["output_video_v2"]
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"ok": True, "result": None},
    {"ok": True, "result": {"jobs": [{"status": "succeeded", "asset_id": "asset_1"}]}},
    {"ok": True, "result": {"status": "externalized", "output_ref": {
        "ref": "toolout_success", "media_type": "application/json", "size_bytes": 90000,
        "sha256": "a" * 64, "preview": "preview", "preview_truncated": True,
    }}},
    {"ok": False, "error": {"code": "host_error", "message": "host message", "source": "gateway"},
     "result": {"code": "provider_error", "message": "provider message",
                "error": {"detail": "nested"}, "jobs": [{"status": "failed"}]}},
    {"ok": False, "error": {"code": "host_error", "message": "host message"},
     "result": {"status": "externalized", "output_ref": {
         "ref": "toolout_failure", "media_type": "application/json", "size_bytes": 90000,
     }}},
    {"ok": False, "error": {"code": "host_error", "message": "host message"}, "result": None},
    {"ok": False, "error": {"code": "host_error", "message": "host message"}},
])
async def test_live_and_replayed_results_have_identical_durable_content(tmp_path, payload):
    db_path = tmp_path / "runtime-result-parity.db"
    db = SessionDB(db_path=db_path)
    result = {"call_id": "call_parity", **payload}
    try:
        db.create_session("session_parity", "api_server")
        db.append_message("session_parity", role="assistant", content=None, tool_calls=[{
            "id": "call_parity", "type": "function",
            "function": {"name": "platform.test", "arguments": "{}"},
        }])
        queue = asyncio.Queue()
        session = RuntimeBridgeSession(
            "run_parity", asyncio.get_running_loop(), queue,
            [{"name": "platform.test", "input_schema": {"type": "object"}}],
            5000, "session_parity", db,
        )
        invocation = asyncio.create_task(asyncio.to_thread(
            session.invoke_platform_tool, "platform.test", {}, "call_parity",
        ))
        try:
            event = await asyncio.wait_for(queue.get(), timeout=5)
            assert event["type"] == "tool_request"
            assert session.submit_result(result)
            live = await asyncio.wait_for(invocation, timeout=5)
        finally:
            if not invocation.done():
                session.submit_result(result)
                await asyncio.wait_for(invocation, timeout=5)
        # Decode the actual run request boundary, then persist via the resume path.
        decoded = decode_runtime_run_request({
            "run_id": "run_parity", "model": "test/model", "intent": "resume",
            "messages": [], "tools": [], "context": {"session_id": "session_parity"},
            "system_context": {"version": "ultra-agent-v1", "mode": "profile"},
            "tool_results": [result],
        })
        resumed = resume_session_db_history(db, "session_parity",
            db.get_messages_as_conversation("session_parity"),
            [item.model_dump(mode="json", exclude_unset=True) for item in decoded.tool_results])
        assert json.loads(resumed[-1]["content"]) == json.loads(live)
        expected = payload.get("result") if payload["ok"] else {
            key: value for key, value in payload.items() if key != "ok"}
        assert json.loads(live) == expected
    finally:
        db.close()
    # A process restart followed by delivery retry must neither conflict nor append.
    reopened = SessionDB(db_path=db_path)
    try:
        history = reopened.get_messages_as_conversation("session_parity")
        assert resume_session_db_history(reopened, "session_parity", history, [result]) == history
        assert len(reopened.get_messages_as_conversation("session_parity")) == 2
    finally:
        reopened.close()
