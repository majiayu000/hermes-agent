from __future__ import annotations

import json

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
            "tool_call_id": "call_video",
            "status": "succeeded",
            "output": {
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
        assert projected == result["output"]
        assert projected["jobs"][0]["error"]["code"] == "content_policy_blocked"
        assert projected["jobs"][1]["output_ids"] == ["output_video_v2"]
    finally:
        db.close()
