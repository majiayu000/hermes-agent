"""web_search / web_extract must surface as private activity events.

The runtime whitelists five native tools; before this coverage the web pair
ran with zero observability. Activities must emit started/completed events
with empty arguments so queries and URLs stay private. Media activities are
stricter and omit the arguments field entirely; keep the distinction explicit.
"""

from __future__ import annotations

import asyncio

from gateway.api_server_runtime import _LOCAL_ACTIVITY_TOOLS, RuntimeBridgeSession


def _make_session(queue: asyncio.Queue) -> RuntimeBridgeSession:
    return RuntimeBridgeSession(
        "run_web",
        asyncio.new_event_loop(),
        queue,
        [],
        1_000,
        "agent_web",
    )


def _drain(session: RuntimeBridgeSession, queue: asyncio.Queue) -> list[dict]:
    # emit() schedules via call_soon_threadsafe when no loop is running;
    # spin the session loop once so queued callbacks land in the queue.
    session.loop.run_until_complete(asyncio.sleep(0))
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_web_native_tools_are_local_activities():
    assert {"web_search", "web_extract"} <= _LOCAL_ACTIVITY_TOOLS


def test_web_search_emits_private_started_and_completed_events():
    queue: asyncio.Queue = asyncio.Queue()
    session = _make_session(queue)
    try:
        session.start_local_activity(
            "call_ws", "web_search", {"query": "secret question"}
        )
        session.complete_local_activity(
            "call_ws", "web_search", {"query": "secret question"}, '{"ok": true}'
        )
        events = _drain(session, queue)
        assert [e["type"] for e in events] == [
            "activity_started",
            "activity_completed",
        ]
        started, completed = events
        assert started["payload"]["name"] == "web_search"
        # Arguments must stay private: no query text crosses the event stream.
        assert started["payload"]["arguments"] == {}
        assert completed["payload"]["status"] == "completed"
        assert "arguments" not in completed["payload"]
        assert "secret question" not in str(events)
    finally:
        session.loop.close()


def test_web_extract_failure_reports_failed_status_without_arguments():
    queue: asyncio.Queue = asyncio.Queue()
    session = _make_session(queue)
    try:
        session.start_local_activity(
            "call_we", "web_extract", {"url": "https://example.com/private"}
        )
        session.complete_local_activity(
            "call_we",
            "web_extract",
            {"url": "https://example.com/private"},
            '{"success": false, "error": "fetch failed"}',
        )
        events = _drain(session, queue)
        assert [e["type"] for e in events] == [
            "activity_started",
            "activity_completed",
        ]
        completed = events[1]["payload"]
        assert completed["name"] == "web_extract"
        assert completed["status"] == "failed"
        assert completed["error"]["code"] == "runtime_activity_failed"
        assert "https://example.com/private" not in str(events)
    finally:
        session.loop.close()


def test_nested_error_envelope_reports_failed_activity():
    queue: asyncio.Queue = asyncio.Queue()
    session = _make_session(queue)
    try:
        session.start_local_activity(
            "call_va", "video_analyze", {"video_url": "private.mp4"}
        )
        session.complete_local_activity(
            "call_va",
            "video_analyze",
            {"video_url": "private.mp4"},
            '{"error":{"code":"repeated_non_retryable_tool_call",'
            '"message":"The earlier video analysis failed.","retryable":false}}',
        )
        events = _drain(session, queue)
        completed = events[1]["payload"]
        assert completed["status"] == "failed"
        assert completed["error"] == {
            "code": "runtime_activity_failed",
            "message": "The earlier video analysis failed.",
            "retryable": False,
        }
        assert "private.mp4" not in str(events)
    finally:
        session.loop.close()


def test_non_activity_tool_emits_nothing():
    queue: asyncio.Queue = asyncio.Queue()
    session = _make_session(queue)
    try:
        session.start_local_activity("call_t", "terminal", {"command": "ls"})
        session.complete_local_activity("call_t", "terminal", {"command": "ls"}, "ok")
        assert _drain(session, queue) == []
    finally:
        session.loop.close()
