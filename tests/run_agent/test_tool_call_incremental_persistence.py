"""Behavior contracts for incremental tool-call persistence (#49045).

A destructive or process-terminating tool that runs during tool execution
must not lose the just-executed assistant(tool_calls) block or the tool
results that were produced before it fired.  These tests pin the contract:

    1. run_conversation flushes the assistant tool-call turn to the session
       DB BEFORE handing control to _execute_tool_calls (so a tool that
       restarts/kills the process never orphans the tool-call block).
    2. The SEQUENTIAL tool path flushes each tool result to the session DB
       immediately after appending it — BEFORE the next tool dispatches.
    3. The CONCURRENT tool path flushes each tool result in append order.

These exercise the REAL production dispatch surfaces:

    * sequential -> ``run_agent.handle_function_call`` (tool_executor ~1256/1298)
    * concurrent -> ``agent._invoke_tool`` (tool_executor ~539)

Mocking the genuine dispatch surface keeps the tests deterministic (no real
``web_search`` / network) AND mutation-survivable: the ordering assertions
read snapshots captured at flush time, so removing any production flush call
makes the corresponding assertion fail.
"""

import copy
from types import SimpleNamespace
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_dispatch_helpers import DeferredToolResult, make_tool_result_message
from agent.turn_finalizer import finalize_turn
from gateway.runtime_session_history import (
    RuntimeSessionStateError,
    resume_session_db_history,
)
from hermes_state import SessionDB
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            model="test/model",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name="web_search", arguments="{}", call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


# ---------------------------------------------------------------------------
# Contract 1: run_conversation persists the assistant tool-call block BEFORE
# tool execution begins.
# ---------------------------------------------------------------------------
def test_run_conversation_flushes_assistant_tool_call_before_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tool_call]),
        _mock_response(content="done", finish_reason="stop"),
    ]

    # Record a deep snapshot of the message list at every flush so the
    # assertion does not depend on later mutations.
    flush_snapshots: list[list] = []

    def _record_flush(messages, conversation_history=None):
        flush_snapshots.append(copy.deepcopy(messages))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    # Capture observations at execute time into module-level lists rather than
    # asserting inside _execute_tool_calls — run_conversation's outer loop
    # swallows exceptions, so an in-callback assertion would never surface.
    executed = {"count": 0}
    snapshot_at_execute: list = []

    def _fake_execute(assistant_message, messages, effective_task_id, api_call_count=0):
        executed["count"] += 1
        # Record the DB state observed at the moment tool execution begins.
        snapshot_at_execute.append(
            copy.deepcopy(flush_snapshots[-1]) if flush_snapshots else None
        )
        # Simulate the tool producing a result (as the real path would).
        messages.append(make_tool_result_message("web_search", "search result", "c1"))

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=_fake_execute),
    ):
        result = agent.run_conversation("search something")

    assert executed["count"] == 1, "_execute_tool_calls was never reached"
    # The assistant tool-call block MUST have been flushed before execution.
    last = snapshot_at_execute[0]
    assert last is not None, "no flush occurred before tool execution"
    assert last[-1]["role"] == "assistant"
    assert last[-1]["tool_calls"][0]["id"] == "c1"
    assert result["final_response"] == "done"


def test_required_session_persistence_failure_stops_before_tool_execution():
    agent = _make_agent()
    tool_call = _mock_tool_call(call_id="c1")
    agent.client.chat.completions.create.return_value = _mock_response(
        content="",
        finish_reason="tool_calls",
        tool_calls=[tool_call],
    )
    agent._require_incremental_session_persistence = True

    def _fail_required_flush(*_args, **_kwargs):
        agent._runtime_persistence_failed = True
        raise RuntimeError("state.db unavailable")

    with (
        patch.object(agent, "_flush_messages_to_session_db", side_effect=_fail_required_flush),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls") as execute,
    ):
        result = agent.run_conversation("search something")

    execute.assert_not_called()
    assert result["failed"] is True
    assert result["completed"] is False


def test_required_session_db_append_error_is_not_swallowed():
    agent = _make_agent()
    agent.session_id = "thread_required"
    agent._session_db_created = True
    agent._require_incremental_session_persistence = True
    agent._session_db = SimpleNamespace(
        append_message=MagicMock(side_effect=RuntimeError("database is locked")),
    )

    with pytest.raises(RuntimeError, match="database is locked"):
        agent._flush_messages_to_session_db([{"role": "user", "content": "hello"}])

    assert agent._runtime_persistence_failed is True


def test_runtime_wire_message_id_round_trips_through_real_agent_flush(tmp_path):
    agent = _make_agent()
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "thread_wire_id"
    db.create_session(session_id, "api_server", model="test/model")
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._require_incremental_session_persistence = True
    agent.client.chat.completions.create.return_value = _mock_response(
        content="done",
        finish_reason="stop",
    )

    try:
        with (
            patch.object(agent, "_save_session_log"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation(
                "current Runtime turn",
                runtime_message_id="wire-user-001",
            )

        assert result["final_response"] == "done"
        persisted = db.get_messages_as_conversation(session_id)
        user_messages = [message for message in persisted if message["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "current Runtime turn"
        assert user_messages[0]["message_id"] == "wire-user-001"
        assert db.has_platform_message_id(session_id, "wire-user-001") is True
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Contract 2: the SEQUENTIAL path flushes each tool result immediately, BEFORE
# the next tool dispatches.  Dispatch goes through run_agent.handle_function_call
# (the real production surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_sequential_flushes_each_tool_result_before_next_dispatch():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    # Ordered event log interleaving real dispatches and DB flushes.
    events: list = []

    def _fake_dispatch(function_name, function_args, effective_task_id, **kwargs):
        # The result for call N must have been flushed before call N+1 fires.
        events.append(("dispatch", kwargs.get("tool_call_id")))
        return f"result-{kwargs.get('tool_call_id')}"

    def _record_flush(flush_messages, conversation_history=None):
        # Snapshot the tail tool result that triggered this flush.
        tail = flush_messages[-1]
        events.append(("flush", tail.get("role"), tail.get("tool_call_id")))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch("run_agent.handle_function_call", side_effect=_fake_dispatch) as disp,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    # The mock proves we exercised the REAL sequential dispatch surface.
    assert disp.call_count == 2, "sequential path did not dispatch via handle_function_call"

    # Both tool results landed, in order.
    assert [m["role"] for m in messages] == ["tool", "tool"]
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Ordering contract: each tool result is flushed AFTER its own dispatch
    # and BEFORE the next dispatch. Expected interleaving:
    #   dispatch c1 -> flush c1 -> dispatch c2 -> flush c2
    assert events == [
        ("dispatch", "c1"),
        ("flush", "tool", "c1"),
        ("dispatch", "c2"),
        ("flush", "tool", "c2"),
    ]


def test_execute_tool_calls_sequential_does_not_persist_deferred_runtime_result():
    agent = _make_agent()
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[_mock_tool_call(name="media.generate_image", call_id="call_park")],
    )
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock()

    with patch(
        "run_agent.handle_function_call",
        return_value=DeferredToolResult("call_park"),
    ):
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "thread_session",
        )

    assert messages == []
    agent._flush_messages_to_session_db.assert_not_called()


def test_deferred_runtime_call_closes_unexecuted_sibling_tool_calls():
    agent = _make_agent()
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[
            _mock_tool_call(name="media.generate_image", call_id="call_park"),
            _mock_tool_call(name="media.generate_video", call_id="call_skipped"),
        ],
    )
    messages: list = []
    agent._flush_messages_to_session_db = MagicMock()

    with patch(
        "run_agent.handle_function_call",
        return_value=DeferredToolResult("call_park"),
    ) as dispatch:
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "thread_session",
        )

    dispatch.assert_called_once()
    assert len(messages) == 1
    assert messages[0]["tool_call_id"] == "call_skipped"
    assert "runtime_tool_call_deferred" in messages[0]["content"]
    agent._flush_messages_to_session_db.assert_called_once()


def test_runtime_park_tail_reopens_and_resumes_real_result_exactly_once(tmp_path):
    agent = _make_agent()
    db_path = tmp_path / "state.db"
    session_id = "thread_runtime_park"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id, "api_server", model="test/model")
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    agent._require_incremental_session_persistence = True

    tool_calls = [
        _mock_tool_call(name="media.generate_image", call_id="call_park"),
        _mock_tool_call(name="media.generate_video", call_id="call_sibling"),
    ]
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)
    messages = [
        {
            "role": "user",
            "content": "make both",
            "message_id": "wire-user-park",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        },
    ]
    agent._flush_messages_to_session_db(messages)

    with patch(
        "run_agent.handle_function_call",
        return_value=DeferredToolResult("call_park"),
    ) as dispatch:
        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            session_id,
        )

    dispatch.assert_called_once()
    assert agent._runtime_deferred_tool_call_id == "call_park"
    assert [
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    ] == ["call_sibling"]
    assert "runtime_tool_call_deferred" in messages[-1]["content"]

    with (
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        finalize_turn(
            agent,
            final_response=None,
            api_call_count=1,
            interrupted=True,
            failed=False,
            messages=messages,
            conversation_history=None,
            effective_task_id=session_id,
            turn_id="turn-park",
            user_message="make both",
            original_user_message="make both",
            _should_review_memory=False,
            _turn_exit_reason="runtime_tool_call_deferred",
        )

    assert all(
        message.get("content") != "Operation interrupted."
        for message in messages
    )
    db.close()

    reopened = SessionDB(db_path=db_path)
    history = reopened.get_messages_as_conversation(session_id)
    assistant_calls = {
        call["id"]
        for message in history
        if message["role"] == "assistant"
        for call in message.get("tool_calls") or []
    }
    durable_results = [
        message
        for message in history
        if message["role"] == "tool"
    ]
    assert assistant_calls == {"call_park", "call_sibling"}
    assert [message["tool_call_id"] for message in durable_results] == ["call_sibling"]

    real_result = {
        "call_id": "call_park",
        "ok": True,
        "result": {"asset_id": "asset-real"},
    }
    resumed = resume_session_db_history(
        reopened,
        session_id,
        history,
        [real_result],
    )
    assert [message["tool_call_id"] for message in resumed if message["role"] == "tool"] == [
        "call_sibling",
        "call_park",
    ]
    persisted_once = reopened.get_messages_as_conversation(session_id)
    redelivered = resume_session_db_history(
        reopened,
        session_id,
        persisted_once,
        [real_result],
    )
    assert redelivered == persisted_once
    assert len(reopened.get_messages_as_conversation(session_id)) == len(persisted_once)

    conflicting_result = {
        **real_result,
        "result": {"asset_id": "asset-conflict"},
    }
    with pytest.raises(RuntimeSessionStateError) as conflict:
        resume_session_db_history(
            reopened,
            session_id,
            persisted_once,
            [conflicting_result],
        )
    assert conflict.value.code == "runtime_history_conflict"
    assert len(reopened.get_messages_as_conversation(session_id)) == len(persisted_once)
    reopened.close()


# ---------------------------------------------------------------------------
# Contract 3: the CONCURRENT path flushes each collected tool result in append
# order.  Dispatch goes through agent._invoke_tool (the real concurrent
# surface), which we mock for determinism.
# ---------------------------------------------------------------------------
def test_execute_tool_calls_concurrent_flushes_each_tool_result_in_order():
    agent = _make_agent()
    tool_calls = [
        _mock_tool_call(name="web_search", call_id="c1"),
        _mock_tool_call(name="web_search", call_id="c2"),
    ]
    messages: list = []
    assistant_message = SimpleNamespace(content="", tool_calls=tool_calls)

    invoked_ids: list = []

    def _fake_invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        invoked_ids.append(tool_call_id)
        return f"result-{tool_call_id}"

    # Each flush must observe exactly one more tool result than the previous
    # flush, in append order — i.e. the tail tool_call_id sequence is c1, c2.
    flushed_tool_ids: list = []
    flush_lengths: list = []

    def _record_flush(flush_messages, conversation_history=None):
        flushed_tool_ids.append(flush_messages[-1]["tool_call_id"])
        flush_lengths.append(len([m for m in flush_messages if m.get("role") == "tool"]))

    agent._flush_messages_to_session_db = MagicMock(side_effect=_record_flush)

    with (
        patch.object(agent, "_invoke_tool", side_effect=_fake_invoke) as inv,
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task-1")

    # Proves the real concurrent dispatch surface was exercised.
    assert inv.call_count == 2, "concurrent path did not dispatch via _invoke_tool"
    assert sorted(invoked_ids) == ["c1", "c2"]

    # Results appended in deterministic order.
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]

    # Each tool result was flushed exactly once, in append order, with the
    # running tool count growing by one each time (1 then 2).  Removing either
    # production flush call breaks one of these assertions.
    assert flushed_tool_ids == ["c1", "c2"]
    assert flush_lengths == [1, 2]


def test_parked_tool_skips_context_compression_and_extra_model_calls():
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response(
        content="", finish_reason="tool_calls", tool_calls=[_mock_tool_call(call_id="call_park")],
    )
    agent.compression_enabled = True
    agent.context_compressor.last_prompt_tokens = 100_000

    def park(*_args, **_kwargs):
        agent._runtime_deferred_tool_call_id = "call_park"
        agent.iteration_budget.exhaust()

    with (
        patch.object(agent, "_persist_session") as persist,
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch.object(agent, "_execute_tool_calls", side_effect=park),
        patch.object(agent.context_compressor, "should_compress", return_value=True),
        patch.object(agent, "_compress_context") as compress,
        patch.object(agent, "_handle_max_iterations") as summarize,
    ):
        agent.run_conversation("test suspension")

    compress.assert_not_called()
    summarize.assert_not_called()
    assert agent.client.chat.completions.create.call_count == 1
    assert persist.called
    assert persist.call_args.args[0][-1]["tool_calls"][0]["id"] == "call_park"
    assert agent._session_messages[-1]["tool_calls"][0]["id"] == "call_park"
