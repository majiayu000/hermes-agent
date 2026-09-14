"""Exercise paid-egress recovery through the real conversation and stream loops."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent.error_types import EgressResultUnknown
from agent.conversation_recovery import prepare_tool_call_retry
from agent.transports.types import NormalizedResponse, ToolCall
from gateway.runtime_egress import _configure_run_llm_egress
from run_agent import AIAgent
from tests.run_agent.test_run_agent import (
    _make_chunk, _make_tc_delta, _make_tool_defs, _mock_response, _mock_tool_call,
)


@pytest.fixture
def managed_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("write_file")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.model_metadata.get_model_context_length", return_value=256000),
    ):
        agent = AIAgent(
            model="test-model", provider="custom", api_key="test-only-grant",
            base_url="https://egress.example.test/internal/llm/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            max_tokens=65536,
        )
    _configure_run_llm_egress(agent, {"grant": agent.api_key, "base_url": agent.base_url}, agent.model)
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        yield agent


def test_completed_truncation_at_cap_corrects_request_before_tool_execution(managed_agent):
    agent = managed_agent
    invalid = '{"path":"report.md","content":"unfinished'
    responses = iter([
        _mock_response(content="", finish_reason="length", tool_calls=[_mock_tool_call("write_file", invalid)]),
        _mock_response(content="", tool_calls=[_mock_tool_call("write_file", '{"path":"report.md","content":"complete"}')]),
        _mock_response(content="Done!"),
    ])
    requests = []

    def complete(**kwargs):
        requests.append(deepcopy(kwargs))
        return next(responses)

    agent.client.chat.completions.create.side_effect = complete
    with patch("run_agent.handle_function_call", return_value='{"success":true}') as execute:
        result = agent.run_conversation("write the report")
    assert result["completed"]
    assert result["final_response"] == "Done!"
    execute.assert_called_once()
    assert len(requests) == 3
    assert requests[0]["max_tokens"] == requests[1]["max_tokens"] == 65536
    assert requests[0]["messages"] != requests[1]["messages"]
    import json
    assert invalid in json.loads(requests[1]["messages"][-1]["content"].split("Previous response evidence:\n")[1])["tool_calls"][0]["arguments"]


@pytest.mark.parametrize("kind", ["replay", "unknown", "transport", "provider_502"])
def test_uncertain_operation_never_activates_fallback(managed_agent, kind):
    agent = managed_agent
    request = httpx.Request("POST", agent.base_url)
    expected = "egress_request_replayed" if kind == "replay" else "egress_result_unknown"
    if kind == "provider_502":
        expected = "server_error"
    if kind == "transport":
        error = openai.APIConnectionError(request=request)
    else:
        response = httpx.Response(409 if kind == "replay" else 502, request=request)
        body = {"message": "upstream 502 diagnostic"} if kind == "provider_502" else {"code": expected}
        error = openai.APIStatusError("original model operation unavailable", response=response, body=body)
    agent.client.chat.completions.create.side_effect = error
    with patch.object(agent, "_try_activate_fallback", return_value=True) as fallback:
        result = agent.run_conversation("continue")
    assert result["failed"]
    assert result["failure_reason"] == expected
    assert agent.client.chat.completions.create.call_count == 1
    fallback.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
def test_managed_stream_disconnect_never_reopens(managed_agent, partial):
    agent = managed_agent

    def chunks():
        if partial:
            yield _make_chunk(content="partial result")
        raise httpx.RemoteProtocolError("peer closed connection")

    agent.client.chat.completions.create.return_value = chunks()
    with pytest.raises(httpx.RemoteProtocolError):
        agent._interruptible_streaming_api_call({"model": "test-model", "messages": []})
    assert agent.client.chat.completions.create.call_count == 1


def test_managed_stream_without_terminal_chunk_is_unknown(managed_agent):
    agent = managed_agent
    agent.client.chat.completions.create.return_value = iter([
        _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"unfinished')]),
    ])
    with pytest.raises(EgressResultUnknown):
        agent._interruptible_streaming_api_call({"model": "test-model", "messages": []})
    assert agent.client.chat.completions.create.call_count == 1


def test_completed_non_chat_response_uses_normalized_tool_evidence(managed_agent):
    message = NormalizedResponse(content="partial", finish_reason="length", tool_calls=[ToolCall(id="call_1", name="write_file", arguments='{"path":')])
    messages, api_messages = [], []
    prepare_tool_call_retry(managed_agent, SimpleNamespace(id="anthropic-message"), message, {}, 1, messages, api_messages)
    assert "write_file" in messages[-1]["content"]
    assert messages == api_messages


def test_uncertain_stub_does_not_become_a_corrected_paid_request(managed_agent):
    from hermes_constants import PARTIAL_STREAM_STUB_ID
    messages, api_messages = [], []
    with pytest.raises(EgressResultUnknown):
        message = NormalizedResponse(content=None, tool_calls=None, finish_reason="length")
        prepare_tool_call_retry(managed_agent, SimpleNamespace(id=PARTIAL_STREAM_STUB_ID), message, {}, 1, messages, api_messages)
    assert messages == api_messages == []
