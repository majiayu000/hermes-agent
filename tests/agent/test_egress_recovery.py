"""Paid request failures retain their identity and never silently resubmit."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from agent.auxiliary_client import _create_openai_client, _to_async_client, async_call_llm, call_llm
from agent.error_classifier import classify_api_error
from agent.run_scoped_auxiliary import bind_run_scoped_auxiliary, reset_run_scoped_auxiliary
from gateway.api_server_runtime import _runtime_failure_code
from gateway.runtime_contract import runtime_error_envelope


@pytest.mark.parametrize("status,code", [(409, "egress_request_replayed"), (502, "egress_result_unknown")])
def test_paid_operation_failure_survives_runtime_projection(status, code):
    response = httpx.Response(status, request=httpx.Request("POST", "https://egress.example.test/v1/chat/completions"))
    error = openai.APIStatusError("Original operation result unavailable", response=response, body={"code": code})
    classified = classify_api_error(error, provider="custom", model="test-model")
    assert classified.reason.value == code
    assert not classified.retryable
    assert not classified.should_fallback
    assert not classified.should_rotate_credential
    projected = _runtime_failure_code({"failed": True, "failure_reason": classified.reason.value})
    assert projected == code
    envelope = runtime_error_envelope(projected, support_id="run-test")
    assert not envelope["retryable"]
    assert "model" in envelope["message"].lower()
    assert "incompatible" not in envelope["message"].lower()


def test_sync_auxiliary_disables_hidden_sdk_retries():
    with patch("agent.auxiliary_client.OpenAI") as constructor, patch("agent.auxiliary_requests.build_keepalive_http_client", return_value=None):
        _create_openai_client(api_key="test-only-key", base_url="https://egress.example.test/v1")
    assert constructor.call_args.kwargs["max_retries"] == 0


def test_managed_sync_vision_transport_failure_does_not_resubmit():
    client = MagicMock()
    client.base_url = "https://egress.example.test/v1"
    failure = openai.APIConnectionError(request=httpx.Request("POST", client.base_url))
    client.chat.completions.create.side_effect = failure
    binding = bind_run_scoped_auxiliary({"vision": {"model": "test-model", "base_url": client.base_url, "grant": "test-only-grant"}})
    try:
        with patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=("custom", client, "test-model")):
            with pytest.raises(openai.APIConnectionError):
                call_llm(task="vision", messages=[{"role": "user", "content": "inspect"}], max_tokens=32)
        assert client.chat.completions.create.call_count == 1
    finally:
        reset_run_scoped_auxiliary(binding)


def test_managed_pre_submission_failure_remains_retryable():
    response = httpx.Response(502, request=httpx.Request("POST", "https://egress.example.test/v1"), headers={"X-Ultra-Submission-State": "not-started"})
    error = openai.APIStatusError("gateway could not begin submission", response=response, body={"message": "gateway unavailable"})
    classified = classify_api_error(error, managed_egress=True)
    assert classified.reason.value == "server_error"
    assert classified.retryable


def test_generic_provider_conflict_is_not_invented_as_an_egress_replay():
    response = httpx.Response(409, request=httpx.Request("POST", "https://provider.example.test/v1"), headers={"x-should-retry": "false"})
    error = openai.APIStatusError("provider revision conflict", response=response, body={"message": "revision conflict"})
    classified = classify_api_error(error)
    assert classified.status_code == 409
    assert classified.reason.value not in {"egress_request_replayed", "egress_result_unknown"}


@pytest.mark.asyncio
async def test_async_auxiliary_disables_hidden_sdk_retries():
    with openai.OpenAI(api_key="test-only-key", base_url="https://egress.example.test/v1") as sync:
        with patch("agent.auxiliary_requests.build_keepalive_http_client", return_value=None), patch("agent.auxiliary_client._apply_user_default_headers", return_value=None):
            client, model = _to_async_client(sync, "test-model")
        try:
            assert client.max_retries == 0
            assert model == "test-model"
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_managed_vision_transport_failure_does_not_resubmit():
    client = MagicMock()
    client.base_url = "https://egress.example.test/v1"
    failure = openai.APIConnectionError(request=httpx.Request("POST", client.base_url))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="unexpected second result"))])
    client.chat.completions.create = AsyncMock(side_effect=[failure, response])
    binding = bind_run_scoped_auxiliary({"vision": {"model": "test-model", "base_url": client.base_url, "grant": "test-only-grant"}})
    try:
        with patch("agent.auxiliary_client.resolve_vision_provider_client", return_value=("custom", client, "test-model")):
            with pytest.raises(openai.APIConnectionError):
                await async_call_llm(task="vision", messages=[{"role": "user", "content": "inspect"}], max_tokens=32)
        assert client.chat.completions.create.await_count == 1
    finally:
        reset_run_scoped_auxiliary(binding)
