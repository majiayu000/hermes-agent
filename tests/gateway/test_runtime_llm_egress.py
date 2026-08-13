from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.api_server_runtime import (
    _RUNTIME_LLM_MAX_OUTPUT_TOKENS,
    _configure_run_llm_egress,
    _runtime_auxiliary_llm_egress,
    _runtime_failure_code,
    _runtime_llm_egress,
)
from gateway.runtime_contract import RUNTIME_CAPABILITIES, runtime_error_envelope


def capability(**overrides):
    value = {
        "base_url": "http://agent-orchestrator:8093/internal/llm/v1",
        "grant": "ueg_" + "a" * 43,
        "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
    }
    value.update(overrides)
    return value


def test_runtime_llm_egress_validates_private_capability():
    assert _runtime_llm_egress(capability(), required=True)["grant"].startswith("ueg_")
    assert _runtime_llm_egress(None, required=False) is None
    with pytest.raises(ValueError, match="must contain"):
        _runtime_llm_egress(None, required=True)
    with pytest.raises(ValueError, match="base_url"):
        _runtime_llm_egress(capability(base_url="https://api.atlascloud.ai/v1"), required=True)
    with pytest.raises(ValueError, match="grant"):
        _runtime_llm_egress(capability(grant="shared-api-key"), required=True)
    with pytest.raises(ValueError, match="expired"):
        _runtime_llm_egress(
            capability(expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()),
            required=True,
        )


def test_runtime_vision_llm_egress_requires_exact_model_bound_capability():
    value = {
        **capability(),
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
    }
    assert _runtime_auxiliary_llm_egress(
        value,
        field_name="vision_llm_egress",
    ) == value
    with pytest.raises(ValueError, match="vision_llm_egress.model"):
        _runtime_auxiliary_llm_egress(
            {**capability(), "model": ""},
            field_name="vision_llm_egress",
        )
    with pytest.raises(ValueError, match="vision_llm_egress must contain"):
        _runtime_auxiliary_llm_egress(
            {**value, "provider": "ambient"},
            field_name="vision_llm_egress",
        )


def test_configure_run_llm_egress_rebuilds_run_scoped_client():
    value = capability()
    agent = SimpleNamespace(
        model="moonshotai/kimi-k3",
        provider="custom",
        api_key=value["grant"],
        base_url=value["base_url"],
        api_mode="chat_completions",
    )
    _configure_run_llm_egress(agent, value, "moonshotai/kimi-k3")
    assert agent.max_tokens == _RUNTIME_LLM_MAX_OUTPUT_TOKENS
    agent.api_key = "shared-key"
    with pytest.raises(ValueError, match="inconsistent"):
        _configure_run_llm_egress(agent, value, "moonshotai/kimi-k3")


def test_runtime_contract_and_billing_failure_are_account_aware():
    assert "llm_egress" in RUNTIME_CAPABILITIES
    assert "vision_llm_egress" in RUNTIME_CAPABILITIES
    assert _runtime_failure_code({"error": 'HTTP 402: {"msg":"insufficient balance"}'}) == "insufficient_credits"
    envelope = runtime_error_envelope("insufficient_credits", support_id="run_1")
    assert envelope["code"] == "insufficient_credits"
    assert envelope["retryable"] is False


@patch("gateway.platforms.api_server.AIOHTTP_AVAILABLE", True)
def test_run_scoped_agent_creation_never_resolves_shared_credentials():
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig())
    runtime = {
        "api_key": "ueg_" + "a" * 43,
        "base_url": "http://agent-orchestrator:8093/internal/llm/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "max_tokens": _RUNTIME_LLM_MAX_OUTPUT_TOKENS,
    }
    with (
        patch("gateway.run._resolve_runtime_agent_kwargs") as shared_resolver,
        patch("gateway.run._resolve_gateway_model") as shared_model,
        patch("gateway.run._load_gateway_config", return_value={}),
        patch("run_agent.AIAgent", return_value=MagicMock()) as agent_class,
    ):
        adapter._create_agent(
            runtime_overrides=runtime,
            model_override="zai-org/glm-5.2",
        )

    shared_resolver.assert_not_called()
    shared_model.assert_not_called()
    assert agent_class.call_args.kwargs["api_key"] == runtime["api_key"]
    assert agent_class.call_args.kwargs["model"] == "zai-org/glm-5.2"
    assert agent_class.call_args.kwargs["max_tokens"] == _RUNTIME_LLM_MAX_OUTPUT_TOKENS


def test_runtime_budget_exhaustion_is_safe_and_not_retryable():
    from agent.error_classifier import FailoverReason, classify_api_error

    error = RuntimeError(
        "Error code: 429 - {'error': {'code': 'run_budget_exhausted', "
        "'message': 'The model request could not be completed.'}}"
    )
    error.status_code = 429
    error.body = {
        "error": {
            "code": "run_budget_exhausted",
            "message": "The model request could not be completed.",
        }
    }
    classified = classify_api_error(error, provider="custom", model="zai-org/glm-5.2")
    assert classified.reason == FailoverReason.run_budget_exhausted
    assert classified.retryable is False
    assert classified.should_rotate_credential is False
    assert classified.should_fallback is False
    assert _runtime_failure_code({
        "failed": True,
        "failure_reason": "run_budget_exhausted",
    }) == "runtime_budget_exhausted"
    envelope = runtime_error_envelope("runtime_budget_exhausted", support_id="run_budget")
    assert envelope["retryable"] is False
    assert "safety limit" in envelope["message"]


@pytest.mark.asyncio
async def test_run_scoped_vision_egress_never_falls_back_to_ambient_provider():
    from agent import auxiliary_client
    from agent.run_scoped_auxiliary import (
        bind_run_scoped_auxiliary,
        reset_run_scoped_auxiliary,
    )

    class _Completions:
        async def create(self, **_kwargs):
            raise RuntimeError("HTTP 402 payment required")

    client = SimpleNamespace(
        base_url="http://agent-orchestrator:8093/internal/llm/v1",
        chat=SimpleNamespace(completions=_Completions()),
    )
    value = {
        **capability(),
        "model": "qwen/qwen3-vl-235b-a22b-thinking",
    }
    token = bind_run_scoped_auxiliary({"vision": value})
    try:
        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                return_value=("custom", client, value["model"]),
            ) as resolver,
            patch("agent.auxiliary_client._try_configured_fallback_chain") as configured_fallback,
            patch("agent.auxiliary_client._try_main_agent_model_fallback") as main_fallback,
        ):
            with pytest.raises(RuntimeError, match="402"):
                await auxiliary_client.async_call_llm(
                    task="vision",
                    messages=[{"role": "user", "content": "inspect"}],
                )

        assert resolver.call_args.kwargs["provider"] == "custom"
        assert resolver.call_args.kwargs["model"] == value["model"]
        assert resolver.call_args.kwargs["base_url"] == value["base_url"]
        assert resolver.call_args.kwargs["api_key"] == value["grant"]
        configured_fallback.assert_not_called()
        main_fallback.assert_not_called()
    finally:
        reset_run_scoped_auxiliary(token)
