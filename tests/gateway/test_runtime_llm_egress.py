from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.api_server_runtime import (
    _configure_run_llm_egress,
    _runtime_failure_code,
    _runtime_llm_egress,
    _runtime_vision_llm_egress,
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


def test_runtime_vision_llm_egress_requires_a_fixed_model():
    value = capability(model="google/gemini-3.1-flash-lite")
    assert _runtime_vision_llm_egress(value) == value
    assert _runtime_vision_llm_egress(None) is None
    with pytest.raises(ValueError, match="must contain"):
        _runtime_vision_llm_egress(capability())
    with pytest.raises(ValueError, match="model is invalid"):
        _runtime_vision_llm_egress(capability(model="bad model"))


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
    agent.api_key = "shared-key"
    with pytest.raises(ValueError, match="inconsistent"):
        _configure_run_llm_egress(agent, value, "moonshotai/kimi-k3")


def test_runtime_contract_and_billing_failure_are_account_aware():
    assert "llm_egress" in RUNTIME_CAPABILITIES
    assert "vision_llm_egress" in RUNTIME_CAPABILITIES
    assert "session_db_rebootstrap/v1" in RUNTIME_CAPABILITIES
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
        "max_tokens": None,
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
