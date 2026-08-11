from contextvars import copy_context

import pytest

from agent.auxiliary_client import (
    _resolve_task_provider_model,
    clear_runtime_auxiliary_overrides,
    set_runtime_auxiliary_override,
)


@pytest.fixture(autouse=True)
def clear_runtime_override():
    clear_runtime_auxiliary_overrides()
    yield
    clear_runtime_auxiliary_overrides()


def test_runtime_vision_override_precedes_process_configuration():
    set_runtime_auxiliary_override(
        "vision",
        provider="custom",
        model="google/gemini-3.1-flash-lite",
        base_url="http://agent-orchestrator:8093/internal/llm/v1",
        api_key="ueg_" + "a" * 43,
    )

    assert _resolve_task_provider_model(
        "vision",
        provider="atlas",
        model="google/gemini-3-flash-preview",
    ) == (
        "custom",
        "google/gemini-3.1-flash-lite",
        "http://agent-orchestrator:8093/internal/llm/v1",
        "ueg_" + "a" * 43,
        "chat_completions",
    )


def test_runtime_vision_override_does_not_leak_to_a_fresh_context():
    isolated = copy_context()
    set_runtime_auxiliary_override(
        "vision",
        provider="custom",
        model="google/gemini-3.1-flash-lite",
        base_url="http://agent-orchestrator:8093/internal/llm/v1",
        api_key="ueg_" + "a" * 43,
    )

    assert isolated.run(
        _resolve_task_provider_model,
        "vision",
        "",
        "",
    ) != (
        "custom",
        "google/gemini-3.1-flash-lite",
        "http://agent-orchestrator:8093/internal/llm/v1",
        "ueg_" + "a" * 43,
        "chat_completions",
    )
