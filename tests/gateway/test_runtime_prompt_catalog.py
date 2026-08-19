from __future__ import annotations

import pytest

from gateway.runtime_prompt_catalog import (
    load_runtime_prompt_profile,
    resolve_runtime_system_context,
)


def test_ultra_agent_v1_profile_preserves_migrated_prompt_bytes():
    profile = load_runtime_prompt_profile("ultra-agent-v1")

    assert profile.version == "ultra-agent-v1"
    assert profile.digest == (
        "sha256:8111fda1e279e61eab2bf71e95184ef26da53800e05f6c08773d63de07fd50bc"
    )
    assert len(profile.modules) == 20
    assert profile.modules[0] == "01-identity.md"
    assert profile.modules[-1] == "20-runtime-environment.md"
    assert profile.text.startswith("<!-- module 1/20:")


def test_profile_context_resolves_locally_without_prompt_body():
    profile = resolve_runtime_system_context(
        {"version": "ultra-agent-v1", "mode": "profile"}
    )
    assert profile.text
    assert profile.digest.startswith("sha256:")


@pytest.mark.parametrize(
    "context",
    [
        {"version": "missing-profile", "mode": "profile"},
        {
            "version": "ultra-agent-v1",
            "mode": "profile",
            "stable": "orchestrator prose is forbidden",
        },
        {"version": "../ultra-agent-v1", "mode": "profile"},
    ],
)
def test_profile_context_fails_closed(context):
    with pytest.raises(ValueError):
        resolve_runtime_system_context(context)
