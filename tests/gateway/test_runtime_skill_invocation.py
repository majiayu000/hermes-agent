from __future__ import annotations

import hashlib

import pytest

from gateway.runtime_skill_invocation import (
    compile_invoked_skill_messages,
    compile_runtime_skill_context,
)
from gateway.runtime_skill_projection import RuntimeSkillProjection


def _projection(
    name: str,
    body: str,
    *,
    routing_mode: str = "primary",
) -> RuntimeSkillProjection:
    skill = (
        "---\n"
        f"name: {name}\n"
        f"description: {name} description.\n"
        "kind: method\n"
        "---\n"
        f"{body}\n"
    ).encode()
    return RuntimeSkillProjection(
        files={"SKILL.md": skill},
        routing_mode=routing_mode,
        kind="method",
    )


def _system_context(stable: str = "ultra-agent-v1") -> dict[str, str]:
    version = "ultra-agent-v1"
    mode = "replace"
    digest = "sha256:" + hashlib.sha256(
        f"{version}\n{mode}\n{stable}".encode()
    ).hexdigest()
    return {
        "version": version,
        "mode": mode,
        "stable": stable,
        "digest": digest,
    }


def test_explicit_runtime_skills_use_native_hermes_history_scaffolding(monkeypatch):
    monkeypatch.setattr("tools.skill_usage.bump_use", lambda _name: None)
    projections = {
        "briefing": _projection("briefing", "Collect requirements."),
        "production": _projection("production", "Create the deliverable."),
    }

    loaded = compile_invoked_skill_messages(
        ["briefing", "production"],
        projections,
        user_instruction="Make a 25-shot product video.",
        session_id="thread-1",
    )

    assert [name for name, _ in loaded] == ["briefing", "production"]
    combined = "\n\n".join(message for _, message in loaded)
    assert combined.count("[IMPORTANT: The user has invoked the") == 2
    assert "Collect requirements." in combined
    assert "Create the deliverable." in combined
    assert combined.count("Make a 25-shot product video.") == 1
    assert "Make a 25-shot product video." not in loaded[0][1]
    assert "Make a 25-shot product video." in loaded[1][1]


@pytest.mark.parametrize(
    ("invoked", "match"),
    [
        (["missing"], "unavailable"),
        (["helper"], "unavailable"),
        (["root", "root"], "invalid or duplicate"),
        (["Bad_Name"], "invalid or duplicate"),
        (["root"] * 9, "at most 8"),
    ],
)
def test_explicit_runtime_skill_validation_fails_closed(invoked, match):
    projections = {
        "root": _projection("root", "Run root."),
        "helper": _projection(
            "helper",
            "Internal helper.",
            routing_mode="dependency_only",
        ),
    }

    with pytest.raises(ValueError, match=match):
        compile_invoked_skill_messages(
            invoked,
            projections,
            user_instruction="go",
            session_id="thread-1",
        )


def test_thread_skill_catalog_is_stable_while_run_state_is_ephemeral(monkeypatch):
    monkeypatch.setattr("tools.skill_usage.bump_use", lambda _name: None)
    context = compile_runtime_skill_context(
        system_context=_system_context(),
        invoked_skills=["root"],
        projections={"root": _projection("root", "Pinned body v1.")},
        normalized_messages=[
            {"message_id": "message-1", "role": "user", "content": "go"}
        ],
        user_instruction="go",
        session_id="thread-1",
        run_state={"attempt": 2},
        runtime_context=None,
        attachment_references=None,
    )

    assert context.stable_instructions.startswith("ultra-agent-v1")
    assert "<available_skills>" in context.stable_instructions
    assert '"attempt":2' not in context.stable_instructions
    assert '"attempt":2' in context.runtime_overlay
    assert "Pinned body v1." in context.invoked_messages[0][1]
