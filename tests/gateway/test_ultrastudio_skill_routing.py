from __future__ import annotations

import os

import pytest

import gateway.ultrastudio_skill_routing as routing_module
from gateway.ultrastudio_skill_routing import (
    discover_skill_metadata,
    format_allowed_skills,
    workflow_routing,
)


def workflow(name: str, priority: int) -> dict:
    return {
        "name": name,
        "description": f"{name} description.",
        "category": "workflow-generation",
        "routing": {
            "priority": priority,
            "triggers": [f"{name} one", f"{name} two", f"{name} three"],
            "negative": [f"not {name}"],
        },
    }


def test_workflow_routing_requires_complete_metadata():
    assert workflow_routing(workflow("specific", 80)) == (
        80,
        ["specific one", "specific two", "specific three"],
        ["not specific"],
    )
    for routing in (
        None,
        {"priority": 101, "triggers": ["a", "b", "c"], "negative": ["d"]},
        {"priority": 50, "triggers": ["a"], "negative": ["d"]},
        {"priority": 50, "triggers": ["a", "b", "c"], "negative": []},
    ):
        item = workflow("broken", 50)
        item["routing"] = routing
        with pytest.raises(ValueError, match="routing metadata"):
            workflow_routing(item)


def test_allowed_skill_index_orders_specific_routes_before_fallbacks():
    discovered = [
        workflow("fallback", 10),
        workflow("specific", 80),
        {
            "name": "helper",
            "description": "Supporting guidance.",
            "category": "creative",
        },
    ]

    prompt = format_allowed_skills(
        {"fallback", "specific", "helper"},
        discovered,
    )

    assert prompt.index("- specific:") < prompt.index("- fallback:")
    assert prompt.index("- fallback:") < prompt.index("- helper:")
    assert "priority=80" in prompt
    assert "applies=specific one; specific two; specific three" in prompt
    assert "not=not specific" in prompt
    assert "- helper: Supporting guidance." in prompt


def test_empty_allowed_skill_index_explicitly_denies_skill_claims():
    prompt = format_allowed_skills(set(), [
        workflow("installed-but-not-bound", 80),
    ])

    assert "No skills are available for this run." in prompt
    assert "Do not claim that a skill is available" in prompt
    assert "installed-but-not-bound" not in prompt
    assert prompt.startswith("\n\n<available_skills>\n")
    assert prompt.endswith("\n</available_skills>")


@pytest.fixture
def skill_discovery(tmp_path, monkeypatch):
    """One skill root with one SKILL.md, instrumented scan counters."""
    import agent.skill_utils as skill_utils
    import tools.skills_tool as skills_tool

    root = tmp_path / "skills"
    (root / "demo").mkdir(parents=True)
    (root / "demo" / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")

    calls = {"find": 0, "parse": 0}

    def fake_find_all_skills(**_kwargs):
        calls["find"] += 1
        return [{"name": "demo", "description": "Demo skill.", "category": "creative"}]

    def fake_parse_frontmatter(_text):
        calls["parse"] += 1
        return (
            {
                "name": "demo",
                "routing": {"priority": 5, "triggers": ["a", "b", "c"], "negative": ["d"]},
            },
            "",
        )

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", root)
    monkeypatch.setattr(skills_tool, "_find_all_skills", fake_find_all_skills)
    monkeypatch.setattr(skills_tool, "_parse_frontmatter", fake_parse_frontmatter)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    monkeypatch.setattr(routing_module, "_CACHE_KEY", None)
    monkeypatch.setattr(routing_module, "_CACHE_RESULT", [])
    return root, calls


def test_discovery_cache_hit_returns_cached_result_without_rescan(skill_discovery):
    _root, calls = skill_discovery
    first = discover_skill_metadata()
    assert calls == {"find": 1, "parse": 1}
    assert first == [{
        "name": "demo",
        "description": "Demo skill.",
        "category": "creative",
        "routing": {"priority": 5, "triggers": ["a", "b", "c"], "negative": ["d"]},
    }]

    second = discover_skill_metadata()
    assert second == first
    assert calls == {"find": 1, "parse": 1}

    # The cache hands out copies: caller-side mutation must not leak back.
    second.append({"name": "junk"})
    assert discover_skill_metadata() == first
    assert calls == {"find": 1, "parse": 1}


def test_discovery_cache_invalidates_when_root_mtime_changes(skill_discovery):
    root, calls = skill_discovery
    discover_skill_metadata()
    assert calls == {"find": 1, "parse": 1}

    stat = root.stat()
    os.utime(root, (stat.st_atime + 10, stat.st_mtime + 10))
    discover_skill_metadata()
    assert calls == {"find": 2, "parse": 2}


def test_allowed_skill_index_rejects_missing_and_isolates_unroutable_workflow(caplog):
    with pytest.raises(ValueError, match="allowed skills unavailable: missing"):
        format_allowed_skills({"missing"}, [])
    prompt = format_allowed_skills(
        {"broken", "healthy"},
        [
            {
                "name": "broken",
                "description": "Broken.",
                "category": "workflow-generation",
            },
            {
                "name": "healthy",
                "description": "Healthy.",
                "category": "method",
            },
        ],
    )
    assert "broken" not in prompt
    assert "- healthy: Healthy." in prompt
    assert "Isolating unroutable run-bound Skill broken" in caplog.text
