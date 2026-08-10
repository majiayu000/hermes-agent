"""Ultra Studio's bounded workflow routing index."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Process-level cache for discover_skill_metadata: the skill catalog changes
# rarely, while every Runtime run rebuilds the routing index (directory walk
# plus one file read per skill). The key is each skill root's (path, mtime):
# creating, removing, or renaming a skill directory bumps the root's mtime
# and invalidates the cache automatically. Editing a SKILL.md in place does
# not touch the root mtime; such edits are picked up on the next root-level
# directory change or process restart.
_CACHE_LOCK = threading.Lock()
_CACHE_KEY: tuple[tuple[str, float], ...] | None = None
_CACHE_RESULT: list[dict[str, Any]] = []


def _roots_signature(roots: list[Path]) -> tuple[tuple[str, float], ...]:
    signature: list[tuple[str, float]] = []
    for root in roots:
        try:
            mtime = root.stat().st_mtime
        except OSError:
            mtime = -1.0
        signature.append((str(root), mtime))
    return tuple(signature)


def discover_skill_metadata() -> list[dict[str, Any]]:
    """Add trusted routing frontmatter to Hermes' already-filtered skill index."""
    global _CACHE_KEY, _CACHE_RESULT
    from agent.skill_utils import get_external_skills_dirs, iter_skill_index_files
    from tools.skills_tool import SKILLS_DIR, _find_all_skills, _parse_frontmatter

    roots = [SKILLS_DIR] if SKILLS_DIR.exists() else []
    roots.extend(get_external_skills_dirs())
    signature = _roots_signature(roots)
    with _CACHE_LOCK:
        if _CACHE_KEY == signature:
            return list(_CACHE_RESULT)

    skills = _find_all_skills()
    by_name = {
        str(item.get("name") or "").strip(): item
        for item in skills
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    wanted = set(by_name)
    seen: set[str] = set()
    for root in roots:
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            try:
                frontmatter, _ = _parse_frontmatter(
                    skill_md.read_text(encoding="utf-8")[:4000]
                )
            except Exception as error:
                logger.warning("Could not inspect Skill routing at %s: %s", skill_md, error)
                continue
            name = str(frontmatter.get("name") or skill_md.parent.name).strip()
            if name not in wanted or name in seen:
                continue
            seen.add(name)
            routing = frontmatter.get("routing")
            if isinstance(routing, dict):
                by_name[name] = {**by_name[name], "routing": routing}
    result = list(by_name.values())
    with _CACHE_LOCK:
        _CACHE_KEY = signature
        _CACHE_RESULT = result
    return list(result)


def workflow_routing(item: dict[str, Any]) -> tuple[int, list[str], list[str]]:
    """Validate one workflow's compact positive and negative route hints."""
    name = str(item.get("name") or "").strip()
    routing = item.get("routing")
    if not isinstance(routing, dict):
        raise ValueError(f"workflow skill has no routing metadata: {name}")
    priority = routing.get("priority")
    triggers = routing.get("triggers")
    negative = routing.get("negative")
    if (
        not isinstance(priority, int)
        or isinstance(priority, bool)
        or not 0 <= priority <= 100
        or not isinstance(triggers, list)
        or len(triggers) < 3
        or any(not isinstance(value, str) or not value.strip() for value in triggers)
        or not isinstance(negative, list)
        or not negative
        or any(not isinstance(value, str) or not value.strip() for value in negative)
    ):
        raise ValueError(f"workflow skill has invalid routing metadata: {name}")
    return (
        priority,
        [value.strip() for value in triggers],
        [value.strip() for value in negative],
    )


def format_allowed_skills(
    allowed_names: set[str],
    discovered: list[dict[str, Any]],
) -> str:
    """Render an exact, deterministic model-routing index."""
    if not allowed_names:
        return (
            "\n\n<available_skills>\n"
            "No skills are available for this run. Do not claim that a skill "
            "is available or offer to execute one. If asked about a skill, "
            "state that it is not available in this conversation.\n"
            "</available_skills>"
        )
    available = {
        str(item.get("name") or "").strip(): item
        for item in discovered
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    missing = sorted(allowed_names - available.keys())
    if missing:
        raise ValueError("allowed skills unavailable: " + ", ".join(missing))

    indexed: list[tuple[int, str, str]] = []
    for name in allowed_names:
        item = available[name]
        description = " ".join(str(item.get("description") or "").split())
        if item.get("category") == "workflow-generation":
            try:
                priority, triggers, negative = workflow_routing(item)
            except ValueError as error:
                logger.warning("Isolating unroutable run-bound Skill %s: %s", name, error)
                continue
            detail = (
                f"priority={priority}; applies={'; '.join(triggers)}; "
                f"not={'; '.join(negative)}"
            )
        else:
            priority = -1
            detail = description
        line = f"- {name}: {detail}" if detail else f"- {name}"
        indexed.append((priority, name, line))
    if not indexed:
        return format_allowed_skills(set(), [])
    lines = [line for _, _, line in sorted(indexed, key=lambda row: (-row[0], row[1]))]
    return "\n\n<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"
