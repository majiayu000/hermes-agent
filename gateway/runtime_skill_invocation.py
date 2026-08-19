"""Compile structured Runtime Skill inputs into Hermes-owned context."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from agent.skill_commands import build_loaded_skill_invocation_message
from gateway.runtime_prompt_context import (
    attachment_reference_prompt,
    run_state_prompt,
    verified_activity_prompt,
)
from gateway.runtime_prompt_catalog import resolve_runtime_system_context
from gateway.runtime_skill_projection import (
    RuntimeSkillProjection,
    projection_skill_metadata,
    view_skill,
)

_SKILL_ALIAS = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class CompiledRuntimeSkillContext:
    stable_instructions: str
    runtime_overlay: str
    invoked_messages: list[tuple[str, str]]
    system_context_version: str
    system_context_mode: str
    system_context_digest: str


def format_available_skills_prompt(
    allowed_names: set[str],
    metadata: list[dict[str, Any]] | Mapping[str, RuntimeSkillProjection],
) -> str:
    """Render the stable bootstrap index inside Hermes, not Orchestrator."""
    from gateway.ultrastudio_skill_routing import format_allowed_skills

    if isinstance(metadata, Mapping):
        metadata = projection_skill_metadata(metadata)
    return format_allowed_skills(allowed_names, metadata)


def compile_invoked_skill_messages(
    invoked_skills: Any,
    projections: Mapping[str, RuntimeSkillProjection],
    *,
    user_instruction: str,
    session_id: str,
) -> list[tuple[str, str]]:
    """Expand explicit roots from verified projections into Hermes history."""
    if invoked_skills is None:
        return []
    if not isinstance(invoked_skills, list) or len(invoked_skills) > 8:
        raise ValueError("invoked_skills must be an array with at most 8 names")

    seen: set[str] = set()
    loaded: list[tuple[str, str]] = []
    for index, raw_name in enumerate(invoked_skills):
        if not isinstance(raw_name, str):
            raise ValueError("invoked_skills must contain only names")
        name = raw_name.strip()
        if (
            not _SKILL_ALIAS.fullmatch(name)
            or len(name) > 128
            or name in seen
        ):
            raise ValueError("invoked_skills contains an invalid or duplicate name")
        seen.add(name)
        projection = projections.get(name)
        if projection is None or projection.routing_mode == "dependency_only":
            raise ValueError(f"invoked Skill {name!r} is unavailable")

        payload = json.loads(view_skill(name, projection, {}))
        if not payload.get("success") or not isinstance(payload.get("content"), str):
            raise ValueError(f"invoked Skill {name!r} could not be loaded")
        loaded.append(
            (
                name,
                build_loaded_skill_invocation_message(
                    payload,
                    name,
                    user_instruction=(
                        user_instruction
                        if index == len(invoked_skills) - 1
                        else ""
                    ),
                    session_id=session_id,
                ),
            )
        )
    return loaded


def compile_runtime_skill_context(
    *,
    system_context: Any,
    invoked_skills: Any,
    projections: Mapping[str, RuntimeSkillProjection],
    normalized_messages: list[dict[str, Any]],
    user_instruction: str,
    session_id: str,
    run_state: Any,
    runtime_context: Any,
    attachment_references: Any,
) -> CompiledRuntimeSkillContext:
    """Compile stable bootstrap context and the current attempt overlay."""
    prompt_profile = resolve_runtime_system_context(system_context)
    routing_metadata = projection_skill_metadata(projections)
    selectable_names = {
        str(item.get("name") or "").strip()
        for item in routing_metadata
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    }
    return CompiledRuntimeSkillContext(
        stable_instructions=(
            prompt_profile.text
            + format_available_skills_prompt(selectable_names, routing_metadata)
        ),
        runtime_overlay=(
            run_state_prompt(run_state)
            + verified_activity_prompt(runtime_context, normalized_messages)
            + attachment_reference_prompt(attachment_references)
        ),
        invoked_messages=compile_invoked_skill_messages(
            invoked_skills,
            projections,
            user_instruction=user_instruction,
            session_id=session_id,
        ),
        system_context_version=prompt_profile.version,
        system_context_mode=str(system_context.get("mode") or "").strip(),
        system_context_digest=prompt_profile.digest,
    )
