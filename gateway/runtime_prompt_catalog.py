"""Hermes-owned, immutable Runtime system-prompt profiles."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROFILE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PROFILE_ROOT = Path(__file__).with_name("runtime_prompts")


@dataclass(frozen=True)
class RuntimePromptProfile:
    version: str
    text: str
    digest: str
    modules: tuple[str, ...]


def load_runtime_prompt_profile(version: str) -> RuntimePromptProfile:
    """Load one named profile in deterministic module order."""
    version = str(version or "").strip()
    if not _PROFILE_NAME.fullmatch(version) or len(version) > 128:
        raise ValueError("system_context profile version is invalid")
    profile_dir = _PROFILE_ROOT / version
    try:
        modules = tuple(
            sorted(
                path.name
                for path in profile_dir.iterdir()
                if path.is_file() and path.suffix == ".md"
            )
        )
    except OSError as exc:
        raise ValueError(f"system_context profile {version!r} is unavailable") from exc
    if not modules:
        raise ValueError(f"system_context profile {version!r} is unavailable")

    sections: list[str] = []
    for module in modules:
        try:
            body = (profile_dir / module).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise ValueError(
                f"system_context profile {version!r} could not be loaded"
            ) from exc
        if not body:
            raise ValueError(f"system_context profile module {module!r} is empty")
        sections.append(body)
    text = "\n\n---\n\n".join(sections)
    if len(text.encode("utf-8")) < 1024:
        raise ValueError(f"system_context profile {version!r} is incomplete")
    return RuntimePromptProfile(
        version=version,
        text=text,
        digest="sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
        modules=modules,
    )


def resolve_runtime_system_context(system_context: Any) -> RuntimePromptProfile:
    """Resolve a Hermes profile or a legacy replacement artifact."""
    if not isinstance(system_context, dict):
        raise ValueError("trusted system_context is required")
    mode = str(system_context.get("mode") or "").strip()
    if mode == "profile":
        if set(system_context) != {"version", "mode"}:
            raise ValueError("profile system_context contains unsupported fields")
        return load_runtime_prompt_profile(str(system_context.get("version") or ""))
    if mode != "replace":
        raise ValueError("system_context mode is invalid")

    from gateway.runtime_prompt_context import replacement_system_prompt

    text = replacement_system_prompt(system_context)
    version = str(system_context["version"]).strip()
    return RuntimePromptProfile(
        version=version,
        text=text,
        digest=str(system_context["digest"]).strip(),
        modules=(),
    )
