"""Read exact, run-bound Skill snapshots without global Skill discovery."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from agent.skill_utils import parse_frontmatter

_MAX_SKILL_FILE_BYTES = 8 << 20
_MAX_SKILL_SNAPSHOT_BYTES = 32 << 20
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeSkillProjection:
    files: Mapping[str, bytes]
    routing_mode: str
    kind: str


def projection_skill_metadata(
    projections: Mapping[str, RuntimeSkillProjection],
) -> list[dict[str, Any]]:
    """Build routing metadata from the exact run-bound Skill snapshot."""
    result: list[dict[str, Any]] = []
    for name in sorted(projections):
        projection = projections[name]
        if projection.routing_mode == "dependency_only":
            continue
        try:
            content = projection.files["SKILL.md"].decode("utf-8")
            frontmatter, _ = parse_frontmatter(content)
            if not isinstance(frontmatter, dict):
                raise ValueError("frontmatter is not an object")
        except (KeyError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("Isolating unreadable run-bound Skill metadata for %s: %s", name, exc)
            continue
        description = frontmatter.get("description")
        if not isinstance(description, str):
            description = ""
        routing = frontmatter.get("routing")
        kind = projection.kind or str(frontmatter.get("kind") or "").strip()
        item: dict[str, Any] = {
            "name": name,
            "description": " ".join(description.split()),
            "category": (
                "workflow-generation"
                if kind == "workflow" or isinstance(routing, dict)
                else kind or "skill"
            ),
        }
        if isinstance(routing, dict):
            item["routing"] = routing
        result.append(item)
    return result


def resolve_skill_projections(
    skill_manifest: object,
    no_manifest_sentinel: object,
) -> dict[str, RuntimeSkillProjection]:
    if skill_manifest is no_manifest_sentinel:
        return {}
    if not isinstance(skill_manifest, dict):
        raise ValueError("skill_manifest must be an object")
    skills = skill_manifest.get("skills")
    if not isinstance(skills, list):
        raise ValueError("skill_manifest.skills must be an array")

    projections: dict[str, RuntimeSkillProjection] = {}
    total_size = 0
    for skill in skills:
        if not isinstance(skill, dict):
            raise ValueError("skill_manifest.skills must contain only objects")
        name = str(skill.get("runtime_alias") or "").strip()
        routing_mode = str(skill.get("routing_mode") or "").strip()
        kind = str(skill.get("kind") or "").strip()
        raw_files = skill.get("files")
        if not name or not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"skill_manifest files are required for {name or 'skill'}")
        if name in projections:
            raise ValueError(f"skill_manifest contains duplicate runtime_alias {name}")
        if routing_mode not in {"primary", "domain", "dependency_only"}:
            raise ValueError(f"skill_manifest routing_mode is invalid for {name}")

        files: dict[str, bytes] = {}
        for raw_file in raw_files:
            path, body = _verified_file(name, raw_file)
            if path in files:
                raise ValueError(f"skill_manifest contains duplicate file for {name}")
            total_size += len(body)
            if total_size > _MAX_SKILL_SNAPSHOT_BYTES:
                raise ValueError("skill_manifest files exceed transport limit")
            files[path] = body
        if "SKILL.md" not in files:
            raise ValueError(f"skill_manifest root file is required for {name}")
        projections[name] = RuntimeSkillProjection(
            files=dict(files), routing_mode=routing_mode, kind=kind
        )
    return projections


def _verified_file(name: str, raw_file: object) -> tuple[str, bytes]:
    if not isinstance(raw_file, dict):
        raise ValueError(f"skill_manifest files must contain only objects for {name}")
    raw_path = raw_file.get("path")
    raw_size = raw_file.get("size_bytes")
    raw_digest = raw_file.get("sha256")
    encoded = raw_file.get("content_base64")
    if not isinstance(raw_path, str) or not _safe_relative_path(raw_path):
        raise ValueError(f"skill_manifest file path is invalid for {name}")
    if (
        isinstance(raw_size, bool)
        or not isinstance(raw_size, int)
        or raw_size < 0
        or raw_size > _MAX_SKILL_FILE_BYTES
    ):
        raise ValueError(f"skill_manifest file size is invalid for {name}")
    if not isinstance(raw_digest, str) or not _HEX_DIGEST.fullmatch(raw_digest):
        raise ValueError(f"skill_manifest file digest is invalid for {name}")
    max_encoded_bytes = 4 * ((_MAX_SKILL_FILE_BYTES + 2) // 3)
    if not isinstance(encoded, str) or len(encoded) > max_encoded_bytes:
        raise ValueError(f"skill_manifest file content is invalid for {name}")
    try:
        body = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"skill_manifest file content is invalid for {name}") from exc
    if len(body) != raw_size or hashlib.sha256(body).hexdigest() != raw_digest:
        raise ValueError(f"skill_manifest file differs from signed inventory for {name}")
    return raw_path, body


def _safe_relative_path(raw_path: str) -> bool:
    if "\\" in raw_path:
        return False
    path = PurePosixPath(raw_path)
    return bool(
        raw_path
        and not path.is_absolute()
        and path.parts
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def view_skill(
    name: str,
    projection: RuntimeSkillProjection,
    args: Mapping[str, object],
) -> str:
    raw_file_path = str(args.get("file_path") or "").strip()
    path = raw_file_path or "SKILL.md"
    if not _safe_relative_path(path):
        return _file_error(name, "Skill file path is invalid.")
    body = projection.files.get(path)
    if body is None:
        return _file_error(name, "Skill file is unavailable.")
    try:
        content = body.decode("utf-8")
    except UnicodeDecodeError:
        return _file_error(name, "Skill file is not readable text.")
    if raw_file_path:
        return json.dumps(
            {
                "success": True,
                "name": name,
                "file": raw_file_path,
                "content": content,
                "file_type": PurePosixPath(path).suffix,
            },
            ensure_ascii=False,
        )

    parsed, _ = parse_frontmatter(content)
    frontmatter = parsed if isinstance(parsed, dict) else {}
    linked_files: dict[str, list[str]] = {}
    for relative_path in sorted(projection.files):
        if relative_path == "SKILL.md":
            continue
        category = relative_path.split("/", 1)[0]
        linked_files.setdefault(category, []).append(relative_path)
    return json.dumps(
        {
            "success": True,
            "name": name,
            "description": str(frontmatter.get("description") or ""),
            "content": content,
            "path": f"{name}/SKILL.md",
            "linked_files": linked_files or None,
            "usage_hint": (
                "To view linked files, call skill_view(name, file_path)."
                if linked_files
                else None
            ),
            "required_environment_variables": [],
            "required_commands": [],
            "missing_required_environment_variables": [],
            "missing_credential_files": [],
            "missing_required_commands": [],
            "setup_needed": False,
            "setup_skipped": False,
            "readiness_status": "available",
        },
        ensure_ascii=False,
    )


def _file_error(name: str, message: str) -> str:
    return json.dumps(
        {"success": False, "error": f"Skill '{name}': {message}"},
        ensure_ascii=False,
        separators=(",", ":"),
    )
