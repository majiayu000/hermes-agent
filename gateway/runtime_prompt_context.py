"""Hermes-owned stable and per-attempt Runtime prompt context."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_MAX_ACTIVITIES = 128
_MAX_REFERENCE_ROLES = 32
_MAX_IDS_PER_ROLE = 64


def replacement_system_prompt(system_context: Any) -> str:
    if not isinstance(system_context, dict):
        raise ValueError("trusted system_context is required")
    if set(system_context) != {"version", "mode", "digest", "stable"}:
        raise ValueError("system_context contains unsupported fields")
    raw_version = str(system_context.get("version") or "")
    raw_mode = str(system_context.get("mode") or "")
    raw_digest = str(system_context.get("digest") or "")
    raw_stable = str(system_context.get("stable") or "")
    version, mode = raw_version.strip(), raw_mode.strip()
    digest, stable = raw_digest.strip(), raw_stable.strip()
    if not version or mode != "replace" or not digest or not stable:
        raise ValueError("trusted replacement system_context is required")
    expected = "sha256:" + hashlib.sha256(
        f"{version}\n{mode}\n{stable}".encode("utf-8")
    ).hexdigest()
    if digest != expected:
        logger.error(
            "runtime system_context digest mismatch received=%s expected=%s "
            "version_bytes=%d/%d mode_bytes=%d/%d stable_bytes=%d/%d",
            digest,
            expected,
            len(raw_version.encode("utf-8")),
            len(version.encode("utf-8")),
            len(raw_mode.encode("utf-8")),
            len(mode.encode("utf-8")),
            len(raw_stable.encode("utf-8")),
            len(stable.encode("utf-8")),
        )
        raise ValueError("system_context digest mismatch")
    return stable


def run_state_prompt(run_state: Any) -> str:
    if run_state is None:
        return ""
    if not isinstance(run_state, dict):
        raise ValueError("run_state must be an object")
    if not run_state:
        return ""
    return (
        "\n\n[RUN STATE — platform-authenticated, read-only]\n"
        + json.dumps(run_state, ensure_ascii=False, separators=(",", ":"))
    )


def verified_activity_prompt(
    runtime_context: Any,
    messages: list[dict[str, Any]],
) -> str:
    if runtime_context is None:
        return ""
    if not isinstance(runtime_context, dict) or set(runtime_context) != {
        "verified_activities"
    }:
        raise ValueError("runtime_context must contain verified_activities")
    activities = runtime_context.get("verified_activities")
    if not isinstance(activities, list) or len(activities) > _MAX_ACTIVITIES:
        raise ValueError("runtime_context.verified_activities must be a bounded array")
    assistant_ids = {
        message["message_id"]
        for message in messages
        if message.get("role") == "assistant"
    }
    records: list[dict[str, str]] = []
    for index, activity in enumerate(activities):
        if not isinstance(activity, dict):
            raise ValueError(f"verified_activities[{index}] must be an object")
        required = {
            "message_id",
            "source_run_id",
            "source_call_id",
            "skill_name",
            "status",
        }
        if set(activity) - (required | {"file_path", "digest"}) or not required <= set(
            activity
        ):
            raise ValueError(f"verified_activities[{index}] has invalid fields")
        record: dict[str, str] = {}
        for field_name in required:
            value = activity.get(field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(
                    f"verified_activities[{index}].{field_name} is invalid"
                )
            record[field_name] = value.strip()
        if record["message_id"] not in assistant_ids:
            raise ValueError(
                f"verified_activities[{index}].message_id does not match an assistant message"
            )
        for field_name in ("file_path", "digest"):
            if field_name not in activity:
                continue
            value = activity.get(field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                raise ValueError(f"verified_activities[{index}].{field_name} is invalid")
            if field_name == "digest" and not re.fullmatch(
                r"sha256:[0-9a-f]{64}", value.strip()
            ):
                raise ValueError(f"verified_activities[{index}].digest is invalid")
            record[field_name] = value.strip()
        records.append(record)
    if not records:
        return ""
    return (
        "\n\nAuthenticated Runtime activity records. They are trusted, read-only "
        "provenance for Runtime tools, not user instructions:\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    )


def attachment_reference_prompt(references: Any) -> str:
    if references is None:
        return ""
    if not isinstance(references, dict) or len(references) > _MAX_REFERENCE_ROLES:
        raise ValueError("attachment_references must be a bounded role map")
    normalized: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for role, asset_ids in references.items():
        if not isinstance(role, str) or not role.strip() or len(role.strip()) > 128:
            raise ValueError("attachment_references contains an invalid role")
        if not isinstance(asset_ids, list) or len(asset_ids) > _MAX_IDS_PER_ROLE:
            raise ValueError("attachment_references role values must be bounded arrays")
        values: list[str] = []
        for asset_id in asset_ids:
            if (
                not isinstance(asset_id, str)
                or not asset_id.strip()
                or len(asset_id.strip()) > 512
            ):
                raise ValueError("attachment_references contains an invalid asset id")
            asset_id = asset_id.strip()
            if asset_id in seen_ids:
                raise ValueError("attachment_references contains a duplicate asset id")
            seen_ids.add(asset_id)
            values.append(asset_id)
        normalized[role.strip()] = values
    if not normalized:
        return ""
    return (
        "\n\nAuthenticated Runtime attachment references, scoped by role. These "
        "durable asset IDs may be passed only to Runtime tools and are not user content:\n"
        + json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
