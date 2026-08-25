"""Authoritative Run-facing definitions for Hermes native media Tools."""

from __future__ import annotations

import copy
from typing import Any


def native_image_tool_definition() -> dict[str, Any]:
    from tools import image_analyze as _image_analyze  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("image_analyze")
    if entry is None:
        raise RuntimeError("Hermes image_analyze tool is not registered")
    return project_runtime_media_tool({
        "type": "function",
        "function": {**entry.schema, "name": entry.name},
    })


def native_video_tool_definition() -> dict[str, Any]:
    from tools import vision_tools as _vision_tools  # noqa: F401
    from tools.registry import registry

    entry = registry.get_entry("video_analyze")
    if entry is None:
        raise RuntimeError("Hermes video_analyze tool is not registered")
    return project_runtime_media_tool({
        "type": "function",
        "function": {**entry.schema, "name": entry.name},
    })


def project_runtime_media_tool(definition: dict[str, Any]) -> dict[str, Any]:
    """Project stable Run references without renaming the underlying Tool."""
    projected = copy.deepcopy(definition)
    function = projected.get("function") or {}
    name = str(function.get("name") or "")
    properties = (function.get("parameters") or {}).get("properties") or {}
    if name == "image_analyze":
        function["description"] = (
            "Analyze one or more images owned by this Run. Pass each image as a "
            "run-bound asset_id, a run-owned output_id, or an HTTPS URL."
        )
        for field_name in ("image_url", "image_paths"):
            if isinstance(properties.get(field_name), dict):
                properties[field_name]["description"] = (
                    "One reference or an array of references. Each value must be a "
                    "run-bound asset_id, a run-owned output_id, or an HTTPS URL."
                )
    elif name == "video_analyze":
        function["description"] = (
            "Analyze a video owned by this Run. Pass video_url as a run-bound "
            "asset_id or a run-owned output_id. The source stays in the data plane; "
            "the model receives only bounded derived evidence."
        )
        if isinstance(properties.get("video_url"), dict):
            properties["video_url"]["description"] = (
                "A run-bound asset_id or a run-owned output_id."
            )
    return projected
