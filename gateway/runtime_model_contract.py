"""Exact model contracts for delegated media tools; execution stays in the gateway."""
from __future__ import annotations

import re
from typing import Any

_MODEL_SCHEMA_ENVELOPE_FIELDS = frozenset({"request_id", "model", "medias"})
_MEDIA_ROLE_PARAMETER_ALIASES = {
    "start_frame": ("first_frame", "first_frame_img", "image", "input_image"),
    "end_frame": ("last_frame", "last_image", "end_image", "last_frame_img"),
    "reference_video": (
        "reference_videos",
        "reference_video",
        "videos",
        "video",
        "input_video",
    ),
    "audio": ("audio", "audio_url", "input_audio"),
    "reference": ("reference_images", "reference_image", "images", "image", "input_image"),
    "reference_image": ("reference_images", "reference_image", "images", "image", "input_image"),
    "moodboard": ("reference_images", "reference_image", "images", "image", "input_image"),
    "style_reference": ("reference_images", "reference_image", "images", "image", "input_image"),
    "foundation_reference": ("reference_images", "reference_image", "images", "image", "input_image"),
    "character_reference": ("reference_images", "reference_image", "images", "image", "input_image"),
    "element_reference": ("reference_images", "reference_image", "images", "image", "input_image"),
    "product_photo": ("reference_images", "reference_image", "images", "image", "input_image"),
    "logo": ("reference_images", "reference_image", "images", "image", "input_image"),
    "character_sheet": ("reference_images", "reference_image", "images", "image", "input_image"),
    "storyboard": ("reference_images", "reference_image", "images", "image", "input_image"),
    "user_upload": ("reference_images", "reference_image", "images", "image", "input_image"),
    "parcel_photo": ("reference_images", "reference_image", "images", "image", "input_image"),
}
_PLATFORM_MANAGED_MEDIA_PARAMETERS = frozenset(
    _MEDIA_ROLE_PARAMETER_ALIASES
) | frozenset(
    parameter_name
    for aliases in _MEDIA_ROLE_PARAMETER_ALIASES.values()
    for parameter_name in aliases
)
def _requires_model_parameter_contract(tool_name: str) -> bool:
    return tool_name == "media.estimate_cost" or tool_name.startswith("media.generate_")
def _model_parameter_contract(
    result: Any,
) -> tuple[str, str, dict[str, dict[str, Any]]] | None:
    """Extract one exact model contract from Runtime-private control data."""
    if not isinstance(result, dict):
        return None
    if set(result) != {"model", "parameters", "observed_schema_digest"}:
        return None
    model = str(result.get("model") or "").strip()
    digest = str(result.get("observed_schema_digest") or "").strip()
    if not model or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        return None
    parameters = result.get("parameters")
    if not isinstance(parameters, list) or not parameters:
        return None
    contract: dict[str, dict[str, Any]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            return None
        if set(parameter) - {
            "name", "type", "required", "default", "options", "description"
        }:
            return None
        name = str(parameter.get("name") or "").strip()
        parameter_type = str(parameter.get("type") or "").strip()
        if not name or not parameter_type or name in contract:
            return None
        contract[name] = {
            "type": parameter_type,
            "required": parameter.get("required") is True,
            "options": list(parameter.get("options") or []),
            "description": str(parameter.get("description") or ""),
        }
        if "default" in parameter:
            contract[name]["default"] = parameter.get("default")
    return model, digest, contract


def _parameter_value_matches_type(value: Any, parameter_type: str) -> bool:
    parameter_type = parameter_type.lower()
    if parameter_type == "string":
        return isinstance(value, str)
    if parameter_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if parameter_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if parameter_type == "boolean":
        return isinstance(value, bool)
    if parameter_type == "array":
        return isinstance(value, list)
    if parameter_type == "object":
        return isinstance(value, dict)
    return False


def _arbitrary_image_size_is_valid(value: Any, description: str) -> bool:
    if not isinstance(value, str) or "arbitrary" not in description.lower():
        return False
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value.strip().lower())
    if match is None:
        return False
    width, height = int(match.group(1)), int(match.group(2))
    return (
        width % 16 == 0
        and height % 16 == 0
        and 1 / 3 <= width / height <= 3
        and max(width, height) <= 3840
        and width * height <= 3840 * 2160
    )


def _model_parameter_is_supplied_by_medias(name: str, request: dict[str, Any]) -> bool:
    """Return whether the platform medias envelope supplies one provider field."""
    medias = request.get("medias")
    if not isinstance(medias, list) or not medias:
        return False
    for media in medias:
        if not isinstance(media, dict):
            continue
        role = str(media.get("role") or "").strip().lower()
        if not role:
            continue
        if name == role or name in _MEDIA_ROLE_PARAMETER_ALIASES.get(role, ()):
            return True
    return False


def _model_request_contract_error(
    tool_name: str,
    args: Any,
    contracts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any] | None:
    """Validate media requests against contracts fetched for their exact models."""
    if not _requires_model_parameter_contract(tool_name):
        return None
    requests = args.get("requests") if isinstance(args, dict) else None
    if not isinstance(requests, list) or not requests:
        return None  # The platform tool schema owns the generic envelope error.
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            continue
        model = str(request.get("model") or "").strip()
        if not model:
            continue
        contract = contracts.get(model)
        if contract is None:
            return {
                "error": {
                    "code": "model_schema_required",
                    "message": (
                        f"Runtime did not resolve the exact input contract for model "
                        f"{model!r} before calling {tool_name}."
                    ),
                    "retryable": True,
                }
            }
        writer_source = tool_name == "media.generate_video" and "writer_tool_call_id" in request
        supplied = set(request) - _MODEL_SCHEMA_ENVELOPE_FIELDS
        if writer_source:
            supplied.discard("writer_tool_call_id")
        provider_media_fields = sorted(supplied & _PLATFORM_MANAGED_MEDIA_PARAMETERS)
        if provider_media_fields:
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": (
                        f"requests[{index}] contains provider-managed media parameters: "
                        f"{', '.join(provider_media_fields)}. Keep Runtime asset and output "
                        f"references in requests[{index}].medias; the platform maps them to "
                        "provider parameters at the execution boundary."
                    ),
                    "retryable": False,
                }
            }
        literal_contract = {
            name: parameter
            for name, parameter in contract.items()
            if name not in _PLATFORM_MANAGED_MEDIA_PARAMETERS
        }
        unknown = sorted(supplied - set(literal_contract))
        if unknown:
            allowed_parts = []
            for parameter_name in sorted(literal_contract):
                options = literal_contract[parameter_name]["options"]
                if options:
                    allowed_parts.append(f"{parameter_name}={options}")
                else:
                    allowed_parts.append(parameter_name)
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": (
                        f"requests[{index}] contains parameters not declared by model "
                        f"{model!r}: {', '.join(unknown)}. Allowed literal parameters: "
                        f"{'; '.join(allowed_parts)}"
                    ),
                    "retryable": False,
                }
            }
        medias = request.get("medias")
        if isinstance(medias, list) and medias:
            compatible_media_parameters = [
                name
                for name in contract
                if (
                    name in _PLATFORM_MANAGED_MEDIA_PARAMETERS
                    and _model_parameter_is_supplied_by_medias(name, request)
                )
            ]
            if not compatible_media_parameters:
                roles = sorted({
                    str(media.get("role") or "").strip()
                    for media in medias
                    if isinstance(media, dict) and str(media.get("role") or "").strip()
                })
                return {
                    "error": {
                        "code": "invalid_tool_arguments",
                        "message": (
                            f"Model {model!r} cannot accept the supplied platform media "
                            f"roles: {', '.join(roles) or 'unknown'}. Preserve medias and "
                            "use the workflow-declared model for that media stage."
                        ),
                        "retryable": False,
                    }
                }
        missing_literal = sorted(
            name for name, parameter in literal_contract.items()
            if parameter["required"] and name not in request
            and not (writer_source and name == "prompt")
        )
        if missing_literal:
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": (
                        f"requests[{index}] is missing required model parameters: "
                        f"{', '.join(missing_literal)}"
                    ),
                    "retryable": False,
                }
            }
        missing_media = sorted(
            name
            for name, parameter in contract.items()
            if (
                name in _PLATFORM_MANAGED_MEDIA_PARAMETERS
                and parameter["required"]
                and not _model_parameter_is_supplied_by_medias(name, request)
            )
        )
        if missing_media:
            return {
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": (
                        f"requests[{index}] requires platform media input for model "
                        f"{model!r}. Provide Runtime asset or output references in medias; "
                        "do not send provider fields such as "
                        f"{', '.join(missing_media)}."
                    ),
                    "retryable": False,
                }
            }
        for name in sorted(supplied):
            parameter = literal_contract[name]
            value = request[name]
            if not _parameter_value_matches_type(value, parameter["type"]):
                return {
                    "error": {
                        "code": "invalid_tool_arguments",
                        "message": (
                            f"requests[{index}].{name} must have model-declared type "
                            f"{parameter['type']}"
                        ),
                        "retryable": False,
                    }
                }
            options = parameter["options"]
            if options and value not in options and not (
                name == "size"
                and _arbitrary_image_size_is_valid(value, parameter["description"])
            ):
                return {
                    "error": {
                        "code": "invalid_tool_arguments",
                        "message": (
                            f"requests[{index}].{name} is not allowed by model {model!r}; "
                            f"use one of {options}"
                        ),
                        "retryable": False,
                    }
                }
    return None

