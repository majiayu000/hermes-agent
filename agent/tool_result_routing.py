"""Capability and projection helpers for image-bearing tool results.

User image input and image content inside a ``role=tool`` result are separate
provider contracts.  This module keeps the latter fail-closed: only an
explicit, integration-verified capability may place a data URL in a tool
message.  Reference-only and unknown routes retain text/media references while
removing pixels before the request is sent.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


TOOL_RESULT_IMAGE_EMBED_DATA_URL = "embed_data_url"
TOOL_RESULT_IMAGE_ATTACH_BY_REF = "attach_by_ref"
TOOL_RESULT_IMAGE_REJECT = "reject"
TOOL_RESULT_IMAGE_MODES = frozenset({
    TOOL_RESULT_IMAGE_EMBED_DATA_URL,
    TOOL_RESULT_IMAGE_ATTACH_BY_REF,
    TOOL_RESULT_IMAGE_REJECT,
})


def _normalize_tool_result_image_mode(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in TOOL_RESULT_IMAGE_MODES else None


def _configured_tool_result_image_mode(
    cfg: Optional[Dict[str, Any]],
    provider: str,
    model: str,
) -> Optional[str]:
    """Return an exact config override without consulting a live catalog."""
    if not isinstance(cfg, dict):
        return None

    model_cfg_raw = cfg.get("model")
    model_cfg: Dict[str, Any] = (
        model_cfg_raw if isinstance(model_cfg_raw, dict) else {}
    )
    top_level = _normalize_tool_result_image_mode(
        model_cfg.get("tool_result_image_mode")
    )
    if top_level is not None:
        return top_level

    config_provider = str(model_cfg.get("provider") or "").strip()
    providers_raw = cfg.get("providers")
    providers_cfg: Dict[str, Any] = (
        providers_raw if isinstance(providers_raw, dict) else {}
    )
    for provider_name in dict.fromkeys(filter(None, (provider, config_provider))):
        provider_entry_raw = providers_cfg.get(provider_name)
        provider_entry: Dict[str, Any] = (
            provider_entry_raw if isinstance(provider_entry_raw, dict) else {}
        )
        models_raw = provider_entry.get("models")
        models: Dict[str, Any] = models_raw if isinstance(models_raw, dict) else {}
        model_entry_raw = models.get(model)
        model_entry: Dict[str, Any] = (
            model_entry_raw if isinstance(model_entry_raw, dict) else {}
        )
        configured = _normalize_tool_result_image_mode(
            model_entry.get("tool_result_image_mode")
        )
        if configured is not None:
            return configured

    custom_providers = cfg.get("custom_providers")
    if not isinstance(custom_providers, list):
        return None
    candidate_names: set[str] = set()
    for provider_name in filter(None, (provider, config_provider)):
        candidate_names.add(provider_name)
        if provider_name.startswith("custom:"):
            candidate_names.add(provider_name[len("custom:"):])
        else:
            candidate_names.add(f"custom:{provider_name}")
    for provider_entry_raw in custom_providers:
        if not isinstance(provider_entry_raw, dict):
            continue
        provider_name = str(provider_entry_raw.get("name") or "").strip()
        if provider_name not in candidate_names:
            continue
        models_raw = provider_entry_raw.get("models")
        models = models_raw if isinstance(models_raw, dict) else {}
        model_entry_raw = models.get(model)
        model_entry = model_entry_raw if isinstance(model_entry_raw, dict) else {}
        configured = _normalize_tool_result_image_mode(
            model_entry.get("tool_result_image_mode")
        )
        if configured is not None:
            return configured
    return None


def _resolve_tool_result_image_mode(
    provider: str,
    model: str,
    cfg: Optional[Dict[str, Any]],
    *,
    runtime_override: Any = None,
) -> str:
    """Resolve the tool-image contract; unknown routes always reject pixels."""
    runtime_mode = _normalize_tool_result_image_mode(runtime_override)
    if runtime_mode is not None:
        return runtime_mode

    configured = _configured_tool_result_image_mode(cfg, provider, model)
    if configured is not None:
        return configured

    try:
        from providers import get_provider_profile

        profile = get_provider_profile(provider)
    except Exception:
        profile = None
    if profile is None:
        return TOOL_RESULT_IMAGE_REJECT

    # Compatibility for external provider plugins that explicitly declared
    # the old boolean.  An omitted legacy field is not evidence of support.
    legacy_support = getattr(profile, "supports_vision_tool_messages", None)
    if legacy_support is True:
        return TOOL_RESULT_IMAGE_EMBED_DATA_URL
    if legacy_support is False:
        return TOOL_RESULT_IMAGE_REJECT

    profile_mode = _normalize_tool_result_image_mode(
        getattr(profile, "tool_result_image_mode", None)
    )
    return profile_mode or TOOL_RESULT_IMAGE_REJECT


def _has_inline_image_tool_result(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and isinstance(message.get("content"), list)
        and any(
            isinstance(part, dict)
            and part.get("type") in {"image_url", "input_image"}
            for part in message["content"]
        )
        for message in messages
    )


def _should_retry_tool_result_projection(
    status_code: Any,
    failure_reason: Any,
    messages: Any,
    *,
    already_attempted: bool,
) -> bool:
    """Choose the one safe payload variant without relying on error text."""
    if already_attempted:
        return False
    normalized_reason = str(getattr(failure_reason, "value", failure_reason) or "")
    if normalized_reason == "multimodal_tool_content_unsupported":
        return True
    non_payload_reasons = {
        "auth",
        "auth_permanent",
        "billing",
        "content_policy_blocked",
        "invalid_encrypted_content",
        "llama_cpp_grammar_pattern",
        "long_context_tier",
        "model_not_found",
        "oauth_long_context_beta_forbidden",
        "provider_policy_blocked",
        "rate_limit",
        "thinking_signature",
    }
    return (
        status_code in {400, 422}
        and normalized_reason not in non_payload_reasons
        and _has_inline_image_tool_result(messages)
    )


def _strip_inline_image_tool_results(messages: Any) -> bool:
    """Replace tool-result image lists with their reference/text projection."""
    if not isinstance(messages, list):
        return False
    changed = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        had_image = False
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"image_url", "input_image"}:
                had_image = True
                continue
            if part_type in {"text", "input_text"}:
                text = str(part.get("text") or "").strip()
                if text:
                    text_parts.append(text)
        if not had_image:
            continue
        message["content"] = (
            "\n\n".join(text_parts)
            if text_parts
            else "[image content omitted; use the retained media reference or image_analyze]"
        )
        changed = True
    return changed


__all__ = [
    "TOOL_RESULT_IMAGE_ATTACH_BY_REF",
    "TOOL_RESULT_IMAGE_EMBED_DATA_URL",
    "TOOL_RESULT_IMAGE_MODES",
    "TOOL_RESULT_IMAGE_REJECT",
]
