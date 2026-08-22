"""Image analysis through Hermes' auxiliary vision model.

This module owns the public image-analysis contract for one to sixteen images:
every input is resolved and validated, all images are sent to the vision model
in one request, and the result is returned as text in the standard Hermes JSON
tool envelope.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Optional

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from hermes_constants import get_hermes_dir
from tools.registry import registry
from tools.vision_tools import (
    _EMBED_MAX_DIMENSION,
    _MAX_BASE64_BYTES,
    _RESIZE_TARGET_BYTES,
    _VISION_MAX_DOWNLOAD_BYTES,
    _detect_image_mime_type,
    _download_image,
    _image_exceeds_dimension,
    _image_to_base64_data_url,
    _is_image_size_error,
    _resize_image_for_vision,
    _validate_image_url_async,
    check_vision_requirements,
)
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_MAX_IMAGES = 16
_RETRY_TOTAL_BASE64_BYTES = 16 * 1024 * 1024


def _normalize_image_field(value: Any, field_name: str) -> list[str]:
    """Normalize one schema field to a list while preserving input order."""
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"{field_name} must be a string or an array of strings")

    normalized: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{field_name}[{index}] must be a non-empty string"
            )
        normalized.append(item.strip())
    return normalized


def _normalize_image_sources(
    image_url: Optional[str | list[str]],
    image_paths: Optional[str | list[str]],
) -> list[str]:
    """Merge both accepted fields and enforce the tool-wide 1–16 limit."""
    sources = [
        *_normalize_image_field(image_url, "image_url"),
        *_normalize_image_field(image_paths, "image_paths"),
    ]
    if not sources:
        raise ValueError("Provide at least one image in image_url or image_paths")
    if len(sources) > _MAX_IMAGES:
        raise ValueError(
            f"image_analyze accepts at most {_MAX_IMAGES} images per call; "
            f"received {len(sources)}"
        )
    return sources


async def _resolve_image_source(
    source: str,
    *,
    index: int,
) -> tuple[Path, bool]:
    """Resolve one local/remote source and report whether it needs cleanup."""
    resolved_source = source[len("file://"):] if source.startswith("file://") else source
    local_path = Path(os.path.expanduser(resolved_source))
    if local_path.is_file():
        local_size = local_path.stat().st_size
        if local_size > _VISION_MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Image {index} is too large "
                f"({local_size} bytes, "
                f"max {_VISION_MAX_DOWNLOAD_BYTES})"
            )
        return local_path, False

    if not await _validate_image_url_async(source):
        raise ValueError(
            f"Invalid image source at position {index}. Provide an HTTP/HTTPS "
            "URL, file:// URI, or valid local file path."
        )

    blocked = check_website_access(source)
    if blocked:
        raise PermissionError(blocked["message"])

    temp_dir = get_hermes_dir("cache/vision", "temp_vision_images")
    temp_path = temp_dir / f"image_analyze_{uuid.uuid4()}.img"
    try:
        await _download_image(source, temp_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return temp_path, True


def _encode_image(path: Path, *, index: int) -> str:
    """Validate and encode one image, shrinking only when required."""
    mime_type = _detect_image_mime_type(path)
    if not mime_type:
        raise ValueError(
            f"Image {index} is not a supported image file "
            "(JPEG, PNG, GIF, BMP, WebP, or SVG)."
        )

    data_url = _image_to_base64_data_url(path, mime_type=mime_type)
    if (
        len(data_url) > _MAX_BASE64_BYTES
        or _image_exceeds_dimension(path, _EMBED_MAX_DIMENSION)
    ):
        data_url = _resize_image_for_vision(
            path,
            mime_type=mime_type,
            max_base64_bytes=_RESIZE_TARGET_BYTES,
            max_dimension=_EMBED_MAX_DIMENSION,
        )
    if len(data_url) > _MAX_BASE64_BYTES:
        raise ValueError(
            f"Image {index} remains too large after resizing "
            f"({len(data_url)} base64 bytes, max {_MAX_BASE64_BYTES})."
        )
    return data_url


def _fit_combined_payload(
    paths: list[Path],
    data_urls: list[str],
    *,
    max_total_bytes: int,
    max_image_bytes: Optional[int] = None,
) -> list[str]:
    """Shrink images proportionally when their combined request is too large."""
    combined_size = sum(len(item) for item in data_urls)
    individual_sizes_fit = (
        max_image_bytes is None
        or all(len(item) <= max_image_bytes for item in data_urls)
    )
    if combined_size <= max_total_bytes and individual_sizes_fit:
        return data_urls

    per_image_budget = max(512 * 1024, max_total_bytes // len(paths))
    if max_image_bytes is not None:
        per_image_budget = min(per_image_budget, max_image_bytes)
    resized: list[str] = []
    for index, path in enumerate(paths, start=1):
        mime_type = _detect_image_mime_type(path)
        if not mime_type:
            raise ValueError(f"Image {index} is not a supported image file")
        resized.append(
            _resize_image_for_vision(
                path,
                mime_type=mime_type,
                max_base64_bytes=per_image_budget,
                max_dimension=_EMBED_MAX_DIMENSION,
            )
        )

    combined_size = sum(len(item) for item in resized)
    if combined_size > max_total_bytes:
        raise ValueError(
            "Combined image payload remains too large after resizing "
            f"({combined_size} base64 bytes, max {max_total_bytes})."
        )
    return resized


def _build_analysis_content(
    data_urls: list[str],
    question: str,
) -> list[dict[str, Any]]:
    """Build one ordered multimodal user message for the full image set."""
    count = len(data_urls)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Analyze the following {count} image"
                f"{'s' if count != 1 else ''} together. Fully describe each "
                "image individually in order, then answer the question across "
                "the complete set. Compare or cross-reference images when "
                "relevant. If a requested detail is not visible, say so "
                f"explicitly.\n\nQuestion: {question}"
            ),
        }
    ]
    for index, data_url in enumerate(data_urls, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Image {index}:"},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
    return content


def _vision_call_settings(image_count: int) -> tuple[float, float, int]:
    """Resolve existing auxiliary vision settings without adding new config."""
    timeout = 120.0
    temperature = 0.1
    max_tokens = min(8000, 2000 + (400 * image_count))
    try:
        from hermes_cli.config import cfg_get, load_config

        config = load_config()
        vision_config = cfg_get(config, "auxiliary", "vision", default={})
        if vision_config.get("timeout") is not None:
            timeout = float(vision_config["timeout"])
        if vision_config.get("temperature") is not None:
            temperature = float(vision_config["temperature"])
        if vision_config.get("max_tokens") is not None:
            max_tokens = int(vision_config["max_tokens"])
    except Exception as exc:
        logger.debug("Could not resolve image analysis settings: %s", exc)
    return timeout, temperature, max_tokens


async def image_analyze_tool(
    *,
    image_url: Optional[str | list[str]] = None,
    image_paths: Optional[str | list[str]] = None,
    question: str,
) -> str:
    """Analyze one to sixteen images together in a single vision-model call."""
    temp_paths: list[Path] = []
    provider_submission_started = False
    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            raise RuntimeError("Interrupted")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        sources = _normalize_image_sources(image_url, image_paths)
        resolved_paths: list[Path] = []
        for index, source in enumerate(sources, start=1):
            path, should_cleanup = await _resolve_image_source(
                source,
                index=index,
            )
            resolved_paths.append(path)
            if should_cleanup:
                temp_paths.append(path)

        data_urls = [
            _encode_image(path, index=index)
            for index, path in enumerate(resolved_paths, start=1)
        ]
        data_urls = _fit_combined_payload(
            resolved_paths,
            data_urls,
            max_total_bytes=_MAX_BASE64_BYTES,
        )

        timeout, temperature, max_tokens = _vision_call_settings(len(sources))
        messages = [
            {
                "role": "user",
                "content": _build_analysis_content(data_urls, question.strip()),
            }
        ]
        call_kwargs: dict[str, Any] = {
            "task": "vision",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        model = os.getenv("AUXILIARY_VISION_MODEL", "").strip()
        if model:
            call_kwargs["model"] = model

        try:
            provider_submission_started = True
            response = await async_call_llm(**call_kwargs)
        except Exception as exc:
            if not _is_image_size_error(exc):
                raise
            logger.info(
                "Vision provider rejected the combined image payload; "
                "resizing and retrying once"
            )
            data_urls = _fit_combined_payload(
                resolved_paths,
                data_urls,
                max_total_bytes=_RETRY_TOTAL_BASE64_BYTES,
                max_image_bytes=_RESIZE_TARGET_BYTES,
            )
            messages[0]["content"] = _build_analysis_content(
                data_urls,
                question.strip(),
            )
            response = await async_call_llm(**call_kwargs)

        analysis = extract_content_or_reasoning(response)
        if not analysis:
            logger.warning("Image analysis returned empty content; retrying once")
            response = await async_call_llm(**call_kwargs)
            analysis = extract_content_or_reasoning(response)
        if not analysis:
            raise RuntimeError("Vision model returned no analysis")

        return json.dumps(
            {
                "success": True,
                "analysis": analysis,
                "image_count": len(sources),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.error("image_analyze failed: %s", exc, exc_info=True)
        normalized_error = str(exc).lower()
        retryable = provider_submission_started and any(hint in normalized_error for hint in (
            "timeout", "timed out", "connection", "temporarily unavailable",
            "rate limit", "429", "500", "502", "503", "504", "offline",
        ))
        error_code = (
            "image_analysis_provider_unavailable"
            if retryable
            else "image_analysis_provider_failed"
            if provider_submission_started
            else "invalid_image_input"
        )
        return json.dumps(
            {
                "success": False,
                "error": f"Error analyzing images: {exc}",
                "error_code": error_code,
                "retryable": retryable,
                "provider_submission_started": provider_submission_started,
                "analysis": (
                    "The images could not be analyzed. Correct the reported "
                    "input or provider error and retry."
                ),
            },
            ensure_ascii=False,
        )
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning(
                    "Could not delete temporary image %s: %s",
                    path,
                    exc,
                    exc_info=True,
                )


IMAGE_ANALYZE_SCHEMA: dict[str, Any] = {
    "name": "image_analyze",
    "description": (
        "Analyze one or more images using a vision model in a single call. "
        "Pass a string for one image or an array for multiple images. URLs "
        "and local paths can be mixed. The model sees the full set together "
        "and describes each image before answering the question. Accepts at "
        "most 16 images per call."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_url": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": _MAX_IMAGES,
                    },
                ],
                "description": (
                    "One image or an array of images. Each entry is an "
                    "HTTP(S) URL, file:// URI, or local file path."
                ),
            },
            "image_paths": {
                "anyOf": [
                    {"type": "string"},
                    {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": _MAX_IMAGES,
                    },
                ],
                "description": (
                    "Alias of image_url with the same shape. Entries from "
                    "both fields are analyzed together."
                ),
            },
            "question": {
                "type": "string",
                "description": (
                    "The specific question or request about the image set."
                ),
            },
        },
        "required": ["question"],
        "anyOf": [
            {"required": ["image_url"]},
            {"required": ["image_paths"]},
        ],
    },
}


def _handle_image_analyze(
    args: dict[str, Any],
    **_: Any,
) -> Awaitable[str]:
    return image_analyze_tool(
        image_url=args.get("image_url"),
        image_paths=args.get("image_paths"),
        question=args.get("question", ""),
    )


registry.register(
    name="image_analyze",
    toolset="vision",
    schema=IMAGE_ANALYZE_SCHEMA,
    handler=_handle_image_analyze,
    check_fn=check_vision_requirements,
    is_async=True,
    emoji="👁️",
)
