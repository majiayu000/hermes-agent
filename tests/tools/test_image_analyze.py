"""Tests for the multi-image ``image_analyze`` tool."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.image_analyze import (
    IMAGE_ANALYZE_SCHEMA,
    _build_analysis_content,
    _encode_image,
    _fit_combined_payload,
    _handle_image_analyze,
    _normalize_image_sources,
    _resolve_image_source,
    _vision_call_settings,
    image_analyze_tool,
)


_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _vision_response(text: str) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    response.choices = [choice]
    return response


class TestNormalizeImageSources:
    def test_merges_both_fields_in_order(self):
        assert _normalize_image_sources(
            ["https://example.com/one.png", "/tmp/two.png"],
            "file:///tmp/three.png",
        ) == [
            "https://example.com/one.png",
            "/tmp/two.png",
            "file:///tmp/three.png",
        ]

    @pytest.mark.parametrize("value", [None, []])
    def test_rejects_empty_input(self, value):
        with pytest.raises(ValueError, match="at least one image"):
            _normalize_image_sources(value, None)

    def test_rejects_blank_string(self):
        with pytest.raises(ValueError, match=r"image_url\[0\].*non-empty"):
            _normalize_image_sources("", None)

    def test_rejects_non_string_items(self):
        with pytest.raises(ValueError, match=r"image_url\[1\]"):
            _normalize_image_sources(cast(Any, ["ok.png", 3]), None)

    def test_rejects_non_string_field(self):
        with pytest.raises(ValueError, match="string or an array"):
            _normalize_image_sources(cast(Any, {"path": "x.png"}), None)

    def test_rejects_more_than_sixteen_combined(self):
        with pytest.raises(ValueError, match="at most 16"):
            _normalize_image_sources(["x.png"] * 10, ["y.png"] * 7)


class TestBuildAnalysisContent:
    def test_labels_each_image_and_includes_question(self):
        content = _build_analysis_content(
            ["data:image/png;base64,ONE", "data:image/png;base64,TWO"],
            "Which is brighter?",
        )
        assert content[0]["type"] == "text"
        assert "Which is brighter?" in content[0]["text"]
        assert [part["text"] for part in content if part["type"] == "text"][1:] == [
            "Image 1:",
            "Image 2:",
        ]
        image_parts = [part for part in content if part["type"] == "image_url"]
        assert len(image_parts) == 2


class TestImagePreparation:
    @pytest.mark.asyncio
    async def test_resolves_file_uri_without_cleanup(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        path, should_cleanup = await _resolve_image_source(
            f"file://{image}",
            index=1,
        )
        assert path == image
        assert should_cleanup is False

    @pytest.mark.asyncio
    async def test_rejects_oversized_local_file(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch("tools.image_analyze._VISION_MAX_DOWNLOAD_BYTES", 1),
            pytest.raises(ValueError, match="too large"),
        ):
            await _resolve_image_source(str(image), index=1)

    @pytest.mark.asyncio
    async def test_rejects_invalid_and_policy_blocked_urls(self):
        with (
            patch(
                "tools.image_analyze._validate_image_url_async",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(ValueError, match="Invalid image source"),
        ):
            await _resolve_image_source("ftp://example.com/image.png", index=1)

        with (
            patch(
                "tools.image_analyze._validate_image_url_async",
                new=AsyncMock(return_value=True),
            ),
            patch(
                "tools.image_analyze.check_website_access",
                return_value={"message": "blocked by policy"},
            ),
            pytest.raises(PermissionError, match="blocked by policy"),
        ):
            await _resolve_image_source(
                "https://example.com/image.png",
                index=1,
            )

    @pytest.mark.asyncio
    async def test_failed_remote_download_removes_partial_file(self, tmp_path):
        destinations = []

        async def fail_after_partial_write(_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"partial")
            destinations.append(destination)
            raise RuntimeError("download failed")

        with (
            patch(
                "tools.image_analyze._validate_image_url_async",
                new=AsyncMock(return_value=True),
            ),
            patch("tools.image_analyze.check_website_access", return_value=None),
            patch("tools.image_analyze.get_hermes_dir", return_value=tmp_path),
            patch(
                "tools.image_analyze._download_image",
                new=AsyncMock(side_effect=fail_after_partial_write),
            ),
            pytest.raises(RuntimeError, match="download failed"),
        ):
            await _resolve_image_source(
                "https://example.com/image.png",
                index=1,
            )

        assert len(destinations) == 1
        assert not destinations[0].exists()

    def test_encode_resizes_oversized_or_tall_image(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch(
                "tools.image_analyze._image_to_base64_data_url",
                return_value="x" * 100,
            ),
            patch("tools.image_analyze._MAX_BASE64_BYTES", 50),
            patch(
                "tools.image_analyze._resize_image_for_vision",
                return_value="small",
            ) as mock_resize,
        ):
            assert _encode_image(image, index=1) == "small"
        mock_resize.assert_called_once()

    def test_encode_rejects_image_that_cannot_be_reduced(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch(
                "tools.image_analyze._image_to_base64_data_url",
                return_value="x" * 100,
            ),
            patch("tools.image_analyze._MAX_BASE64_BYTES", 50),
            patch(
                "tools.image_analyze._resize_image_for_vision",
                return_value="x" * 60,
            ),
            pytest.raises(ValueError, match="remains too large"),
        ):
            _encode_image(image, index=1)

    def test_fit_combined_payload_resizes_and_enforces_limit(self, tmp_path):
        first = tmp_path / "first.png"
        second = tmp_path / "second.png"
        first.write_bytes(_TINY_PNG)
        second.write_bytes(_TINY_PNG)

        with patch(
            "tools.image_analyze._resize_image_for_vision",
            side_effect=["a" * 4, "b" * 4],
        ) as mock_resize:
            result = _fit_combined_payload(
                [first, second],
                ["x" * 8, "y" * 8],
                max_total_bytes=10,
            )
        assert result == ["a" * 4, "b" * 4]
        assert mock_resize.call_count == 2

        with (
            patch(
                "tools.image_analyze._resize_image_for_vision",
                side_effect=["a" * 6, "b" * 6],
            ),
            pytest.raises(ValueError, match="Combined image payload"),
        ):
            _fit_combined_payload(
                [first, second],
                ["x" * 8, "y" * 8],
                max_total_bytes=10,
            )

    def test_vision_call_settings_uses_existing_config(self):
        with (
            patch(
                "hermes_cli.config.load_config",
                return_value={"auxiliary": {"vision": {}}},
            ),
            patch(
                "hermes_cli.config.cfg_get",
                return_value={
                    "timeout": "45",
                    "temperature": "0.2",
                    "max_tokens": "6000",
                },
            ),
        ):
            assert _vision_call_settings(3) == (45.0, 0.2, 6000)


class TestImageAnalyzeTool:
    @pytest.mark.asyncio
    async def test_sends_all_images_in_one_model_call(self, tmp_path):
        first = tmp_path / "first.png"
        second = tmp_path / "second.png"
        first.write_bytes(_TINY_PNG)
        second.write_bytes(_TINY_PNG)

        with (
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(return_value=_vision_response("Joint analysis")),
            ) as mock_call,
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 4000),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(first),
                    image_paths=[str(second)],
                    question="Compare them",
                )
            )

        assert result == {
            "success": True,
            "analysis": "Joint analysis",
            "image_count": 2,
        }
        mock_call.assert_awaited_once()
        await_args = mock_call.await_args
        assert await_args is not None
        messages = await_args.kwargs["messages"]
        image_parts = [
            part
            for part in messages[0]["content"]
            if part["type"] == "image_url"
        ]
        assert len(image_parts) == 2
        assert "Compare them" in messages[0]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_invalid_image_fails_without_calling_model(self, tmp_path):
        invalid = tmp_path / "not-an-image.txt"
        invalid.write_text("not an image")

        with patch(
            "tools.image_analyze.async_call_llm",
            new=AsyncMock(),
        ) as mock_call:
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(invalid),
                    question="Read it",
                )
            )

        assert result["success"] is False
        assert "supported image file" in result["error"]
        assert result["error_code"] == "invalid_image_input"
        assert result["provider_submission_started"] is False
        mock_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_remote_temp_file_is_deleted(self, tmp_path):
        downloaded: list = []

        async def fake_download(_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_TINY_PNG)
            downloaded.append(destination)
            return destination

        with (
            patch(
                "tools.image_analyze._validate_image_url_async",
                new=AsyncMock(return_value=True),
            ),
            patch("tools.image_analyze.check_website_access", return_value=None),
            patch("tools.image_analyze.get_hermes_dir", return_value=tmp_path),
            patch(
                "tools.image_analyze._download_image",
                new=AsyncMock(side_effect=fake_download),
            ),
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(return_value=_vision_response("Remote analysis")),
            ),
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url="https://example.com/image.png",
                    question="Describe it",
                )
            )

        assert result["success"] is True
        assert len(downloaded) == 1
        assert not downloaded[0].exists()

    @pytest.mark.asyncio
    async def test_retries_once_on_empty_model_output(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        responses = [_vision_response(""), _vision_response("Recovered")]

        with (
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(side_effect=responses),
            ) as mock_call,
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(image),
                    question="Describe it",
                )
            )

        assert result["success"] is True
        assert result["analysis"] == "Recovered"
        assert mock_call.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_size_rejection_with_reduced_payload(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        calls = [
            RuntimeError("413 payload too large"),
            _vision_response("Reduced analysis"),
        ]

        with (
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(side_effect=calls),
            ) as mock_call,
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
            patch(
                "tools.image_analyze._fit_combined_payload",
                side_effect=lambda _paths, urls, **_kwargs: urls,
            ) as mock_fit,
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(image),
                    question="Describe it",
                )
            )

        assert result["analysis"] == "Reduced analysis"
        assert mock_call.await_count == 2
        assert mock_fit.call_count == 2
        assert mock_fit.call_args_list[1].kwargs["max_image_bytes"] > 0

    @pytest.mark.asyncio
    async def test_non_size_provider_error_is_reported(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(side_effect=RuntimeError("provider offline")),
            ),
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(image),
                    question="Describe it",
                )
            )
        assert result["success"] is False
        assert "provider offline" in result["error"]
        assert result["error_code"] == "image_analysis_provider_unavailable"
        assert result["retryable"] is True
        assert result["provider_submission_started"] is True

    @pytest.mark.asyncio
    async def test_empty_retry_is_an_explicit_error(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(return_value=_vision_response("")),
            ),
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(image),
                    question="Describe it",
                )
            )
        assert result["success"] is False
        assert "returned no analysis" in result["error"]

    @pytest.mark.asyncio
    async def test_passes_internal_model_override_without_public_schema(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        with (
            patch.dict(
                "tools.image_analyze.os.environ",
                {"AUXILIARY_VISION_MODEL": "configured/vision"},
            ),
            patch(
                "tools.image_analyze.async_call_llm",
                new=AsyncMock(return_value=_vision_response("Analysis")),
            ) as mock_call,
            patch(
                "tools.image_analyze._vision_call_settings",
                return_value=(30.0, 0.1, 2400),
            ),
        ):
            result = json.loads(
                await image_analyze_tool(
                    image_url=str(image),
                    question="Describe it",
                )
            )
        assert result["success"] is True
        await_args = mock_call.await_args
        assert await_args is not None
        assert await_args.kwargs["model"] == "configured/vision"

    @pytest.mark.asyncio
    async def test_question_is_required(self, tmp_path):
        image = tmp_path / "image.png"
        image.write_bytes(_TINY_PNG)
        result = json.loads(
            await image_analyze_tool(
                image_url=str(image),
                question="",
            )
        )
        assert result["success"] is False
        assert "question must be a non-empty string" in result["error"]


class TestImageAnalyzeRegistration:
    def test_schema_matches_multi_image_contract(self):
        parameters = cast(dict[str, Any], IMAGE_ANALYZE_SCHEMA["parameters"])
        properties = cast(dict[str, Any], parameters["properties"])
        assert IMAGE_ANALYZE_SCHEMA["name"] == "image_analyze"
        assert properties["image_url"]["anyOf"][1]["maxItems"] == 16
        assert properties["image_paths"]["anyOf"][1]["maxItems"] == 16
        assert parameters["required"] == ["question"]

    def test_registered_as_async_vision_tool(self):
        from tools.registry import registry

        entry = registry.get_entry("image_analyze")
        assert entry is not None
        assert entry.toolset == "vision"
        assert entry.is_async is True
        assert entry.handler is _handle_image_analyze

    def test_handler_forwards_both_aliases(self):
        with patch(
            "tools.image_analyze.image_analyze_tool",
            new=AsyncMock(return_value="{}"),
        ) as mock_tool:
            coroutine = _handle_image_analyze(
                {
                    "image_url": ["one.png"],
                    "image_paths": "two.png",
                    "question": "Compare",
                }
            )
            assert asyncio.iscoroutine(coroutine)
            coroutine.close()

        mock_tool.assert_called_once_with(
            image_url=["one.png"],
            image_paths="two.png",
            question="Compare",
        )

    def test_exposed_by_core_vision_and_mcp_surfaces(self):
        from agent.tool_dispatch_helpers import _PARALLEL_SAFE_TOOLS
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        from toolsets import _HERMES_CORE_TOOLS, TOOLSETS

        assert "image_analyze" in _HERMES_CORE_TOOLS
        vision_tools = cast(list[str], TOOLSETS["vision"]["tools"])
        assert "image_analyze" in vision_tools
        assert "image_analyze" in _PARALLEL_SAFE_TOOLS
        assert "image_analyze" in EXPOSED_TOOLS
