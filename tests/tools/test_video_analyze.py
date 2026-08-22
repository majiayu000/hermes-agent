"""Tests for video_analyze tool in tools/vision_tools.py."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from tools.video_frame_analysis import ExtractedVideoFrame, VideoFrameExtractionError
from tools.vision_tools import (
    _detect_video_mime_type,
    _handle_video_analyze,
    video_analyze_tool,
    VIDEO_ANALYZE_SCHEMA,
)


# ---------------------------------------------------------------------------
# _detect_video_mime_type
# ---------------------------------------------------------------------------


class TestDetectVideoMimeType:
    """Extension-based MIME detection for video files."""

    def test_mp4(self, tmp_path):
        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_webm(self, tmp_path):
        p = tmp_path / "clip.webm"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/webm"

    def test_mov(self, tmp_path):
        p = tmp_path / "clip.mov"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mov"

    def test_avi_fallback_mp4(self, tmp_path):
        p = tmp_path / "clip.avi"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_mkv_fallback_mp4(self, tmp_path):
        p = tmp_path / "clip.mkv"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"

    def test_mpeg(self, tmp_path):
        p = tmp_path / "clip.mpeg"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mpeg"

    def test_mpg(self, tmp_path):
        p = tmp_path / "clip.mpg"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mpeg"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "clip.flv"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) is None

    def test_case_insensitive(self, tmp_path):
        p = tmp_path / "clip.MP4"
        p.write_bytes(b"\x00" * 10)
        assert _detect_video_mime_type(p) == "video/mp4"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestVideoAnalyzeSchema:
    """Schema structure is correct."""

    def test_schema_name(self):
        assert VIDEO_ANALYZE_SCHEMA["name"] == "video_analyze"

    def test_schema_has_required_fields(self):
        params = VIDEO_ANALYZE_SCHEMA["parameters"]
        assert "video_url" in params["properties"]
        assert "question" in params["properties"]
        assert "include_transcript" in params["properties"]
        assert params["required"] == ["video_url", "question"]

    def test_schema_description_mentions_video(self):
        assert "video" in VIDEO_ANALYZE_SCHEMA["description"].lower()


# ---------------------------------------------------------------------------
# _handle_video_analyze handler
# ---------------------------------------------------------------------------


class TestHandleVideoAnalyze:
    """Tests for the registry handler wrapper."""

    def test_returns_awaitable(self, tmp_path, monkeypatch):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "test"})
            result = _handle_video_analyze({"video_url": str(video_file), "question": "what is this?"})
            # Should return an awaitable (coroutine)
            assert asyncio.iscoroutine(result)
            # Clean up the unawaited coroutine
            result.close()

    def test_uses_auxiliary_video_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "google/gemini-2.5-flash")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "other-model")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-2.5-flash"

    def test_falls_back_to_vision_model_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUXILIARY_VIDEO_MODEL", "")
        monkeypatch.setenv("AUXILIARY_VISION_MODEL", "google/gemini-flash")

        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({"video_url": "/tmp/test.mp4", "question": "test"})
            )
            args = mock_tool.call_args[0]
            assert args[2] == "google/gemini-flash"

    def test_forwards_explicit_transcript_request(self):
        with patch("tools.vision_tools.video_analyze_tool", new_callable=AsyncMock) as mock_tool:
            mock_tool.return_value = json.dumps({"success": True, "analysis": "ok"})
            asyncio.get_event_loop().run_until_complete(
                _handle_video_analyze({
                    "video_url": "/tmp/test.mp4",
                    "question": "test",
                    "include_transcript": True,
                })
            )
            assert mock_tool.call_args[0][3] is True


# ---------------------------------------------------------------------------
# video_analyze_tool — integration-style tests with mocked LLM
# ---------------------------------------------------------------------------


class TestVideoAnalyzeTool:
    """Core video analysis function tests."""

    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    @staticmethod
    def _sampled_frames(tmp_path):
        frames = []
        for index, (timestamp, ratio) in enumerate(
            ((1.5, 0.15), (5.0, 0.5), (8.5, 0.85)),
            start=1,
        ):
            path = tmp_path / f"frame-{index}.jpg"
            path.write_bytes(f"jpeg-{index}".encode())
            frames.append(ExtractedVideoFrame(path, timestamp, ratio))
        return frames

    def test_local_file_success(self, tmp_path, monkeypatch):
        """Analyze a local video file — happy path."""
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"\x00" * 1024)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A short video showing a demo."

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="A short video showing a demo.",
        ):
            result = self._run(video_analyze_tool(str(video), "What is this?"))

        data = json.loads(result)
        assert data["success"] is True
        assert "demo" in data["analysis"].lower()
        assert data["sampled_frames"] == [
            {"timestamp_seconds": 1.5, "ratio": 0.15},
            {"timestamp_seconds": 5.0, "ratio": 0.5},
            {"timestamp_seconds": 8.5, "ratio": 0.85},
        ]

    def test_local_file_not_found(self, tmp_path):
        """Non-existent file raises appropriate error."""
        result = self._run(video_analyze_tool("/nonexistent/video.mp4", "What?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "invalid video source" in data["analysis"].lower()
        assert data["error_code"] == "video_analysis_failed"
        assert data["retryable"] is False

    def test_unsupported_format(self, tmp_path):
        """Unsupported extension raises error."""
        video = tmp_path / "clip.flv"
        video.write_bytes(b"\x00" * 100)

        result = self._run(video_analyze_tool(str(video), "What is this?"))
        data = json.loads(result)
        assert data["success"] is False
        assert "unsupported video format" in data["analysis"].lower()
        assert data["retryable"] is False

    def test_frame_extraction_failure_prevents_provider_submission(self, tmp_path):
        video = tmp_path / "invalid.mp4"
        video.write_bytes(b"\x00" * 100)

        with patch(
            "tools.vision_tools.extract_video_frames",
            side_effect=VideoFrameExtractionError("ffprobe binary not found"),
        ), patch("tools.vision_tools.async_call_llm", new_callable=AsyncMock) as llm:
            result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is False
        assert "ffprobe binary not found" in data["analysis"].lower()
        assert data["error_code"] == "video_analysis_failed"
        assert data["retryable"] is False
        llm.assert_not_awaited()

    def test_interrupt_check(self, tmp_path):
        """Tool respects interrupt flag."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        with patch("tools.interrupt.is_interrupted", return_value=True):
            result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is False
        assert data["error_code"] == "video_analysis_interrupted"
        assert data["retryable"] is False

    def test_empty_response_is_terminal(self, tmp_path):
        """An empty paid response fails closed instead of calling the model again."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        call_count = 0
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Video analysis result."

        async def fake_llm(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch("tools.vision_tools.async_call_llm", side_effect=fake_llm), patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="",
        ):
            result = self._run(video_analyze_tool(str(video), "What?"))

        data = json.loads(result)
        assert data["success"] is False
        assert data["error_code"] == "video_analysis_failed"
        assert data["retryable"] is False
        assert call_count == 1

    def test_file_scheme_stripped(self, tmp_path):
        """file:// prefix is stripped correctly."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch("tools.vision_tools.extract_content_or_reasoning", return_value="OK"):
            result = self._run(video_analyze_tool(f"file://{video}", "What?"))

        data = json.loads(result)
        assert data["success"] is True

    def test_api_message_format(self, tmp_path):
        """The LLM receives timestamped JPEG samples, never the complete video."""
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)

        captured_kwargs = {}

        async def capture_llm(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "OK"
            return mock_response

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch("tools.vision_tools.async_call_llm", side_effect=capture_llm), patch(
            "tools.vision_tools.extract_content_or_reasoning",
            return_value="OK",
        ):
            self._run(video_analyze_tool(str(video), "Describe this"))

        messages = captured_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 7
        assert content[0]["type"] == "text"
        assert "sampled server-side" in content[0]["text"]
        assert [part["type"] for part in content[1:]] == [
            "text", "image_url", "text", "image_url", "text", "image_url",
        ]
        assert "00:01.500" in content[1]["text"]
        assert "00:05.000" in content[3]["text"]
        assert "00:08.500" in content[5]["text"]
        assert all(
            part["image_url"]["url"].startswith("data:image/jpeg;base64,")
            for part in content[2::2]
        )
        assert all(part["type"] != "video_url" for part in content)

    def test_optional_audio_transcription_is_returned_explicitly(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Visual analysis."

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch("tools.vision_tools.extract_content_or_reasoning", return_value="Visual analysis."):
            with patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={
                    "success": True,
                    "transcript": "Exact spoken words.",
                    "provider": "local",
                },
            ):
                result = self._run(
                    video_analyze_tool(
                        str(video),
                        "Describe this",
                        include_transcript=True,
                    )
                )

        data = json.loads(result)
        assert data["success"] is True
        assert data["analysis"] == "Visual analysis."
        assert data["transcription"] == {
            "success": True,
            "transcript": "Exact spoken words.",
            "provider": "local",
        }

    def test_transcription_failure_is_not_silently_hidden(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"\x00" * 100)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]

        with patch(
            "tools.vision_tools.extract_video_frames",
            return_value=self._sampled_frames(tmp_path),
        ), patch(
            "tools.vision_tools.async_call_llm",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch("tools.vision_tools.extract_content_or_reasoning", return_value="Visual analysis."):
            with patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={
                    "success": False,
                    "transcript": "",
                    "error": "STT is disabled",
                },
            ):
                result = self._run(
                    video_analyze_tool(
                        str(video),
                        "Describe this",
                        include_transcript=True,
                    )
                )

        data = json.loads(result)
        assert data["success"] is True
        assert data["transcription"]["success"] is False
        assert data["transcription"]["error"] == "STT is disabled"


# ---------------------------------------------------------------------------
# Toolset registration
# ---------------------------------------------------------------------------


class TestVideoToolsetRegistration:
    """Verify the tool is registered correctly."""

    def test_registered_in_video_toolset(self):
        from tools.registry import registry
        entry = registry.get_entry("video_analyze")
        assert entry is not None
        assert entry.toolset == "video"
        assert entry.is_async is True
        assert entry.emoji == "🎬"

    def test_not_in_core_tools(self):
        """video_analyze should NOT be in _HERMES_CORE_TOOLS (default disabled)."""
        from toolsets import _HERMES_CORE_TOOLS
        assert "video_analyze" not in _HERMES_CORE_TOOLS

    def test_in_video_toolset_definition(self):
        """Toolset 'video' should contain video_analyze."""
        from toolsets import TOOLSETS
        assert "video" in TOOLSETS
        assert "video_analyze" in TOOLSETS["video"]["tools"]
