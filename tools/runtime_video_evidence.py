"""Analyze bounded video evidence supplied by the Ultra Studio data plane."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
from tools.video_frame_analysis import build_video_evidence_analysis_message


logger = logging.getLogger(__name__)


async def analyze_runtime_video_evidence(
    evidence: dict[str, Any],
    user_prompt: str,
    model: str | None,
    include_transcript: bool,
) -> str:
    """Call vision and optional speech models without materializing source video."""
    try:
        messages = [build_video_evidence_analysis_message(user_prompt, evidence)]
        timeout, temperature = _vision_settings()
        call_kwargs: dict[str, Any] = {
            "task": "vision",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 4000,
            "timeout": timeout,
        }
        if model:
            call_kwargs["model"] = model
        response = await async_call_llm(**call_kwargs)
        analysis = extract_content_or_reasoning(response)
        if not analysis:
            raise RuntimeError("Video analysis model returned an empty response")
        frames = evidence.get("frames") or []
        result: dict[str, Any] = {
            "success": True,
            "analysis": analysis,
            "sampled_frames": [
                {
                    "timestamp_seconds": round(float(frame["timestamp_seconds"]), 3),
                    "ratio": float(frame["ratio"]),
                }
                for frame in frames
            ],
            "evidence": {
                "source_digest": evidence.get("source_digest"),
                "analyzer_version": evidence.get("analyzer_version"),
                "cache_status": evidence.get("cache_status"),
                "sampling": evidence.get("sampling"),
            },
        }
        if include_transcript:
            result["transcription"] = await _transcribe_audio_proxy(
                evidence.get("audio_proxy")
            )
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("Runtime video evidence analysis failed: %s", exc, exc_info=True)
        from tools.video_analysis_errors import classify_video_analysis_error

        error_code, retryable, analysis = classify_video_analysis_error(exc)
        return json.dumps({
            "success": False,
            "error": f"Error analyzing video evidence: {exc}",
            "error_code": error_code,
            "retryable": retryable,
            "analysis": analysis,
        }, indent=2, ensure_ascii=False)


def _vision_settings() -> tuple[float, float]:
    timeout = 180.0
    temperature = 0.1
    try:
        from hermes_cli.config import cfg_get, load_config

        config = cfg_get(load_config(), "auxiliary", "vision", default={})
        if config.get("timeout") is not None:
            timeout = max(float(config["timeout"]), 180.0)
        if config.get("temperature") is not None:
            temperature = float(config["temperature"])
    except Exception:
        logger.debug("Using default Runtime video evidence vision settings", exc_info=True)
    return timeout, temperature


async def _transcribe_audio_proxy(audio_proxy: Any) -> dict[str, Any]:
    if not isinstance(audio_proxy, dict) or not audio_proxy.get("data"):
        return {
            "success": False,
            "transcript": "",
            "error": "The bounded video evidence contains no audio proxy.",
        }
    audio_bytes = base64.b64decode(str(audio_proxy["data"]), validate=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="hermes-video-evidence-", suffix=".ogg", delete=False
        ) as temporary:
            temporary.write(audio_bytes)
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o600)
        from tools.transcription_tools import transcribe_audio

        transcription = await asyncio.to_thread(transcribe_audio, str(temporary_path))
        if not isinstance(transcription, dict):
            raise RuntimeError("Audio transcription returned an invalid result")
        return transcription
    except Exception as exc:
        logger.error("Runtime audio proxy transcription failed", exc_info=True)
        return {"success": False, "transcript": "", "error": str(exc)}
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
