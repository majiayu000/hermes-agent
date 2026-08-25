from __future__ import annotations

import base64
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.runtime_media_references import (
    invoke_video_analyze,
    prepare_video_evidence,
    validate_video_evidence,
)
from tools.video_frame_analysis import build_video_evidence_analysis_message


def _blob(data: bytes) -> dict[str, object]:
    return {
        "mime_type": "image/jpeg",
        "sha256": hashlib.sha256(data).hexdigest(),
        "data": base64.b64encode(data).decode(),
    }


def _evidence(reference_id: str = "asset_video") -> dict[str, object]:
    return {
        "reference_id": reference_id,
        "source_digest": hashlib.sha256(b"source").hexdigest(),
        "source_mime_type": "video/mp4",
        "source_size_bytes": 512 << 20,
        "duration_seconds": 30.0,
        "width": 1920,
        "height": 1080,
        "sampling": "uniform_midpoint",
        "analyzer_version": "video-evidence-v1",
        "cache_status": "miss",
        "frames": [
            {**_blob(f"frame-{index}".encode()), "timestamp_seconds": index * 10.0 + 5.0, "ratio": (index + 0.5) / 3}
            for index in range(3)
        ],
    }


def test_video_evidence_accepts_large_source_metadata_but_only_bounded_frames():
    evidence = _evidence()
    assert validate_video_evidence(evidence, "asset_video") is evidence
    message = build_video_evidence_analysis_message("Summarize", evidence)
    image_parts = [part for part in message["content"] if part["type"] == "image_url"]
    assert len(image_parts) == 3
    assert all(part["image_url"]["url"].startswith("data:image/jpeg;base64,") for part in image_parts)
    assert base64.b64encode(b"source").decode() not in str(message)


def test_video_evidence_rejects_digest_mismatch():
    evidence = _evidence()
    evidence["frames"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        validate_video_evidence(evidence, "asset_video")


def test_video_evidence_control_uses_opaque_reference_and_returns_projection():
    evidence = _evidence("output_video")
    session = SimpleNamespace(
        lock=threading.RLock(),
        pending_controls={},
        deadline_seconds=1.0,
        unbounded_tool_wait_seconds=1.0,
        interrupted=threading.Event(),
    )

    def emit(event_type, payload):
        assert event_type == "runtime_control_request"
        assert payload == {
            "request_id": payload["request_id"],
            "kind": "video_evidence.prepare",
            "reference_id": "output_video",
            "media_type": "video",
            "include_transcript": True,
        }
        pending = session.pending_controls.pop(payload["request_id"])
        pending.result = {"ok": True, "result": evidence}
        pending.ready.set()

    session.emit = emit
    result, error = prepare_video_evidence(
        session, "output_video", True, "call_video",
    )
    assert error is None
    assert result is evidence
    assert session.pending_controls == {}


def test_video_analyze_returns_completed_result_to_agent(monkeypatch):
    evidence = _evidence("output_video")
    session = SimpleNamespace(
        video_analyze_lock=threading.Lock(),
        lock=threading.RLock(),
        native_non_retryable_failures={},
        _tool_signature_key=lambda name, args: f"{name}:{args['video_url']}",
        _halt_tool_loop=lambda *_args: None,
    )

    monkeypatch.setattr(
        "gateway.runtime_media_references.prepare_video_evidence",
        lambda *_args, **_kwargs: (evidence, None),
    )

    async def analyze(_evidence, _prompt, _model, _include_transcript):
        return json.dumps({"success": True, "analysis": "bounded evidence"})

    monkeypatch.setattr(
        "tools.runtime_video_evidence.analyze_runtime_video_evidence",
        analyze,
    )

    result = invoke_video_analyze(
        session,
        {
            "video_url": "output_video",
            "question": "Inspect it",
            "_runtime_parent_call_id": "call_video",
        },
        None,
    )

    assert isinstance(result, str)
    assert json.loads(result) == {
        "success": True,
        "analysis": "bounded evidence",
    }
