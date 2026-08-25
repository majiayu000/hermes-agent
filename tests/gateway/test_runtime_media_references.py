from __future__ import annotations

import base64
import json
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

from gateway.api_server_runtime import _runtime_attachment_parts
from gateway.runtime_media_references import resolve_media_arguments


def _session(result: dict, media_dir: str) -> SimpleNamespace:
    session = SimpleNamespace(
        allowed_image_references={},
        allowed_video_references={},
        allowed_image_paths=set(),
        allowed_video_paths=set(),
        pending_controls={},
        lock=threading.RLock(),
        deadline_seconds=1.0,
        unbounded_tool_wait_seconds=1.0,
        interrupted=threading.Event(),
    )

    def emit(_event_type: str, payload: dict) -> None:
        pending = session.pending_controls[payload["request_id"]]
        pending.result = result
        pending.ready.set()

    session.emit = emit
    session.materialize_media_reference = lambda attachment: _runtime_attachment_parts(
        [attachment], image_dir=media_dir
    )
    return session


def test_asset_id_is_resolved_to_private_image_path() -> None:
    with tempfile.TemporaryDirectory() as media_dir:
        session = _session({
            "ok": True,
            "result": {
                "role": "runtime_reference",
                "reference_id": "asset_brand",
                "asset_id": "asset_brand",
                "filename": "brand.png",
                "media_type": "image",
                "mime_type": "image/png",
                "data": base64.b64encode(b"png-bytes").decode(),
            },
        }, media_dir)
        args, error = resolve_media_arguments(
            session, {"image_url": "asset_brand"}, ("image_url",), "image", "call_1"
        )
        assert error is None
        path = Path(args["image_url"])
        assert path.is_file()
        assert path.read_bytes() == b"png-bytes"
        assert session.allowed_image_references["asset_brand"] == str(path)


def test_https_reference_does_not_request_platform_resolution() -> None:
    with tempfile.TemporaryDirectory() as media_dir:
        session = _session({}, media_dir)
        args, error = resolve_media_arguments(
            session,
            {"image_url": "https://static.example/image.png"},
            ("image_url",),
            "image",
            "call_2",
        )
        assert error is None
        assert args["image_url"] == "https://static.example/image.png"
        assert session.pending_controls == {}


def test_plain_http_reference_is_rejected_before_provider_submission() -> None:
    with tempfile.TemporaryDirectory() as media_dir:
        session = _session({}, media_dir)
        _, error = resolve_media_arguments(
            session,
            {"image_url": "http://static.example/image.png"},
            ("image_url",),
            "image",
            "call_http",
        )
        payload = json.loads(error)
        assert payload["error"]["code"] == "insecure_media_reference"
        assert payload["error"]["provider_submission_started"] is False


def test_unbound_output_image_returns_structured_pre_submission_error() -> None:
    with tempfile.TemporaryDirectory() as media_dir:
        session = _session({
            "ok": False,
            "error": {
                "code": "output_reference_not_bound",
                "message": "output is not bound",
                "retryable": False,
            },
        }, media_dir)
        _, error = resolve_media_arguments(
            session, {"image_url": "output_foreign"}, ("image_url",), "image", "call_3"
        )
        assert error is not None
        assert '"code":"output_reference_not_bound"' in error
        assert '"provider_submission_started":false' in error
