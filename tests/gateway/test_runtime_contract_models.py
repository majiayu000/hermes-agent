from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from gateway.api_server_runtime import RuntimeBridgeSession
from gateway.runtime_contract_models import (
    RuntimeManifest,
    RuntimeRunRequest,
    RuntimeToolRequest,
    RuntimeToolResult,
    decode_runtime_event,
    decode_runtime_run_request,
    decode_runtime_tool_result,
    encode_runtime_event,
)
from scripts.verify_runtime_contract_projection import LOCAL_ROOT


def _valid_run_request() -> dict[str, object]:
    return {
        "run_id": "run_fixture",
        "model": "test/model",
        "intent": "bootstrap",
        "messages": [{"id": "message_fixture", "role": "user", "content": "hello"}],
        "tools": [],
        "system_context": {
            "version": "fixture/v1",
            "mode": "replace",
            "stable": "trusted rules",
            "digest": "sha256:" + "a" * 64,
        },
        "context": {"session_id": "session_fixture"},
    }


@pytest.mark.parametrize(
    ("model", "schema_name"),
    [
        (RuntimeManifest, "manifest.schema.json"),
        (RuntimeRunRequest, "run-request.schema.json"),
        (RuntimeToolRequest, "tool-request.schema.json"),
        (RuntimeToolResult, "tool-result.schema.json"),
    ],
)
def test_typed_decoder_top_level_fields_match_canonical_schema(model, schema_name):
    schema = json.loads((LOCAL_ROOT / "v1" / schema_name).read_text(encoding="utf-8"))
    assert set(model.model_fields) == set(schema["properties"])
    assert {
        name for name, field in model.model_fields.items() if field.is_required()
    } == set(schema["required"])


def test_run_request_decoder_rejects_missing_required_and_unknown_fields():
    valid = _valid_run_request()
    assert decode_runtime_run_request(valid).run_id == "run_fixture"

    missing = dict(valid)
    missing.pop("tools")
    with pytest.raises(ValidationError):
        decode_runtime_run_request(missing)

    unknown = {**valid, "principal": {"account_id": "browser-supplied"}}
    with pytest.raises(ValidationError):
        decode_runtime_run_request(unknown)


def test_run_request_decoder_accepts_bounded_structured_invoked_skills():
    valid = {**_valid_run_request(), "invoked_skills": ["briefing", "production"]}
    assert decode_runtime_run_request(valid).invoked_skills == [
        "briefing",
        "production",
    ]

    with pytest.raises(ValidationError):
        decode_runtime_run_request(
            {**_valid_run_request(), "invoked_skills": ["Bad_Name"]}
        )
    with pytest.raises(ValidationError):
        decode_runtime_run_request(
            {**_valid_run_request(), "invoked_skills": ["skill"] * 9}
        )


def test_run_request_decoder_accepts_profile_and_keeps_legacy_replace():
    profile = _valid_run_request()
    profile["system_context"] = {
        "version": "ultra-agent-v1",
        "mode": "profile",
    }
    assert decode_runtime_run_request(profile).system_context.mode == "profile"

    with pytest.raises(ValidationError):
        decode_runtime_run_request(
            {
                **profile,
                "system_context": {
                    "version": "ultra-agent-v1",
                    "mode": "profile",
                    "stable": "forbidden",
                },
            }
        )


def test_event_decoder_rejects_unnegotiated_type_and_bad_payload():
    completed = {
        "run_id": "run_fixture",
        "type": "completed",
        "payload": {"finish_reason": "stop", "text": "done"},
    }
    assert decode_runtime_event(completed).type == "completed"

    with pytest.raises(ValidationError):
        decode_runtime_event({"type": "reasoning_delta", "payload": {"delta": "x"}})
    with pytest.raises(ValidationError):
        decode_runtime_event({
            "type": "completed",
            "payload": {"finish_reason": "stop"},
        })


def test_failed_tool_result_requires_typed_error():
    with pytest.raises(ValidationError):
        decode_runtime_tool_result({"call_id": "call_fixture", "ok": False})
    result = decode_runtime_tool_result({
        "call_id": "call_fixture",
        "ok": False,
        "error": {
            "code": "provider_timeout",
            "message": "The provider timed out.",
            "retryable": True,
        },
    })
    assert result.error is not None and result.error.retryable is True


def test_event_contract_failure_does_not_expose_payload():
    loop = asyncio.new_event_loop()
    try:
        session = RuntimeBridgeSession(
            "run_fixture",
            loop,
            asyncio.Queue(),
            [],
            1000,
            "session_fixture",
        )
        with pytest.raises(RuntimeError) as failure:
            session.emit("tool_request", {"credential": "must-not-leak"})
        assert "must-not-leak" not in str(failure.value)
        assert "tool_request" in str(failure.value)
    finally:
        loop.close()


def test_runtime_event_encoder_reuses_strict_decoder():
    encoded = encode_runtime_event({
        "run_id": "run_fixture",
        "type": "completed",
        "payload": {"finish_reason": "stop", "text": "done"},
    })
    assert json.loads(encoded) == {
        "run_id": "run_fixture",
        "type": "completed",
        "payload": {"finish_reason": "stop", "text": "done"},
    }
    with pytest.raises(ValidationError):
        encode_runtime_event({"type": "future_frame", "payload": {}})
