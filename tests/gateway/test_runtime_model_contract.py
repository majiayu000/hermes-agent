"""A gateway-resolved Writer prompt must not be demanded from the Agent again."""
import pytest

from gateway.runtime_model_contract import _model_request_contract_error


def contract():
    return {"test/video": {
        "prompt": {"type": "string", "required": True, "options": [], "description": ""},
        "duration": {"type": "integer", "required": True, "options": [5, 10], "description": ""},
    }}


def test_writer_reference_delegates_only_prompt_resolution():
    request = {"model": "test/video", "writer_tool_call_id": "call_saved", "duration": 5}
    assert _model_request_contract_error("media.generate_video", {"requests": [request]}, contract()) is None
    assert "prompt" not in request
    request["duration"] = 7
    error = _model_request_contract_error("media.generate_video", {"requests": [request]}, contract())
    assert "duration" in error["error"]["message"]


@pytest.mark.parametrize("tool", ["media.generate_image", "media.generate_audio", "media.estimate_cost"])
def test_writer_reference_is_not_a_generic_provider_parameter(tool):
    error = _model_request_contract_error(tool, {"requests": [{"model": "test/video", "writer_tool_call_id": "call_saved", "duration": 5}]}, contract())
    assert "writer_tool_call_id" in error["error"]["message"]


def test_literal_prompt_still_requires_declared_type():
    for request in [{"duration": 5}, {"duration": 5, "prompt": 42}]:
        error = _model_request_contract_error("media.generate_video", {"requests": [{"model": "test/video", **request}]}, contract())
        assert "prompt" in error["error"]["message"]
