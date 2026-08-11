from agent.tool_result_routing import (
    TOOL_RESULT_IMAGE_ATTACH_BY_REF,
    TOOL_RESULT_IMAGE_EMBED_DATA_URL,
    TOOL_RESULT_IMAGE_REJECT,
    _configured_tool_result_image_mode,
    _has_inline_image_tool_result,
    _resolve_tool_result_image_mode,
    _should_retry_tool_result_projection,
    _strip_inline_image_tool_results,
)


def _image_tool_messages():
    return [{
        "role": "tool",
        "tool_call_id": "call-image",
        "content": [
            {"type": "text", "text": "asset_id=asset-1; url=https://media/image.png"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,cGl4ZWxz"},
            },
        ],
    }]


def test_unknown_provider_fails_closed():
    assert (
        _resolve_tool_result_image_mode("unknown-provider", "model", {})
        == TOOL_RESULT_IMAGE_REJECT
    )


def test_runtime_override_has_highest_precedence():
    cfg = {"model": {"tool_result_image_mode": "embed_data_url"}}
    assert _resolve_tool_result_image_mode(
        "anthropic",
        "claude",
        cfg,
        runtime_override="attach_by_ref",
    ) == TOOL_RESULT_IMAGE_ATTACH_BY_REF


def test_exact_provider_model_override_is_resolved():
    cfg = {
        "providers": {
            "custom": {
                "models": {
                    "verified-vlm": {
                        "tool_result_image_mode": "embed_data_url",
                    },
                },
            },
        },
    }
    assert (
        _configured_tool_result_image_mode(cfg, "custom", "verified-vlm")
        == TOOL_RESULT_IMAGE_EMBED_DATA_URL
    )


def test_generic_400_uses_actual_payload_to_select_safe_variant():
    messages = _image_tool_messages()
    assert _has_inline_image_tool_result(messages) is True
    assert _should_retry_tool_result_projection(
        400,
        "format_error",
        messages,
        already_attempted=False,
    ) is True
    assert _should_retry_tool_result_projection(
        400,
        "format_error",
        messages,
        already_attempted=True,
    ) is False


def test_non_payload_400_does_not_select_image_projection():
    assert _should_retry_tool_result_projection(
        400,
        "format_error",
        [{"role": "tool", "content": "text"}],
        already_attempted=False,
    ) is False


def test_payload_variant_precedes_context_heuristic():
    assert _should_retry_tool_result_projection(
        400,
        "context_overflow",
        _image_tool_messages(),
        already_attempted=False,
    ) is True


def test_confirmed_billing_400_does_not_retry_payload_variant():
    assert _should_retry_tool_result_projection(
        400,
        "billing",
        _image_tool_messages(),
        already_attempted=False,
    ) is False


def test_projection_removes_pixels_but_preserves_reference_text():
    messages = _image_tool_messages()
    assert _strip_inline_image_tool_results(messages) is True
    assert messages[0]["content"] == (
        "asset_id=asset-1; url=https://media/image.png"
    )
    assert "data:image" not in messages[0]["content"]
