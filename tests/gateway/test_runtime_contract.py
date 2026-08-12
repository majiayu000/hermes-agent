from gateway.runtime_contract import (
    RUNTIME_DRIVER_FRAME_TYPES,
    runtime_error_envelope,
    runtime_health_contract,
)


def test_runtime_health_contract_is_explicit_and_versioned():
    contract = runtime_health_contract()
    assert contract["runtime_protocol_version"] == "2"
    assert contract["runtime_frame_types"] == list(RUNTIME_DRIVER_FRAME_TYPES)
    assert "checkpoint" not in contract["runtime_frame_types"]
    assert {
        "delegated_tools",
        "interrupt",
        "model_contract_control",
        "session_db_rebootstrap/v1",
        "system_context.replace",
        "llm_egress",
    } <= set(contract["runtime_capabilities"])


def test_runtime_error_envelope_never_includes_raw_exception_text():
    error = runtime_error_envelope(
        "../../private/config bearer-secret",
        support_id="run_1",
    )
    assert error == {
        "code": "unexpected_error",
        "message": "The request could not be completed.",
        "retryable": False,
        "reason": "unexpected_error",
        "source": "runtime",
        "support_id": "run_1",
    }


def test_runtime_unavailable_is_safe_and_retryable():
    error = runtime_error_envelope("runtime_unavailable", support_id="run_2")
    assert error["message"] == "The creation service is temporarily unavailable."
    assert error["retryable"] is True


def test_model_incompatible_is_safe_and_not_retryable():
    error = runtime_error_envelope("model_incompatible", support_id="run-model")
    assert error["message"] == (
        "The selected model could not accept the generated result."
    )
    assert error["retryable"] is False
