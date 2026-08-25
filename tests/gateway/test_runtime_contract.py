from gateway.runtime_contract import (
    RUNTIME_CONTRACT_MAJOR,
    RUNTIME_CONTRACT_MINOR,
    RUNTIME_CONTRACT_SCHEMA_DIGEST,
    RUNTIME_DRIVER_FRAME_TYPES,
    RUNTIME_PROTOCOL_VERSION,
    RUNTIME_RUN_REQUEST_SCHEMA_DIGEST,
    runtime_error_envelope,
    runtime_health_contract,
    runtime_manifest_contract,
)


def test_runtime_health_contract_is_explicit_and_versioned():
    contract = runtime_health_contract()
    assert contract["runtime_protocol_version"] == RUNTIME_PROTOCOL_VERSION
    assert contract["runtime_frame_types"] == list(RUNTIME_DRIVER_FRAME_TYPES)
    assert "checkpoint" not in contract["runtime_frame_types"]
    assert {
        "abort_attempt",
        "cancel_run",
        "delegated_tools",
        "suspend_attempt",
        "system_context.replace",
        "llm_egress",
    } <= set(contract["runtime_capabilities"])
    assert "interrupt" not in contract["runtime_capabilities"]


def test_runtime_manifest_contract_is_negotiable_and_contains_real_limits():
    manifest = runtime_manifest_contract(
        runtime_build="git:" + "a" * 40,
        max_request_bytes=98_000_000,
        max_tool_result_bytes=10_000_000,
    )

    assert manifest == {
        "runtime": "hermes",
        "runtime_build": "git:" + "a" * 40,
        "contract": {
            "major": RUNTIME_CONTRACT_MAJOR,
            "min_minor": RUNTIME_CONTRACT_MINOR,
            "max_minor": RUNTIME_CONTRACT_MINOR,
            "schema_digests": [RUNTIME_CONTRACT_SCHEMA_DIGEST],
        },
        "intents": ["bootstrap", "new_turn", "resume", "retry", "rebootstrap"],
        "features": [
            "invoked_skills.v1",
            "llm_egress.v1",
            "media_reference_resolution.v1",
            "session_db_rebootstrap.v1",
            "system_prompt_profiles.v1",
            "tool_result_replay.v1",
                "typed_run_control.v1",
                "video_evidence_projection.v1",
                "vision_llm_egress.v1",
        ],
        "limits": {
            "max_request_bytes": 98_000_000,
            "max_tool_result_bytes": 10_000_000,
        },
    }
    assert RUNTIME_RUN_REQUEST_SCHEMA_DIGEST.startswith("sha256:")
    assert len(RUNTIME_RUN_REQUEST_SCHEMA_DIGEST) == len("sha256:") + 64
    assert RUNTIME_CONTRACT_SCHEMA_DIGEST.startswith("sha256:")
    assert len(RUNTIME_CONTRACT_SCHEMA_DIGEST) == len("sha256:") + 64
    assert RUNTIME_CONTRACT_SCHEMA_DIGEST != RUNTIME_RUN_REQUEST_SCHEMA_DIGEST


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
