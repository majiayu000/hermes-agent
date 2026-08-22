from tools.video_analysis_errors import classify_video_analysis_error


def test_transient_video_provider_failure_is_retryable() -> None:
    code, retryable, _ = classify_video_analysis_error(
        RuntimeError("provider connection timed out with HTTP 503")
    )
    assert code == "video_analysis_failed"
    assert retryable is True


def test_video_billing_and_input_failures_are_terminal() -> None:
    assert classify_video_analysis_error(RuntimeError("HTTP 402 credits"))[:2] == (
        "video_analysis_insufficient_credits",
        False,
    )
    assert classify_video_analysis_error(RuntimeError("payload exceeds size limit"))[:2] == (
        "video_analysis_input_too_large",
        False,
    )
