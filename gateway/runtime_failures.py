"""Project Agent failure reasons into the Runtime error contract."""

_FAILURE_REASON_CODES = {
    "egress_request_replayed": "egress_request_replayed",
    "egress_result_unknown": "egress_result_unknown",
    "billing": "insufficient_credits",
    "content_policy_blocked": "content_policy_blocked",
    "format_error": "model_incompatible",
    "multimodal_tool_content_unsupported": "model_incompatible",
    "timeout": "provider_timeout",
    "overloaded": "provider_unavailable",
    "rate_limit": "provider_unavailable",
    "run_budget_exhausted": "run_budget_exhausted",
    "server_error": "provider_unavailable",
}


def _runtime_failure_code(result: object) -> str:
    if not isinstance(result, dict):
        return "runtime_unavailable"
    if result.get("turn_exit_reason") in {
        "empty_response_exhausted",
        "all_retries_exhausted_no_response",
    }:
        return "provider_empty_stream"
    reason = str(result.get("failure_reason") or "").strip().lower()
    if reason in _FAILURE_REASON_CODES:
        return _FAILURE_REASON_CODES[reason]
    error = str(result.get("error") or "").strip().lower()
    if "insufficient balance" in error or "insufficient credit" in error or "http 402" in error:
        return "insufficient_credits"
    if error.startswith("content_policy_blocked:"):
        return "content_policy_blocked"
    return "runtime_unavailable"
