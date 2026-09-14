"""HTTP, error-code, and message classification for model requests."""
from __future__ import annotations

from typing import Optional
from agent.error_types import ClassifiedError, FailoverReason
from agent.error_patterns import (
    _BILLING_PATTERNS,
    _RATE_LIMIT_PATTERNS,
    _OVERLOADED_PATTERNS,
    _USAGE_LIMIT_PATTERNS,
    _USAGE_LIMIT_TRANSIENT_SIGNALS,
    _PAYLOAD_TOO_LARGE_PATTERNS,
    _IMAGE_TOO_LARGE_PATTERNS,
    _MULTIMODAL_TOOL_CONTENT_PATTERNS,
    _CONTEXT_OVERFLOW_PATTERNS,
    _MODEL_NOT_FOUND_PATTERNS,
    _REQUEST_VALIDATION_PATTERNS,
    _PROVIDER_POLICY_BLOCKED_PATTERNS,
    _AUTH_PATTERNS,
    _TIMEOUT_MESSAGE_PATTERNS,
)

def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on HTTP status code with message-aware refinement."""
    if status_code == 401:
        # Not retryable on its own — credential pool rotation and
        # provider-specific refresh (Codex, Anthropic, Nous) run before
        # the retryability check in run_agent.py.  If those succeed, the
        # loop `continue`s.  If they fail, retryable=False ensures we
        # hit the client-error abort path (which tries fallback first).
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # OpenRouter 403 "key limit exceeded" is actually billing. Other
        # providers also use 403 for account-plan or credit exhaustion.
        if (
            "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        # Nous API currently surfaces HA/NAS credit depletion as a paid model
        # becoming unavailable on the Free Tier, returned as 404 rather than
        # 402. Treat that as entitlement/billing exhaustion, not a missing
        # model, so the retry loop can show credit/top-up guidance.
        if any(p in error_msg for p in _BILLING_PATTERNS):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        # OpenRouter policy-block 404 — distinct from "model not found".
        # The model exists; the user's account privacy setting excludes the
        # only endpoint serving it. Falling back to another provider won't
        # help (same account setting applies).  The error body already
        # contains the fix URL, so just surface it.
        if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
            return result_fn(
                FailoverReason.provider_policy_blocked,
                retryable=False,
                should_fallback=False,
            )
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # Generic 404 with no "model not found" signal — could be a wrong
        # endpoint path (common with local llama.cpp / Ollama / vLLM when
        # the URL is slightly misconfigured), a proxy routing glitch, or
        # a transient backend issue.  Classifying these as model_not_found
        # silently falls back to a different provider and tells the model
        # the model is missing, which is wrong and wastes a turn.  Treat
        # as unknown so the retry loop surfaces the real error instead.
        return result_fn(
            FailoverReason.unknown,
            retryable=True,
        )

    if status_code == 413:
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        # Already checked long_context_tier above. Some providers (notably
        # Z.AI / Zhipu) reuse HTTP 429 for server-wide overload — same status
        # code as a true per-credential rate limit, but the credential is
        # valid and the correct recovery is "back off and retry the same key",
        # NOT "rotate the credential" (which exhausts the pool while the
        # endpoint is still busy, and does nothing for a single-key user).
        # Disambiguate on the error body so an overload 429 takes the
        # transient-overload path instead of burning the pool. (#14038)
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(
                FailoverReason.overloaded,
                retryable=True,
            )
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider, model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Some OpenAI-compatible gateways return request-validation errors
        # with a 5xx status (codex.nekos.me returns 502 for unknown/
        # unsupported parameters). These are deterministic — every retry
        # gets the identical rejection — so the generic "5xx → retryable
        # server_error" rule turns one bad request into a retry flood.
        # Detect the unambiguous request-validation signals (in either the
        # message text or the structured error code) and fail fast.
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in {"invalid_request_error", "unknown_parameter",
                                      "unsupported_parameter"}
        ):
            return result_fn(
                FailoverReason.format_error,
                retryable=False,
                should_fallback=True,
            )
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        return result_fn(FailoverReason.overloaded, retryable=True)

    # Other 4xx — non-retryable
    if 400 <= status_code < 500:
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Other 5xx — retryable
    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: billing exhaustion vs transient usage limit.

    The key insight from OpenClaw: some 402s are transient rate limits
    disguised as payment errors.  "Usage limit, try again in 5 minutes"
    is NOT a billing problem — it's a periodic quota that resets.
    """
    # Check for transient usage-limit signals first
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)

    if has_usage_limit and has_transient_signal:
        # Transient quota — treat as rate limit, not billing
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Confirmed billing exhaustion
    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic."""

    # Multimodal tool content rejected from 400.  Must be checked BEFORE
    # image_too_large because the recovery is different (strip image parts
    # from tool messages, mark the model as no-list-tool-content for the
    # rest of the session) and BEFORE context_overflow because some of the
    # patterns ("text is not set") are ambiguous in isolation but become
    # specific when combined with a 400 on a request known to contain
    # multimodal tool content.
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large from 400 (Anthropic's 5 MB per-image check fires this way).
    # Must be checked BEFORE context_overflow because messages can trip both
    # patterns ("exceeds" + "image") and image-shrink is a cheaper recovery.
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Invalid encrypted reasoning replay blob (OpenAI Responses API).  Must be
    # checked BEFORE context_overflow because some surfaces emit messages that
    # contain context-like phrasing ("encrypted content … could not be
    # verified") which could otherwise trip the context_overflow heuristics.
    # ``error_msg`` is lowercased upstream — match accordingly.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or (
            "encrypted content for item" in error_msg
            and "could not be verified" in error_msg
        )
    ):
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    # Request-validation errors (unsupported / unknown parameter) MUST be
    # checked BEFORE context_overflow.  A GPT-5 model rejecting max_tokens
    # returns:
    #   "Unsupported parameter: 'max_tokens' is not supported with this model.
    #    Use 'max_completion_tokens' instead."
    # That string contains the literal substring "max_tokens", which is one of
    # the _CONTEXT_OVERFLOW_PATTERNS — so without this guard the 400 is
    # misclassified as context_overflow, routed into the compression loop,
    # re-sent with the same bad parameter, and ends in "Cannot compress
    # further".  These errors are deterministic (every retry gets the identical
    # rejection), so classify as a non-retryable format_error and fall back.
    #
    # NOTE: we deliberately do NOT key off the generic ``invalid_request_error``
    # code here — OpenAI stamps that same code on genuine context-overflow 400s,
    # so matching it would mis-route real overflows away from compression. The
    # unambiguous signals are the explicit "unsupported/unknown parameter"
    # message text and the specific parameter-level error codes.
    if (
        any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS
            if p != "invalid_request_error")
        or error_code_lower in {"unknown_parameter", "unsupported_parameter"}
    ):
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Context overflow from 400
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Some providers return model-not-found as 400 instead of 404 (e.g. OpenRouter).
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Some providers return rate limit / billing errors as 400 instead of 429/402.
    # Check these patterns before falling through to format_error.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Generic 400 + large session → probable context overflow
    # Anthropic sometimes returns a bare "Error" message when context is too large
    err_body_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            err_body_msg = str(err_obj.get("message") or "").strip().lower()
        # Responses API (and some providers) use flat body: {"message": "..."}
        if not err_body_msg:
            err_body_msg = str(body.get("message") or "").strip().lower()
    is_generic = len(err_body_msg) < 30 or err_body_msg in {"error", ""}
    # Absolute token/message-count thresholds are only a proxy for smaller
    # context windows.  Large-context sessions can have many messages while
    # still being far below their actual token budget.
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256000 and (approx_tokens > 80000 or num_messages > 80)
    )

    if is_generic and is_large:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Non-retryable format error
    return result_fn(
        FailoverReason.format_error,
        retryable=False,
        should_fallback=True,
    )


# ── Error code classification ───────────────────────────────────────────

def _classify_by_error_code(
    error_code: str, error_msg: str, result_fn,
) -> Optional[ClassifiedError]:
    """Classify by structured error codes from the response body."""
    code_lower = error_code.lower()

    if code_lower in {"resource_exhausted", "throttled", "rate_limit_exceeded"}:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
        )

    if code_lower in {
        "insufficient_quota",
        "billing_not_active",
        "payment_required",
        "insufficient_credits",
        "no_usable_credits",
        "balance_depleted",
        "model_not_supported_on_free_tier",
    }:
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if code_lower in {"model_not_found", "model_not_available", "invalid_model"}:
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    if code_lower in {"context_length_exceeded", "max_tokens_exceeded"}:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if code_lower == "invalid_encrypted_content":
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    return None


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(
    error_msg: str,
    error_type: str,
    *,
    approx_tokens: int,
    context_length: int,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on error message patterns when no status code is available."""

    # Payload-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    # Multimodal tool content patterns (from message text when no status_code)
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Usage-limit patterns need the same disambiguation as 402: some providers
    # surface "usage limit" errors without an HTTP status code.  A transient
    # signal ("try again", "resets at", …) means it's a periodic quota, not
    # billing exhaustion.
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    if has_usage_limit:
        has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
        if has_transient_signal:
            return result_fn(
                FailoverReason.rate_limit,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Overloaded / server-busy patterns — must come BEFORE the rate_limit and
    # billing checks so that a message-only "overloaded" (no 503/529 status,
    # e.g. some Anthropic-compatible proxies) classifies as a transient
    # overload (backoff + retry) instead of falling through to `unknown` or
    # incorrectly triggering credential rotation.
    if any(p in error_msg for p in _OVERLOADED_PATTERNS):
        return result_fn(
            FailoverReason.overloaded,
            retryable=True,
        )

    # Billing patterns
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Rate limit patterns
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Context overflow patterns
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Auth patterns
    # Auth errors should NOT be retried directly — the credential is invalid and
    # retrying with the same key will always fail.  Set retryable=False so the
    # caller triggers credential rotation (should_rotate_credential=True) or
    # provider fallback rather than an immediate retry loop.
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Provider policy-block (aggregator-side guardrail) — check before
    # model_not_found so we don't mis-label as a missing model.
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )

    # Model not found patterns
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Timeout message patterns — generic exception types (e.g. RuntimeError)
    # raised by local shims or custom providers that internally wrap a
    # subprocess/HTTP timeout.  Classified as transport timeout so the retry
    # loop rebuilds the client instead of treating the turn as an empty
    # model response.
    if any(p in error_msg for p in _TIMEOUT_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    return None
