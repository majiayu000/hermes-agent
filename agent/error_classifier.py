"""Classify model API failures without changing provider or execution facts."""
from __future__ import annotations

import logging
from typing import Optional

from agent.error_types import ClassifiedError, FailoverReason
from agent.error_patterns import (
    _CONTENT_POLICY_BLOCKED_PATTERNS,
    _TRANSPORT_ERROR_TYPES,
    _SERVER_DISCONNECT_PATTERNS,
    _SSL_TRANSIENT_PATTERNS,
)
from agent.error_status import (
    _classify_by_status,
    _classify_402,
    _classify_400,
    _classify_by_error_code,
    _classify_by_message,
)

logger = logging.getLogger(__name__)

def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
    managed_egress: bool = False,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation.

    Priority-ordered pipeline:
      1. Special-case provider-specific patterns (thinking sigs, tier gates)
      2. HTTP status code + message-aware refinement
      3. Error code classification (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL/TLS transient alert patterns → retry as timeout
      6. Server disconnect + large session → context overflow
      7. Transport error heuristics
      8. Fallback: unknown (retryable with backoff)

    Args:
        error: The exception from the API call.
        provider: Current provider name (e.g. "openrouter", "anthropic").
        model: Current model slug.
        approx_tokens: Approximate token count of the current context.
        context_length: Maximum context length for the current model.

    Returns:
        ClassifiedError with reason and recovery action hints.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429
    # so downstream rate-limit handling (classifier reason, pool rotation,
    # fallback gating) fires correctly instead of misclassifying as generic.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)

    # Build a comprehensive error message string for pattern matching.
    # str(error) alone may not include the body message (e.g. OpenAI SDK's
    # APIStatusError.__str__ returns the first arg, not the body).  Append
    # the body message so patterns like "try again" in 402 disambiguation
    # are detected even when only present in the structured body.
    #
    # Also extract metadata.raw — OpenRouter wraps upstream provider errors
    # inside {"error": {"message": "Provider returned error", "metadata":
    # {"raw": "<actual error JSON>"}}} and the real error message (e.g.
    # "context length exceeded") is only in the inner JSON.
    _raw_msg = str(error).lower()
    _body_msg = ""
    _metadata_msg = ""
    if isinstance(body, dict):
        _err_obj = body.get("error", {})
        if isinstance(_err_obj, dict):
            _body_msg = str(_err_obj.get("message") or "").lower()
            # Parse metadata.raw for wrapped provider errors
            _metadata = _err_obj.get("metadata", {})
            if isinstance(_metadata, dict):
                _raw_json = _metadata.get("raw") or ""
                if isinstance(_raw_json, str) and _raw_json.strip():
                    try:
                        import json
                        _inner = json.loads(_raw_json)
                        if isinstance(_inner, dict):
                            _inner_err = _inner.get("error", {})
                            if isinstance(_inner_err, dict):
                                _metadata_msg = str(_inner_err.get("message") or "").lower()
                    except (json.JSONDecodeError, TypeError):
                        pass
        if not _body_msg:
            _body_msg = str(body.get("message") or "").lower()
    # Combine all message sources for pattern matching
    parts = [_raw_msg]
    if _body_msg and _body_msg not in _raw_msg:
        parts.append(_body_msg)
    if _metadata_msg and _metadata_msg not in _raw_msg and _metadata_msg not in _body_msg:
        parts.append(_metadata_msg)
    error_msg = " ".join(parts)
    provider_lower = (provider or "").strip().lower()
    model_lower = (model or "").strip().lower()

    def _result(reason: FailoverReason, **overrides) -> ClassifiedError:
        defaults = {
            "reason": reason,
            "status_code": status_code,
            "provider": provider,
            "model": model,
            "message": _extract_message(error, body),
        }
        defaults.update(overrides)
        return ClassifiedError(**defaults)

    if managed_egress:
        import httpx
        from openai import APIConnectionError

        if isinstance(error, (httpx.TransportError, APIConnectionError, TimeoutError, ConnectionError)):
            return _result(FailoverReason.egress_result_unknown, retryable=False)

    # A reserved paid operation cannot be retried or sent to a fallback.
    if error_code.lower() in {"egress_request_replayed", "egress_result_unknown"}:
        return _result(FailoverReason(error_code.lower()), retryable=False)

    if managed_egress and status_code is not None and 500 <= status_code <= 599:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        if headers.get("X-Ultra-Submission-State") != "not-started":
            # Keep the provider failure, but do not replay its reserved paid
            # request unless the gateway proves submission never began.
            return _result(FailoverReason.server_error, retryable=False)

    # ── 1. Provider-specific patterns (highest priority) ────────────

    # The Ultra Studio LLM egress proxy uses HTTP 429 for its durable,
    # per-run safety ledger. This is not an upstream provider throttle: the
    # same run cannot replenish its ledger, rotate credentials, or succeed on
    # an unchanged retry. Preserve the structured code before generic 429
    # classification turns it into a transient rate limit.
    if error_code.lower() == "run_budget_exhausted":
        return _result(
            FailoverReason.run_budget_exhausted,
            retryable=False,
            should_rotate_credential=False,
            should_fallback=False,
        )

    # Provider content-policy / safety-filter block. The provider has made a
    # deterministic refusal decision about THIS prompt — retrying unchanged
    # just reproduces the same refusal and burns paid attempts. Must run
    # before status-based classification so a 400 safety block isn't
    # downgraded to a generic ``format_error`` and a status-less block
    # (OpenAI Codex SDK can raise without one) isn't left in the retryable
    # ``unknown`` bucket. See issue #18028.
    if any(p in error_msg for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        return _result(
            FailoverReason.content_policy_blocked,
            retryable=False,
            should_fallback=True,
        )

    # Anthropic thinking block recovery (400).  Two distinct failure modes,
    # same recovery (strip all reasoning_details and retry without thinking
    # blocks — see the thinking_signature handler in conversation_loop.py):
    #   1. Signature mismatch: a thinking block is signed against the full
    #      turn content; any upstream mutation (context compression, session
    #      truncation, message merging) invalidates the signature.
    #      Pattern: "signature" + "thinking".
    #   2. Frozen-block mutation: Anthropic rejects any change to the
    #      thinking/redacted_thinking blocks in the *latest* assistant
    #      message — "`thinking` or `redacted_thinking` blocks in the latest
    #      assistant message cannot be modified. These blocks must remain as
    #      they were in the original response."  This carries no "signature"
    #      token, so the original pattern missed it and the turn hard-aborted
    #      as a non-retryable client error instead of self-healing.
    #      Pattern: "thinking" + ("cannot be modified" | "must remain as they were").
    # Don't gate on provider — OpenRouter proxies Anthropic errors, so the
    # provider may be "openrouter" even though the error is Anthropic-specific.
    # The combined patterns are unique enough.
    if (
        status_code == 400
        and "thinking" in error_msg
        and (
            "signature" in error_msg
            or "cannot be modified" in error_msg
            or "must remain as they were" in error_msg
        )
    ):
        return _result(
            FailoverReason.thinking_signature,
            retryable=True,
            should_compress=False,
        )

    # Anthropic long-context tier gate (429 "extra usage" + "long context")
    if (
        status_code == 429
        and "extra usage" in error_msg
        and "long context" in error_msg
    ):
        return _result(
            FailoverReason.long_context_tier,
            retryable=True,
            should_compress=True,
        )

    # Anthropic OAuth subscription rejects the 1M-context beta header.
    # Observed error body: "The long context beta is not yet available for
    # this subscription." Returned as HTTP 400 from native Anthropic when
    # the subscription doesn't include 1M context, even though the request
    # carries ``anthropic-beta: context-1m-2025-08-07``. The recovery path
    # in run_agent.py rebuilds the Anthropic client with the beta stripped
    # and retries once. Pattern is narrow enough that it won't collide with
    # the 429 tier-gate pattern above (different status, different phrase).
    if (
        status_code == 400
        and "long context beta" in error_msg
        and "not yet available" in error_msg
    ):
        return _result(
            FailoverReason.oauth_long_context_beta_forbidden,
            retryable=True,
            should_compress=False,
        )

    # llama.cpp's ``json-schema-to-grammar`` converter (used by its OAI
    # server to build GBNF tool-call parsers) rejects regex escape classes
    # like ``\d``/``\w``/``\s`` and most ``format`` values. MCP servers
    # routinely emit ``"pattern": "\\d{4}-\\d{2}-\\d{2}"`` for date/phone/
    # email params. llama.cpp surfaces this as HTTP 400 with one of a few
    # recognizable phrases; on match we strip ``pattern``/``format`` from
    # ``self.tools`` in the retry loop and retry once. Cloud providers are
    # unaffected — they accept these keywords and we never hit this branch.
    if (
        status_code == 400
        and (
            "error parsing grammar" in error_msg
            or "json-schema-to-grammar" in error_msg
            or (
                "unable to generate parser" in error_msg
                and "template" in error_msg
            )
        )
    ):
        return _result(
            FailoverReason.llama_cpp_grammar_pattern,
            retryable=True,
            should_compress=False,
        )

    # xAI Grok subscription entitlement errors.
    #
    # xAI returns "You have either run out of available resources or do not
    # have an active Grok subscription" through two distinct code paths:
    #
    #   • HTTP 403 — status_code is set; _classify_by_status (step 2) routes
    #     it to FailoverReason.auth correctly, and _is_entitlement_failure
    #     then prevents the credential-refresh loop.
    #
    #   • SSE ``type=error`` frame — surfaced as _StreamErrorEvent with
    #     status_code=None.  _classify_by_status is skipped entirely, and
    #     "grok subscription" / "out of available resources" appear in none
    #     of the message-pattern lists below.  Without this guard the error
    #     falls through to FailoverReason.unknown (retryable=True), burning
    #     max_retries before the agent stops — and _is_entitlement_failure
    #     is never called because it only runs under FailoverReason.auth.
    #
    # Both X Premium+ and SuperGrok subscribers hit this path when their
    # subscription tier does not cover the requested model or feature.
    if (
        "do not have an active grok subscription" in error_msg
        or ("out of available resources" in error_msg and "grok" in error_msg)
    ):
        return _result(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    # ── 2. HTTP status code classification ──────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code, error_msg, error_code, body,
            provider=provider_lower, model=model_lower,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(
        error_msg, error_type,
        approx_tokens=approx_tokens,
        context_length=context_length,
        result_fn=_result,
    )
    if classified is not None:
        return classified

    # ── 5. SSL/TLS transient errors → retry as timeout (not compression) ──
    # SSL alerts mid-stream are transport hiccups, not server-side context
    # overflow signals.  Classify before the disconnect check so a large
    # session doesn't incorrectly trigger context compression when the real
    # cause is a flaky TLS handshake.  Also matches when the error is
    # wrapped in a generic exception whose message string carries the SSL
    # alert text but the type isn't ssl.SSLError (happens with some SDKs
    # that re-raise without chaining).
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Must come BEFORE generic transport error catch — a disconnect on
    # a large session is more likely context overflow than a transient
    # transport hiccup.  Without this ordering, RemoteProtocolError
    # always maps to timeout regardless of session size.

    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and not status_code:
        # Reasoning-model override: a transport disconnect on a reasoning
        # model is much more likely the upstream proxy idle-killing a
        # long thinking stream than a true context overflow — even on
        # large sessions.  The default disconnect+large-session routing
        # below would otherwise send the user into the compression
        # branch (should_compress=True) and silently delete
        # conversation history on a phantom context-length error.
        # Reasoning models have multi-minute thinking phases that
        # routinely exceed the cloud gateway's idle window (NVIDIA
        # NIM ~120s — first-party repro at NVIDIA/NemoClaw#4846;
        # OpenAI worker / Anthropic stream-idle similar).  The
        # per-reasoning-model stale-timeout floor in
        # agent/reasoning_timeouts.py raises the stale-detector
        # threshold to tolerate long thinking, so a true
        # transport-layer failure here is recoverable via the retry
        # path — not via context compression.  Reclassify as timeout.
        # (Part 1 of Fixes #52310.)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute token/message-count thresholds are only a proxy for smaller
        # context windows.  Large-context sessions can have hundreds of
        # messages while still being far below their actual token budget.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7. Transport / timeout heuristics ───────────────────────────

    if error_type in _TRANSPORT_ERROR_TYPES or isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 8. Fallback: unknown ────────────────────────────────────────

    return _result(FailoverReason.unknown, retryable=True)



def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
    current = error
    for _ in range(5):  # Max depth to prevent infinite loops
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs use .status instead of .status_code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        # Walk cause chain
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_error_body(error: Exception) -> dict:
    """Extract the structured error body from an SDK exception or its cause chain."""
    current = error
    for _ in range(5):  # Match _extract_status_code() traversal depth.
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            return body
        # Some errors have .response.json()
        response = getattr(current, "response", None)
        if response is not None:
            try:
                json_body = response.json()
                if isinstance(json_body, dict):
                    return json_body
            except Exception:
                pass
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return {}


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    if not body:
        return ""

    def _code_from_payload(payload) -> str:
        """Extract a code/type from a nested error payload dict (defensive)."""
        if not isinstance(payload, dict):
            return ""
        payload_error = payload.get("error", {})
        if isinstance(payload_error, dict):
            nested = payload_error.get("code") or payload_error.get("type") or ""
            if isinstance(nested, str) and nested.strip() and nested.strip() != "400":
                return nested.strip()
        code = payload.get("code") or payload.get("error_code") or ""
        if isinstance(code, (str, int)):
            text = str(code).strip()
            if text and text != "400":
                return text
        return ""

    error_obj = body.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip() and code.strip() != "400":
            return code.strip()

        # Some providers wrap the real JSON error body as a string inside
        # error.message — peek into it for a nested code (e.g. Responses API
        # surfaces ``invalid_encrypted_content`` this way).
        message = error_obj.get("message")
        if isinstance(message, str) and message.strip().startswith("{"):
            import json
            try:
                inner = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                inner = None
            nested_code = _code_from_payload(inner)
            if nested_code:
                return nested_code

    # Top-level code
    code = body.get("code") or body.get("error_code") or ""
    if isinstance(code, (str, int)):
        text = str(code).strip()
        if text and text != "400":
            return text
    return ""


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message."""
    # Try structured body first
    if body:
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
    # Fallback to str(error)
    return str(error)[:500]
