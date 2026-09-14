"""Assemble a completed Chat Completions stream without losing tool evidence."""
import json
import logging
import uuid
from types import SimpleNamespace

from agent.message_sanitization import _repair_tool_call_arguments
from agent.error_types import EgressResultUnknown
from hermes_constants import PARTIAL_STREAM_STUB_ID, FINISH_REASON_LENGTH

logger = logging.getLogger(__name__)


def assemble_chat_stream_response(
    content_parts: list, reasoning_parts: list, tool_calls_acc: dict,
    finish_reason: str | None, role: str, model_name: str, usage_obj: object,
    *, managed_egress: bool = False,
) -> SimpleNamespace:
    if managed_egress and finish_reason is None:
        raise EgressResultUnknown()
    # Build mock response matching non-streaming shape
    full_content = "".join(content_parts) or None
    mock_tool_calls = None
    has_truncated_tool_args = False
    had_invalid_tool_args_before_repair = False
    if tool_calls_acc:
        mock_tool_calls = []
        for idx in sorted(tool_calls_acc):
            tc = tool_calls_acc[idx]
            arguments = tc["function"]["arguments"]
            tool_name = tc["function"]["name"] or "?"
            if arguments and arguments.strip():
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    # Preserve the raw stream shape before attempting a
                    # best-effort repair.  A clean stream close with no
                    # finish_reason and syntactically incomplete tool
                    # arguments is a dropped stream, even if balancing a
                    # lone opening brace happens to produce valid JSON.
                    # Repair remains useful when the provider supplied an
                    # explicit terminal finish_reason.
                    had_invalid_tool_args_before_repair = True
                    # Attempt repair before flagging as truncated.
                    # Models like GLM-5.1 via Ollama produce trailing
                    # commas, unclosed brackets, Python None, etc.
                    # Without repair, these hit the truncation handler
                    # and kill the session.  _repair_tool_call_arguments
                    # returns None for unrepairable args so the truncation
                    # path fails closed without inventing an empty object.
                    repaired = _repair_tool_call_arguments(arguments, tool_name)
                    if repaired is not None:
                        # Successfully repaired — use the fixed args
                        arguments = repaired
                    else:
                        # Unrepairable — flag for truncation handling
                        has_truncated_tool_args = True
            mock_tool_calls.append(SimpleNamespace(
                id=tc["id"],
                type=tc["type"],
                extra_content=tc.get("extra_content"),
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=arguments,
                ),
            ))

    # Zero-chunk guard: stream yielded nothing usable — a provider/upstream
    # error or malformed SSE, not a legitimate empty completion. Raise so the
    # retry machinery handles it instead of fabricating a successful turn.
    if (
        finish_reason is None
        and not content_parts
        and not reasoning_parts
        and not tool_calls_acc
    ):
        raise RuntimeError(
            "Provider returned an empty stream with no finish_reason "
            "(possible upstream error or malformed SSE response)."
        )

    # A stream that delivered a tool call but only partial/unparseable
    # JSON args splits into two very different cases:
    #
    #   1. Provider sent finish_reason="length" → a genuine output-cap
    #      truncation.  Boosting max_tokens on retry is the right move.
    #
    #   2. Provider sent NO finish_reason (the SSE simply stopped after
    #      the opening "{" with no terminator and no [DONE]) → the
    #      upstream dropped/stalled the connection mid tool-call.  This
    #      is NOT an output cap — the model never reported hitting one.
    #      Some dedicated endpoints (e.g. NVIDIA Nemotron Ultra on the
    #      Nous dedicated endpoint) stall for minutes during large
    #      tool-arg generation, then close the stream cleanly without a
    #      finish_reason.  Stamping "length" here sends it down the
    #      max_tokens-boost truncation path, which retries 3× to no
    #      effect and finally reports the misleading "Response truncated
    #      due to output length limit" — the red herring this guards
    #      against.  Route it through the partial-stream-stub path
    #      instead so the loop reports an honest mid-tool-call stream
    #      drop and fails fast rather than escalating output budget.
    _tool_args_dropped_no_finish = (
        had_invalid_tool_args_before_repair and finish_reason is None
    )
    if _tool_args_dropped_no_finish:
        _dropped_names = [
            (tool_calls_acc[idx]["function"]["name"] or "?")
            for idx in sorted(tool_calls_acc)
        ]
        logger.warning(
            "Stream ended with no finish_reason while a tool call's "
            "arguments were still incomplete (tools=%s); treating as a "
            "mid-tool-call stream drop, not an output-length truncation.",
            _dropped_names,
        )
        full_reasoning = "".join(reasoning_parts) or None
        mock_message = SimpleNamespace(
            role=role,
            content=full_content,
            tool_calls=None,
            reasoning_content=full_reasoning,
        )
        mock_choice = SimpleNamespace(
            index=0,
            message=mock_message,
            finish_reason=FINISH_REASON_LENGTH,
        )
        return SimpleNamespace(
            id=PARTIAL_STREAM_STUB_ID,
            model=model_name,
            choices=[mock_choice],
            usage=usage_obj,
            _dropped_tool_names=_dropped_names or None,
        )

    effective_finish_reason = finish_reason or "stop"
    if has_truncated_tool_args:
        effective_finish_reason = "length"

    full_reasoning = "".join(reasoning_parts) or None
    mock_message = SimpleNamespace(
        role=role,
        content=full_content,
        tool_calls=mock_tool_calls,
        reasoning_content=full_reasoning,
    )
    mock_choice = SimpleNamespace(
        index=0,
        message=mock_message,
        finish_reason=effective_finish_reason,
    )
    return SimpleNamespace(
        id="stream-" + str(uuid.uuid4()),
        model=model_name,
        choices=[mock_choice],
        usage=usage_obj,
    )
