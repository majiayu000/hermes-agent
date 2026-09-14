"""Prepare recovery from a completed but truncated model tool call."""
from __future__ import annotations

from typing import TYPE_CHECKING
import json
from hermes_constants import PARTIAL_STREAM_STUB_ID
from agent.error_types import EgressResultUnknown

if TYPE_CHECKING:
    from run_agent import AIAgent
    from agent.transports.types import NormalizedResponse


def prepare_tool_call_retry(
    agent: AIAgent, response: object, message: NormalizedResponse, api_kwargs: dict, attempt: int,
    messages: list[dict], api_messages: list[dict],
) -> None:
    is_stub_stall = getattr(response, "id", "") == PARTIAL_STREAM_STUB_ID
    if is_stub_stall and getattr(agent, "_run_llm_egress", False) is True:
        raise EgressResultUnknown()
    if is_stub_stall:
        # The stream broke mid tool-call (network /
        # peer-closed connection), not a real output
        # cap — say so instead of "max output tokens".
        agent._buffer_vprint(
            f"⚠️  Stream interrupted mid tool-call — "
            f"retrying ({attempt}/3)..."
        )
    else:
        agent._buffer_vprint(
            f"⚠️  Truncated tool call detected — "
            f"retrying API call "
            f"({attempt}/3)..."
        )
    # Boost max_tokens on each retry so the model has
    # more room to complete the tool-call JSON. A
    # network stall doesn't need a bigger budget, but
    # a genuine output-cap truncation does, and the
    # boost is harmless for the stall case.
    _tc_boost_base = agent.max_tokens if agent.max_tokens else 4096
    _tc_boost = _tc_boost_base * (attempt + 1)
    _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
    if _tc_requested_cap is not None:
        _tc_boost = max(_tc_boost, _tc_requested_cap)
    _tc_boost_cap = max(32768, _tc_requested_cap or 0)
    agent._ephemeral_max_output_tokens = min(_tc_boost, _tc_boost_cap)
    if not is_stub_stall:
        # A completed malformed generation is evidence for a correction, not
        # an identical transport retry. Preserve it as text: invalid tool-call
        # JSON must not enter the executable assistant/tool message sequence.
        evidence = {
            "content": message.content,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in (message.tool_calls or [])
            ],
        }
        correction = {
            "role": "user",
            "content": (
                "The previous completed response contained an incomplete tool call. "
                "No tool from that response was executed. Correct the arguments "
                "and return a complete tool call within the output budget. "
                "Previous response evidence:\n" + json.dumps(evidence, ensure_ascii=False)
            ),
        }
        messages.append(correction)
        if api_messages is not messages:
            api_messages.append(dict(correction))
