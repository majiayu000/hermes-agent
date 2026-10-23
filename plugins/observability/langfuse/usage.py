"""Provider usage normalization and estimated cost for trace display."""
from __future__ import annotations
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def _usage_and_cost(response: Any, *, provider: str, api_mode: str, model: str, base_url: str) -> tuple[dict[str, int], dict[str, float]]:
    usage_details: Dict[str, int] = {}
    cost_details: Dict[str, float] = {}
    raw_usage = getattr(response, "usage", None)
    if not raw_usage:
        return usage_details, cost_details

    try:
        from agent.usage_pricing import estimate_usage_cost, normalize_usage

        canonical = normalize_usage(raw_usage, provider=provider, api_mode=api_mode)
        # Langfuse usage_details keys follow a naming convention:
        #   - Dashboard sums all keys containing "input" as input total
        #   - Dashboard sums all keys containing "output" as output total
        #   - If no "total" key, Langfuse derives it from all usage types
        # Use Anthropic-style key names so cache tokens roll into the
        # dashboard input total automatically.
        # Ref: https://langfuse.com/docs/model-usage-and-cost
        usage_details = {
            "input": canonical.input_tokens,
            "output": canonical.output_tokens,
        }
        if canonical.cache_read_tokens:
            usage_details["cache_read_input_tokens"] = canonical.cache_read_tokens
        if canonical.cache_write_tokens:
            usage_details["cache_creation_input_tokens"] = canonical.cache_write_tokens
        if canonical.reasoning_tokens:
            usage_details["reasoning_tokens"] = canonical.reasoning_tokens
        cost = estimate_usage_cost(
            model,
            canonical,
            provider=provider,
            base_url=base_url,
            api_key="",
        )
        if cost.amount_usd is not None:
            # Langfuse cost_details keys must match usage_details keys.
            # Provide per-type breakdown so dashboard can show cost by type.
            try:
                from agent.usage_pricing import get_pricing_entry
                from decimal import Decimal
                _ONE_M = Decimal("1000000")
                entry = get_pricing_entry(model, provider=provider, base_url=base_url)
                if entry:
                    if entry.input_cost_per_million is not None and canonical.input_tokens:
                        cost_details["input"] = float(Decimal(canonical.input_tokens) * entry.input_cost_per_million / _ONE_M)
                    if entry.output_cost_per_million is not None and canonical.output_tokens:
                        cost_details["output"] = float(Decimal(canonical.output_tokens) * entry.output_cost_per_million / _ONE_M)
                    if entry.cache_read_cost_per_million is not None and canonical.cache_read_tokens:
                        cost_details["cache_read_input_tokens"] = float(Decimal(canonical.cache_read_tokens) * entry.cache_read_cost_per_million / _ONE_M)
                    if entry.cache_write_cost_per_million is not None and canonical.cache_write_tokens:
                        cost_details["cache_creation_input_tokens"] = float(Decimal(canonical.cache_write_tokens) * entry.cache_write_cost_per_million / _ONE_M)
                else:
                    cost_details["total"] = float(cost.amount_usd)
            except Exception:
                cost_details["total"] = float(cost.amount_usd)
    except Exception as exc:  # pragma: no cover - fail-open
        logger.debug("usage normalization failed: %s", exc)

    return usage_details, cost_details


def _summary_usage_and_cost(usage: dict, *, model: str, provider: str, base_url: str) -> tuple[dict, dict]:
    # post_api_request passes a pre-built CanonicalUsage summary dict.
    # Use Langfuse-convention key names: "input", "output", and
    # "cache_read_input_tokens" / "cache_creation_input_tokens" so the
    # dashboard sums cache tokens into the input total automatically.
    _input = usage.get("input_tokens", 0)
    _output = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
    _cache_read = usage.get("cache_read_tokens", 0)
    _cache_write = usage.get("cache_write_tokens", 0)
    _reasoning = usage.get("reasoning_tokens", 0)
    usage_details = {
        "input": _input,
        "output": _output,
    }
    if _cache_read:
        usage_details["cache_read_input_tokens"] = _cache_read
    if _cache_write:
        usage_details["cache_creation_input_tokens"] = _cache_write
    if _reasoning:
        usage_details["reasoning_tokens"] = _reasoning
    cost_details = {}
    # Estimate per-type cost from the summary if possible
    try:
        from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, get_pricing_entry
        from decimal import Decimal
        _ONE_M = Decimal("1000000")
        _cu = CanonicalUsage(
            input_tokens=_input,
            output_tokens=_output,
            cache_read_tokens=_cache_read,
            cache_write_tokens=_cache_write,
            reasoning_tokens=_reasoning,
        )
        entry = get_pricing_entry(model, provider=provider, base_url=base_url)
        if entry:
            if entry.input_cost_per_million is not None and _input:
                cost_details["input"] = float(Decimal(_input) * entry.input_cost_per_million / _ONE_M)
            if entry.output_cost_per_million is not None and _output:
                cost_details["output"] = float(Decimal(_output) * entry.output_cost_per_million / _ONE_M)
            if entry.cache_read_cost_per_million is not None and _cache_read:
                cost_details["cache_read_input_tokens"] = float(Decimal(_cache_read) * entry.cache_read_cost_per_million / _ONE_M)
            if entry.cache_write_cost_per_million is not None and _cache_write:
                cost_details["cache_creation_input_tokens"] = float(Decimal(_cache_write) * entry.cache_write_cost_per_million / _ONE_M)
        else:
            _cost = estimate_usage_cost(model, _cu, provider=provider, base_url=base_url, api_key="")
            if _cost.amount_usd is not None:
                cost_details["total"] = float(_cost.amount_usd)
    except Exception:
        pass
    return usage_details, cost_details
