"""Validate and bind the Runtime run-scoped LLM capability."""
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

def _runtime_llm_egress(value: object, *, required: bool) -> dict[str, str] | None:
    if value is None and not required:
        return None
    if not isinstance(value, dict) or set(value) != {"base_url", "grant", "expires_at"}:
        raise ValueError("llm_egress must contain base_url, grant, and expires_at")
    base_url = str(value.get("base_url") or "").strip().rstrip("/")
    grant = str(value.get("grant") or "").strip()
    expires_at = str(value.get("expires_at") or "").strip()
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/internal/llm/v1"
    ):
        raise ValueError("llm_egress.base_url is invalid")
    if not re.fullmatch(r"ueg_[A-Za-z0-9_-]{43}", grant):
        raise ValueError("llm_egress.grant is invalid")
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("llm_egress.expires_at is invalid") from exc
    if expiry.tzinfo is None or expiry.astimezone(timezone.utc) <= datetime.now(timezone.utc):
        raise ValueError("llm_egress grant is expired")
    return {"base_url": base_url, "grant": grant, "expires_at": expires_at}


def _runtime_vision_llm_egress(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "model", "base_url", "grant", "expires_at",
    }:
        raise ValueError(
            "vision_llm_egress must contain model, base_url, grant, and expires_at"
        )
    model = str(value.get("model") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,511}", model):
        raise ValueError("vision_llm_egress.model is invalid")
    capability = _runtime_llm_egress(
        {key: value[key] for key in ("base_url", "grant", "expires_at")},
        required=True,
    )
    return {"model": model, **capability}


def _configure_run_llm_egress(agent: object, capability: dict[str, str] | None, model: object) -> None:
    if capability is None:
        return
    requested_model = str(model or "").strip()
    if not requested_model:
        raise ValueError("model is required")
    if (
        str(getattr(agent, "model", "") or "").strip() != requested_model
        or str(getattr(agent, "provider", "") or "").strip() != "custom"
        or str(getattr(agent, "api_key", "") or "") != capability["grant"]
        or str(getattr(agent, "base_url", "") or "").rstrip("/")
        != capability["base_url"]
    ):
        raise ValueError("agent run-scoped LLM egress configuration is inconsistent")
    agent._run_llm_egress = True
