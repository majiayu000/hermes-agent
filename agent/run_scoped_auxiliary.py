"""Request-scoped auxiliary model capabilities for managed Runtime runs.

The values live in a ContextVar so concurrent gateway runs cannot see each
other's grants.  They are never bridged through environment variables, tool
arguments, or process-global provider configuration.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Mapping


_RUN_SCOPED_AUXILIARY: ContextVar[dict[str, dict[str, str]]] = ContextVar(
    "run_scoped_auxiliary",
    default={},
)


def bind_run_scoped_auxiliary(
    capabilities: Mapping[str, Mapping[str, str]],
) -> Token[dict[str, dict[str, str]]]:
    """Bind validated auxiliary capabilities to the current Runtime request."""
    copied = {
        str(task): {str(key): str(value) for key, value in capability.items()}
        for task, capability in capabilities.items()
    }
    return _RUN_SCOPED_AUXILIARY.set(copied)


def get_run_scoped_auxiliary(task: str) -> dict[str, str] | None:
    """Return a defensive copy of the capability for one auxiliary task."""
    capability = _RUN_SCOPED_AUXILIARY.get().get(str(task))
    return dict(capability) if capability is not None else None


def reset_run_scoped_auxiliary(
    token: Token[dict[str, dict[str, str]]],
) -> None:
    """Restore the previous request context."""
    _RUN_SCOPED_AUXILIARY.reset(token)
