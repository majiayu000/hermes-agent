"""Capability-specific availability checks for Web model tools."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_web_extract_available() -> bool:
    """Return whether the resolved provider can currently extract URLs."""
    from tools.web_tools import _ensure_web_plugins_loaded

    _ensure_web_plugins_loaded()

    from agent.web_search_registry import get_active_extract_provider

    provider = get_active_extract_provider()
    if provider is None or not provider.supports_extract():
        return False
    try:
        return bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "web extract provider %s availability check raised %s",
            provider.name,
            exc,
        )
        return False
