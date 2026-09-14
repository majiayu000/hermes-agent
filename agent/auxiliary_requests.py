"""HTTP client options and transport retries shared by auxiliary LLM calls."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar, Optional, Dict, Any

from agent.process_bootstrap import build_keepalive_http_client

logger = logging.getLogger(__name__)
_Response = TypeVar("_Response")

def _openai_http_client_kwargs(
    base_url: Optional[str],
    *,
    async_mode: bool = False,
) -> Dict[str, Any]:
    """Inject keepalive httpx client with env-only proxy (not macOS system proxy)."""
    client = build_keepalive_http_client(str(base_url or ""), async_mode=async_mode)
    if client is None:
        return {"max_retries": 0}
    return {"max_retries": 0, "http_client": client}



def invoke_auxiliary_request(
    request: Callable[[], _Response], *, task: str | None,
    is_transient: Callable[[Exception], bool], allow_retry: bool = True,
) -> _Response:
    try:
        return request()
    except Exception as error:
        if not allow_retry or not is_transient(error):
            raise
        logger.info("Auxiliary %s: transient transport error; retrying once on the same provider before fallback: %s", task or "call", error)
        return request()


async def invoke_async_auxiliary_request(
    request: Callable[[], Awaitable[_Response]], *, task: str | None,
    is_transient: Callable[[Exception], bool], allow_retry: bool = True,
) -> _Response:
    try:
        return await request()
    except Exception as error:
        if not allow_retry or not is_transient(error):
            raise
        logger.info("Auxiliary %s (async): transient transport error; retrying once on the same provider before fallback: %s", task or "call", error)
        return await request()
