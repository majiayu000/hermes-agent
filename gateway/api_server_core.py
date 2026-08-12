"""Extracted API-server adapter methods.

This module is mechanically split from gateway.platforms.api_server.
"""

from __future__ import annotations

from datetime import datetime, timezone

from gateway.api_server_shared import *
from gateway.api_server_audit import log_api_decision, request_id_headers
from gateway.config import is_runtime_driver_only
from gateway.principal_headers import parse_principal_scope_headers
from gateway.runtime_provenance import (
    RuntimeProvenance,
    RuntimeProvenanceConfig,
    collect_runtime_provenance,
)
from gateway.runtime_contract import runtime_health_contract
from gateway.runtime_capability_auth import (
    RuntimeCapabilityConfig,
    RuntimeCapabilityError,
    RuntimeCapabilityVerifier,
)


_API_SERVER_CLI_INHERITED_TOOLSETS = frozenset({"video_gen"})


def _resolve_api_server_toolsets(
    user_config: dict,
    *,
    include_default_mcp_servers: bool = True,
) -> List[str]:
    """Resolve the API-server agent toolsets.

    The API server predates the standalone creative web panel, so its
    built-in composite intentionally omits default-off interactive media
    tools.  When the user has not configured ``platform_toolsets.api_server``
    explicitly, mirror the CLI's enabled video generation switch so the web
    agent can do the same creative work as the local agent.  A saved
    ``api_server`` list remains authoritative.
    """
    from hermes_cli.tools_config import _get_platform_tools

    config = user_config if isinstance(user_config, dict) else {}
    try:
        enabled = set(
            _get_platform_tools(
                config,
                "api_server",
                include_default_mcp_servers=include_default_mcp_servers,
            )
        )
    except TypeError as exc:
        if "include_default_mcp_servers" not in str(exc):
            raise
        enabled = set(_get_platform_tools(config, "api_server"))
    platform_toolsets = config.get("platform_toolsets") or {}
    api_server_explicit = (
        isinstance(platform_toolsets, dict)
        and "api_server" in platform_toolsets
    )
    if not api_server_explicit:
        try:
            cli_enabled = set(
                _get_platform_tools(
                    config,
                    "cli",
                    include_default_mcp_servers=include_default_mcp_servers,
                )
            )
        except TypeError as exc:
            if "include_default_mcp_servers" not in str(exc):
                raise
            cli_enabled = set(_get_platform_tools(config, "cli"))
        enabled.update(_API_SERVER_CLI_INHERITED_TOOLSETS & cli_enabled)

    return sorted(enabled)


class APIServerCoreMixin:
    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.API_SERVER)
        extra = config.extra or {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        # Active run agent/task references for stop support
        self._active_run_agents: Dict[str, Any] = {}
        self._active_run_tasks: Dict[str, "asyncio.Task"] = {}
        # Active session chat streams: session_id -> stream control state.
        # Session ACL is still checked by the stop endpoint before this map is
        # consulted, so callers cannot stop or probe another principal's run.
        self._active_session_streams: Dict[str, Dict[str, Any]] = {}
        # Pollable run status for dashboards and external control-plane UIs.
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active approval session key for each run_id.  The approval core
        # resolves requests by session key, while API clients address the
        # in-flight run by run_id.
        self._run_approval_sessions: Dict[str, str] = {}
        self._session_db: Optional[Any] = None  # Lazy-init SessionDB for session continuity
        self._max_concurrent_runs: int = self._resolve_max_concurrent_runs()
        self._inflight_agent_runs: int = 0
        self._runtime_started_at = datetime.now(timezone.utc)
        self._runtime_provenance: Optional[RuntimeProvenance] = None
        self._runtime_capability_verifier: Optional[RuntimeCapabilityVerifier] = None

    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_max_concurrent_runs() -> int:
        """Read the concurrent-run cap from config.yaml; 0 disables it."""
        default = 10
        try:
            from hermes_cli.config import cfg_get, load_config

            raw = cfg_get(
                load_config(),
                "gateway",
                "api_server",
                "max_concurrent_runs",
                default=default,
            )
            value = int(raw)
        except Exception:
            return default
        return max(0, value)

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in {"default", "custom"}:
                return profile
        except Exception:
            pass
        return "hermes-agent"

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    @staticmethod
    def _clean_log_value(value: Any, *, max_len: int = 200) -> str:
        """Sanitize request metadata before it reaches security logs."""
        if value is None:
            return ""
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        return text[:max_len]

    def _request_audit_context(self, request: "web.Request") -> Dict[str, str]:
        """Return non-secret source metadata for security/audit warnings."""
        peer_ip = ""
        try:
            peer = request.transport.get_extra_info("peername") if request.transport else None
            if isinstance(peer, (tuple, list)) and peer:
                peer_ip = str(peer[0])
        except Exception:
            peer_ip = ""

        return {
            "remote": self._clean_log_value(getattr(request, "remote", "") or peer_ip),
            "peer_ip": self._clean_log_value(peer_ip),
            "forwarded_for": self._clean_log_value(request.headers.get("X-Forwarded-For", "")),
            "real_ip": self._clean_log_value(request.headers.get("X-Real-IP", "")),
            "method": self._clean_log_value(request.method, max_len=16),
            "path": self._clean_log_value(request.path_qs, max_len=500),
            "user_agent": self._clean_log_value(request.headers.get("User-Agent", ""), max_len=300),
        }

    def _request_audit_log_suffix(self, request: "web.Request") -> str:
        ctx = self._request_audit_context(request)
        fields = [f"{key}={value!r}" for key, value in ctx.items() if value]
        return " ".join(fields) if fields else "source='unknown'"

    def _cron_origin_from_request(self, request: "web.Request") -> Dict[str, str]:
        """Persist safe API source metadata on cron jobs created over HTTP."""
        ctx = self._request_audit_context(request)
        origin = {
            "platform": "api_server",
            "chat_id": "api",
        }
        if ctx.get("remote"):
            origin["source_ip"] = ctx["remote"]
        if ctx.get("peer_ip"):
            origin["peer_ip"] = ctx["peer_ip"]
        if ctx.get("forwarded_for"):
            origin["forwarded_for"] = ctx["forwarded_for"]
        if ctx.get("real_ip"):
            origin["real_ip"] = ctx["real_ip"]
        if ctx.get("user_agent"):
            origin["user_agent"] = ctx["user_agent"]
        return origin

    @staticmethod
    def _bind_api_server_session(
        *,
        chat_id: str = "",
        session_key: str = "",
        session_id: str = "",
        principal_scope: Optional[Dict[str, Any]] = None,
    ) -> list:
        """Bind request-scoped contextvars for an API-server agent run."""
        from gateway.session_context import set_session_vars

        scope = principal_scope or {}
        return set_session_vars(
            platform="api_server",
            chat_id=chat_id,
            session_key=session_key,
            session_id=session_id,
            tenant_id=str(scope.get("tenant_id") or ""),
            workspace_id=str(scope.get("workspace_id") or ""),
            project_id=str(scope.get("project_id") or ""),
            user_id=str(scope.get("user_id") or ""),
            roles=scope.get("roles"),
            sandbox_id=str(scope.get("sandbox_id") or ""),
            sandbox_status=str(scope.get("sandbox_status") or "active"),
            sandbox_expires_at=scope.get("sandbox_expires_at"),
            async_delivery=False,
        )

    def _concurrency_limited_response(self) -> Optional["web.Response"]:
        """Return 429 when the shared API-server agent cap is exhausted."""
        limit = self._max_concurrent_runs
        if limit <= 0:
            return None
        inflight = self._inflight_agent_runs + len(self._run_streams)
        if inflight >= limit:
            return web.json_response(
                _openai_error(
                    f"Too many concurrent runs (max {limit})",
                    err_type="rate_limit_error",
                    code="rate_limit_exceeded",
                ),
                status=429,
                headers={"Retry-After": "1"},
            )
        return None

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        General API mode refuses to start without API_SERVER_KEY. Ultra Studio
        runtime-driver-only mode authenticates its private routes separately
        with service identity plus an Ed25519 operation capability.
        """
        if not self._api_key:
            return None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        logger.warning(
            "API server rejected invalid API key: %s",
            self._request_audit_log_suffix(request),
        )
        log_api_decision(
            request,
            action="auth.check",
            result="denied",
            status=401,
            reason="invalid_api_key",
        )
        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
            headers=request_id_headers(request),
        )

    async def _check_runtime_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """Authenticate the private runtime surface with a bound capability."""
        if not is_runtime_driver_only():
            return self._check_auth(request)
        try:
            if self._runtime_capability_verifier is None:
                self._runtime_capability_verifier = RuntimeCapabilityVerifier(
                    RuntimeCapabilityConfig.from_env()
                )
            request["runtime_capability"] = await self._runtime_capability_verifier.verify(request)
            return None
        except RuntimeCapabilityError as exc:
            logger.warning("Runtime capability rejected: %s", exc)
            return web.json_response(
                {"error": {"message": "Invalid runtime capability", "type": "invalid_request_error", "code": "invalid_runtime_capability"}},
                status=401,
                headers=request_id_headers(request),
            )

    # ------------------------------------------------------------------
    # Session header helpers
    # ------------------------------------------------------------------

    # Soft length cap for session identifiers.  Headers are bounded in
    # aggregate by aiohttp (``client_max_size`` / default 8 KiB per
    # header), but we impose a tighter limit on the session headers so a
    # caller can't burn memory by passing a multi-kilobyte "session key".
    # 256 chars is well above any realistic stable channel identifier
    # (e.g. ``agent:main:webui:dm:user-42``) while staying small enough
    # that the sanitized form is safe to pass into Honcho / state.db.
    _MAX_SESSION_HEADER_LEN = 256

    def _parse_session_key_header(
        self, request: "web.Request"
    ) -> tuple[Optional[str], Optional["web.Response"]]:
        """Extract and validate the ``X-Hermes-Session-Key`` header.

        The session key is a stable per-channel identifier that scopes
        long-term memory (e.g. Honcho sessions) across transcripts.  It
        is independent of ``X-Hermes-Session-Id``: callers may send
        either, both, or neither.

        Returns ``(session_key, None)`` on success (with an empty/absent
        header yielding ``None`` for the key), or ``(None, error_response)``
        on validation failure.

        Security: like session continuation, accepting a caller-supplied
        memory scope requires API-key authentication so that an
        unauthenticated client on a local-only server can't inject itself
        into another user's long-term memory scope by guessing a key.
        """
        raw = request.headers.get("X-Hermes-Session-Key", "").strip()
        if not raw:
            return None, None

        if not self._api_key:
            logger.warning(
                "X-Hermes-Session-Key rejected: no API key configured. "
                "Set API_SERVER_KEY to enable long-term memory scoping."
            )
            return None, web.json_response(
                _openai_error(
                    "X-Hermes-Session-Key requires API key authentication. "
                    "Configure API_SERVER_KEY to enable this feature."
                ),
                status=403,
            )

        # Reject control characters that could enable header injection on
        # the echo path.
        if re.search(r'[\r\n\x00]', raw):
            return None, web.json_response(
                {"error": {"message": "Invalid session key", "type": "invalid_request_error"}},
                status=400,
            )

        if len(raw) > self._MAX_SESSION_HEADER_LEN:
            return None, web.json_response(
                {"error": {"message": "Session key too long", "type": "invalid_request_error"}},
                status=400,
            )

        return raw, None

    def _parse_principal_scope_headers(
        self, request: "web.Request"
    ) -> tuple[Dict[str, Any], Optional["web.Response"]]:
        """Parse optional multi-user scope headers for this agent turn."""
        scope, error = parse_principal_scope_headers(
            request.headers,
            api_key_configured=bool(self._api_key),
            max_len=self._MAX_SESSION_HEADER_LEN,
        )
        if error is None:
            return scope, None

        status = 403 if "API key" in error else 400
        logger.warning("Principal scope headers rejected: %s", error)
        log_api_decision(
            request,
            action="principal.parse",
            result="denied",
            status=status,
            reason=error,
        )
        return {}, web.json_response(_openai_error(error), status=status, headers=request_id_headers(request))

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        stream_delta_callback=None,
        reasoning_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        gateway_session_key: Optional[str] = None,
        clarify_callback=None,
        runtime_overrides: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        ``gateway_session_key`` is a stable per-channel identifier supplied
        by the client (via ``X-Hermes-Session-Key``).  Unlike ``session_id``
        which scopes the short-term transcript and rotates on /new, this
        key is meant to persist across transcripts so long-term memory
        providers (e.g. Honcho) can scope their per-chat state correctly
        — matching the semantics of the native gateway's ``session_key``.
        """
        from run_agent import AIAgent
        from gateway.run import (
            _current_max_iterations,
            _resolve_runtime_agent_kwargs,
            _resolve_gateway_model,
            _load_gateway_config,
            GatewayRunner,
        )

        runtime_kwargs = (
            dict(runtime_overrides)
            if runtime_overrides is not None
            else _resolve_runtime_agent_kwargs()
        )
        reasoning_config = GatewayRunner._load_reasoning_config()
        model = str(model_override or "").strip() or _resolve_gateway_model()

        user_config = _load_gateway_config()
        enabled_toolsets = _resolve_api_server_toolsets(user_config)

        max_iterations = _current_max_iterations()

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        fallback_model = GatewayRunner._load_fallback_model()

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            reasoning_callback=reasoning_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            clarify_callback=clarify_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            gateway_session_key=gateway_session_key,
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health — simple health check."""
        if self._runtime_provenance is None:
            self._runtime_provenance = collect_runtime_provenance(
                RuntimeProvenanceConfig(
                    listen_host=self._host,
                    listen_port=self._port,
                    cors_origins=self._cors_origins,
                    model_name=self._model_name,
                    max_concurrent_runs=self._max_concurrent_runs,
                ),
                startup_timestamp=self._runtime_started_at,
            )
        return web.json_response(
            {
                "status": "ok",
                "platform": "hermes-agent",
                **runtime_health_contract(),
                **self._runtime_provenance.to_dict(),
            }
        )

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed — rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  No authentication required.
        """
        from gateway.status import (
            derive_gateway_busy,
            derive_gateway_drainable,
            parse_active_agents,
            read_runtime_status,
        )

        runtime = read_runtime_status() or {}
        gw_state = runtime.get("gateway_state")
        gw_active = parse_active_agents(runtime.get("active_agents", 0))
        return web.json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "version": _hermes_version(),
            "gateway_state": gw_state,
            "platforms": runtime.get("platforms", {}),
            "active_agents": gw_active,
            "gateway_busy": derive_gateway_busy(
                gateway_running=True,
                gateway_state=gw_state,
                active_agents=gw_active,
            ),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=True,
                gateway_state=gw_state,
            ),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models — return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": self._model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": self._model_name,
                    "parent": None,
                }
            ],
        })

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities — advertise the stable API surface.

        External UIs and orchestrators use this endpoint to discover the API
        server's plugin-safe contract without scraping docs or assuming that
        every Hermes version exposes the same endpoints.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "model": self._model_name,
            "auth": {
                "type": "bearer",
                "required": bool(self._api_key),
            },
            "runtime": {
                "mode": "server_agent",
                "tool_execution": "server",
                "split_runtime": False,
                "description": (
                    "The API server creates a server-side Hermes AIAgent; "
                    "tools execute on the API-server host unless a future "
                    "explicit split-runtime mode is enabled."
                ),
            },
            "features": {
                "chat_completions": True,
                "chat_completions_streaming": True,
                "responses_api": True,
                "responses_streaming": True,
                "run_submission": True,
                "run_status": True,
                "run_events_sse": True,
                "run_stop": True,
                "run_approval_response": True,
                "tool_progress_events": True,
                "approval_events": True,
                "session_resources": True,
                "session_chat": True,
                "session_chat_streaming": True,
                "session_chat_stop": True,
                "session_chat_approval": True,
                "session_chat_prompt": True,
                "session_fork": True,
                "admin_config_rw": False,
                "jobs_admin": False,
                "memory_write_api": False,
                "skills_api": True,
                "audio_api": False,
                "realtime_voice": False,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
                "cors": bool(self._cors_origins),
            },
            "endpoints": {
                "health": {"method": "GET", "path": "/health"},
                "health_detailed": {"method": "GET", "path": "/health/detailed"},
                "models": {"method": "GET", "path": "/v1/models"},
                "chat_completions": {"method": "POST", "path": "/v1/chat/completions"},
                "responses": {"method": "POST", "path": "/v1/responses"},
                "runs": {"method": "POST", "path": "/v1/runs"},
                "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
                "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
                "run_approval": {"method": "POST", "path": "/v1/runs/{run_id}/approval"},
                "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
                "skills": {"method": "GET", "path": "/v1/skills"},
                "toolsets": {"method": "GET", "path": "/v1/toolsets"},
                "sessions": {"method": "GET", "path": "/api/sessions"},
                "session_create": {"method": "POST", "path": "/api/sessions"},
                "session": {"method": "GET", "path": "/api/sessions/{session_id}"},
                "session_update": {"method": "PATCH", "path": "/api/sessions/{session_id}"},
                "session_delete": {"method": "DELETE", "path": "/api/sessions/{session_id}"},
                "session_messages": {"method": "GET", "path": "/api/sessions/{session_id}/messages"},
                "session_fork": {"method": "POST", "path": "/api/sessions/{session_id}/fork"},
                "session_chat": {"method": "POST", "path": "/api/sessions/{session_id}/chat"},
                "session_chat_stream": {"method": "POST", "path": "/api/sessions/{session_id}/chat/stream"},
                "session_chat_stop": {"method": "POST", "path": "/api/sessions/{session_id}/chat/stop"},
                "session_chat_approval": {"method": "POST", "path": "/api/sessions/{session_id}/chat/approval"},
                "session_chat_prompt": {"method": "POST", "path": "/api/sessions/{session_id}/chat/prompt"},
            },
        })

    async def _handle_skills(self, request: "web.Request") -> "web.Response":
        """GET /v1/skills — list installed skills visible to the API-server agent.

        Read-only listing intended for external clients that need to know
        which skills are available without sending a chat message and asking
        the model. Mirrors what the gateway/CLI surfaces through
        ``/skills list``, but as a deterministic JSON payload.

        Returns the same skill metadata (name, description, category) the
        skills hub uses internally. Disabled skills are excluded so the
        listing matches what the agent actually loads.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from tools.skills_tool import _find_all_skills, _sort_skills
            skills = _sort_skills(_find_all_skills(skip_disabled=False))
        except Exception:
            logger.exception("GET /v1/skills failed")
            return web.json_response(
                _openai_error("Failed to enumerate skills", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "data": skills,
        })

    async def _handle_toolsets(self, request: "web.Request") -> "web.Response":
        """GET /v1/toolsets — list toolsets and their resolved tools.

        Returns the toolset surface the api_server platform actually exposes
        to its agent: each toolset's enabled/configured state plus the
        concrete tool names it expands to. This is the deterministic
        equivalent of what a client would otherwise have to recover by
        asking the model what tools it can call.
        """
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        try:
            from hermes_cli.config import load_config
            from hermes_cli.tools_config import (
                _get_effective_configurable_toolsets,
                _toolset_has_keys,
            )
            from toolsets import resolve_toolset

            config = load_config()
            enabled_toolsets = set(_resolve_api_server_toolsets(config, include_default_mcp_servers=False))
            data: List[Dict[str, Any]] = []
            for name, label, desc in _get_effective_configurable_toolsets():
                try:
                    tools = sorted(set(resolve_toolset(name)))
                except Exception:
                    tools = []
                is_enabled = name in enabled_toolsets
                data.append({
                    "name": name,
                    "label": label,
                    "description": desc,
                    "enabled": is_enabled,
                    "configured": _toolset_has_keys(name, config),
                    "tools": tools,
                })
        except Exception:
            logger.exception("GET /v1/toolsets failed")
            return web.json_response(
                _openai_error("Failed to enumerate toolsets", err_type="server_error"),
                status=500,
            )

        return web.json_response({
            "object": "list",
            "platform": "api_server",
            "data": data,
        })

    # ------------------------------------------------------------------
    # /api/sessions — thin client/session resource API
    # ------------------------------------------------------------------
