"""Extracted API-server adapter methods.

This module is mechanically split from gateway.platforms.api_server.
"""

from __future__ import annotations

from gateway.api_server_shared import *
from gateway.api_server_audit import request_audit_middleware
from gateway.config import is_runtime_driver_only
from gateway.runtime_capability_auth import (
    RuntimeCapabilityConfig,
    RuntimeCapabilityError,
    RuntimeCapabilityVerifier,
)


class APIServerLifecycleMixin:
    def _setup_routes(self) -> None:
        """Register HTTP routes on ``self._app``.

        With HERMES_RUNTIME_DRIVER_ONLY set, only the health probe and the
        private Run Orchestrator Runtime Driver contract are registered —
        every other entrypoint (chat completions, sessions, responses, jobs,
        runs) is left off the router so it 404s instead of bypassing
        upstream billing.
        """
        assert self._app is not None
        runtime_driver_only = is_runtime_driver_only()
        if runtime_driver_only:
            logger.info(
                "[%s] HERMES_RUNTIME_DRIVER_ONLY enabled: registering only "
                "/healthz and /v1/runtime/* routes",
                self.name,
            )
        else:
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_get("/v1/capabilities", self._handle_capabilities)
            self._app.router.add_get("/v1/skills", self._handle_skills)
            self._app.router.add_get("/v1/toolsets", self._handle_toolsets)
            # Session/client control surface (thin wrappers over SessionDB + _run_agent)
            self._app.router.add_get("/api/sessions", self._handle_list_sessions)
            self._app.router.add_post("/api/sessions", self._handle_create_session)
            self._app.router.add_get("/api/sessions/{session_id}", self._handle_get_session)
            self._app.router.add_patch("/api/sessions/{session_id}", self._handle_patch_session)
            self._app.router.add_delete("/api/sessions/{session_id}", self._handle_delete_session)
            self._app.router.add_get("/api/sessions/{session_id}/messages", self._handle_session_messages)
            self._app.router.add_post("/api/sessions/{session_id}/fork", self._handle_fork_session)
            self._app.router.add_post("/api/sessions/{session_id}/chat", self._handle_session_chat)
            self._app.router.add_post("/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream)
            self._app.router.add_post("/api/sessions/{session_id}/chat/stop", self._handle_session_chat_stop)
            self._app.router.add_post("/api/sessions/{session_id}/chat/approval", self._handle_session_chat_approval)
            self._app.router.add_post("/api/sessions/{session_id}/chat/prompt", self._handle_session_chat_prompt)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management API
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            self._app.router.add_post("/api/cron/fire", self._handle_cron_fire)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}", self._handle_get_run)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/runs/{run_id}/approval", self._handle_run_approval)
            self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)
        # Private Run Orchestrator Runtime Driver contract.
        self._app.router.add_get("/healthz", self._handle_health)
        self._app.router.add_get("/v1/runtime/manifest", self._handle_runtime_manifest)
        self._app.router.add_post("/v1/runtime/runs", self._handle_runtime_run)
        self._app.router.add_post("/v1/runtime/runs/{run_id}/tool-results", self._handle_runtime_tool_result)
        self._app.router.add_post("/v1/runtime/runs/{run_id}/control-results", self._handle_runtime_control_result)
        self._app.router.add_post("/v1/runtime/runs/{run_id}/suspend", self._handle_runtime_suspend)
        self._app.router.add_post("/v1/runtime/runs/{run_id}/cancel", self._handle_runtime_cancel)
        self._app.router.add_post("/v1/runtime/runs/{run_id}/abort", self._handle_runtime_abort)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", self.name)
            return False
        runtime_driver_only = is_runtime_driver_only()

        try:
            mws = [
                mw
                for mw in (
                    request_audit_middleware,
                    cors_middleware,
                    body_limit_middleware,
                    security_headers_middleware,
                )
                if mw is not None
            ]
            self._app = web.Application(
                middlewares=mws,
                client_max_size=MAX_RUNTIME_REQUEST_BYTES,
            )
            assert self._app is not None
            self._setup_routes()
            # Store the adapter after native routes are registered. Local Hermes-Relay
            # bootstrap shims use this key as a feature-detection hook; registering
            # native routes first lets those shims no-op instead of shadowing the
            # upstream session-control handlers.
            self._app["api_server_adapter"] = self

            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)

            # Runtime-driver-only mode uses a separate caller credential plus
            # per-request Ed25519 capabilities. Refuse to expose the private
            # execution surface unless both verifier inputs are valid.
            if runtime_driver_only:
                try:
                    self._runtime_capability_verifier = RuntimeCapabilityVerifier(
                        RuntimeCapabilityConfig.from_env()
                    )
                except RuntimeCapabilityError as exc:
                    logger.error("[%s] Refusing to start: %s", self.name, exc)
                    return False

            # Refuse to start without authentication. The general API server can
            # dispatch terminal-capable agent work, so every deployment needs
            # an explicit API_SERVER_KEY regardless of bind address. Driver-only
            # mode was already gated above and deliberately does not share it.
            if not runtime_driver_only and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: API_SERVER_KEY is required for the API server, "
                    "including loopback-only binds on %s.",
                    self.name, self._host,
                )
                return False

            # Refuse to start network-accessible with a placeholder or weak key.
            # Ported from openclaw/openclaw#64586; entropy floor raised to 16 in
            # the June 2026 hermes-0day hardening (an 8-char key dispatching
            # terminal-capable agent work on a public bind is brute-forceable).
            if not runtime_driver_only and is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=16):
                        logger.error(
                            "[%s] Refusing to start: API_SERVER_KEY is a "
                            "placeholder or too short (<16 chars) for a "
                            "network-accessible bind. This endpoint dispatches "
                            "terminal-capable agent work — a guessable key is "
                            "remote code execution. Generate a strong secret "
                            "(e.g. `openssl rand -hex 32`) and set "
                            "API_SERVER_KEY before exposing it on %s.",
                            self.name, self._host,
                        )
                        return False
                except ImportError:
                    pass

            # Port conflict detection — fail fast if port is already in use
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use. Set a different port in config.yaml: platforms.api_server.port', self.name, self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._mark_connected()
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                self.name, self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", self.name, e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server and release all owned resources.

        Closes the ResponseStore SQLite connection in addition to stopping
        the aiohttp web server. Without this, every adapter instance leaks
        2 file descriptors (the database file and its WAL sidecar) — the
        reconnect loop in ``gateway.run`` constructs a fresh adapter on
        every retry, so 2 fds/retry × 300s backoff cap ≈ 12 fds/hour, which
        exhausts the default 2560 fd limit after ~12h of failed reconnects
        and turns the whole gateway into a zombie
        (OSError: [Errno 24] Too many open files, #37011).
        """
        self._mark_disconnected()
        if self._response_store is not None:
            try:
                self._response_store.close()
            except Exception:
                logger.debug(
                    "Failed to close response store for %s", self.name, exc_info=True,
                )
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("[%s] API server stopped", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used — HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }
