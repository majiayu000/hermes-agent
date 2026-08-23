"""Tests for the HERMES_RUNTIME_DRIVER_ONLY / HERMES_DISABLE_CRON switches.

HERMES_RUNTIME_DRIVER_ONLY must restrict the API server router to the health
probe plus the private ``/v1/runtime/*`` Runtime Driver contract, so the
billing-bypass entrypoints (/v1/chat/completions, /api/sessions/*/chat,
/v1/responses, /api/jobs, ...) 404 instead of dispatching agent work.

HERMES_DISABLE_CRON (or runtime-driver-only mode) must skip the background
cron scheduler thread.
"""

import pytest

from gateway.config import (
    PlatformConfig,
    is_cron_disabled,
    is_runtime_driver_only,
)

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server import APIServerAdapter
from gateway.runtime_capability_auth import (
    RuntimeCapabilityConfig,
    RuntimeCapabilityVerifier,
)

RUNTIME_ROUTES = {
    ("GET", "/healthz"),
    ("GET", "/v1/runtime/manifest"),
    ("POST", "/v1/runtime/runs"),
    ("POST", "/v1/runtime/runs/{run_id}/tool-results"),
    ("POST", "/v1/runtime/runs/{run_id}/control-results"),
    ("POST", "/v1/runtime/runs/{run_id}/suspend"),
    ("POST", "/v1/runtime/runs/{run_id}/cancel"),
    ("POST", "/v1/runtime/runs/{run_id}/abort"),
}

# A representative sample of the billing-bypass surface that must disappear
# in runtime-driver-only mode.
BLOCKED_ROUTES = {
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/responses"),
    ("POST", "/api/sessions/{session_id}/chat"),
    ("GET", "/api/jobs"),
    ("POST", "/api/jobs"),
    ("POST", "/v1/runs"),
    ("GET", "/api/sessions"),
    ("GET", "/v1/models"),
}


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _registered_routes(adapter: APIServerAdapter) -> set:
    """Build the app, run _setup_routes, and return {(method, path)}."""
    adapter._app = web.Application()
    adapter._setup_routes()
    return {
        (route.method, route.resource.canonical)
        for route in adapter._app.router.routes()
        if route.method != "HEAD"  # aiohttp auto-adds HEAD for GET routes
    }


# ---------------------------------------------------------------------------
# Env flag helpers (gateway.config)
# ---------------------------------------------------------------------------


class TestEnvFlags:
    def test_defaults_are_off(self, monkeypatch):
        monkeypatch.delenv("HERMES_RUNTIME_DRIVER_ONLY", raising=False)
        monkeypatch.delenv("HERMES_DISABLE_CRON", raising=False)
        assert is_runtime_driver_only() is False
        assert is_cron_disabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE"])
    def test_runtime_driver_only_truthy(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", value)
        assert is_runtime_driver_only() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_runtime_driver_only_falsy(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", value)
        assert is_runtime_driver_only() is False

    def test_disable_cron_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_RUNTIME_DRIVER_ONLY", raising=False)
        monkeypatch.setenv("HERMES_DISABLE_CRON", "1")
        assert is_cron_disabled() is True
        assert is_runtime_driver_only() is False

    def test_runtime_driver_only_implies_cron_disabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_DISABLE_CRON", raising=False)
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
        assert is_cron_disabled() is True


# ---------------------------------------------------------------------------
# Route registration (_setup_routes)
# ---------------------------------------------------------------------------


class TestSetupRoutes:
    def test_default_mode_keeps_all_routes(self, monkeypatch):
        monkeypatch.delenv("HERMES_RUNTIME_DRIVER_ONLY", raising=False)
        routes = _registered_routes(_make_adapter())
        assert RUNTIME_ROUTES <= routes
        assert BLOCKED_ROUTES <= routes

    def test_runtime_only_registers_exactly_runtime_surface(self, monkeypatch):
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
        routes = _registered_routes(_make_adapter())
        assert routes == RUNTIME_ROUTES

    def test_runtime_only_blocks_billing_bypass_routes(self, monkeypatch):
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
        routes = _registered_routes(_make_adapter())
        assert routes.isdisjoint(BLOCKED_ROUTES)

    @pytest.mark.asyncio
    async def test_runtime_only_http_404_for_blocked_routes(self, monkeypatch):
        """Blocked entrypoints must 404 over real HTTP; /healthz must serve."""
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
        adapter = _make_adapter()
        adapter._app = web.Application()
        adapter._setup_routes()
        client = TestClient(TestServer(adapter._app))
        await client.start_server()
        try:
            resp = await client.get("/healthz")
            assert resp.status == 200

            for method, path in (
                ("POST", "/v1/chat/completions"),
                ("POST", "/v1/responses"),
                ("POST", "/api/sessions/abc/chat"),
                ("GET", "/api/jobs"),
                ("POST", "/v1/runs"),
            ):
                resp = await client.request(method, path)
                assert resp.status == 404, f"{method} {path} must 404"
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_runtime_manifest_requires_service_identity_without_operation_capability(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "1")
        adapter = _make_adapter()
        adapter._runtime_capability_verifier = RuntimeCapabilityVerifier(
            RuntimeCapabilityConfig(
                service_token="runtime-service-test",
                public_keys={},
            )
        )
        adapter._app = web.Application()
        adapter._setup_routes()
        client = TestClient(TestServer(adapter._app))
        await client.start_server()
        try:
            rejected = await client.get("/v1/runtime/manifest")
            assert rejected.status == 401

            accepted = await client.get(
                "/v1/runtime/manifest",
                headers={
                    "X-Ultra-Service-Authorization": "Bearer runtime-service-test",
                },
            )
            assert accepted.status == 200
            body = await accepted.json()
            assert body["runtime"] == "hermes"
            assert body["contract"]["schema_digests"][0].startswith("sha256:")
            assert "vision_llm_egress.v1" in body["features"]
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# Cron gating decision
# ---------------------------------------------------------------------------


class TestCronGate:
    """gateway.run gates the cron thread on is_cron_disabled(); the decision
    logic itself is what is unit-testable here (the thread start lives in the
    monolithic gateway main loop)."""

    def test_cron_runs_by_default(self, monkeypatch):
        monkeypatch.delenv("HERMES_RUNTIME_DRIVER_ONLY", raising=False)
        monkeypatch.delenv("HERMES_DISABLE_CRON", raising=False)
        assert is_cron_disabled() is False

    def test_cron_skipped_when_disable_cron_set(self, monkeypatch):
        monkeypatch.setenv("HERMES_DISABLE_CRON", "yes")
        assert is_cron_disabled() is True

    def test_cron_skipped_in_runtime_driver_only_mode(self, monkeypatch):
        monkeypatch.delenv("HERMES_DISABLE_CRON", raising=False)
        monkeypatch.setenv("HERMES_RUNTIME_DRIVER_ONLY", "true")
        assert is_cron_disabled() is True
