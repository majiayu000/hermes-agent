import base64
import hashlib
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.runtime_capability_auth import (
    RuntimeCapabilityConfig,
    RuntimeCapabilityError,
    RuntimeCapabilityVerifier,
)


@pytest.fixture
def capability(monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    public_raw = private_key.public_key().public_bytes_raw()
    monkeypatch.setenv("RUNTIME_DRIVER_SERVICE_TOKEN", "runtime-service-test")
    monkeypatch.setenv(
        "RUNTIME_CAPABILITY_PUBLIC_KEYS",
        json.dumps({"orchestrator-test": base64.urlsafe_b64encode(public_raw).decode().rstrip("=")}),
    )
    return private_key


def _claims(body: bytes, *, now: int | None = None) -> dict:
    issued_at = int(time.time()) if now is None else now
    return {
        "iss": "agent-orchestrator",
        "aud": "hermes-runtime",
        "azp": "agent-orchestrator",
        "sub": "agent-orchestrator",
        "iat": issued_at,
        "nbf": issued_at - 5,
        "exp": issued_at + 120,
        "jti": "runtime-test-jti",
        "capability_version": 1,
        "principal": {
            "user_id": "agent-orchestrator",
            "account_id": "ultrastudio-platform",
            "tenant_id": "ultrastudio-platform",
            "workspace_id": "foundation",
            "project_id": "runtime",
        },
        "run_id": "run_auth_test",
        "operation_id": f"runtime.run.start:run_auth_test:{hashlib.sha256(body).hexdigest()}",
        "action": "runtime.run.start",
        "method": "POST",
        "path": "/v1/runtime/runs",
        "body_sha256": hashlib.sha256(body).hexdigest(),
    }


def _token(private_key, claims: dict, *, algorithm: str = "EdDSA") -> str:
    key = private_key if algorithm == "EdDSA" else "shared-secret-used-only-for-negative-test"
    return jwt.encode(
        claims,
        key,
        algorithm=algorithm,
        headers={"kid": "orchestrator-test", "typ": "JWT"},
    )


async def _client(verifier: RuntimeCapabilityVerifier) -> TestClient:
    async def verify(request):
        try:
            await verifier.verify(request)
        except RuntimeCapabilityError:
            return web.Response(status=401)
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/v1/runtime/runs", verify)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_runtime_capability_accepts_exact_ed25519_request_and_rejects_replay(capability):
    body = b'{"run_id":"run_auth_test"}'
    verifier = RuntimeCapabilityVerifier(RuntimeCapabilityConfig.from_env())
    client = await _client(verifier)
    token = _token(capability, _claims(body))
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Ultra-Service-Authorization": "Bearer runtime-service-test",
        "Content-Type": "application/json",
    }
    try:
        assert (await client.post("/v1/runtime/runs", data=body, headers=headers)).status == 204
        assert (await client.post("/v1/runtime/runs", data=body, headers=headers)).status == 401
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["hs256", "expired", "wrong_body", "wrong_service"])
async def test_runtime_capability_rejects_invalid_algorithm_expiry_scope_and_caller(capability, failure):
    body = b'{"run_id":"run_auth_test"}'
    claims = _claims(body)
    algorithm = "EdDSA"
    sent_body = body
    service = "runtime-service-test"
    if failure == "hs256":
        algorithm = "HS256"
    elif failure == "expired":
        claims = _claims(body, now=int(time.time()) - 300)
    elif failure == "wrong_body":
        sent_body = b'{"run_id":"run_auth_test","extra":true}'
    elif failure == "wrong_service":
        service = "wrong-service"
    verifier = RuntimeCapabilityVerifier(RuntimeCapabilityConfig.from_env())
    client = await _client(verifier)
    try:
        response = await client.post(
            "/v1/runtime/runs",
            data=sent_body,
            headers={
                "Authorization": f"Bearer {_token(capability, claims, algorithm=algorithm)}",
                "X-Ultra-Service-Authorization": f"Bearer {service}",
                "Content-Type": "application/json",
            },
        )
        assert response.status == 401
    finally:
        await client.close()


def test_runtime_capability_config_fails_closed_without_independent_inputs(monkeypatch):
    monkeypatch.delenv("RUNTIME_DRIVER_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("RUNTIME_CAPABILITY_PUBLIC_KEYS", raising=False)
    with pytest.raises(RuntimeCapabilityError):
        RuntimeCapabilityConfig.from_env()
