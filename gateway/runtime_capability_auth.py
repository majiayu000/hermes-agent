"""Fail-closed authentication for the private Ultra Studio Runtime contract."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


_AUDIENCE = "hermes-runtime"
_ISSUER = "agent-orchestrator"
_MAX_TTL_SECONDS = 120
_SERVICE_HEADER = "X-Ultra-Service-Authorization"
_EXPECTED_PRINCIPAL = {
    "user_id": "agent-orchestrator",
    "account_id": "ultrastudio-platform",
    "tenant_id": "ultrastudio-platform",
    "workspace_id": "foundation",
    "project_id": "runtime",
}


class RuntimeCapabilityError(ValueError):
    """A private Runtime request failed caller or operation authorization."""


@dataclass(frozen=True)
class RuntimeCapabilityConfig:
    service_token: str
    public_keys: dict[str, Ed25519PublicKey]

    @classmethod
    def from_env(cls) -> "RuntimeCapabilityConfig":
        service_token = os.getenv("RUNTIME_DRIVER_SERVICE_TOKEN", "").strip()
        raw_keys = os.getenv("RUNTIME_CAPABILITY_PUBLIC_KEYS", "").strip()
        if not service_token:
            raise RuntimeCapabilityError("RUNTIME_DRIVER_SERVICE_TOKEN is required")
        try:
            encoded_keys = json.loads(raw_keys)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeCapabilityError("RUNTIME_CAPABILITY_PUBLIC_KEYS must be a JSON object") from exc
        if not isinstance(encoded_keys, dict) or not encoded_keys:
            raise RuntimeCapabilityError("RUNTIME_CAPABILITY_PUBLIC_KEYS must contain at least one key")
        public_keys: dict[str, Ed25519PublicKey] = {}
        for key_id, encoded in encoded_keys.items():
            if not isinstance(key_id, str) or not key_id or not isinstance(encoded, str):
                raise RuntimeCapabilityError("runtime capability key entry is invalid")
            try:
                padded = encoded + "=" * (-len(encoded) % 4)
                raw = base64.urlsafe_b64decode(padded.encode("ascii"))
                public_keys[key_id] = Ed25519PublicKey.from_public_bytes(raw)
            except (ValueError, UnicodeError) as exc:
                raise RuntimeCapabilityError(f"runtime capability public key {key_id!r} is invalid") from exc
        return cls(service_token=service_token, public_keys=public_keys)


class RuntimeCapabilityVerifier:
    """Verify Ed25519 operation capabilities and reject token replay."""

    def __init__(self, config: RuntimeCapabilityConfig):
        self._config = config
        self._seen_jti: dict[str, int] = {}
        self._lock = threading.Lock()

    async def verify(self, request: Any) -> dict[str, Any]:
        service_header = request.headers.get(_SERVICE_HEADER, "")
        expected_service = "Bearer " + self._config.service_token
        if not hmac.compare_digest(service_header, expected_service):
            raise RuntimeCapabilityError("invalid runtime service credential")

        authorization = request.headers.get("Authorization", "")
        if not authorization.startswith("Bearer "):
            raise RuntimeCapabilityError("runtime capability is required")
        token = authorization[7:].strip()
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise RuntimeCapabilityError("runtime capability header is invalid") from exc
        key_id = header.get("kid")
        if header.get("alg") != "EdDSA" or header.get("typ") != "JWT" or key_id not in self._config.public_keys:
            raise RuntimeCapabilityError("runtime capability algorithm or key is invalid")
        try:
            claims = jwt.decode(
                token,
                self._config.public_keys[key_id],
                algorithms=["EdDSA"],
                audience=_AUDIENCE,
                issuer=_ISSUER,
                leeway=5,
                options={
                    "require": [
                        "iss", "aud", "azp", "sub", "iat", "nbf", "exp", "jti",
                        "capability_version", "principal", "run_id", "operation_id",
                        "action", "method", "path", "body_sha256",
                    ]
                },
            )
        except jwt.PyJWTError as exc:
            raise RuntimeCapabilityError("runtime capability signature or claims are invalid") from exc

        now = int(time.time())
        if (
            claims.get("azp") != _ISSUER
            or claims.get("sub") != _ISSUER
            or claims.get("capability_version") != 1
            or claims.get("principal") != _EXPECTED_PRINCIPAL
            or not isinstance(claims.get("iat"), int)
            or not isinstance(claims.get("exp"), int)
            or not isinstance(claims.get("nbf"), int)
            or claims["iat"] <= 0
            or claims["nbf"] > claims["iat"]
            or claims["exp"] <= claims["iat"]
            or claims["exp"] - claims["iat"] > _MAX_TTL_SECONDS
            or claims["iat"] > now + 5
        ):
            raise RuntimeCapabilityError("runtime capability authority is invalid")

        body = await request.read()
        path = request.rel_url.path
        run_id, action = self._operation_scope(path, body)
        body_digest = hashlib.sha256(body).hexdigest()
        expected_operation_id = f"{action}:{run_id}:{body_digest}"
        if (
            claims.get("run_id") != run_id
            or claims.get("operation_id") != expected_operation_id
            or claims.get("action") != action
            or claims.get("method") != request.method
            or claims.get("path") != path
            or claims.get("query", "") != request.rel_url.query_string
            or claims.get("body_sha256") != body_digest
        ):
            raise RuntimeCapabilityError("runtime capability does not bind this request")

        jti = claims.get("jti")
        if not isinstance(jti, str) or not jti:
            raise RuntimeCapabilityError("runtime capability id is invalid")
        with self._lock:
            self._seen_jti = {value: expiry for value, expiry in self._seen_jti.items() if expiry >= now}
            if jti in self._seen_jti:
                raise RuntimeCapabilityError("runtime capability was replayed")
            self._seen_jti[jti] = claims["exp"]
        return claims

    @staticmethod
    def _operation_scope(path: str, body: bytes) -> tuple[str, str]:
        if path == "/v1/runtime/runs":
            try:
                decoded = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeCapabilityError("runtime request body is invalid") from exc
            run_id = decoded.get("run_id") if isinstance(decoded, dict) else None
            action = "runtime.run.start"
        else:
            prefix = "/v1/runtime/runs/"
            if not path.startswith(prefix):
                raise RuntimeCapabilityError("runtime capability path is invalid")
            remainder = path[len(prefix):]
            parts = remainder.split("/", 1)
            if len(parts) != 2 or parts[1] not in {"tool-results", "control-results", "interrupt"}:
                raise RuntimeCapabilityError("runtime capability path is invalid")
            run_id, suffix = parts
            action = f"runtime.run.{suffix}"
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeCapabilityError("runtime run id is required")
        return run_id, action
