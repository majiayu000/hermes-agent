"""Release-provenance contract tests for the Hermes Runtime health surface."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.runtime_provenance import (
    RuntimeProvenanceConfig,
    collect_runtime_provenance,
)


def _git(root, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_checkout(tmp_path):
    (tmp_path / "gateway").mkdir()
    (tmp_path / "hermes_cli").mkdir()
    (tmp_path / "gateway" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "hermes_cli" / "__init__.py").write_text(
        '__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "tests@example.invalid")
    _git(tmp_path, "config", "user.name", "Hermes Tests")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "fixture")
    return tmp_path


def test_collects_clean_source_identity_and_artifact_digest(tmp_path):
    root = _source_checkout(tmp_path)
    config = RuntimeProvenanceConfig(
        listen_host="127.0.0.1",
        listen_port=9120,
        cors_origins=("https://panel.example",),
        model_name="hermes-agent",
        max_concurrent_runs=10,
    )

    manifest = collect_runtime_provenance(
        config,
        startup_timestamp=datetime(2026, 7, 23, 1, 2, 3, tzinfo=timezone.utc),
        project_root=root,
        version="1.2.3",
        schema_version="17",
    )

    assert manifest.service_name == "hermes-runtime"
    assert manifest.release_id == "development"
    assert manifest.git_commit == _git(root, "rev-parse", "HEAD")
    assert manifest.build_timestamp == "unknown"
    assert manifest.binary_sha256.startswith("sha256:")
    assert manifest.schema_version == "17"
    assert manifest.config_digest.startswith("sha256:")
    assert manifest.startup_timestamp == "2026-07-23T01:02:03Z"
    assert manifest.source_dirty is False
    assert manifest.provenance_complete is False
    assert set(manifest.provenance_errors) == {
        "release identity unavailable",
        "build_timestamp unavailable",
    }


def test_dirty_source_changes_digest_and_is_never_complete(tmp_path):
    root = _source_checkout(tmp_path)
    config = RuntimeProvenanceConfig(
        listen_host="127.0.0.1",
        listen_port=9120,
        cors_origins=(),
        model_name="hermes-agent",
        max_concurrent_runs=10,
    )
    startup = datetime(2026, 7, 23, tzinfo=timezone.utc)
    clean = collect_runtime_provenance(
        config,
        startup_timestamp=startup,
        project_root=root,
        version="1.2.3",
        schema_version="17",
    )

    (root / "gateway" / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = collect_runtime_provenance(
        config,
        startup_timestamp=startup,
        project_root=root,
        version="1.2.3",
        schema_version="17",
    )

    assert dirty.source_dirty is True
    assert dirty.binary_sha256 != clean.binary_sha256
    assert dirty.provenance_complete is False
    assert "source checkout is dirty" in dirty.provenance_errors


@pytest.mark.asyncio
async def test_healthz_exposes_provenance_without_api_key():
    secret = "runtime-secret-that-must-never-appear"
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 9120,
                "key": secret,
                "cors_origins": ["https://panel.example"],
            },
        )
    )
    app = web.Application()
    app.router.add_get("/healthz", adapter._handle_health)

    async with TestClient(TestServer(app)) as client:
        response = await client.get("/healthz")
        assert response.status == 200
        body = await response.json()

    assert {
        "service_name",
        "release_id",
        "version",
        "git_commit",
        "build_timestamp",
        "binary_sha256",
        "schema_version",
        "config_digest",
        "startup_timestamp",
        "source_dirty",
        "provenance_complete",
        "provenance_errors",
    } <= body.keys()
    assert body["service_name"] == "hermes-runtime"
    assert body["runtime_protocol_version"] == "2"
    assert "error" in body["runtime_frame_types"]
    assert {
        "delegated_tools",
        "interrupt",
        "system_context.replace",
    } <= set(body["runtime_capabilities"])
    assert body["status"] == "ok"
    assert secret not in json.dumps(body)
