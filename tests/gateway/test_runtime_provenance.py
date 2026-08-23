"""Release-provenance contract tests for the Hermes Runtime health surface."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import gateway.runtime_provenance as provenance
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.runtime_contract import RUNTIME_PROTOCOL_VERSION
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


def _config() -> RuntimeProvenanceConfig:
    return RuntimeProvenanceConfig(
        listen_host="127.0.0.1",
        listen_port=9120,
        cors_origins=("https://panel.example",),
        model_name="hermes-agent",
        max_concurrent_runs=10,
    )


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


def _installed_tree(tmp_path):
    root = tmp_path / "installed"
    (root / "gateway").mkdir(parents=True)
    (root / "hermes_cli").mkdir()
    (root / "tools").mkdir()
    (root / "gateway" / "runtime.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "hermes_cli" / "__init__.py").write_text(
        '__version__ = "1.2.3"\n',
        encoding="utf-8",
    )
    (root / "tools" / "tool.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "hermes-agent"\nversion = "1.2.3"\n',
        encoding="utf-8",
    )
    return root


def _write_metadata(root, **overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": 1,
        "git_commit": "a" * 40,
        "release_id": "1.2.3",
        "build_timestamp": "2026-07-23T01:02:03Z",
        "runtime_sha256": provenance._runtime_artifact_digest(root),
    }
    payload.update(overrides)
    (root / provenance.BUILD_METADATA_FILENAME).write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _collect(root, **kwargs):
    return collect_runtime_provenance(
        _config(),
        startup_timestamp=datetime(2026, 7, 23, 1, 2, 3, tzinfo=timezone.utc),
        project_root=root,
        version="1.2.3",
        schema_version="17",
        **kwargs,
    )


def _copy_build_cli(root) -> Path:
    script = root / "gateway" / "runtime_provenance.py"
    shutil.copy2(Path(provenance.__file__), script)
    return script


def test_clean_source_keeps_live_git_identity_and_is_incomplete(tmp_path):
    root = _source_checkout(tmp_path)

    manifest = _collect(root)

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
    clean = _collect(root)

    (root / "gateway" / "runtime.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = _collect(root)

    assert dirty.source_dirty is True
    assert dirty.binary_sha256 != clean.binary_sha256
    assert dirty.provenance_complete is False
    assert "source checkout is dirty" in dirty.provenance_errors


def test_partial_metadata_never_upgrades_a_clean_source_checkout(tmp_path):
    root = _source_checkout(tmp_path)
    (root / ".git" / "info" / "exclude").write_text(
        f"/{provenance.BUILD_METADATA_FILENAME}\n",
        encoding="utf-8",
    )
    (root / provenance.BUILD_METADATA_FILENAME).write_text(
        '{"format_version": 1, "git_commit": "secret"}\n',
        encoding="utf-8",
    )

    manifest = _collect(root)

    assert manifest.source_dirty is False
    assert manifest.git_commit == _git(root, "rev-parse", "HEAD")
    assert manifest.release_id == "development"
    assert manifest.provenance_complete is False


def test_build_cli_supports_development_and_legacy_sha_only(tmp_path, monkeypatch):
    root = _installed_tree(tmp_path)
    monkeypatch.setattr(provenance, "_PROJECT_ROOT", root)
    metadata_path = root / provenance.BUILD_METADATA_FILENAME
    legacy_sha_path = root / provenance.LEGACY_BUILD_SHA_FILENAME
    metadata_path.write_text("stale metadata\n", encoding="utf-8")
    legacy_sha_path.write_text("stale sha\n", encoding="ascii")

    assert provenance._build_cli(["stamp-build", "", "", ""]) == 0

    assert not metadata_path.exists()
    assert not legacy_sha_path.exists()

    commit = "a" * 40
    assert provenance._build_cli(["stamp-build", commit, "", ""]) == 0

    assert not metadata_path.exists()
    assert legacy_sha_path.read_text(encoding="ascii") == f"{commit}\n"
    assert not [path for path in root.iterdir() if path.name.startswith("..hermes_")]


@pytest.mark.parametrize(
    ("git_commit", "release_id", "build_timestamp"),
    [
        ("a" * 40, "1.2.3", ""),
        ("a" * 40, "development", "2026-07-23T01:02:03Z"),
    ],
)
def test_build_cli_rejects_partial_or_invalid_release_inputs(
    tmp_path,
    monkeypatch,
    capsys,
    git_commit: str,
    release_id: str,
    build_timestamp: str,
):
    root = _installed_tree(tmp_path)
    monkeypatch.setattr(provenance, "_PROJECT_ROOT", root)

    assert (
        provenance._build_cli([
            "stamp-build",
            git_commit,
            release_id,
            build_timestamp,
        ])
        == 2
    )
    stderr = capsys.readouterr().err

    assert str(root) not in stderr
    assert git_commit not in stderr
    assert not (root / provenance.BUILD_METADATA_FILENAME).exists()
    assert not (root / provenance.LEGACY_BUILD_SHA_FILENAME).exists()


def test_complete_installed_metadata_reports_exact_verified_fields(tmp_path):
    root = _installed_tree(tmp_path)
    script = _copy_build_cli(root)
    commit = "a" * 40
    timestamp = "2026-07-23T01:02:03Z"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "stamp-build",
            commit,
            "1.2.3",
            timestamp,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(
        (root / provenance.BUILD_METADATA_FILENAME).read_text(encoding="utf-8")
    )

    manifest = _collect(root)

    assert set(payload) == {
        "format_version",
        "git_commit",
        "release_id",
        "build_timestamp",
        "runtime_sha256",
    }
    assert manifest.release_id == payload["release_id"]
    assert manifest.git_commit == payload["git_commit"]
    assert manifest.build_timestamp == payload["build_timestamp"]
    assert manifest.binary_sha256 == payload["runtime_sha256"]
    assert manifest.source_dirty is False
    assert manifest.provenance_complete is True
    assert manifest.provenance_errors == ()
    assert (root / provenance.LEGACY_BUILD_SHA_FILENAME).read_text(
        encoding="ascii"
    ) == f"{commit}\n"


def test_runtime_digest_is_deterministic_and_excludes_metadata(tmp_path):
    root = _installed_tree(tmp_path)
    first = provenance._runtime_artifact_digest(root)
    _write_metadata(root, release_id="release-a")
    second = provenance._runtime_artifact_digest(root)

    (root / "gateway" / "runtime.py").write_text("VALUE = 3\n", encoding="utf-8")
    third = provenance._runtime_artifact_digest(root)

    assert first.startswith("sha256:")
    assert second == first
    assert third != first


def test_missing_metadata_fails_closed_for_an_installed_tree(tmp_path):
    root = _installed_tree(tmp_path)
    legacy_sha = "a" * 40
    (root / ".hermes_build_sha").write_text(f"{legacy_sha}\n", encoding="utf-8")

    manifest = _collect(root)

    assert manifest.release_id == "development"
    assert manifest.git_commit == "unknown"
    assert manifest.build_timestamp == "unknown"
    assert manifest.binary_sha256 == "unknown"
    assert manifest.source_dirty is False
    assert manifest.provenance_complete is False
    assert "build metadata unavailable" in manifest.provenance_errors
    assert manifest.git_commit != legacy_sha


def test_legacy_sha_file_alone_cannot_fabricate_release_fields(tmp_path):
    root = _installed_tree(tmp_path)
    (root / ".hermes_build_sha").write_text("legacy-file-content\n", encoding="utf-8")

    manifest = _collect(root)

    assert manifest.git_commit == "unknown"
    assert manifest.release_id == "development"
    assert manifest.build_timestamp == "unknown"
    assert "legacy-file-content" not in json.dumps(manifest.to_dict())


@pytest.mark.parametrize(
    "metadata_text",
    [
        "{not-json",
        '{"format_version": 1, "git_commit": "a"}',
        json.dumps({
            "format_version": 1,
            "git_commit": "a" * 40,
            "release_id": "unknown",
            "build_timestamp": "2026-07-23T01:02:03Z",
            "runtime_sha256": "sha256:" + "a" * 64,
        }),
        json.dumps({
            "format_version": 1,
            "git_commit": "a" * 40,
            "release_id": "1.2.3",
            "build_timestamp": "not-a-timestamp",
            "runtime_sha256": "sha256:" + "a" * 64,
            "credential": "do-not-return-me",
        }),
        "x" * (provenance._BUILD_METADATA_MAX_BYTES + 1),
    ],
)
def test_malformed_or_partial_metadata_cannot_fabricate_provenance(
    tmp_path,
    metadata_text: str,
):
    root = _installed_tree(tmp_path)
    (root / provenance.BUILD_METADATA_FILENAME).write_text(
        metadata_text,
        encoding="utf-8",
    )

    manifest = _collect(root)

    assert manifest.release_id == "development"
    assert manifest.git_commit == "unknown"
    assert manifest.build_timestamp == "unknown"
    assert manifest.binary_sha256 == "unknown"
    assert manifest.provenance_complete is False
    assert "do-not-return-me" not in json.dumps(manifest.to_dict())


def test_digest_mismatch_invalidates_a_build_record(tmp_path):
    root = _installed_tree(tmp_path)
    _write_metadata(root, runtime_sha256="sha256:" + "b" * 64)

    manifest = _collect(root)

    assert manifest.release_id == "development"
    assert manifest.git_commit == "unknown"
    assert manifest.build_timestamp == "unknown"
    assert manifest.binary_sha256 == "unknown"
    assert manifest.source_dirty is False
    assert manifest.provenance_complete is False
    assert "runtime artifact digest mismatch" in manifest.provenance_errors


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
    assert body["runtime_protocol_version"] == RUNTIME_PROTOCOL_VERSION
    assert "error" in body["runtime_frame_types"]
    assert {
        "abort_attempt",
        "cancel_run",
        "delegated_tools",
        "suspend_attempt",
        "system_context.replace",
    } <= set(body["runtime_capabilities"])
    assert "interrupt" not in body["runtime_capabilities"]
    assert body["status"] == "ok"
    assert secret not in json.dumps(body)


@pytest.mark.asyncio
async def test_healthz_reports_complete_build_contract(monkeypatch, tmp_path):
    root = _installed_tree(tmp_path)
    provenance.stamp_build_provenance(
        root,
        "a" * 40,
        "1.2.3",
        "2026-07-23T01:02:03Z",
    )
    payload = json.loads(
        (root / provenance.BUILD_METADATA_FILENAME).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(provenance, "_PROJECT_ROOT", root)
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"host": "127.0.0.1", "port": 9120, "key": "secret"},
        )
    )

    body = json.loads((await adapter._handle_health(None)).text)

    assert body["release_id"] == payload["release_id"]
    assert body["git_commit"] == payload["git_commit"]
    assert body["build_timestamp"] == payload["build_timestamp"]
    assert body["binary_sha256"] == payload["runtime_sha256"]
    assert body["source_dirty"] is False
    assert body["provenance_complete"] is True
