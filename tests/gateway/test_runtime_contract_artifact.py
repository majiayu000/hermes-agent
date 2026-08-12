from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from gateway.runtime_contract import (
    RUNTIME_CONTRACT_MAJOR,
    RUNTIME_CONTRACT_MINOR,
    RUNTIME_CONTRACT_SCHEMA_DIGEST,
    RUNTIME_DRIVER_FRAME_TYPES,
    RUNTIME_MANIFEST_FEATURES,
    RUNTIME_RUN_INTENTS,
)
from scripts.verify_runtime_contract_projection import (
    LOCAL_ROOT,
    SCHEMA_NAMES,
    bundle_digest,
    verify_projection,
)


def test_runtime_contract_projection_is_complete_and_self_consistent():
    schema_dir = LOCAL_ROOT / "v1"
    assert tuple(sorted(path.name for path in schema_dir.glob("*.schema.json"))) == SCHEMA_NAMES
    assert bundle_digest(schema_dir) == RUNTIME_CONTRACT_SCHEMA_DIGEST

    for name in SCHEMA_NAMES:
        document = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        metadata = document["x-ultra-contract"]
        assert metadata["major"] == RUNTIME_CONTRACT_MAJOR
        assert metadata["minor"] == RUNTIME_CONTRACT_MINOR

    compatibility = yaml.safe_load(
        (LOCAL_ROOT / "compatibility.yaml").read_text(encoding="utf-8")
    )
    assert tuple(sorted(compatibility["schemas"])) == SCHEMA_NAMES
    assert tuple(compatibility["intents"]) == RUNTIME_RUN_INTENTS
    assert tuple(compatibility["features"]) == RUNTIME_MANIFEST_FEATURES
    assert tuple(compatibility["runtime_frame_types"]) == RUNTIME_DRIVER_FRAME_TYPES


def test_projection_gate_accepts_exact_artifact_and_rejects_drift(tmp_path: Path):
    canonical = tmp_path / "runtime"
    shutil.copytree(LOCAL_ROOT, canonical)
    assert verify_projection(canonical) == RUNTIME_CONTRACT_SCHEMA_DIGEST

    drifted = canonical / "v1" / "error.schema.json"
    drifted.write_bytes(drifted.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="error.schema.json"):
        verify_projection(canonical)


def test_bundle_digest_binds_file_names_and_bytes(tmp_path: Path):
    schema_dir = tmp_path / "v1"
    schema_dir.mkdir()
    for name in SCHEMA_NAMES:
        (schema_dir / name).write_bytes(name.encode("utf-8"))
    first = bundle_digest(schema_dir)
    (schema_dir / SCHEMA_NAMES[0]).write_bytes(b"changed")
    second = bundle_digest(schema_dir)
    assert first != second
    assert first == "sha256:" + hashlib.sha256(
        b"".join(
            name.encode("utf-8") + b"\0" + name.encode("utf-8") + b"\0"
            for name in SCHEMA_NAMES
        )
    ).hexdigest()
