"""Shared paths, immutable run identity, and access to the original PFAD code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results/first_arrival_refinement/rebuild_v2"
REVISION = ROOT / "artifacts/pfad_tcsvt_revision_20260908"
SPECIMENS = tuple(f"specimen{i:02}" for i in range(1, 15))
ANATOMIES = ("foot", "tibia", "fibula")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_pfad():
    from pfad import core
    return core


def experiment_directory(config_path: Path) -> tuple[dict, Path]:
    config = json.loads(config_path.read_text())
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return config, REVISION / "experiments" / identity


def frozen_policies() -> dict[str, dict]:
    data = json.loads(
        (RESULT / "selector_attentive_loocv/nested_policy/loocv_protocol.json").read_text()
    )
    return {f["heldout_specimen"]: f["policy_selection"]["chosen"] for f in data["folds"]}
