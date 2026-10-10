"""Fail-closed helpers shared by the source-strict reproduction runners.

The runners deliberately stop when a source-defined value is absent.  A
completion value can be used in an exploratory audit, but it must never enter
the source-faithful reproduction path silently.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class MissingSourceEvidence(RuntimeError):
    """Raised when a strict reproduction contract is not closed."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MissingSourceEvidence(f"missing required contract: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MissingSourceEvidence(f"contract must be a JSON object: {path}")
    return value


def require_resolved(config: dict[str, Any], fields: Iterable[str]) -> None:
    unresolved = []
    for field in fields:
        value = config.get(field)
        if value is None or value == "UNRESOLVED" or value == "REQUEST_AUTHOR":
            unresolved.append(field)
    if unresolved:
        raise MissingSourceEvidence(
            "strict reproduction is blocked; unresolved source fields: "
            + ", ".join(unresolved)
        )


def write_failure(path: Path, *, method: str, message: str, config_path: Path) -> None:
    payload = {
        "schema": "source-strict.failure.v1",
        "method": method,
        "status": "BLOCKED",
        "message": message,
        "config": str(config_path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_base(*, run_id: str, method: str, config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "source-strict.run-manifest.v1",
        "run_id": run_id,
        "method": method,
        "status": "PENDING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "configuration": config,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_faithful_claim_allowed": False,
    }
