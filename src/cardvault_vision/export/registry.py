"""Production model registry (Phase 17).

Immutable, versioned model artifacts with the provenance needed to answer "what is
running in production, and can I reproduce it?".

Layout::

    exports/<task>/<version>/          # immutable artifact (see onnx_export)
    exports/<task>/production.json     # pointer to the promoted version

Rules enforced here:

* A version directory is **immutable**: re-exporting onto an existing version is
  an error, not a silent overwrite. Publishing ``1.0.1`` is cheap; discovering
  that ``1.0.0`` changed under you is expensive.
* Promotion requires a passing ``validation.json`` (Phase 15) and, optionally, a
  passing regression gate against the current production model (Phase 18).
* Every entry records the dataset version and hash, architecture, metrics, git
  commit, checksum and export format.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..errors import ExportError
from ..logging_utils import get_logger
from ..reproducibility import write_json

_LOG = get_logger("export.registry")

PRODUCTION_FILE = "production.json"
REGISTRY_FILE = "registry.json"


@dataclass
class ModelEntry:
    """One registered model version."""

    task: str
    version: str
    architecture: str
    path: str
    checksum_sha256: str
    exported_at: str
    dataset_version: str | None = None
    dataset_hash: str | None = None
    git_commit: str | None = None
    metrics: dict = field(default_factory=dict)
    format: str = "onnx"
    validated: bool = False
    promoted_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> ModelEntry:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def load_registry(exports_root: str | Path = "exports") -> dict:
    path = Path(exports_root) / REGISTRY_FILE
    if not path.exists():
        return {"schema_version": "1.0", "models": [], "productions": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(registry: dict, exports_root: str | Path = "exports") -> Path:
    return write_json(Path(exports_root) / REGISTRY_FILE, registry)


def register_export(
    export_dir: str | Path,
    *,
    exports_root: str | Path | None = None,
    metrics: dict | None = None,
    notes: str | None = None,
    allow_overwrite: bool = False,
) -> ModelEntry:
    """Register an exported artifact in the registry index."""
    directory = Path(export_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ExportError(f"Not an export directory (no metadata.json): {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = Path(exports_root) if exports_root else directory.parent.parent
    registry = load_registry(root)

    task = metadata["task"]
    version = metadata["version"]
    for entry in registry["models"]:
        if entry["task"] == task and entry["version"] == version:
            if not allow_overwrite:
                raise ExportError(
                    f"{task} {version} is already registered. Published versions are "
                    "immutable - bump the version instead (e.g. "
                    f"{_bump_patch(version)})."
                )
            registry["models"].remove(entry)

    validation_path = directory / "validation.json"
    validated = False
    if validation_path.exists():
        validated = json.loads(
            validation_path.read_text(encoding="utf-8")).get("status") == "APPROVED"

    entry = ModelEntry(
        task=task,
        version=version,
        architecture=metadata.get("architecture", "unknown"),
        path=str(directory),
        checksum_sha256=metadata.get("checksum_sha256", ""),
        exported_at=metadata.get("exported_at", ""),
        dataset_version=(metadata.get("dataset") or {}).get("version"),
        dataset_hash=(metadata.get("dataset") or {}).get("content_hash"),
        git_commit=metadata.get("git_commit"),
        metrics=metrics or metadata.get("metrics") or {},
        format=metadata.get("format", "onnx"),
        validated=validated,
        notes=notes,
    )
    registry["models"].append(entry.to_dict())
    registry["models"].sort(key=lambda item: (item["task"], _version_key(item["version"])))
    save_registry(registry, root)
    return entry


def _bump_patch(version: str) -> str:
    parts = (version.split(".") + ["0", "0"])[:3]
    try:
        parts[2] = str(int(parts[2]) + 1)
    except ValueError:
        parts[2] = "1"
    return ".".join(parts)


def _version_key(version: str) -> tuple:
    parts = []
    for piece in version.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts) + (0,) * (4 - len(parts))


def promote(
    task: str,
    version: str,
    *,
    exports_root: str | Path = "exports",
    require_validation: bool = True,
    gate_report: dict | None = None,
    force: bool = False,
) -> dict:
    """Point production at ``version`` for ``task``.

    Promotions are the only mutation the registry allows. ``gate_report`` is the
    output of a Phase 18 regression comparison; when supplied, a failing gate
    blocks the promotion unless ``force``.
    """
    root = Path(exports_root)
    registry = load_registry(root)
    entry = next((item for item in registry["models"]
                  if item["task"] == task and item["version"] == version), None)
    if entry is None:
        available = [item["version"] for item in registry["models"] if item["task"] == task]
        raise ExportError(
            f"{task} {version} is not registered. Available: {available or 'none'}"
        )
    if require_validation and not entry.get("validated"):
        raise ExportError(
            f"{task} {version} has no passing validation report. Run "
            "`vision export validate --path <dir>` first (or pass --force)."
        )
    if gate_report is not None and not gate_report.get("passed", False) and not force:
        raise ExportError(
            "regression gate failed, refusing to promote: "
            + "; ".join(gate_report.get("reasons", []))
        )

    previous = registry["productions"].get(task)
    entry["promoted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    registry["productions"][task] = {
        "version": entry["version"],
        "path": entry["path"],
        "checksum_sha256": entry["checksum_sha256"],
        "promoted_at": entry["promoted_at"],
        "previous_version": (previous or {}).get("version"),
    }
    save_registry(registry, root)
    write_json(root / task / PRODUCTION_FILE, registry["productions"][task])
    _LOG.info("Promoted %s %s to production (was %s)", task, version,
              (previous or {}).get("version", "none"))
    return registry["productions"][task]


def production_model(task: str, exports_root: str | Path = "exports") -> dict | None:
    """What CardVault should load for ``task``."""
    root = Path(exports_root)
    registry = load_registry(root)
    return registry.get("productions", {}).get(task)


def list_versions(task: str | None = None,
                  exports_root: str | Path = "exports") -> list[dict]:
    registry = load_registry(exports_root)
    models = registry.get("models", [])
    if task:
        models = [item for item in models if item["task"] == task]
    return models


def artifact_manifest(task: str, version: str,
                      exports_root: str | Path = "exports") -> dict:
    """Everything a consumer needs to load this model, in one document.

    This is the Phase 16 integration contract: a lightweight client can fetch this
    one JSON file and know which files to download, in which order, and how to
    verify them.
    """
    directory = Path(exports_root) / task / version
    if not directory.is_dir():
        raise ExportError(f"No such artifact: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    files = {}
    for name in ("model.onnx", "labels.json", "preprocessing.json", "contract.json"):
        path = directory / name
        if path.exists():
            files[name] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    return {
        "task": task,
        "version": version,
        "format": metadata.get("format", "onnx"),
        "checksum_sha256": metadata.get("checksum_sha256"),
        "input_spec": metadata.get("input_spec"),
        "output_spec": metadata.get("output_spec"),
        "files": files,
        "preprocessing": metadata.get("input_spec"),
        "validated": (directory / "validation.json").exists(),
        "requires": {"python": False, "pytorch": False},
        "notes": [
            "Load model.onnx with any ONNX Runtime; no Python training code needed.",
            "Decode outputs using contract.json.",
            "Preprocess using preprocessing.json.",
        ],
    }
