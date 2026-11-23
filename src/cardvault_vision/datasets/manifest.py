"""Dataset manifest: the versioned index of a dataset.

A manifest is the *only* thing downstream code needs. It records:

* what the dataset is (kind, name, version, description),
* where its files live (relative paths, resolved against ``root``),
* every sample with its annotation and filesize/hash,
* full provenance (source manifest, config fingerprint, seed, git commit),
* a **content hash** over all ``(image_id, sha256)`` pairs.

Because the content hash is order-independent, two independently generated
shards of the same job produce a manifest with the same content hash — which is
how "same SOURCE + CONFIG + SEED" is verified in CI.

Manifest kinds
--------------
``source``    clean card images + ground truth (input to the synthetic engine)
``real``      real photographs with annotations
``synthetic`` generated captures with automatic ground truth
``golden``    the permanent, immutable evaluation set (Phase 19)
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import canonical_json, fingerprint
from ..errors import DatasetError
from ..logging_utils import get_logger
from ..reproducibility import ReproRecord, write_json
from ..version import SCHEMA_VERSION
from .hashing import dataset_content_hash
from .schema import SampleRecord

_LOG = get_logger("datasets.manifest")

MANIFEST_KINDS = ("source", "real", "synthetic", "golden")
MANIFEST_FILENAME = "manifest.json"


@dataclass
class DatasetVersion:
    """Version metadata for a dataset.

    ``version`` is human-facing; ``content_hash`` is the machine truth. A
    golden dataset also carries ``frozen_at`` and must never change after it.
    """

    version: str = "0.0.0"
    content_hash: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    frozen_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        out = {"version": self.version, "created_at": self.created_at}
        if self.content_hash:
            out["content_hash"] = self.content_hash
        if self.frozen_at:
            out["frozen_at"] = self.frozen_at
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetVersion:
        return cls(
            version=str(data.get("version", "0.0.0")),
            content_hash=data.get("content_hash"),
            created_at=str(data.get("created_at", "")),
            frozen_at=data.get("frozen_at"),
            notes=data.get("notes"),
        )

    @property
    def is_frozen(self) -> bool:
        return self.frozen_at is not None


@dataclass
class DatasetManifest:
    """Versioned index of a dataset on disk."""

    name: str
    kind: str = "source"
    root: Path = field(default_factory=Path)
    samples: list[SampleRecord] = field(default_factory=list)
    version: DatasetVersion = field(default_factory=DatasetVersion)
    provenance: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    #: Path of the manifest itself, when loaded from disk.
    path: Path | None = None

    # -- construction -----------------------------------------------------
    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.kind not in MANIFEST_KINDS:
            raise DatasetError(f"Unknown manifest kind {self.kind!r}; expected {MANIFEST_KINDS}")

    @classmethod
    def create(
        cls,
        name: str,
        *,
        kind: str = "source",
        root: str | Path = ".",
        version: str = "0.0.0",
        description: str | None = None,
    ) -> DatasetManifest:
        return cls(
            name=name,
            kind=kind,
            root=Path(root),
            version=DatasetVersion(version=version, notes=description),
            metadata={"description": description} if description else {},
        )

    # -- sample management ------------------------------------------------
    def add(self, sample: SampleRecord) -> None:
        if self.version.is_frozen:
            raise DatasetError(
                f"Dataset {self.name!r} v{self.version.version} is frozen "
                f"(frozen_at={self.version.frozen_at}); frozen datasets are immutable"
            )
        self.samples.append(sample)

    def extend(self, samples: Iterable[SampleRecord]) -> None:
        for sample in samples:
            self.add(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self.samples)

    def get(self, image_id: str) -> SampleRecord:
        for sample in self.samples:
            if sample.image_id == image_id:
                return sample
        raise KeyError(image_id)

    # -- integrity --------------------------------------------------------
    @property
    def card_ids(self) -> set[str]:
        return {s.card_id for s in self.samples}

    def content_hash(self) -> str:
        """Order-independent hash over every ``(image_id, sha256)``."""
        missing = [s.image_id for s in self.samples if not s.sha256]
        if missing:
            raise DatasetError(
                f"{len(missing)} sample(s) lack a sha256 (e.g. {missing[0]!r}); "
                "run `vision dataset validate --fix` to populate hashes"
            )
        return dataset_content_hash([(s.image_id, s.sha256 or "") for s in self.samples])

    def config_hash(self) -> str:
        return fingerprint(self.config) if self.config else ""

    def seal(self, *, version: str | None = None) -> DatasetManifest:
        """Compute and stamp the content hash (call once generation is complete)."""
        if version:
            self.version.version = version
        self.version.content_hash = self.content_hash()
        return self

    # -- persistence ------------------------------------------------------
    def to_dict(self, *, with_annotations: bool = True) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "kind": self.kind,
            "root": str(self.root),
            "version": self.version.to_dict(),
            "counts": {
                "samples": len(self.samples),
                "cards": len(self.card_ids),
            },
            "config": self.config,
            "config_fingerprint": self.config_hash() or None,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "samples": [s.to_dict(with_annotation=with_annotations) for s in self.samples],
        }

    def save(
        self,
        path: str | Path | None = None,
        *,
        with_annotations: bool = True,
        compute_hash: bool = True,
    ) -> Path:
        """Write the manifest atomically. Returns the path written."""
        if compute_hash and self.samples and any(s.sha256 for s in self.samples):
            self.version.content_hash = self.content_hash()
        target = Path(path) if path else (self.root / MANIFEST_FILENAME)
        write_json(target, self.to_dict(with_annotations=with_annotations))
        self.path = target
        return target

    @classmethod
    def load(cls, path: str | Path, *, root: str | Path | None = None,
             lazy: bool = False) -> DatasetManifest:
        """Load a manifest.

        ``root`` overrides the recorded root, which makes a dataset relocatable
        (essential for CI and for mounting datasets outside the repo).
        """
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_FILENAME
        if not p.exists():
            raise DatasetError(f"Manifest not found: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Manifest {p} is not valid JSON: {exc}") from exc
        schema = str(data.get("schema_version", "0.0"))
        if schema.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            _LOG.warning("Manifest %s has schema %s (this build expects %s)", p, schema, SCHEMA_VERSION)
        resolved_root = Path(root) if root is not None else Path(data.get("root", p.parent))
        manifest = cls(
            name=str(data.get("name", p.stem)),
            kind=str(data.get("kind", "source")),
            root=resolved_root,
            samples=[] if lazy else [SampleRecord.from_dict(s) for s in data.get("samples", [])],
            version=DatasetVersion.from_dict(data.get("version", {})),
            provenance=dict(data.get("provenance") or {}),
            config=dict(data.get("config") or {}),
            metadata=dict(data.get("metadata") or {}),
            schema_version=schema,
            path=p,
        )
        if lazy:
            manifest.metadata["_raw_samples"] = data.get("samples", [])
        return manifest

    # -- helpers ----------------------------------------------------------
    def resolve(self, image_path: str) -> Path:
        """Resolve a manifest-relative image path against ``root``."""
        candidate = Path(image_path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def attach_provenance(self, record: ReproRecord, **extra: Any) -> DatasetManifest:
        self.provenance = {**record.to_dict(), **extra}
        return self

    def summary(self) -> dict:
        """Compact description for logs and CLI output."""
        return {
            "name": self.name,
            "kind": self.kind,
            "version": self.version.version,
            "samples": len(self.samples),
            "cards": len(self.card_ids),
            "content_hash": self.version.content_hash,
            "root": str(self.root),
        }

    def manifest_checksum(self) -> str:
        """Checksum of the manifest body (excluding volatile fields)."""
        body = self.to_dict()
        body.pop("version", None)
        return fingerprint(canonical_json(body), length=32)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DatasetManifest(name={self.name!r}, kind={self.kind!r}, "
                f"samples={len(self.samples)}, version={self.version.version!r})")
