"""Reproducibility framework.

The promise CardVault Vision makes is not "we set a seed somewhere" — it is
that a dataset or a model can be **rebuilt bit-for-bit** from a small record:

    source manifest + config + seed + code version

This module provides:

* :func:`seed_everything` - seed Python, NumPy and (when installed) torch, and
  optionally force deterministic kernels.
* :class:`ReproRecord` - the provenance block written next to every dataset and
  model: code version, git commit, config fingerprint, seeds, hardware, and
  library versions.
* :func:`set_deterministic_algorithms` - strict mode for benchmark comparisons.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT, canonical_json, fingerprint
from .logging_utils import get_logger
from .rng import normalize_seed
from .version import SCHEMA_VERSION, __version__

_LOG = get_logger("repro")


def git_commit(short: bool = False) -> str | None:
    """Current git commit, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", *(["--short"] if short else []), "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    commit = out.stdout.strip()
    return commit or None


def git_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes (``None`` if unknown)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def library_versions() -> dict[str, str]:
    """Versions of the libraries that can change results."""
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except Exception:  # pragma: no cover
        pass
    for mod_name, key in (("torch", "torch"), ("cv2", "opencv"), ("onnx", "onnx"),
                          ("onnxruntime", "onnxruntime")):
        try:
            module = __import__(mod_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except Exception:
            continue
    return versions


@dataclass
class ReproRecord:
    """Provenance written alongside every generated dataset and exported model."""

    seed: int
    config_fingerprint: str | None = None
    source_fingerprint: str | None = None
    code_version: str = __version__
    schema_version: str = SCHEMA_VERSION
    git_commit: str | None = None
    git_dirty: bool | None = None
    python: str = field(default_factory=lambda: platform.python_version())
    platform: str = field(default_factory=platform.platform)
    libraries: dict[str, str] = field(default_factory=library_versions)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        seed: int | str | None,
        *,
        config: Any = None,
        source: Any = None,
        **extra: Any,
    ) -> ReproRecord:
        return cls(
            seed=normalize_seed(seed),
            config_fingerprint=fingerprint(config) if config is not None else None,
            source_fingerprint=fingerprint(source) if source is not None else None,
            git_commit=git_commit(),
            git_dirty=git_dirty(),
            extra=extra,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


def seed_everything(seed: int | str | None, *, deterministic: bool = False,
                    warn_only: bool = True) -> int:
    """Seed every RNG we can reach and return the normalised seed.

    ``deterministic=True`` also disables cuDNN benchmarking and asks torch for
    deterministic algorithms. That can be slower, and some ops have no
    deterministic implementation — ``warn_only`` decides whether that is a
    warning or an error.
    """
    seed = normalize_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed % (2**63))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**63))
        if deterministic:
            set_deterministic_algorithms(warn_only=warn_only)
    except ImportError:
        pass
    _LOG.debug("Seeded global RNGs with %d (deterministic=%s)", seed, deterministic)
    return seed


def set_deterministic_algorithms(warn_only: bool = True) -> None:
    """Force torch into deterministic mode (no-op without torch)."""
    try:
        import torch
    except ImportError:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except Exception as exc:  # pragma: no cover
        _LOG.warning("Could not enable deterministic algorithms: %s", exc)


def assert_reproducible(record: ReproRecord, expected: dict) -> None:
    """Raise when a run's provenance diverges from what was recorded.

    Used by ``vision dataset validate --reproduce`` and by regression tests that
    must prove a dataset can be regenerated.
    """
    from .errors import ValidationError

    mismatches = []
    for key, want in expected.items():
        got = record.to_dict().get(key)
        if got != want:
            mismatches.append(f"{key}: recorded={want!r} actual={got!r}")
    if mismatches:
        raise ValidationError("Reproducibility check failed:\n  " + "\n  ".join(mismatches))


def write_json(path: str | os.PathLike, payload: Any) -> Path:
    """Atomically write indented JSON (``.tmp`` then rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(p)
    return p


def read_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def executable_versions() -> dict:
    """Environment snapshot used by benchmarks."""
    from .hardware import detect_hardware

    return {
        "sys": sys.version,
        "hardware": detect_hardware().to_dict(),
        "libraries": library_versions(),
        "git_commit": git_commit(),
    }
