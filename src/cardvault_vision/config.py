"""Layered configuration system.

Configuration in CardVault Vision is *data*, not code, and it is resolved in a
fixed layering order so a run is always explainable:

    1. built-in defaults (this module / the owning dataclass)
    2. config files, left to right  (later files win)
    3. CLI ``--set key=value`` overrides (dotted paths win last)

Every resolved config can emit a **fingerprint**: a BLAKE2b hash of its
canonical JSON form. That fingerprint is what makes "the same SOURCE + CONFIG +
SEED produces the same dataset" checkable — it is written into every synthetic
manifest and every exported model's metadata.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

def _find_repo_root(start: Path) -> Path:
    """Locate the repository root from this module's location.

    An editable install puts this file at ``<repo>/src/cardvault_vision/config.py``
    (three levels up) while a wheel install puts it in site-packages, where
    walking up lands somewhere meaningless. Walk up looking for the markers that
    actually identify the repo, and fall back to the parent of the package rather
    than to a guessed depth.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").is_dir():
            return candidate
    # A wheel install: configs/ is not shipped, so point at the package's parent
    # and let the caller treat a missing config dir as "no repo configs".
    return start.parents[1]


#: Repository root (…/CardVault-Vision), derived from this file's location.
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, *overrides: Mapping[str, Any] | None) -> dict:
    """Recursively merge mappings; later values win.

    Lists are replaced wholesale (not concatenated) because merging lists
    silently is a common source of confusing configuration.
    """
    out = copy.deepcopy(dict(base))
    for override in overrides:
        if not override:
            continue
        if not isinstance(override, Mapping):
            raise ConfigError(f"Cannot merge non-mapping override: {type(override).__name__}")
        for key, value in override.items():
            if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
                out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
            else:
                out[key] = copy.deepcopy(value)
    return out


def parse_value(text: str) -> Any:
    """Interpret a CLI ``--set`` value using JSON, with a string fallback.

    ``true``/``false``/``null``/numbers/JSON objects and lists are converted;
    anything else stays a plain string (``null`` -> ``None``).
    """
    lowered = text.strip()
    if lowered == "":
        return ""
    try:
        return json.loads(lowered)
    except json.JSONDecodeError:
        return text


def set_dotted(cfg: dict, dotted: str, value: Any) -> dict:
    """Set ``cfg["a"]["b"]["c"] = value`` from the path ``"a.b.c"``."""
    if not dotted:
        raise ConfigError("Empty configuration key")
    node = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        if part in node and not isinstance(node[part], dict):
            raise ConfigError(
                f"Cannot set {dotted!r}: {part!r} is already a scalar value"
            )
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return cfg


def get_dotted(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Read a dotted path, returning ``default`` when absent."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def load_yaml(path: str | os.PathLike) -> dict:
    """Load a YAML mapping from disk with precise error messages."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:  # pragma: no cover - message passthrough
        raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {p} must be a mapping, got {type(data).__name__}")
    return data


def canonical_json(obj: Any) -> str:
    """Stable JSON serialisation used for hashing and checksums."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(obj: Any, length: int = 16) -> str:
    """Short, stable hash of a config/dataset description."""
    return hashlib.blake2b(canonical_json(obj).encode("utf-8"), digest_size=32).hexdigest()[:length]


class Config:
    """A resolved, hashable configuration mapping with provenance."""

    def __init__(self, data: Mapping[str, Any] | None = None, *, sources: list[str] | None = None):
        self._data: dict = copy.deepcopy(dict(data or {}))
        self.sources: list[str] = list(sources or [])

    # -- construction -----------------------------------------------------
    @classmethod
    def load(
        cls,
        *paths: str | os.PathLike,
        defaults: Mapping[str, Any] | None = None,
        overrides: Mapping[str, Any] | None = None,
        sets: list[str] | None = None,
    ) -> "Config":
        """Build a config from defaults, YAML files, dict overrides and ``k=v`` sets."""
        merged: dict = copy.deepcopy(dict(defaults or {}))
        used: list[str] = []
        for path in paths:
            resolved = _resolve_or_path(path)
            data = load_yaml(resolved)
            merged = deep_merge(merged, data)
            used.append(str(resolved))
        if overrides:
            merged = deep_merge(merged, overrides)
        for item in sets or []:
            if "=" not in item:
                raise ConfigError(f"--set expects key=value, got {item!r}")
            key, raw = item.split("=", 1)
            set_dotted(merged, key.strip(), parse_value(raw))
            used.append(f"set:{key.strip()}")
        return cls(merged, sources=used)

    # -- mapping protocol -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        return get_dotted(self._data, dotted, default)

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = get_dotted(self._data, dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"Missing required configuration key: {dotted}")
        return value

    def set(self, dotted: str, value: Any) -> "Config":
        set_dotted(self._data, dotted, value)
        return self

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and get_dotted(self._data, key, _MISSING) is not _MISSING

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(fingerprint={self.fingerprint()}, keys={sorted(self._data)})"

    # -- provenance -------------------------------------------------------
    def fingerprint(self) -> str:
        """Hash of the fully resolved configuration."""
        return fingerprint(self._data)

    def describe(self) -> dict:
        """A JSON-safe description embedded into run metadata."""
        return {"fingerprint": self.fingerprint(), "sources": self.sources, "values": self._data}

    def save(self, path: str | os.PathLike) -> Path:
        """Write the resolved config as YAML (a run's exact recipe)."""
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=True)
        return p


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover
        return "<missing>"


_MISSING = _Missing()


def _resolve_or_path(path: str | os.PathLike) -> Path:
    """Accept a real path or a bare config name such as ``train-ocr``.

    ``--config train-ocr`` is the documented form, so resolving names here keeps
    the CLI honest instead of only accepting full paths.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix in {".yaml", ".yml"} or os.sep not in str(path):
        try:
            return resolve_config_path(str(path))
        except ConfigError:
            pass
    return candidate


def resolve_config_path(name: str, *, search: tuple[Path, ...] = ()) -> Path:
    """Find a named config, accepting a path, a path without extension, or a
    name relative to the repository ``configs/`` directory."""
    candidate = Path(name)
    search_dirs = (Path.cwd(), *search, CONFIG_DIR, REPO_ROOT)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate
    names = [candidate]
    if candidate.suffix not in {".yaml", ".yml"}:
        names += [candidate.with_suffix(".yaml"), candidate.with_suffix(".yml")]
    for base in search_dirs:
        if not base.is_dir():
            continue
        for candidate_path in names:
            for probe in (base / candidate_path, base / "configs" / candidate_path):
                if probe.exists():
                    return probe
    raise ConfigError(
        f"Could not resolve config {name!r}. Looked in: "
        + ", ".join(str(d) for d in search_dirs)
    )
