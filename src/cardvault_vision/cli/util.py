"""Shared CLI helpers: output formatting and common argument groups."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..console import (
    bullet_list,
    format_table,
    heading,
    kv_block,
)

__all__ = [
    "add_common_config_args",
    "bullet_list",
    "emit",
    "format_table",
    "heading",
    "kv_block",
    "print_json",
    "resolve_manifest",
    "write_output",
]


def add_common_config_args(parser: argparse.ArgumentParser, *, seed: bool = True) -> None:
    """Attach the flags almost every command needs."""
    if seed:
        parser.add_argument("--seed", type=int, default=None,
                            help="random seed (default: random, but recorded)")
    parser.add_argument("--config", action="append", default=[], metavar="FILE",
                        help="YAML config file (repeatable; later files win)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value using a dotted path")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def resolve_manifest(path: str | Path) -> Path:
    """Accept a manifest file, or a directory containing ``manifest.json``."""
    p = Path(path)
    if p.is_dir():
        candidate = p / "manifest.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No manifest.json in {p}")
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return p


def print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def emit(args: argparse.Namespace, payload: Any, human: str) -> int:
    """Print JSON or the human-readable rendering, depending on ``--json``."""
    if getattr(args, "json", False):
        print_json(payload)
    else:
        print(human)
    return 0


def write_output(path: str | Path | None, payload: Any) -> None:
    """Write JSON to a file when a path is given."""
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def rows_from_counts(counts: dict[str, int], *, label: str = "class",
                     limit: int | None = None) -> list[list[str]]:
    """Turn a ``{name: count}`` mapping into sorted table rows with percentages."""
    total = sum(counts.values()) or 1
    items: Iterable[tuple[str, int]] = sorted(counts.items(), key=lambda kv: -kv[1])
    rows = []
    for name, count in list(items)[:limit] if limit else items:
        rows.append([str(name), str(count), f"{count / total:.1%}"])
    return rows
