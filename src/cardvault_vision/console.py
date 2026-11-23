"""Terminal formatting helpers (no dependencies).

Kept separate from the CLI so library code can produce the same tables and blocks
in reports and docs.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterable, Sequence


def terminal_width(default: int = 100) -> int:
    try:
        return max(60, min(shutil.get_terminal_size().columns, 160))
    except (OSError, ValueError):  # pragma: no cover
        return default


def heading(text: str, char: str = "=") -> str:
    width = min(terminal_width(), max(len(text) + 4, 40))
    return f"\n{text}\n{char * width}"


def subheading(text: str) -> str:
    return f"\n-- {text} " + "-" * max(0, min(terminal_width(), 60) - len(text) - 4)


def kv_block(pairs: Iterable[tuple[str, object]], *, indent: int = 2,
             key_width: int | None = None) -> str:
    pairs = [(str(k), _fmt(v)) for k, v in pairs]
    width = key_width or (max((len(k) for k, _ in pairs), default=0) + 1)
    pad = " " * indent
    return "\n".join(f"{pad}{k:<{width}} : {v}" for k, v in pairs)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]],
                 *, max_width: int | None = None) -> str:
    if not rows:
        return "  (empty)"
    cells = [[str(c) for c in row] for row in rows]
    widths = [max(len(str(h)), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]
    limit = max_width or terminal_width()
    while sum(widths) + 2 * (len(widths) - 1) > limit and max(widths) > 8:
        widest = widths.index(max(widths))
        widths[widest] -= 1
    def render(row: Sequence[str]) -> str:
        return "  ".join(f"{c[:w]:<{w}}" for c, w in zip(row, widths, strict=True)).rstrip()
    lines = [render(headers), "  ".join("-" * w for w in widths)]
    lines.extend(render(row) for row in cells)
    return "\n".join("  " + line for line in lines)


def bullet_list(items: Iterable[str], char: str = "-") -> str:
    return "\n".join(f"  {char} {item}" for item in items)


def progress_bar(done: int, total: int, *, width: int = 28, prefix: str = "") -> str:
    if total <= 0:
        return ""
    fraction = min(1.0, done / total)
    filled = int(round(width * fraction))
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix}[{bar}] {done}/{total} ({fraction:5.1%})"


def supports_color() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        return f"{value:.4g}"
    if value is None:
        return "-"
    return str(value)
