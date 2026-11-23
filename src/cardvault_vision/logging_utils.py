"""Logging with human-readable console output and machine-readable run logs.

Two distinct needs:

* **Operator logs** - one line per event, readable in a terminal or CI log.
* **Run records** - append-only JSON Lines under ``experiments/<run>/`` so a
  training curve or generation job can be inspected or plotted later, and so
  two runs can be diffed.

Nothing here imports torch or numpy, so logging is safe to use at import time.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOGGER_NAME = "cardvault_vision"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


class _ColorFormatter(logging.Formatter):
    """Minimal ANSI colouring, disabled when not writing to a TTY."""

    COLORS = {"DEBUG": "\033[38;5;244m", "INFO": "\033[38;5;39m",
              "WARNING": "\033[38;5;214m", "ERROR": "\033[38;5;196m",
              "CRITICAL": "\033[48;5;196m"}
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt, datefmt=_DATE_FORMAT)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.use_color:
            return text
        color = self.COLORS.get(record.levelname)
        return f"{color}{text}{self.RESET}" if color else text


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: str | os.PathLike | None = None,
    force: bool = False,
    use_color: bool | None = None,
) -> logging.Logger:
    """Configure the ``cardvault_vision`` logger tree.

    Idempotent: repeated calls from the CLI and from library code will not stack
    duplicate handlers unless ``force=True``.
    """
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level {level!r}")
    logger.setLevel(numeric)
    if _configured and not force:
        logger.setLevel(numeric)
        return logger
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    if use_color is None:
        use_color = bool(getattr(sys.stderr, "isatty", lambda: False)())
    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(_ColorFormatter(_DEFAULT_FORMAT, use_color))
    logger.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    logger.propagate = False
    _configured = True
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger, e.g. ``get_logger("synthetic.geometry")``."""
    if not _configured:
        setup_logging(os.environ.get("CVV_LOG_LEVEL", "INFO"))
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


class RunRecorder:
    """Append-only JSONL record of a run (metrics, events, provenance).

    >>> rec = RunRecorder("experiments/demo")            # doctest: +SKIP
    >>> rec.event("start", profile="phone")              # doctest: +SKIP
    >>> rec.metric("cer", 0.081, step=1000)              # doctest: +SKIP
    """

    def __init__(self, run_dir: str | os.PathLike, *, quiet: bool = False) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._quiet = quiet
        self.log = get_logger("run")

    def _write(self, payload: dict) -> None:
        payload.setdefault("time", time.time())
        payload.setdefault("iso_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    def event(self, name: str, **fields: Any) -> None:
        self._write({"type": "event", "event": name, **fields})
        if not self._quiet:
            extras = " ".join(f"{k}={v}" for k, v in fields.items())
            self.log.info("%s %s", name, extras)

    def metric(self, name: str, value: float, *, step: int | None = None, **extra: Any) -> None:
        payload: dict[str, Any] = {"type": "metric", "name": name, "value": float(value)}
        if step is not None:
            payload["step"] = int(step)
        payload.update(extra)
        self._write(payload)

    def artifact(self, path: str | os.PathLike, **meta: Any) -> None:
        self._write({"type": "artifact", "path": str(path), **meta})

    def write_summary(self, summary: dict) -> Path:
        """Write ``summary.json`` (and refresh ``summary.json`` atomically)."""
        out = self.dir / "summary.json"
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
        tmp.replace(out)
        return out


@contextmanager
def timed(logger: logging.Logger, label: str, *, level: int = logging.INFO):
    """Log the wall-clock duration of a block.

    >>> with timed(get_logger(), "generate 100 images"):  # doctest: +SKIP
    ...     ...
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s finished in %.2fs", label, time.perf_counter() - start)
