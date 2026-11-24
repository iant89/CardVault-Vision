"""Client for the CardVault Image Engine alignment bridge.

The engine is a TypeScript package. Rather than shipping a second runtime inside
CardVault Vision, the Python side talks to a small Node process
(``align_bridge.mjs``) over stdin/stdout: one JSON job in, one JSON result out.
That keeps the dependency to "a Node binary and a built engine", both of which
are optional and absent by default - :func:`engine_available` reports what is
actually present so callers can skip work rather than crash.

The engine is *not* on npm, so it is consumed from a checkout. Point
``CVV_IMAGE_ENGINE`` at its ``dist`` directory (the one holding ``index.js``).

Scope: this module aligns and grades. It deliberately does not re-implement the
engine's image operations in Python, because two implementations of the same
filter is two sources of truth.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigError
from ..logging_utils import get_logger

_LOG = get_logger("integrations.image_engine")

#: Bridge script, resolved relative to this file so an installed package works.
BRIDGE_SCRIPT = Path(__file__).with_name("align_bridge.mjs")

#: CardVault Vision layout names -> engine profile names. The engine itself maps
#: these to its exported constants.
PROFILE_NAMES = (
    "standard-trading-card",
    "pokemon",
    "magic-the-gathering",
    "yu-gi-oh",
    "sports",
)


class EngineUnavailable(ConfigError):
    """Raised when the Node bridge or the built engine cannot be used."""


def engine_path(explicit: str | Path | None = None) -> Path | None:
    """Locate the built engine's ``dist`` directory, or ``None``."""
    candidate = explicit or os.environ.get("CVV_IMAGE_ENGINE")
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_dir() and (path / "index.js").is_file():
        return path
    if path.is_file() and path.name == "index.js":
        return path.parent
    return None


def engine_available(explicit: str | Path | None = None) -> bool:
    """True when both a Node binary and a built engine are present."""
    return shutil.which("node") is not None and engine_path(explicit) is not None


@dataclass
class AlignmentResult:
    """The engine's verdict for one image."""

    status: str
    corners: list[tuple[float, float]] | None
    confidence: float | None
    candidates: list[dict]
    total_ms: float
    engine_path: str

    @property
    def aligned(self) -> bool:
        return self.status == "ALIGNED" and self.corners is not None


def align(
    image: object,
    *,
    profile: str | None = None,
    options: dict | None = None,
    engine: str | Path | None = None,
    node: str | None = None,
    timeout: float = 300.0,
) -> AlignmentResult:
    """Align one ``(H, W, 3|4)`` or ``(H, W)`` uint8 array via the engine.

    ``image`` is a numpy array, which is the shape the dataset loader already
    produces; importing numpy here keeps the no-numpy CI job out of this module's
    way until it is actually called.
    """
    import numpy as np

    # Validate what the caller passed before looking for binaries: a bad profile
    # or a float image is a programming error, and reporting "Node not found"
    # instead sends the reader looking in the wrong place.
    if profile is not None and profile not in PROFILE_NAMES:
        raise ConfigError(
            f"Unknown image-engine profile {profile!r}; expected one of {list(PROFILE_NAMES)}"
        )

    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ConfigError(f"image must be uint8, got {array.dtype}")
    if array.ndim == 2:
        height, width = array.shape
        fmt = "GRAY"
        payload = array
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        height, width, channels = array.shape
        fmt = "RGBA" if channels == 4 else "RGB"
        payload = array
    else:
        raise ConfigError(
            f"unsupported image shape {array.shape}; expected (H, W), (H, W, 3) or (H, W, 4)"
        )

    resolved = engine_path(engine)
    if resolved is None:
        raise EngineUnavailable(
            "The image engine is not available. Build it and point "
            "CVV_IMAGE_ENGINE at its dist directory (the one containing index.js)."
        )
    binary = node or shutil.which("node")
    if binary is None:
        raise EngineUnavailable("Node is required for the image engine bridge but "
                                "was not found on PATH.")

    job = {
        "operation": "align",
        "width": int(width),
        "height": int(height),
        "format": fmt,
        "pixels": base64.b64encode(np.ascontiguousarray(payload).tobytes()).decode("ascii"),
        "enginePath": str(resolved),
    }
    if profile:
        job["profile"] = profile
    if options:
        job["options"] = options

    try:
        completed = subprocess.run(
            [binary, str(BRIDGE_SCRIPT)],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CVV_IMAGE_ENGINE": str(resolved)},
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - environment
        raise EngineUnavailable(f"image engine bridge timed out after {timeout}s") from exc

    if completed.returncode != 0:
        raise EngineUnavailable(
            f"image engine bridge exited {completed.returncode}: "
            f"{completed.stderr.strip()[:400]}"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EngineUnavailable(
            f"image engine bridge returned non-JSON output: {completed.stdout[:200]!r}"
        ) from exc
    if not result.get("ok"):
        raise EngineUnavailable(
            f"image engine rejected the job: {result.get('error', 'unknown error')}"
        )

    geometry = result.get("geometry") or None
    corners = None
    if geometry and geometry.get("corners"):
        corners = [(float(point["x"]), float(point["y"])) for point in geometry["corners"]]
    return AlignmentResult(
        status=str(result.get("status", "UNKNOWN")),
        corners=corners,
        confidence=None if geometry is None else float(geometry.get("confidence", 0.0)),
        candidates=list(result.get("candidates") or []),
        total_ms=float((result.get("timings") or {}).get("totalMs", 0.0)),
        engine_path=str(result.get("enginePath", resolved)),
    )
