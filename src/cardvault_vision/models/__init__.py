"""Model architectures (Phases 3, 4, 5, 6, 8).

Two families:

* **OCR heads** - CRNN + CTC (the baseline) and a lightweight temporal-convolution
  variant, sharing one char-set/decoding contract.
* **Card-type classifier** - a small CNN with a mobile-friendly head.

Torch is an optional dependency. Importing this package never requires it; the
builders raise :class:`~cardvault_vision.errors.MissingDependencyError` with an
install hint if you ask for a model without torch available. That keeps the
dataset and synthetic engines usable on a minimal install.

The registry in :mod:`cardvault_vision.models.registry` maps architecture names
(used in YAML and the CLI) to builders, so experiments can be described as data.
"""

from __future__ import annotations

from ..errors import MissingDependencyError


def require_torch(feature: str) -> None:
    """Raise a helpful error when torch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised without torch
        raise MissingDependencyError(feature, "torch", "train") from exc


from .classifier import ClassifierConfig, build_classifier  # noqa: E402
from .ocr import (
    CRNN,
    OCRModelConfig,
    LightweightTransformerOCR,
    OCR_ARCHITECTURES,
    TemporalConvOCR,
    build_ocr,
)  # noqa: E402
from .registry import MODEL_REGISTRY, build_model, describe_models  # noqa: E402

__all__ = [
    "CRNN",
    "ClassifierConfig",
    "LightweightTransformerOCR",
    "MODEL_REGISTRY",
    "OCRModelConfig",
    "OCR_ARCHITECTURES",
    "TemporalConvOCR",
    "build_classifier",
    "build_model",
    "build_ocr",
    "describe_models",
    "require_torch",
]
