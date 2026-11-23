"""Model registry: architecture names -> builders, described as data.

Experiments reference models by name (``name-ocr: crnn``), so adding an
architecture is a matter of registering it here and nothing else changes.
"""

from __future__ import annotations

from typing import Callable

from ..errors import ConfigError
from .classifier import ClassifierConfig, build_classifier
from .ocr import OCRModelConfig, OCR_ARCHITECTURES, build_ocr

#: Task name -> {architecture name: description}.
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "name-ocr": {
        "crnn": "CNN + BiGRU + CTC baseline (Phase 3)",
        "crnn_small": "Shallower CRNN with a quarter of the parameters",
        "gru": "CNN + unidirectional GRU + CTC",
        "tcn": "CNN + dilated temporal convolutions + CTC",
        "transformer": "CNN + lightweight Transformer encoder + CTC",
    },
    "number-ocr": {
        "crnn": "CNN + BiGRU + CTC with a constrained digit charset (Phase 5)",
        "crnn_small": "Smaller CRNN for on-device number reading",
        "tcn": "Temporal-convolution head for fast number decoding",
    },
    "card-classifier": {
        "card-type-cnn": "Compact CNN over the card frame (Phase 6)",
    },
}

TASKS = tuple(MODEL_REGISTRY)


def build_model(task: str, architecture: str | None = None, **kwargs):
    """Build a model for ``task`` using ``architecture`` (torch required)."""
    if task not in MODEL_REGISTRY:
        raise ConfigError(f"Unknown task {task!r}; expected one of {list(TASKS)}")
    available = MODEL_REGISTRY[task]
    architecture = architecture or next(iter(available))
    if architecture not in available:
        raise ConfigError(
            f"Unknown architecture {architecture!r} for {task}; "
            f"expected one of {sorted(available)}"
        )
    if task == "card-classifier":
        classes = kwargs.pop("class_names", None)
        config = ClassifierConfig(**{k: v for k, v in kwargs.items()
                                     if k in ("channels", "dropout", "input_size")})
        if classes:
            config.class_names = tuple(classes)
        return config.build()
    charset = kwargs.pop("charset", "")
    if not charset:
        from ..synthetic.text import number_charset, universal_charset

        charset = number_charset() if task == "number-ocr" else universal_charset()
    return build_ocr(architecture, num_classes=len(charset) + 2, **kwargs)


def describe_models() -> dict[str, dict[str, str]]:
    """The registry as a plain dict (for ``vision doctor`` and docs)."""
    return {task: dict(architectures) for task, architectures in MODEL_REGISTRY.items()}


def default_architecture(task: str) -> str:
    if task not in MODEL_REGISTRY:
        raise ConfigError(f"Unknown task {task!r}")
    return next(iter(MODEL_REGISTRY[task]))
