"""Training loops, datasets and losses (Phases 3-6, 12).

``torch`` is an optional dependency: everything in this package imports it
lazily through :func:`cardvault_vision.models.require_torch`, so the rest of the
library (dataset management, synthetic generation, export validation) keeps
working on a machine where the training extra was never installed.

The public surface is deliberately small - a training entry point for each task
plus the checkpoint helpers needed to resume or export:

    from cardvault_vision.training import TrainConfig, train_ocr, load_checkpoint
"""

from __future__ import annotations

from .data import (
    ClassifierDataset,
    DataLoaderConfig,
    OCRDataset,
    OCRLineTransform,
    build_loader,
    collate_classifier,
    collate_ocr,
    configure_threads,
    estimate_steps,
    resolve_batch_size,
)
from .losses import (
    classification_loss,
    compute_class_weights,
    ctc_loss,
    ctc_loss_from_widths,
    focal_loss,
)
from .trainer import (
    EpochMetrics,
    TrainConfig,
    TrainResult,
    build_from_checkpoint,
    load_checkpoint,
    resolve_device,
    train_classifier,
    train_ocr,
)

__all__ = [
    # configuration and results
    "TrainConfig",
    "TrainResult",
    "EpochMetrics",
    # training entry points
    "train_ocr",
    "train_classifier",
    # checkpoints
    "load_checkpoint",
    "build_from_checkpoint",
    "resolve_device",
    # datasets
    "OCRDataset",
    "ClassifierDataset",
    "OCRLineTransform",
    "DataLoaderConfig",
    "build_loader",
    "collate_ocr",
    "collate_classifier",
    "configure_threads",
    "estimate_steps",
    "resolve_batch_size",
    # losses
    "ctc_loss",
    "ctc_loss_from_widths",
    "classification_loss",
    "compute_class_weights",
    "focal_loss",
]
