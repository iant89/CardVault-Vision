"""CardVault Vision - the CardVault model factory.

CardVault Vision owns everything between "a physical card exists" and "a
versioned inference artifact that CardVault can consume":

    datasets -> synthetic scan engine -> models -> benchmarks -> exports

The package is import-light: heavy optional dependencies (PyTorch, ONNX
Runtime) are imported lazily so the dataset and synthetic engines work on a
plain CPU machine.

Typical programmatic use::

    from cardvault_vision.datasets import DatasetManifest
    from cardvault_vision.synthetic import SyntheticConfig, generate_dataset

    source = DatasetManifest.load("datasets/manifests/source.json")
    generate_dataset(source, SyntheticConfig(profile="phone"), out="datasets/synthetic/run1")
"""

from __future__ import annotations

from .errors import (
    CardVaultVisionError,
    ConfigError,
    DatasetError,
    ExportError,
    GenerationError,
    ValidationError,
)
from .version import __version__

__all__ = [
    "__version__",
    "CardVaultVisionError",
    "ConfigError",
    "DatasetError",
    "ExportError",
    "GenerationError",
    "ValidationError",
]
