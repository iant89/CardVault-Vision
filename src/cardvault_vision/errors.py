"""Exception hierarchy.

Every error raised deliberately by CardVault Vision derives from
:class:`CardVaultVisionError`, which makes CLI error handling predictable: the
CLI prints the message and exits non-zero instead of dumping a traceback.
"""

from __future__ import annotations


class CardVaultVisionError(Exception):
    """Base class for every deliberate CardVault Vision failure."""


class ConfigError(CardVaultVisionError):
    """A configuration file or override is missing, malformed or contradictory."""


class DatasetError(CardVaultVisionError):
    """A dataset is missing, corrupt or internally inconsistent."""


class ValidationError(CardVaultVisionError):
    """Data, annotations or a model export failed validation."""


class GenerationError(CardVaultVisionError):
    """The synthetic scan engine could not produce a usable sample."""


class ExportError(CardVaultVisionError):
    """Model export or export validation failed."""


class NotTrainedError(CardVaultVisionError):
    """An operation needs a trained checkpoint that does not exist yet."""


class MissingDependencyError(CardVaultVisionError):
    """An optional dependency (torch, onnx, onnxruntime) is not installed."""

    def __init__(self, feature: str, packages: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires optional dependencies: {packages}.\n"
            f"Install them with:  pip install 'cardvault-vision[{extra}]'"
        )
        self.feature = feature
        self.packages = packages
        self.extra = extra
