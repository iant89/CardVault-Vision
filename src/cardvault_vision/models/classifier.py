"""Card-type classifier (Phase 6).

Determines which ecosystem a card belongs to: Pokémon, Magic, Yu-Gi-Oh!, one of
the sports, or "unknown". The class list comes from
:data:`cardvault_vision.datasets.schema.CARD_TYPES`, including ``UNKNOWN`` and
``OTHER``, because a router that cannot decline to answer will confidently route
a hockey card into the Pokémon OCR path.

The architecture is deliberately small: this model runs on every scan *before* any
OCR, so its CPU latency matters more than a point of accuracy. Latency is measured
rather than assumed (see :mod:`cardvault_vision.benchmarking`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..datasets.schema import CARD_TYPES
from ..errors import ConfigError
from .ocr import _torch


def build_classifier(
    *,
    num_classes: int = len(CARD_TYPES),
    channels: Sequence[int] = (32, 64, 128, 192),
    dropout: float = 0.2,
    input_size: int = 160,
):
    """A compact CNN classifier.

    ``input_size`` is the square input edge. 160 px keeps card-type cues (frame
    colour, text block layout, logo position) visible while staying cheap.
    """
    torch = _torch()
    nn = torch.nn

    def block(in_ch: int, out_ch: int, stride: int = 2):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    class _Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            stages = []
            in_ch = 3
            for index, out_ch in enumerate(channels):
                stages.append(block(in_ch, out_ch, stride=2 if index > 0 else 1))
                in_ch = out_ch
            self.features = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(in_ch, num_classes),
            )

        @property
        def input_size(self) -> int:
            return input_size

        def forward(self, x):
            return self.classifier(self.pool(self.features(x)))

    return _Classifier()


@dataclass
class ClassifierConfig:
    """Rebuild recipe for the classifier."""

    class_names: tuple[str, ...] = CARD_TYPES
    channels: tuple[int, ...] = (32, 64, 128, 192)
    dropout: float = 0.2
    input_size: int = 160

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def build(self):
        return build_classifier(
            num_classes=self.num_classes, channels=self.channels,
            dropout=self.dropout, input_size=self.input_size,
        )

    def to_dict(self) -> dict:
        return {
            "task": "classification",
            "architecture": "card-type-cnn",
            "class_names": list(self.class_names),
            "input_size": self.input_size,
            "channels": list(self.channels),
            "dropout": self.dropout,
        }

    def validate(self) -> None:
        if not self.class_names:
            raise ConfigError("classifier needs at least one class")
        if self.input_size < 32 or self.input_size > 1024:
            raise ConfigError("input_size must be within 32..1024")
