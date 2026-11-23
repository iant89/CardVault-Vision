"""OCR architectures for name and number reading (Phases 3, 4, 5).

All heads share the same contract, which is what makes them comparable under the
roadmap's "same dataset, same split, same metrics, same budget" rule:

    input:  (N, 1, 32, W) float32 grayscale, height fixed at 32
    output: (T, N, C) logits, with C = 2 + len(charset)

``T`` is the sequence length (``width / 2**num_blocks`` for the CNN backbones;
use ``model.output_length(width)`` rather than hard-coding it) and ``C`` includes
the CTC blank at index 1 (see :func:`cardvault_vision.synthetic.text.charset_labels`).

Design notes
------------
* Height is fixed at 32 px and width is dynamic. Fixed height keeps the CNN
  feature map aligned; dynamic width means a long card name does not have to be
  squeezed into a fixed frame, which is the single biggest accuracy lever for
  names.
* The baseline is CRNN (CNN + BiGRU + CTC) because it is well understood and
  cheap. The alternatives exist to answer one question with measurements: does a
  more modern sequence model actually pay for itself?
* Every architecture exposes ``output_length(input_width)`` so training code and
  export metadata agree on the time dimension without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..errors import ConfigError
from . import require_torch

#: Input height expected by all OCR heads.
INPUT_HEIGHT = 32


def _torch():
    require_torch("OCR models")
    import torch

    return torch


class _Backboneless(Exception):  # pragma: no cover - internal guard
    pass


def build_backbone(in_channels: int = 1, *, channels: Sequence[int] = (64, 128, 256, 256)):
    """CNN feature extractor shared by the CRNN-family heads.

    Down-samples height by ``2**len(channels)`` and width by the same factor, so a
    4-block backbone gives one feature column per 16 input pixels. BatchNorm is used rather than InstanceNorm because it
    exports to ONNX more predictably and trains faster at these sizes.
    """
    torch = _torch()
    nn = torch.nn

    layers = []
    in_ch = in_channels
    for i, out_ch in enumerate(channels):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        # CRNN-standard asymmetric pooling: every block halves the height, but
        # only the first two blocks halve the width. Halving width on all four
        # blocks (the naive symmetric choice) leaves a 200 px name crop with ~13
        # CTC time steps - fewer than the characters in the name - so CTC can
        # never align and the loss is undefined. Keeping width reduction at 4x
        # gives 4 sequential steps per character, which is what CTC needs.
        #
        #   height: 32 -> 16 -> 8 -> 4 -> 2   (adaptive pool then collapses to 1)
        #   width:  W -> W/2 -> W/4 -> W/4 -> W/4
        layers.append(nn.MaxPool2d((2, 2) if i < 2 else (2, 1), (2, 2) if i < 2 else (2, 1)))
        in_ch = out_ch
    return nn.Sequential(*layers)


#: Number of leading blocks that also halve the width (see :func:`build_backbone`).
WIDTH_POOL_BLOCKS = 2


def width_reduction(num_blocks: int) -> int:
    """Width down-sampling factor of :func:`build_backbone`.

    Only the first :data:`WIDTH_POOL_BLOCKS` blocks pool across width, so the
    factor saturates at ``2 ** WIDTH_POOL_BLOCKS`` no matter how deep the
    backbone is. Keeping the sequence axis coarse is essential: a 4x reduction
    on a 32 px tall line leaves enough time steps for CTC to align long names.
    """
    return 2 ** min(max(1, num_blocks), WIDTH_POOL_BLOCKS)


def _sequence_length(input_width: int, num_blocks: int) -> int:
    """Number of CTC time steps produced for a given input width."""
    return max(1, input_width // width_reduction(num_blocks))


class CRNN:
    """CRNN + CTC: the Phase 3 baseline.

    >>> model = build_ocr("crnn", num_classes=200)      # doctest: +SKIP
    >>> logits = model(torch.zeros(2, 1, 32, 256))      # doctest: +SKIP
    """

    @staticmethod
    def build(*, num_classes: int, hidden: int = 256, rnn_layers: int = 2,
              channels: Sequence[int] = (64, 128, 256, 256), dropout: float = 0.1,
              bidirectional: bool = True):
        torch = _torch()
        nn = torch.nn

        class _CRNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = build_backbone(1, channels=channels)
                self.pool = nn.AdaptiveAvgPool2d((1, None))
                feat = channels[-1]
                self.rnn = nn.LSTM(
                    feat, hidden, num_layers=rnn_layers, bidirectional=bidirectional,
                    batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0,
                )
                self.dropout = nn.Dropout(dropout)
                factor = 2 if bidirectional else 1
                self.head = nn.Linear(hidden * factor, num_classes)

            def output_length(self, input_width: int) -> int:
                return _sequence_length(int(input_width), len(channels))

            def forward(self, x):
                features = self.cnn(x)                    # N, C, 2, T
                features = self.pool(features)            # N, C, 1, W/4
                features = features.squeeze(2).permute(0, 2, 1)   # N, T, C
                sequence, _ = self.rnn(features)
                return self.head(self.dropout(sequence)).permute(1, 0, 2)  # T, N, C

        return _CRNN()


class TemporalConvOCR:
    """CNN + temporal convolutions + CTC.

    A much shallower sequence model than an RNN but with a wide receptive field
    via dilated convolutions. Far cheaper at inference, which matters for the
    number OCR running on every card.
    """

    @staticmethod
    def build(*, num_classes: int, hidden: int = 256, layers: int = 4,
              kernel: int = 3, dropout: float = 0.1,
              channels: Sequence[int] = (64, 128, 256, 256)):
        torch = _torch()
        nn = torch.nn

        class _TCN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = build_backbone(1, channels=channels)
                self.pool = nn.AdaptiveAvgPool2d((1, None))
                blocks = []
                for i in range(layers):
                    dilation = 2 ** i
                    blocks.append(nn.Sequential(
                        nn.Conv1d(channels[-1], hidden, kernel, padding=dilation,
                                  dilation=dilation),
                        nn.BatchNorm1d(hidden),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden, hidden, 1),
                        nn.GELU(),
                    ))
                self.blocks = nn.Sequential(*blocks)
                self.head = nn.Linear(hidden, num_classes)

            def output_length(self, input_width: int) -> int:
                return _sequence_length(int(input_width), len(channels))

            def forward(self, x):
                features = self.cnn(x)
                features = self.pool(features).squeeze(2)          # N, C, T
                features = self.blocks(features)
                return self.head(features.permute(0, 2, 1)).permute(1, 0, 2)

        return _TCN()


class LightweightTransformerOCR:
    """CNN + a few Transformer encoder layers + CTC.

    Self-attention over the whole line handles long names and distant context
    better than an RNN, at the cost of quadratic cost in sequence length. Included
    to be measured, not assumed.
    """

    @staticmethod
    def build(*, num_classes: int, hidden: int = 192, layers: int = 3,
              heads: int = 6, dropout: float = 0.1,
              channels: Sequence[int] = (64, 128, 256, 256)):
        torch = _torch()
        nn = torch.nn

        class _TransformerOCR(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.cnn = build_backbone(1, channels=channels)
                self.pool = nn.AdaptiveAvgPool2d((1, None))
                self.project = nn.Linear(channels[-1], hidden)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
                    dropout=dropout, batch_first=True, norm_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.head = nn.Linear(hidden, num_classes)

            def output_length(self, input_width: int) -> int:
                return _sequence_length(int(input_width), len(channels))

            def forward(self, x):
                features = self.cnn(x)
                features = self.pool(features).squeeze(2).permute(0, 2, 1)
                encoded = self.encoder(self.project(features))
                return self.head(encoded).permute(1, 0, 2)

        return _TransformerOCR()


class GRUOCR:
    """CNN + GRU + CTC (unidirectional / small): a cheaper CRNN variant."""

    @staticmethod
    def build(*, num_classes: int, hidden: int = 192, rnn_layers: int = 1,
              channels: Sequence[int] = (32, 64, 128, 128), dropout: float = 0.05):
        return CRNN.build(num_classes=num_classes, hidden=hidden, rnn_layers=rnn_layers,
                          channels=channels, dropout=dropout, bidirectional=False)


#: Architecture name -> builder.
OCR_ARCHITECTURES = {
    "crnn": CRNN.build,
    "crnn_small": lambda **kw: CRNN.build(
        **{**{"hidden": 128, "rnn_layers": 1, "channels": (32, 64, 128, 128)}, **kw}
    ),
    "gru": GRUOCR.build,
    "tcn": TemporalConvOCR.build,
    "transformer": LightweightTransformerOCR.build,
}


def build_ocr(architecture: str = "crnn", *, num_classes: int, **kwargs):
    """Build an OCR model by architecture name."""
    if num_classes < 3:
        raise ConfigError("num_classes must be at least 3 (pad, blank, one character)")
    key = architecture.lower()
    if key not in OCR_ARCHITECTURES:
        raise ConfigError(
            f"Unknown OCR architecture {architecture!r}; "
            f"expected one of {sorted(OCR_ARCHITECTURES)}"
        )
    return OCR_ARCHITECTURES[key](num_classes=num_classes, **kwargs)


@dataclass
class OCRModelConfig:
    """Everything needed to rebuild an OCR head exactly."""

    architecture: str = "crnn"
    charset: str = ""
    input_height: int = INPUT_HEIGHT
    kwargs: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.charset) + 2

    def to_dict(self) -> dict:
        return {
            "task": "ocr",
            "architecture": self.architecture,
            "input_height": self.input_height,
            "charset": self.charset,
            "num_classes": self.num_classes,
            "kwargs": self.kwargs,
        }

    def build(self):
        return build_ocr(self.architecture, num_classes=self.num_classes, **self.kwargs)


def model_summary(model) -> dict:
    """Parameter count, size and per-layer shape summary (torch required)."""
    torch = _torch()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters": int(total),
        "trainable_parameters": int(trainable),
        "size_mb_fp32": round(total * 4 / 1024 / 1024, 2),
        "architecture": type(model).__name__,
    }
