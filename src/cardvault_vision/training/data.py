"""torch data pipeline for OCR and classification training (Phase 12).

Performance rules this module implements, because they are where training time
actually goes:

* **Preprocessing happens on workers, not on the GPU.** Resizing/normalising a
  crop is cheap on CPU and expensive when serialised before a step.
* **Variable-width batching.** OCR lines are not all the same width. Bucketing by
  width (``width_buckets``) avoids padding every batch to the longest name in the
  dataset, which is the difference between 40 % and 95 % GPU utilisation on a
  long-name dataset.
* **``pin_memory`` + ``persistent_workers``** when CUDA is present.
* **Deterministic sampling.** The split determines *which* samples; the seed
  determines *order*. Both are recorded so an epoch can be replayed.

``workers`` (data loading processes) and ``threads`` (CPU numerical libraries) are
separate knobs, as the roadmap requires - see :func:`configure_threads`.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np

from ..errors import ConfigError
from ..logging_utils import get_logger
from ..models import require_torch

_LOG = get_logger("training.data")


def configure_threads(threads: int | None) -> int:
    """Set CPU thread count for numerical libraries; returns the value applied.

    This is deliberately *not* ``workers``: oversubscribing BLAS threads across
    many dataloader workers is the classic cause of a training run that is slower
    with more processes.
    """
    if threads is None:
        return int(os.environ.get("OMP_NUM_THREADS", 0)) or (os.cpu_count() or 1)
    threads = max(1, int(threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    return threads


@dataclass
class DataLoaderConfig:
    """Everything about *how* samples reach the model."""

    batch_size: int = 32
    workers: int = 0
    threads: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int = 4
    drop_last: bool = False
    width_buckets: int = 4
    cache: bool = False

    def resolve_pin_memory(self) -> bool:
        if self.pin_memory is not None:
            return self.pin_memory
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def resolve_persistent(self) -> bool:
        if self.persistent_workers is not None:
            return self.persistent_workers
        return self.workers > 0

    def torch_kwargs(self) -> dict:
        kwargs = {
            "num_workers": self.workers,
            "pin_memory": self.resolve_pin_memory(),
            "drop_last": self.drop_last,
        }
        if self.workers > 0:
            kwargs["persistent_workers"] = self.resolve_persistent()
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


# ---------------------------------------------------------------------------
# OCR line preparation
# ---------------------------------------------------------------------------

_MISSING = object()


class _LazySamples(Sequence):
    """A ``Sequence`` of loaded samples that keeps at most one image alive.

    Callers pass a *loader* instead of a pre-materialised list. Each ``Sample``
    owns its full-resolution image, so a 1418-sample split cost 1.4 GB of RAM
    that the training step never used - it only ever needed the small prepared
    crop, which totals about 3 MB. Two 178-card runs had already pushed peak RSS
    to 2.7 GB of a 3 GB box, so anything larger died.

    Access is sequential in practice (dataset construction and the validation
    loop both walk the indices in order), so a single-slot cache turns a
    per-sample load into a no-op for the common case without holding the corpus.
    """

    def __init__(self, loader) -> None:
        self._loader = loader
        self._index: dict[str, object] = {}
        for position, record in enumerate(loader.records()):
            self._index[record.image_id] = (position, record)
        self._order = sorted(self._index.items(), key=lambda kv: kv[1][0])
        self._cached_id: str | None = None
        self._cached_sample = None

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        image_id, (_, record) = self._order[index]
        if image_id != self._cached_id:
            self._cached_sample = self._loader.load_record(record)
            self._cached_id = image_id
        return self._cached_sample

    def __iter__(self):
        for position in range(len(self)):
            yield self[position]


def _iter_samples(source) -> Iterator:
    """Yield samples from a list or from a ``DatasetLoader``-like object."""
    if isinstance(source, Sequence):
        yield from source
        return
    if hasattr(source, "records"):
        # Stream: one image is alive at a time.
        for record in source.records():
            yield source.load_record(record)
        return
    yield from source


@dataclass
class OCRLineTransform:
    """Turn a field crop into a fixed-height, variable-width tensor.

    Height is fixed (models require it); width is kept at the aspect ratio so the
    glyphs are never horizontally squashed, clamped to a sane range so a
    pathological crop cannot blow up memory.
    """

    input_height: int = 32
    min_width: int = 32
    max_width: int = 1600
    invert: bool = False          # some backbones prefer light-on-dark
    augment: bool = False

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.to_tensor_array(image)

    def to_tensor_array(self, image: np.ndarray) -> np.ndarray:
        from ..synthetic.imaging import resize, to_float

        arr = to_float(image) if image.dtype == np.uint8 else image.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        height, width = arr.shape[:2]
        if height <= 0 or width <= 0:
            raise ConfigError("Cannot transform an empty crop")
        scale = self.input_height / height
        new_width = int(round(width * scale))
        new_width = max(self.min_width, min(self.max_width, new_width))
        resized = resize(arr, (new_width, self.input_height), resample="bilinear")
        if self.invert:
            resized = 1.0 - resized
        # The synthetic engine already provides the geometric and photometric
        # distribution; the transform only resizes, so augmentation here would
        # double-count the same variation. ``augment`` is kept for callers that
        # train on real-only data, where extra jitter is the only augmentation.
        return resized.astype(np.float32)


def collate_ocr(batch: Sequence[tuple[np.ndarray, list[int], str]]):
    """Pad crops to the widest item and build CTC targets.

    CTC needs ``input_lengths`` (per-sample time steps) and concatenated targets
    with ``target_lengths``. Both are Triton-free, plain tensor maths, so this
    works on CPU-only installs.
    """
    require_torch("OCR training")
    import torch

    images, targets, texts = [], [], []
    for crop, encoded, text in batch:
        images.append(crop)
        targets.append(encoded)
        texts.append(text)
    max_width = max(image.shape[1] for image in images)
    tensor = np.zeros((len(images), 1, images[0].shape[0], max_width), dtype=np.float32)
    lengths = []
    for index, image in enumerate(images):
        tensor[index, 0, :, : image.shape[1]] = image
        lengths.append(image.shape[1])
    target_lengths = [max(1, len(t)) for t in targets]
    flat = [index for t in targets for index in (t if t else [1])]   # 1 == blank
    return {
        "images": torch.from_numpy(tensor),
        "widths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(flat, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
    }


class OCRDataset:
    """A torch ``Dataset`` over field crops, with width bucketing support.

    Constructed without torch at import time; the class is only instantiated when
    training actually starts.
    """

    def __init__(
        self,
        samples: Sequence,
        field: str,
        charset: str,
        *,
        input_height: int = 32,
        augment: bool = False,
        max_width: int = 1600,
    ) -> None:
        require_torch("OCR training")
        import torch
        from torch.utils.data import Dataset

        from ..synthetic.text import encode_text, normalize_text

        self.field = field
        self.encode_text = encode_text
        self.normalize_text = normalize_text
        self.charset = charset
        self.transform = OCRLineTransform(input_height=input_height, augment=augment,
                                          max_width=max_width)
        self.items: list[tuple[np.ndarray, list[int], str]] = []
        self.widths: list[int] = []
        skipped = 0
        # ``samples`` may be a DatasetLoader: _iter_samples streams it so only
        # one full image is resident at a time. The prepared crops are what we
        # keep, and they are ~500x smaller.
        for sample in _iter_samples(samples):
            annotation = sample.annotation.fields.get(field)
            if annotation is None or not annotation.text:
                skipped += 1
                continue
            text = normalize_text(annotation.text)
            encoded = encode_text(text, charset)
            if not encoded:
                skipped += 1
                continue
            try:
                crop = sample.crop(annotation.bbox, pad=2)
            except Exception:  # pragma: no cover - degenerate box
                skipped += 1
                continue
            if crop.shape[0] < 6 or crop.shape[1] < 6:
                skipped += 1
                continue
            prepared = self.transform(crop)
            self.items.append((prepared, encoded, text))
            self.widths.append(prepared.shape[1])
        if skipped:
            _LOG.info("Skipped %d sample(s) for field %r (missing/empty/degenerate)",
                      skipped, field)
        if not self.items:
            raise ConfigError(
                f"No usable training samples for field {field!r}. "
                "Generate a synthetic dataset first, and check the field name."
            )

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(self.items)

            def __getitem__(self_inner, index: int):
                return self.items[index]

        self._dataset = _Dataset()

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def as_dataset(self):
        """The underlying ``torch.utils.data.Dataset``."""
        return self._dataset

    def length_batches(self, batch_size: int, *, shuffle: bool, seed: int,
                       buckets: int = 4) -> Iterator[list[int]]:
        """Yield index batches grouped by similar width (less padding waste)."""
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.items))
        if shuffle:
            rng.shuffle(order)
        if buckets <= 1:
            for start in range(0, len(order), batch_size):
                yield list(order[start:start + batch_size])
            return
        ordered = sorted(order, key=lambda i: self.widths[i])
        chunk = max(batch_size * 8, batch_size)
        groups = [ordered[i:i + chunk] for i in range(0, len(ordered), chunk)]
        if shuffle:
            rng.shuffle(groups)
        batches: list[list[int]] = []
        for group in groups:
            if shuffle:
                group = list(rng.permutation(group))
            for start in range(0, len(group), batch_size):
                batches.append([int(i) for i in group[start:start + batch_size]])
        if shuffle:
            rng.shuffle(batches)
        yield from batches


class ClassifierDataset:
    """Whole-card crops for the card-type classifier."""

    def __init__(self, samples: Sequence, class_names: Sequence[str], *,
                 input_size: int = 160, augment: bool = False) -> None:
        require_torch("classifier training")
        from torch.utils.data import Dataset

        from ..synthetic.imaging import resize, to_float

        self.class_names = list(class_names)
        self.index = {name: i for i, name in enumerate(self.class_names)}
        self.input_size = input_size
        self.augment = augment
        resize_fn = resize
        to_float_fn = to_float
        items: list[tuple[np.ndarray, int]] = []
        skipped = 0
        rng = np.random.default_rng(0)
        # Stream a loader rather than materialising the split (see _iter_samples).
        for sample in _iter_samples(samples):
            label = self.index.get(sample.annotation.card_type)
            if label is None:
                skipped += 1
                continue
            image = to_float_fn(sample.image)
            size = input_size
            if augment:
                # Cheap, non-geometric augmentation: brightness/contrast jitter.
                gain = float(rng.uniform(0.85, 1.15))
                bias = float(rng.uniform(-0.06, 0.06))
                image = np.clip(image * gain + bias, 0.0, 1.0)
            resized = resize_fn(image, (size, size), resample="bilinear")
            items.append((resized.transpose(2, 0, 1).astype(np.float32), label))
        if skipped:
            _LOG.info("Skipped %d card(s) with a card type outside the class list", skipped)
        if not items:
            raise ConfigError("No usable samples for classifier training")
        self.items = items

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(self.items)

            def __getitem__(self_inner, index: int):
                return self.items[index]

        self._dataset = _Dataset()

    def __len__(self) -> int:
        return len(self.items)

    def as_dataset(self):
        return self._dataset


def collate_classifier(batch):
    require_torch("classifier training")
    import torch

    images = np.stack([item[0] for item in batch])
    labels = np.array([item[1] for item in batch], dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


def build_loader(dataset, config: DataLoaderConfig, *, shuffle: bool,
                 collate_fn: Callable | None = None):
    """Construct a ``torch.utils.data.DataLoader`` from a resolved config."""
    require_torch("training")
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        generator=None,
        **config.torch_kwargs(),
    )


def estimate_steps(num_samples: int, batch_size: int, epochs: int) -> int:
    return max(1, math.ceil(num_samples / max(batch_size, 1))) * max(epochs, 1)


def resolve_batch_size(requested: int | str, *, device: str, default: int = 32) -> int:
    """Support ``--batch-size auto`` with a conservative, memory-aware guess."""
    if not isinstance(requested, str):
        return max(1, int(requested))
    if requested != "auto":
        raise ConfigError(f"Invalid batch size {requested!r}")
    if device.startswith("cuda"):
        try:
            import torch

            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:  # pragma: no cover
            return default
        return int(max(8, min(256, 4 * vram)))
    return default


class GradientAccumulator:
    """Accumulate gradients to emulate a larger batch than fits in memory."""

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ConfigError("gradient accumulation steps must be >= 1")
        self.steps = steps
        self._count = 0

    def step(self) -> bool:
        """Register a micro-batch; returns True when an optimiser step is due."""
        self._count += 1
        if self._count >= self.steps:
            self._count = 0
            return True
        return False
