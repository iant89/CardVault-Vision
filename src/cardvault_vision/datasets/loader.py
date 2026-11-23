"""Dataset loader: manifest -> (image, annotation) tensors-ready samples.

The loader is deliberately free of torch. It yields numpy arrays and
annotations; the training package wraps it in a torch ``Dataset``. That split
means dataset bugs can be reproduced on a laptop without a GPU, and it keeps the
synthetic engine and the training loop on exactly one definition of "a sample".

Images are loaded lazily and optionally cached. ``max_side`` downscales oversized
captures on load — useful for inspection tooling and for OCR training where a
4000 px phone photo is far more pixels than the head can consume.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..errors import DatasetError
from ..logging_utils import get_logger
from .manifest import DatasetManifest
from .schema import Annotation, BBox, SampleRecord

_LOG = get_logger("datasets.loader")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class Sample:
    """One loaded sample: pixels, annotation and provenance."""

    image_id: str
    card_id: str
    image: np.ndarray  # HxWx3 uint8 RGB
    annotation: Annotation
    split: str | None = None
    profile: str | None = None
    path: Path | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    def crop(self, bbox: BBox, pad: int = 0) -> np.ndarray:
        """Crop a field region, clamped to the image."""
        box = bbox.expand(pad).clip(self.width, self.height)
        x0, y0 = int(round(box.x0)), int(round(box.y0))
        x1, y1 = int(round(box.x1)), int(round(box.y1))
        if x1 <= x0 or y1 <= y0:
            raise DatasetError(f"Empty crop for {self.image_id} ({bbox!r})")
        return self.image[y0:y1, x0:x1]

    def text(self, field_name: str) -> str | None:
        return self.annotation.field_text(field_name)


def load_image(path: str | Path, *, max_side: int | None = None,
               ensure_rgb: bool = True) -> np.ndarray:
    """Load an image as an ``HxWx3`` uint8 RGB array.

    Raises :class:`DatasetError` (not PIL's exception) so callers can treat
    corrupt images as data problems uniformly.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            if ensure_rgb:
                img = img.convert("RGB")
            if max_side and max(img.size) > max_side:
                scale = max_side / max(img.size)
                new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return np.asarray(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DatasetError(f"Cannot read image {path}: {exc}") from exc


class DatasetLoader:
    """Iterate a manifest, loading images on demand.

    >>> loader = DatasetLoader(manifest, max_side=1024)   # doctest: +SKIP
    >>> for sample in loader:                            # doctest: +SKIP
    ...     region = sample.crop(sample.annotation.fields["name"].bbox)
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        *,
        split: str | Sequence[str] | None = None,
        max_side: int | None = None,
        verify_hashes: bool = False,
        limit: int | None = None,
    ) -> None:
        self.manifest = manifest
        self.max_side = max_side
        self.verify_hashes = verify_hashes
        if isinstance(split, str):
            self.splits = {split}
        elif split is None:
            self.splits = None
        else:
            self.splits = set(split)
        self.limit = limit

    def records(self) -> Iterator[SampleRecord]:
        """Yield the manifest records selected by this loader."""
        emitted = 0
        for record in self.manifest.samples:
            if self.splits is not None and record.split not in self.splits:
                continue
            if self.limit is not None and emitted >= self.limit:
                break
            emitted += 1
            yield record

    def __len__(self) -> int:
        if self.splits is None and self.limit is None:
            return len(self.manifest.samples)
        return sum(1 for _ in self.records())

    def load_record(self, record: SampleRecord) -> Sample:
        path = self.manifest.resolve(record.image_path)
        if not path.exists():
            raise DatasetError(
                f"Missing image for {record.image_id}: {path}\n"
                f"Manifest root is {self.manifest.root}"
            )
        if self.verify_hashes and record.sha256:
            from .hashing import file_sha256

            actual = file_sha256(path)
            if actual != record.sha256:
                raise DatasetError(
                    f"Hash mismatch for {record.image_id}: manifest={record.sha256[:12]} "
                    f"file={actual[:12]}"
                )
        image = load_image(path, max_side=self.max_side)
        return Sample(
            image_id=record.image_id,
            card_id=record.card_id,
            image=image,
            annotation=record.annotation,
            split=record.split,
            profile=record.profile,
            path=path,
        )

    def __iter__(self) -> Iterator[Sample]:
        for record in self.records():
            yield self.load_record(record)

    def get(self, image_id: str) -> Sample:
        return self.load_record(self.manifest.get(image_id))

    # -- field-crop iteration (used by both OCR heads) --------------------
    def field_batches(
        self,
        field_name: str,
        *,
        batch_size: int = 32,
        pad: int = 2,
    ) -> Iterator[tuple[list[np.ndarray], list[str], list[str]]]:
        """Yield ``(crops, texts, image_ids)`` for one field.

        Samples whose field is missing or whose crop is degenerate are skipped,
        because "card has no name field" is not an OCR training example.
        """
        crops: list[np.ndarray] = []
        texts: list[str] = []
        ids: list[str] = []
        for sample in self:
            field = sample.annotation.fields.get(field_name)
            if field is None or not field.text:
                continue
            try:
                crop = sample.crop(field.bbox, pad=pad)
            except DatasetError:
                continue
            if crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            crops.append(crop)
            texts.append(field.text)
            ids.append(sample.image_id)
            if len(crops) >= batch_size:
                yield crops, texts, ids
                crops, texts, ids = [], [], []
        if crops:
            yield crops, texts, ids


@lru_cache(maxsize=8)
def _cached_manifest(path: str, root: str | None) -> DatasetManifest:
    return DatasetManifest.load(path, root=root)


def load_manifest_cached(path: str | Path, root: str | Path | None = None) -> DatasetManifest:
    """Load a manifest with a small cache (CLI convenience)."""
    return _cached_manifest(str(path), str(root) if root else None)
