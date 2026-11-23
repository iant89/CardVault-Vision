"""Hashing: content integrity and duplicate detection.

Two different questions, two different tools:

* **"Is this the same file?"** -> SHA-256 (exact, cryptographic).
* **"Is this the same card photographed again?"** -> perceptual hash (DCT-based
  difference hash) with a Hamming-distance threshold. This is what protects
  against a physical card leaking across train/test.

The perceptual hash is implemented with numpy so the dataset engine keeps
working on machines without OpenCV.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

_CHUNK = 1 << 20


def file_sha256(path: str | Path, chunk_size: int = _CHUNK) -> str:
    """Streaming SHA-256 of a file.

    Streaming matters: source scans can be 20 MP TIFFs, and we hash tens of
    thousands of them.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _to_gray_array(image) -> np.ndarray:
    """Accept a PIL image, ndarray or path; return a float grayscale array."""
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        arr = arr.astype(np.float32)
    else:  # PIL image
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Cannot hash an empty or non-2D image")
    return arr


def perceptual_hash(image, hash_size: int = 8, highfreq_factor: int = 4,
                    resize: bool = True) -> int:
    """DCT-based perceptual hash as an integer (``hash_size**2`` bits).

    Robust to resising, mild compression and small brightness shifts, which is
    exactly the variation you get when the same physical card is photographed
    twice.
    """
    arr = _to_gray_array(image)
    if resize:
        from PIL import Image

        img_size = hash_size * highfreq_factor
        resized = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.Resampling.LANCZOS
        )
        arr = np.asarray(resized, dtype=np.float32)
    else:
        # Cheap 2x2 box downscale until small enough (no PIL round-trip).
        while arr.shape[0] > hash_size * highfreq_factor and arr.shape[1] > hash_size * highfreq_factor:
            h, w = arr.shape
            arr = arr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    # Orthonormal-ish 2-D DCT-II computed with a matrix product (no scipy).
    n = hash_size * highfreq_factor
    if arr.shape != (n, n):
        from PIL import Image

        arr = np.asarray(
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
                (n, n), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
    basis = _dct_matrix(n)
    dct = basis @ arr @ basis.T
    low = dct[:hash_size, :hash_size]
    median = np.median(low.flatten()[1:])  # exclude DC term
    bits = (low > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _dct_matrix(n: int) -> np.ndarray:
    """DCT-II basis matrix (cached per size)."""
    cached = _DCT_CACHE.get(n)
    if cached is not None:
        return cached
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    mat = np.cos(np.pi * k * (2 * x + 1) / (2 * n)) * np.sqrt(2.0 / n)
    mat[0, :] = np.sqrt(1.0 / n)
    _DCT_CACHE[n] = mat
    return mat


_DCT_CACHE: dict[int, np.ndarray] = {}


def dhash(image, hash_size: int = 8) -> int:
    """Classic difference hash: compare horizontally adjacent pixels.

    Much cheaper than the DCT hash and adequate for "is this literally the same
    picture again" checks.
    """
    from PIL import Image

    arr = _to_gray_array(image)
    resized = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    """Population count of ``a XOR b``."""
    return int(a ^ b).bit_count()


def dhash_distance(a: int, b: int) -> int:
    """Convenience alias for :func:`hamming` on hashes."""
    return hamming(a, b)


def find_duplicate_groups(
    hashes: dict[str, int],
    *,
    max_distance: int = 4,
    exact_only: bool = False,
) -> list[list[str]]:
    """Group keys whose hashes are within ``max_distance`` bits of each other.

    Union-find over near-matches. With ``max_distance=0`` (or ``exact_only``)
    this degenerates to grouping identical hashes, which is cheap because it is
    a dict lookup rather than a pairwise comparison.

    Complexity is O(n * groups) rather than O(n²) because each key is only
    compared against the *representative* of each existing group.
    """
    if exact_only or max_distance == 0:
        buckets: dict[int, list[str]] = defaultdict(list)
        for key, value in hashes.items():
            buckets[value].append(key)
        return [sorted(g) for g in buckets.values() if len(g) > 1]

    parent: dict[str, str] = {key: key for key in hashes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    representatives: dict[int, str] = {}
    for key, value in hashes.items():
        matched = None
        for rep_hash, rep_key in representatives.items():
            if hamming(value, rep_hash) <= max_distance:
                matched = rep_key
                union(matched, key)
                break
        if matched is None:
            representatives[value] = key

    groups: dict[str, list[str]] = defaultdict(list)
    for key in hashes:
        groups[find(key)].append(key)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def dataset_content_hash(entries: list[tuple[str, str]], *, include_cards: bool = True) -> str:
    """Deterministic hash of a dataset's contents.

    ``entries`` are ``(image_id, sha256)`` pairs; they are sorted so the hash is
    independent of generation order and sharding.
    """
    h = hashlib.sha256()
    for image_id, sha in sorted(entries):
        h.update(image_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(sha.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
