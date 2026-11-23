"""Dataset engine (Phase 1).

Public surface::

    from cardvault_vision.datasets import DatasetManifest, validate_dataset, split_dataset

Responsibilities:

* a manifest format that records provenance and a content hash;
* loading/saving annotations and image resolution;
* image validation and corrupt-file detection;
* exact and perceptual duplicate detection;
* statistics, character and class distribution analysis;
* leakage-free, card-level train/validation/test splitting;
* dataset versioning and hashing.
"""

from __future__ import annotations

from .analysis import (
    CharacterDistribution,
    DatasetStatistics,
    character_distribution,
    class_distribution,
    dataset_statistics,
)
from .hashing import (
    dhash,
    dhash_distance,
    file_sha256,
    find_duplicate_groups,
    perceptual_hash,
)
from .loader import DatasetLoader, Sample
from .manifest import DatasetManifest, DatasetVersion
from .split import SplitPlan, split_dataset
from .validation import (
    ImageIssue,
    ValidationReport,
    detect_corrupt_images,
    validate_annotation,
    validate_dataset,
    validate_image,
)

__all__ = [
    "CharacterDistribution",
    "DatasetLoader",
    "DatasetManifest",
    "DatasetStatistics",
    "DatasetVersion",
    "ImageIssue",
    "Sample",
    "SplitPlan",
    "ValidationReport",
    "character_distribution",
    "class_distribution",
    "dataset_statistics",
    "detect_corrupt_images",
    "dhash",
    "dhash_distance",
    "file_sha256",
    "find_duplicate_groups",
    "perceptual_hash",
    "split_dataset",
    "validate_annotation",
    "validate_dataset",
    "validate_image",
]
