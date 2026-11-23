"""Deterministic random-number plumbing.

CardVault Vision's hard reproducibility guarantee is:

    same SOURCE + CONFIG + SEED  ->  byte-identical dataset

Achieving that needs more than passing one ``numpy.random.Generator`` around:

* every image must be independent of *how many* images came before it, so
  generation can be sharded across processes, resumed, or run in a different
  order and still produce identical output for a given sample index;
* every *sub-system* (geometry, lighting, camera, background, ...) must draw
  from its own stream, so adding a new lighting parameter does not shift the
  random numbers consumed by geometry and silently re-roll an entire dataset.

We therefore use **counter-based seed derivation**: a 64-bit key is hashed with
BLAKE2b out of ``(master_seed, labels..., sample_index)`` to produce an
independent deterministic stream. Nothing about the stream depends on
execution order or on global RNG state.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .errors import ConfigError

#: Upper bound for the integer seeds we accept/produce (numpy's SeedSequence
#: handles arbitrarily large ints, but keeping seeds < 2**63 keeps them tidy in
#: JSON, YAML and CLI arguments).
MAX_SEED = 2**63 - 1


def normalize_seed(seed: int | str | None) -> int:
    """Coerce a user-supplied seed into ``[0, MAX_SEED]``.

    ``None`` produces a fresh random seed (recorded in the run metadata, so the
    run stays reproducible a posteriori). Strings are hashed, which lets the
    CLI accept friendly seeds such as ``--seed cardvault``.
    """
    if seed is None:
        return int.from_bytes(os.urandom(8), "big") >> 1
    if isinstance(seed, str):
        digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") >> 1
    if isinstance(seed, bool):  # bool is an int; reject it explicitly
        raise ConfigError("Seed must be an integer or a string, not a bool")
    if isinstance(seed, float) or not isinstance(seed, int):
        raise ConfigError(f"Seed must be an integer or a string, got {seed!r}")
    if seed < 0:
        raise ConfigError(f"Seed must be non-negative, got {seed}")
    return seed % (MAX_SEED + 1)


def derive_seed(master_seed: int, *labels: object) -> int:
    """Derive a stable, independent 64-bit seed from a master seed and labels.

    The result is a pure function of its arguments: ``derive_seed(7, "geometry",
    12)`` returns the same value on every machine, Python version and process.

    >>> derive_seed(7, "geometry", 12) == derive_seed(7, "geometry", 12)
    True
    >>> derive_seed(7, "geometry", 12) == derive_seed(7, "lighting", 12)
    False
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(int(master_seed)).encode("utf-8"))
    for label in labels:
        h.update(b"\x1f")
        h.update(str(label).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") >> 1


def rng_for(master_seed: int, *labels: object) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for an independent stream."""
    return np.random.default_rng(derive_seed(master_seed, *labels))


#: Names of the independent random sub-streams used by the scan engine. Each
#: stage draws only from its own stream, so stages are order-independent.
STREAM_NAMES = (
    "sample",       # profile choice, mode choice, targeted problem choice
    "source",       # which source card
    "geometry",     # rotation/perspective/tilt/crop
    "camera",       # resolution, lens, focus, blur, noise, compression
    "lighting",     # exposure, colour temperature, shadows
    "material",     # glare/foil/reflectivity behaviour
    "environment",  # background choice and placement
    "occlusion",    # fingers / cards / objects
)


@dataclass(frozen=True)
class RngStreams:
    """A bundle of named, independent generators for one sample.

    ``RngStreams(master_seed, index)`` gives a sample its own set of streams;
    asking for the same stream twice returns the same generator object so a
    stage that draws extra numbers cannot perturb a *different* stage.
    """

    master_seed: int
    sample_index: int
    _cache: dict[str, np.random.Generator] = field(default_factory=dict, repr=False, compare=False)

    def get(self, name: str) -> np.random.Generator:
        """Return (creating on first use) the stream called ``name``."""
        if name not in self._cache:
            self._cache[name] = rng_for(self.master_seed, name, self.sample_index)
        return self._cache[name]

    def child(self, *labels: object) -> np.random.Generator:
        """Return an ad-hoc stream, e.g. ``streams.child("occlusion", "finger", 2)``."""
        return rng_for(self.master_seed, *labels)

    def seeds(self) -> dict[str, int]:
        """Deterministic record of the per-stream seeds, for the annotation."""
        return {name: derive_seed(self.master_seed, name, self.sample_index) for name in STREAM_NAMES}


def sample_indices(
    count: int,
    *,
    start: int = 0,
    shard: int = 0,
    num_shards: int = 1,
) -> range:
    """Indices for one shard of a generation job.

    Sharding by index (not by "split the list") keeps every sample's random
    streams identical no matter how the job is parallelised.
    """
    if count < 0:
        raise ConfigError(f"count must be >= 0, got {count}")
    if num_shards < 1:
        raise ConfigError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ConfigError(f"shard must be in [0, {num_shards}), got {shard}")
    return range(start + shard, start + count, num_shards)


def spawn_generators(master_seed: int, count: int) -> list[np.random.Generator]:
    """Return ``count`` independent generators (handy for worker processes)."""
    return [rng_for(master_seed, "worker", i) for i in range(count)]


def stable_permutation(items: Iterable, seed: int) -> list:
    """Deterministically permute ``items`` (used by train/test splitting)."""
    items = list(items)
    rng = rng_for(seed, "permutation")
    order = rng.permutation(len(items))
    return [items[i] for i in order]
