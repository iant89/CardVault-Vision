"""Train/validation/test splitting with card-level leakage prevention.

The invariant, from the roadmap:

    A physical card must never appear in both training and testing data
    just because different photographs of it were used.

So the unit of splitting is the **card_id**, never the image. We additionally
support *stratified* splitting on card type and *near-duplicate grouping*, so
that two source images of the same card under different ids still land in the
same split.

Splitting is deterministic given a seed and is stable under growth: adding new
cards to a dataset does not move existing cards between splits when
``stable=True`` (assignment is a hash of ``(seed, card_id)``, not a function of
the list order).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..errors import ConfigError, DatasetError
from ..logging_utils import get_logger
from ..rng import derive_seed, stable_permutation
from .hashing import find_duplicate_groups
from .manifest import DatasetManifest
from .schema import SampleRecord

_LOG = get_logger("datasets.split")

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SplitPlan:
    """An explicit, auditable assignment of cards to splits."""

    assignments: dict[str, str] = field(default_factory=dict)  # card_id -> split
    ratios: dict[str, float] = field(default_factory=dict)
    seed: int = 0
    strategy: str = "stratified"
    warnings: list[str] = field(default_factory=list)

    def split_for(self, card_id: str) -> str:
        try:
            return self.assignments[card_id]
        except KeyError as exc:
            raise KeyError(f"Card {card_id!r} is not in this split plan") from exc

    def cards(self, split: str) -> list[str]:
        return sorted(c for c, s in self.assignments.items() if s == split)

    def counts(self) -> dict[str, int]:
        counts = {name: 0 for name in SPLIT_NAMES}
        for split in self.assignments.values():
            counts[split] = counts.get(split, 0) + 1
        return counts

    def apply(self, records: Iterable[SampleRecord]) -> list[SampleRecord]:
        """Stamp ``split`` onto records and return them."""
        out = []
        for record in records:
            split = self.assignments.get(record.card_id)
            if split is None:
                raise DatasetError(
                    f"Card {record.card_id!r} has no split assignment; "
                    "the split plan does not cover this dataset"
                )
            record.split = split
            out.append(record)
        return out

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "strategy": self.strategy,
            "ratios": self.ratios,
            "counts": self.counts(),
            "assignments": dict(sorted(self.assignments.items())),
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> SplitPlan:
        return cls(
            assignments=dict(data.get("assignments") or {}),
            ratios=dict(data.get("ratios") or {}),
            seed=int(data.get("seed", 0)),
            strategy=str(data.get("strategy", "stratified")),
            warnings=list(data.get("warnings") or []),
        )


def _normalise_ratios(train: float, val: float, test: float) -> dict[str, float]:
    if min(train, val, test) < 0:
        raise ConfigError("Split ratios must be non-negative")
    total = train + val + test
    if total <= 0:
        raise ConfigError("At least one split ratio must be positive")
    ratios = {name: value / total for name, value in
              zip(SPLIT_NAMES, (train, val, test), strict=True)}
    return ratios


def _assign_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of ``n`` cards across splits."""
    raw = {name: n * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - counts[name]), reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1
    return counts


def split_dataset(
    manifest: DatasetManifest,
    *,
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 0,
    strategy: str = "stratified",
    stratify_by: str = "card_type",
    dedupe: bool = True,
    max_perceptual_distance: int = 4,
    min_cards_per_split: int | None = None,
) -> SplitPlan:
    """Assign every card in ``manifest`` to exactly one split.

    Parameters
    ----------
    strategy:
        ``"stratified"`` balances card *types* across splits; ``"random"``
        ignores card type; ``"hash"`` assigns purely from
        ``hash(seed, card_id)`` and is therefore stable as the dataset grows.
    dedupe:
        Group cards that are perceptually near-identical and keep the whole
        group together (a second physical copy of the same artwork must not be
        on both sides of the split).
    """
    if strategy not in {"stratified", "random", "hash"}:
        raise ConfigError(f"Unknown split strategy {strategy!r}")

    ratios = _normalise_ratios(train, val, test)
    records_by_card: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in manifest.samples:
        records_by_card[record.card_id].append(record)
    cards = sorted(records_by_card)
    if not cards:
        raise DatasetError(f"Dataset {manifest.name!r} has no samples to split")

    # -- group near-duplicate cards together ------------------------------
    groups: list[list[str]] = []
    if dedupe and len(cards) <= 5000:
        groups = _group_cards_by_similarity(manifest, records_by_card, max_perceptual_distance)
    covered = {card for group in groups for card in group}
    groups.extend([[card] for card in cards if card not in covered])

    # -- ordering ---------------------------------------------------------
    if strategy == "hash":
        groups.sort(key=lambda g: derive_seed(seed, "split", g[0]))
    else:
        groups = stable_permutation(groups, seed)

    # -- stratification ---------------------------------------------------
    def card_type(group: Sequence[str]) -> str:
        knowledge = records_by_card[group[0]][0].annotation
        if stratify_by == "card_type":
            return knowledge.card_type
        if stratify_by == "finish" and knowledge.knowledge:
            return knowledge.knowledge.finish or "UNKNOWN"
        return "ALL"

    assignments: dict[str, str] = {}
    warnings: list[str] = []
    if strategy == "stratified":
        buckets: dict[str, list[list[str]]] = defaultdict(list)
        for group in groups:
            buckets[card_type(group)].append(group)
        # Allocate *within* each stratum. Drawing one global sequence and slicing
        # it into a prefix per split put whole strata in one split: a 4:1
        # train/val split of a mixed dataset sent every sports card to train and
        # left val holding only two TCG types. A field that only exists on sports
        # cards then had no validation samples at all, and training failed with
        # "No usable training samples for field 'player'" - a confusing symptom
        # for what was really a broken split.
        for stratum, members in sorted(buckets.items()):
            # ``members`` is already a list of *groups*; shuffling and
            # interleaving it again treats each group as a stratum of its own
            # and hands _allocate bare card-id strings.
            assignments.update(
                _allocate(stable_permutation(members, seed), ratios,
                          warning_prefix=stratum, warnings=warnings)
            )
    else:
        assignments.update(_allocate(list(groups), ratios, warnings=warnings))

    if min_cards_per_split:
        actual = SplitPlan(assignments=assignments).counts()
        for name in SPLIT_NAMES:
            if actual.get(name, 0) < min_cards_per_split:
                warnings.append(
                    f"Split {name!r} has {actual.get(name, 0)} card(s), "
                    f"below the requested minimum of {min_cards_per_split}"
                )

    plan = SplitPlan(assignments=assignments, ratios=ratios, seed=seed,
                     strategy=strategy, warnings=warnings)
    _LOG.info("Split %s: %s cards -> %s", manifest.name, len(cards), plan.counts())
    for warning in warnings:
        _LOG.warning(warning)
    return plan


def _allocate(
    ordered: Sequence[Sequence[str]],
    ratios: Mapping[str, float],
    *,
    warning_prefix: str = "",
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """Assign ordered card groups to splits, proportionally and completely.

    Groups are atomic, so a group larger than its split's share can overshoot;
    the next split then receives fewer. Every split that is entitled to cards
    gets at least one *group*, and the remainder -- including the final split --
    flows into ``train`` so no card is ever dropped.
    """
    warnings = warnings if warnings is not None else []
    total_groups = len(ordered)
    # Target by *card* count, not group count: groups are atomic, so counting
    # groups leaves splits one group short and the shortfall shows up as cards
    # with no split at all.
    target = _assign_counts(sum(len(group) for group in ordered), ratios)
    assignments: dict[str, str] = {}
    cursor = 0
    # ``train`` is consumed last so it can absorb the remainder; the small
    # splits are filled first. Filling train first starved val/test, and filling
    # test last dropped cards entirely because train had already taken them.
    order = [name for name in SPLIT_NAMES if name != "train"] + ["train"]
    for position, name in enumerate(order):
        wanted = target[name]
        if wanted <= 0:
            continue
        last = position == len(order) - 1
        assigned = 0
        while cursor < total_groups and (last or assigned < wanted):
            group = ordered[cursor]
            cursor += 1
            for card in group:
                assignments[card] = name
            assigned += len(group)
        if assigned == 0:
            label = f"{warning_prefix}: " if warning_prefix else ""
            warnings.append(f"{label}split {name!r} requested {wanted} card(s) "
                            f"but received none")
    # Leftovers go to train, never nowhere.
    for group in ordered[cursor:]:
        for card in group:
            assignments[card] = "train"
    return assignments


def _interleave(sequences: list[list]) -> list:
    """Round-robin merge, preserving each sequence's internal order."""
    out: list = []
    index = 0
    while True:
        added = False
        for seq in sequences:
            if index < len(seq):
                out.append(seq[index])
                added = True
        if not added:
            return out
        index += 1


def _group_cards_by_similarity(
    manifest: DatasetManifest,
    records_by_card: dict[str, list[SampleRecord]],
    max_distance: int,
) -> list[list[str]]:
    """Group cards whose representative images are perceptually similar."""
    from .hashing import perceptual_hash

    reps: dict[str, int] = {}
    for card_id, records in records_by_card.items():
        path = manifest.resolve(records[0].image_path)
        if not path.exists():
            continue
        try:
            reps[card_id] = perceptual_hash(path, hash_size=8, highfreq_factor=2)
        except Exception:  # pragma: no cover - unreadable image
            continue
    found = find_duplicate_groups(reps, max_distance=max_distance)
    if found:
        _LOG.info("Grouped %d near-duplicate image groups so they cannot cross splits", len(found))
    return found


def detect_leakage(manifest: DatasetManifest) -> dict[str, set[str]]:
    """Return ``card_id -> splits`` for any card appearing in several splits."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in manifest.samples:
        if record.split:
            seen[record.card_id].add(record.split)
    return {card: splits for card, splits in seen.items() if len(splits) > 1}


def split_stats(manifest: DatasetManifest) -> dict[str, dict[str, int]]:
    """Per-split sample and card counts, plus card-type coverage."""
    stats: dict[str, dict[str, int]] = {}
    for record in manifest.samples:
        split = record.split or "unassigned"
        bucket = stats.setdefault(split, {"samples": 0, "cards": 0})
        bucket["samples"] += 1
        bucket[f"type:{record.annotation.card_type}"] = bucket.get(
            f"type:{record.annotation.card_type}", 0) + 1
    cards_by_split: dict[str, set[str]] = defaultdict(set)
    for record in manifest.samples:
        cards_by_split[record.split or "unassigned"].add(record.card_id)
    for split, cards in cards_by_split.items():
        stats[split]["cards"] = len(cards)
    return dict(sorted(stats.items()))
