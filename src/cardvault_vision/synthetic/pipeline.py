"""Synthetic dataset generation at scale (Phase 2.9 - 2.13).

Responsibilities:

* turn a :class:`SyntheticConfig` into resolved per-sample configs,
* choose source cards (stratified, so rare card types are not starved),
* drive generation with retries and quality control,
* write ``images/*.jpg`` + ``annotations/*.json`` + ``manifest.json``,
* support sharding and resume so a 100k run can be split across workers or
  restarted after an interruption without regenerating anything,
* record full provenance so the dataset is reproducible from
  ``(source, config, seed)`` alone.

Determinism contract
--------------------
Sample ``i``'s pixels depend only on ``(seed, index, source card chosen for i,
resolved profile for i)``. Everything random inside the sample comes from
``RngStreams(seed, index)``. The source card and profile for index ``i`` are also
drawn from streams keyed by ``i``. Therefore:

* two shards of the same job produce identical output for their indices;
* a resumed run fills exactly the missing indices;
* changing ``count`` does not change any existing sample.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from ..config import fingerprint
from ..datasets.hashing import bytes_sha256, file_sha256
from ..datasets.manifest import DatasetManifest, DatasetVersion
from ..datasets.schema import CardKnowledge, SampleRecord
from ..errors import ConfigError, GenerationError
from ..logging_utils import RunRecorder, get_logger
from ..reproducibility import ReproRecord, write_json
from ..rng import RngStreams, derive_seed, normalize_seed, rng_for, sample_indices
from ..version import SCHEMA_VERSION
from .generator import (
    GenerationConfig,
    GeneratedSample,
    SourceCard,
    _UnusableSample,
    generate_sample,
)
from .profiles import BUILTIN_PROFILES, CaptureProfile, resolve_profiles
from .targeting import PROBLEM_TYPES, apply_problem

_LOG = get_logger("synthetic.pipeline")

ANNOTATION_DIR = "annotations"
IMAGE_DIR = "images"


@dataclass
class SyntheticConfig:
    """User-facing configuration for a generation run.

    This is what the CLI and YAML files populate; :func:`generate_dataset`
    resolves it into per-sample configurations.
    """

    #: Profile spec: a name, a mix preset, or "name:weight,name:weight".
    profile: str = "phone_normal"
    #: Mode: clean / normal / poor / extreme / targeted.
    mode: str = "normal"
    #: Targeted problems (Phase 2.12), e.g. ["glare", "perspective"].
    problems: list[str] = field(default_factory=list)
    #: Probability that any given sample is targeted at a problem (targeted mode).
    problem_rate: float = 0.0
    count: int = 100
    seed: int | None = None
    #: Image size multiplier (1.0 = profile default). Useful for smoke tests.
    render_scale: float = 1.0
    #: Quality-control thresholds.
    min_visibility: float = 0.45
    min_legibility: float = 0.10
    max_attempts: int = 3
    #: Output options.
    image_format: str = "jpg"
    jpeg_quality: int = 92
    save_annotation_sidecars: bool = True
    write_manifest: bool = True
    dataset_name: str = "synthetic"
    version: str = "0.1.0"
    #: Parallelism (processes). 1 = in-process.
    workers: int = 1
    #: Sharding.
    shard: int = 0
    num_shards: int = 1
    #: Resume: skip indices whose files already exist.
    resume: bool = True
    #: Extra profile overrides (deep-merged), for CUSTOM profiles.
    overrides: dict = field(default_factory=dict)
    #: Optional per-sample callback used by callers that want progress updates.
    progress_every: int = 0

    def to_dict(self) -> dict:
        """Every field, not just the ones this class happens to use.

        This dictionary is how a parallel worker reconstructs the config, and the
        config is fingerprinted into every annotation. A partial dump silently
        reverted unlisted fields (dataset name, retry budget, resume, version) to
        their defaults in workers, which changed the fingerprint and therefore
        produced different pixels for the same sample index.
        """
        out: dict = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, list):
                value = list(value)
            elif isinstance(value, dict):
                value = dict(value)
            out[name] = value
        return out

    def validate(self) -> None:
        if self.count < 0:
            raise ConfigError("count must be >= 0")
        if self.mode not in {"clean", "normal", "poor", "extreme", "targeted"}:
            raise ConfigError(
                f"Unknown mode {self.mode!r}; expected clean, normal, poor, extreme or targeted"
            )
        unknown = [p for p in self.problems if p not in PROBLEM_TYPES]
        if unknown:
            raise ConfigError(
                f"Unknown problem(s) {unknown}; expected one of {sorted(PROBLEM_TYPES)}"
            )
        if not 0.0 <= self.problem_rate <= 1.0:
            raise ConfigError("problem_rate must be within [0, 1]")
        if self.render_scale <= 0.05 or self.render_scale > 4:
            raise ConfigError("render_scale must be in (0.05, 4]")
        if self.image_format not in {"jpg", "jpeg", "png"}:
            raise ConfigError("image_format must be jpg, jpeg or png")
        if self.num_shards < 1 or not 0 <= self.shard < self.num_shards:
            raise ConfigError("shard must be within [0, num_shards)")
        if self.workers < 1:
            raise ConfigError("workers must be >= 1")


@dataclass
class GenerationReport:
    """Summary of one generation run."""

    dataset_name: str
    generated: int
    rejected: int
    skipped: int
    duration_s: float
    manifest_path: Path | None
    profile_counts: dict[str, int] = field(default_factory=dict)
    problem_counts: dict[str, int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    seed: int = 0
    output_dir: Path | None = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "generated": self.generated,
            "rejected": self.rejected,
            "skipped": self.skipped,
            "duration_s": round(self.duration_s, 2),
            "images_per_second": round(self.generated / self.duration_s, 3)
            if self.duration_s > 0 else 0.0,
            "manifest": str(self.manifest_path) if self.manifest_path else None,
            "profiles": self.profile_counts,
            "problems": self.problem_counts,
            "rejection_reasons": self.rejection_reasons,
            "seed": self.seed,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

class SourcePool:
    """The set of clean source cards available to a run.

    Selection is *stratified by card type*: a uniform draw over samples would let
    a 500-card Pokémon set drown out 30 hockey cards, and the resulting dataset
    would train a classifier that cannot see hockey.
    """

    def __init__(self, cards: Iterable[SourceCard]) -> None:
        self.cards = list(cards)
        if not self.cards:
            raise ConfigError("Source pool is empty: nothing to generate from")
        self.by_type: dict[str, list[int]] = {}
        for index, card in enumerate(self.cards):
            self.by_type.setdefault(card.card_type, []).append(index)

    def __len__(self) -> int:
        return len(self.cards)

    def choose(self, rng: np.random.Generator, *, uniform: bool = False) -> SourceCard:
        if uniform:
            return self.cards[int(rng.integers(0, len(self.cards)))]
        types = sorted(self.by_type)
        # Every card type equally likely, then uniform within the type.
        weights = np.array([len(self.by_type[t]) for t in types], dtype=np.float64)
        type_weights = (1.0 / weights)
        type_weights /= type_weights.sum()
        chosen_type = types[int(rng.choice(len(types), p=type_weights))]
        pool = self.by_type[chosen_type]
        return self.cards[pool[int(rng.integers(0, len(pool)))]]


def load_source_pool(source: DatasetManifest | str | Path, *,
                     max_side: int | None = None,
                     limit: int | None = None) -> SourcePool:
    """Build a :class:`SourcePool` from a source manifest.

    Field text and ROIs come from the annotation, so a source dataset is fully
    self-describing - the generator never needs a separate sidecar file.
    """
    from ..datasets.loader import load_image

    manifest = source if isinstance(source, DatasetManifest) else DatasetManifest.load(source)
    cards: list[SourceCard] = []
    for record in manifest.samples:
        if limit is not None and len(cards) >= limit:
            break
        path = manifest.resolve(record.image_path)
        try:
            image = load_image(path, max_side=max_side)
        except Exception as exc:  # pragma: no cover - corrupt source
            _LOG.warning("Skipping unreadable source image %s: %s", path, exc)
            continue
        fields = {name: fa.text for name, fa in record.annotation.fields.items() if fa.text}
        rois = {}
        for name, fa in record.annotation.fields.items():
            if fa.roi is not None:
                rois[name] = (fa.roi.x0, fa.roi.y0, fa.roi.x1, fa.roi.y1)
        knowledge = record.annotation.knowledge or CardKnowledge(
            card_id=record.card_id, card_type=record.annotation.card_type,
            name=fields.get("name"), number=fields.get("number"),
        )
        cards.append(SourceCard(
            card_id=record.card_id,
            image=image,
            card_type=record.annotation.card_type,
            fields=fields,
            field_rois=rois,
            knowledge=knowledge,
            finish=knowledge.finish,
        ))
    if not cards:
        raise ConfigError(f"Source dataset {manifest.name!r} yielded no usable cards")
    return SourcePool(cards)


# ---------------------------------------------------------------------------
# Per-sample plan
# ---------------------------------------------------------------------------

@dataclass
class SamplePlan:
    """Everything resolved for one output index."""

    index: int
    source_card: SourceCard
    profile: CaptureProfile
    problems: list[str]
    image_id: str
    config: GenerationConfig


def plan_sample(
    *,
    index: int,
    seed: int,
    pool: SourcePool,
    config: SyntheticConfig,
    profile_weights: dict[str, float],
    other_cards: list[np.ndarray] | None = None,
) -> SamplePlan:
    """Resolve the source card, profile and problem for one index.

    All draws use streams keyed by ``index`` so the plan is stable regardless of
    execution order or sharding.
    """
    stream = RngStreams(master_seed=seed, sample_index=index)

    # -- profile -----------------------------------------------------------
    names = list(profile_weights)
    weights = np.array([profile_weights[n] for n in names], dtype=np.float64)
    weights = weights / weights.sum()
    profile_name = names[int(stream.get("sample").choice(len(names), p=weights))]
    base = profile_name.lower()
    base_key = base if base in BUILTIN_PROFILES else "phone_normal"
    overrides = dict(config.overrides or {})
    if config.mode in {"poor", "extreme", "clean"} and not overrides.get("mode"):
        overrides.setdefault("mode", config.mode)
    profile = CaptureProfile.custom(base_key, overrides) if overrides else \
        CaptureProfile.get(base_key)

    # -- problem targeting -------------------------------------------------
    problems: list[str] = []
    problem_rng = stream.get("sample")
    if config.problems:
        if config.mode == "targeted":
            if problem_rng.random() < max(config.problem_rate, 1e-9):
                problems = [str(problem_rng.choice(config.problems))]
        elif problem_rng.random() < config.problem_rate:
            # In non-targeted modes, apply targeted problems at the requested rate
            # so a mixed dataset can still contain hard cases.
            problems = [str(problem_rng.choice(config.problems))]

    # -- source card -------------------------------------------------------
    # NOTE: the occluder is *not* chosen here. SamplePlan carries no occluder
    # field, so drawing one in this function only consumed RNG draws and threw
    # the result away; the callers below derive it explicitly with
    # rng_for(seed, "other_card", index) so serial and parallel agree.
    card = pool.choose(stream.get("source"))

    profile_data = apply_problem(profile.to_dict(), problems) if problems else profile.to_dict()
    # Preserve the profile's identity: overrides must not turn every sample into
    # "custom", or per-profile dataset statistics become meaningless.
    profile_data["name"] = profile_name
    gen_config = GenerationConfig(
        profile=profile_data,
        mode=config.mode if config.mode != "targeted" else "normal",
        problems=problems,
        min_visibility=config.min_visibility,
        min_legibility=config.min_legibility,
        render_scale=config.render_scale,
        jpeg_on_final=True,
        metadata={"source_card": card.card_id},
    )
    image_id = f"{(problems[0] + '_') if problems else ''}{index:08d}"
    return SamplePlan(
        index=index,
        source_card=card,
        profile=profile,
        problems=problems,
        image_id=image_id,
        config=gen_config,
    )


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------

def _write_image(path: Path, sample: GeneratedSample, config: SyntheticConfig) -> str:
    """Write the image and return its sha256."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(sample.image)
    if config.image_format in {"jpg", "jpeg"}:
        image.save(path, format="JPEG", quality=config.jpeg_quality, subsampling=0,
                   optimize=False)
    else:
        image.save(path, format="PNG", compress_level=6)
    return file_sha256(path)


def _annotation_payload(sample: GeneratedSample, config: SyntheticConfig,
                        source_manifest_hash: str | None) -> dict:
    payload = sample.annotation.to_dict()
    payload["quality"] = sample.quality
    payload["provenance"] = {
        "engine": "cardvault-vision-synthetic",
        "schema_version": SCHEMA_VERSION,
        "profile": sample.profile,
        "mode": config.mode,
        "seed": sample.seed,
        "index": sample.index,
        "source_manifest": source_manifest_hash,
        "config_fingerprint": fingerprint(config.to_dict()),
    }
    return payload


def _build_sample_record(
    sample: GeneratedSample,
    *,
    image_path: str,
    image_hash: str,
) -> SampleRecord:
    return SampleRecord(
        image_id=sample.annotation.image_id,
        card_id=sample.annotation.card_id,
        image_path=image_path,
        annotation=sample.annotation,
        sha256=image_hash,
        width=sample.annotation.width,
        height=sample.annotation.height,
        profile=sample.profile,
        tags=list(sample.annotation.generation.get("problems") or []),
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def generate_dataset(
    source: DatasetManifest | str | Path | SourcePool,
    config: SyntheticConfig | None = None,
    *,
    output_dir: str | Path = "datasets/synthetic/run",
    progress: Callable[[int, int], None] | None = None,
    run_recorder: RunRecorder | None = None,
) -> GenerationReport:
    """Generate a synthetic dataset and write a manifest.

    >>> from cardvault_vision.synthetic import SyntheticConfig, generate_dataset
    >>> generate_dataset("datasets/manifests/source.json",         # doctest: +SKIP
    ...                  SyntheticConfig(count=1000, profile="phone"),
    ...                  output_dir="datasets/synthetic/phone-001")
    """
    config = config or SyntheticConfig()
    config.validate()
    seed = normalize_seed(config.seed)

    started = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / IMAGE_DIR
    annotation_dir = out_dir / ANNOTATION_DIR
    image_dir.mkdir(exist_ok=True)
    if config.save_annotation_sidecars:
        annotation_dir.mkdir(exist_ok=True)

    pool = source if isinstance(source, SourcePool) else load_source_pool(source)
    # Workers reload the source themselves, which needs a concrete path rather
    # than an in-memory pool. Recording it here (instead of relying on an
    # environment variable nobody sets) is what makes a worker use the source the
    # caller actually asked for rather than whatever a previous call cached.
    source_path = source if isinstance(source, (str, Path)) else getattr(
        source, "root", None)
    manifest_hash = None
    if isinstance(source, DatasetManifest):
        manifest_hash = source.version.content_hash or source.manifest_checksum()

    profile_weights = resolve_profiles(config.profile)
    other_cards = _other_cards_for(pool)

    recorder = run_recorder or RunRecorder(out_dir / "run", quiet=True)
    recorder.event("synthetic_start", dataset=config.dataset_name, count=config.count,
                   seed=seed, profile=config.profile, mode=config.mode,
                   workers=config.workers, shard=config.shard, shards=config.num_shards)

    indices = list(sample_indices(config.count, shard=config.shard,
                                  num_shards=config.num_shards))

    # -- resume ------------------------------------------------------------
    skipped = 0
    pending: list[int] = []
    for index in indices:
        plan = plan_sample(index=index, seed=seed, pool=pool, config=config,
                           profile_weights=profile_weights)
        if config.resume and _sample_written(image_dir, annotation_dir,
                                             plan.image_id, config):
            skipped += 1
            continue
        pending.append(index)

    records: list[SampleRecord] = []
    # Existing (resumed) samples must still appear in the new manifest.
    if skipped:
        records.extend(_load_existing_records(image_dir, annotation_dir, config))

    report = GenerationReport(
        dataset_name=config.dataset_name,
        generated=0,
        rejected=0,
        skipped=skipped,
        duration_s=0.0,
        manifest_path=None,
        seed=seed,
        output_dir=out_dir,
    )

    def handle(plan: SamplePlan, other: np.ndarray | None) -> SampleRecord | None:
        sample = generate_sample(
            plan.source_card, plan.config, seed=seed, index=plan.index,
            image_id=plan.image_id, other_card=other,
        )
        relative = f"{IMAGE_DIR}/{plan.image_id}.{config.image_format}"
        path = out_dir / relative
        image_hash = _write_image(path, sample, config)
        if config.save_annotation_sidecars:
            write_json(annotation_dir / f"{plan.image_id}.json",
                       _annotation_payload(sample, config, manifest_hash))
        return _build_sample_record(sample, image_path=relative, image_hash=image_hash)

    total = len(pending)
    completed = 0
    if config.workers > 1 and total > 1:
        records.extend(_generate_parallel(
            pending=pending, seed=seed, pool=pool, config=config,
            profile_weights=profile_weights, out_dir=out_dir, report=report,
            handle=handle, progress=progress, total=total,
            source_path=source_path, other_cards=other_cards,
        ))
    else:
        for index in pending:
            plan = plan_sample(index=index, seed=seed, pool=pool, config=config,
                               profile_weights=profile_weights)
            other = None
            if other_cards:
                # Worker processes must repeat this derivation exactly (see
                # _worker_generate); a sample rendered with a different occluder
                # is a different sample, so the stream is named explicitly.
                chooser = rng_for(seed, "other_card", index)
                other = other_cards[int(chooser.integers(0, len(other_cards)))]
            record, rejected_reason = _attempt(plan, handle, config, other)
            if record is None:
                report.rejected += 1
                key = rejected_reason or "unknown"
                report.rejection_reasons[key] = report.rejection_reasons.get(key, 0) + 1
            else:
                records.append(record)
                report.generated += 1
                report.profile_counts[plan.profile.name] = report.profile_counts.get(
                    plan.profile.name, 0) + 1
                for problem in plan.problems:
                    report.problem_counts[problem] = report.problem_counts.get(problem, 0) + 1
            completed += 1
            if progress is not None:
                progress(completed, total)
            if config.progress_every and completed % config.progress_every == 0:
                _LOG.info("Generated %d/%d samples", completed, total)

    # -- manifest ----------------------------------------------------------
    records.sort(key=lambda r: r.image_id)
    manifest = DatasetManifest(
        name=config.dataset_name,
        kind="synthetic",
        root=out_dir,
        samples=records,
        version=DatasetVersion(version=config.version),
        config=config.to_dict(),
        metadata={
            "num_shards": config.num_shards,
            "shard": config.shard,
            "image_format": config.image_format,
            "profiles": profile_weights,
        },
    )
    provenance = ReproRecord.create(
        seed,
        config=config.to_dict(),
        source=manifest_hash,
        generator="cardvault-vision-synthetic",
        mode=config.mode,
        problems=list(config.problems),
    )
    manifest.provenance = provenance.to_dict()

    manifest_path = None
    if config.write_manifest:
        manifest_path = manifest.save(compute_hash=True)
        write_json(out_dir / "generation.json", report.to_dict() if report.generated else
                   _report_from_records(records, config, seed))
        # Persist the fully-resolved recipe so the run can be repeated verbatim.
        from ..config import Config

        Config(config.to_dict(), sources=["cli"]).save(out_dir / "config.resolved.yaml")

    report.duration_s = time.perf_counter() - started
    report.manifest_path = manifest_path
    recorder.event("synthetic_finished", generated=report.generated,
                   rejected=report.rejected, skipped=report.skipped,
                   duration_s=round(report.duration_s, 2))
    recorder.write_summary(report.to_dict())
    _LOG.info(
        "Generated %d sample(s) in %.1fs (%d rejected, %d skipped) -> %s",
        report.generated, report.duration_s, report.rejected, report.skipped, out_dir,
    )
    return report


def _report_from_records(records: list[SampleRecord], config: SyntheticConfig,
                         seed: int) -> dict:
    profiles: dict[str, int] = {}
    problems: dict[str, int] = {}
    for record in records:
        if record.profile:
            profiles[record.profile] = profiles.get(record.profile, 0) + 1
        for problem in record.tags:
            problems[problem] = problems.get(problem, 0) + 1
    return {"generated": len(records), "profiles": profiles, "problems": problems,
            "seed": seed}


def _attempt(plan: SamplePlan, handle, config: SyntheticConfig,
             other: np.ndarray | None) -> tuple[SampleRecord | None, str | None]:
    """Generate one sample, retrying with a different index derived from the base.

    Retries use ``index * 1000 + attempt`` style derivation so a retry never
    re-rolls the same numbers (which would loop forever on a pathological card).
    """
    last_reason = None
    for attempt in range(max(1, config.max_attempts)):
        working = plan if attempt == 0 else replace(
            plan,
            index=plan.index + attempt * 10_000_019,
            image_id=f"{plan.image_id}_r{attempt}",
            config=replace(plan.config, problems=plan.problems,
                           metadata={**plan.config.metadata, "retry": attempt}),
        )
        try:
            return handle(working, other), None
        except _UnusableSample as exc:
            last_reason = _classify_rejection(exc.quality)
            _LOG.debug("Rejected %s (attempt %d): %s", plan.image_id, attempt + 1, last_reason)
        except GenerationError as exc:
            last_reason = f"generation_error: {exc}"
            _LOG.debug("Generation failed for %s: %s", plan.image_id, exc)
    return None, last_reason


def _classify_rejection(quality: dict) -> str:
    if quality.get("card_visibility", 1.0) < 0.25:
        return "card_not_visible"
    if quality.get("min_legibility", 1.0) < 0.05:
        return "illegible_text"
    if quality.get("mean_luma", 0.5) < 0.03:
        return "too_dark"
    if quality.get("sharpness", 999) < 8.0:
        return "too_blurry"
    return "quality"


def _generate_parallel(*, pending, seed, pool, config, profile_weights, out_dir, report,
                       handle, progress, total,
                       source_path: str | Path | None = None,
                       other_cards: list[np.ndarray] | None = None,
                       ) -> list[SampleRecord]:
    """Generate in worker processes.

    Each worker gets the source pool once. Results are collected as *records*;
    images are written by the workers themselves (each index owns its files, so
    there are no write collisions), and failures are reported back rather than
    raised so one bad sample cannot kill a 100k run.
    """
    records: list[SampleRecord] = []
    payload = [(index, seed, config) for index in pending]
    completed = 0
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = {
            executor.submit(_worker_generate, index, seed, config.to_dict(),
                            out_dir=str(out_dir), profile_weights=profile_weights,
                            source_path=None if source_path is None else str(source_path),
                            # Just the count, never the pixels: the worker rebuilds
                            # the arrays from its own cached pool.
                            other_cards=len(other_cards) if other_cards else 0,
                            ): index
            for index, _, _ in payload
        }
        for future in as_completed(futures):
            try:
                record_dict, rejection = future.result()
            except Exception as exc:  # pragma: no cover - worker crash
                record_dict, rejection = None, f"worker_error: {exc}"
            if record_dict is None:
                report.rejected += 1
                report.rejection_reasons[rejection] = report.rejection_reasons.get(
                    rejection, 0) + 1
            else:
                record = SampleRecord.from_dict(record_dict)
                records.append(record)
                report.generated += 1
                if record.profile:
                    report.profile_counts[record.profile] = report.profile_counts.get(
                        record.profile, 0) + 1
                for problem in record.tags:
                    report.problem_counts[problem] = report.problem_counts.get(problem, 0) + 1
            completed += 1
            if progress is not None:
                progress(completed, total)
    return records


_WORKER_POOL: SourcePool | None = None
_WORKER_POOL_KEY: str | None = None
_WORKER_OTHER_CARDS: list | None = None


def _worker_other_cards() -> list:
    """Occluder candidates derived from the worker's cached source pool.

    These used to be pickled into *every* task payload: eight 880x630x3 arrays
    came to 13.3 MB per task, so a 1500-sample run queued 20 GB of pickles
    through the pool's feeder thread and appeared to hang (the parent stayed
    alive at 3% CPU while the newest image on disk was minutes old). Deriving
    them once per worker process instead costs one load per worker and nothing
    per task, and it stays deterministic because it reads the same pool with the
    same rule as the in-process path.
    """
    global _WORKER_OTHER_CARDS
    if _WORKER_OTHER_CARDS is None:
        _WORKER_OTHER_CARDS = _other_cards_for(_WORKER_POOL) if _WORKER_POOL else []
    return _WORKER_OTHER_CARDS


def _other_cards_for(pool: SourcePool) -> list:
    """The at-most-8 same-type card images used as occluders."""
    if not pool.cards:
        return []
    return [card.image for card in pool.cards
            if card.card_type == pool.cards[0].card_type][:8]


def _worker_generate(index: int, seed: int, config_dict: dict, *,
                     out_dir: str, profile_weights: dict[str, float],
                     source_path: str | None = None,
                     other_cards: int = 0):
    """Worker entry point (module-level so it is picklable)."""
    global _WORKER_POOL, _WORKER_POOL_KEY

    from ..config import Config

    # Key the cache on the source. A process pool created with "spawn" inherits
    # the parent's module state, so a worker that reused an unconditionally
    # cached pool would silently generate from the *previous* dataset's cards.
    key = source_path or os.environ.get("CVV_SOURCE_MANIFEST") or "<in-memory>"
    if _WORKER_POOL is None or _WORKER_POOL_KEY != key:
        resolved = source_path or os.environ.get("CVV_SOURCE_MANIFEST")
        if not resolved:
            raise GenerationError(
                "Parallel generation needs a source path; pass the source dataset "
                "directory or manifest file rather than an in-memory pool."
            )
        _WORKER_POOL = load_source_pool(resolved)
        _WORKER_POOL_KEY = key

    config = SyntheticConfig(**{**_clean_config_dict(config_dict)})
    plan = plan_sample(index=index, seed=seed, pool=_WORKER_POOL, config=config,
                       profile_weights=profile_weights)
    out_path = Path(out_dir)

    pool_others = _worker_other_cards()
    other = None
    if pool_others and other_cards:
        # Same derivation as the in-process path: a sample rendered with a
        # different occluding card is a different image, so serial and parallel
        # runs must choose identically or the determinism guarantee breaks.
        #
        # ``other_cards`` is only a *flag* here (it carries the count, not the
        # pixels): the images are rebuilt from the worker's own cached pool. See
        # _worker_other_cards for why shipping the arrays through the task
        # payload is not acceptable.
        chooser = rng_for(seed, "other_card", index)
        other = pool_others[int(chooser.integers(0, len(pool_others)))]

    def handle(working_plan: SamplePlan, other_card):
        # ``other_card`` may be a retry-adjusted plan's occluder; fall back to the
        # one chosen above when the caller passes None.
        sample = generate_sample(working_plan.source_card, working_plan.config, seed=seed,
                                 index=working_plan.index, image_id=working_plan.image_id,
                                 other_card=other_card if other_card is not None else other)
        relative = f"{IMAGE_DIR}/{working_plan.image_id}.{config.image_format}"
        image_hash = _write_image(out_path / relative, sample, config)
        if config.save_annotation_sidecars:
            write_json(out_path / ANNOTATION_DIR / f"{working_plan.image_id}.json",
                       _annotation_payload(sample, config, None))
        return _build_sample_record(sample, image_path=relative, image_hash=image_hash)

    record, reason = _attempt(plan, handle, config, other)
    return (record.to_dict() if record is not None else None), reason


def _clean_config_dict(data: dict) -> dict:
    """Rebuild a SyntheticConfig kwargs dict from its serialised form."""
    known = set(SyntheticConfig.__dataclass_fields__)
    return {k: v for k, v in data.items() if k in known}


def _sample_written(image_dir: Path, annotation_dir: Path, image_id: str,
                    config: SyntheticConfig) -> bool:
    """Whether ``image_id`` was already written by an earlier run.

    A sample accepted on a retry is stored as ``<image_id>_r<n>``, not as the
    base id. Checking only the base id meant every resumed run regenerated every
    retried sample: wasted work, and a slower run each time it was restarted.
    """
    candidates = [image_id] + [f"{image_id}_r{attempt}"
                               for attempt in range(1, max(config.max_attempts, 1) + 1)]
    for candidate in candidates:
        if not (image_dir / f"{candidate}.{config.image_format}").exists():
            continue
        if not config.save_annotation_sidecars:
            return True
        if (annotation_dir / f"{candidate}.json").exists():
            return True
    return False


def _load_existing_records(image_dir: Path, annotation_dir: Path,
                           config: SyntheticConfig) -> list[SampleRecord]:
    """Reload annotations for a resumed run so the manifest stays complete."""
    records: list[SampleRecord] = []
    if not annotation_dir.exists():
        return records
    for path in sorted(annotation_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            continue
        if "image_id" not in payload:
            continue
        image_name = f"{payload['image_id']}.{config.image_format}"
        image_path = image_dir / image_name
        if not image_path.exists():
            continue
        from ..datasets.schema import Annotation

        annotation = Annotation.from_dict(payload)
        records.append(SampleRecord(
            image_id=annotation.image_id,
            card_id=annotation.card_id,
            image_path=f"{IMAGE_DIR}/{image_name}",
            annotation=annotation,
            sha256=file_sha256(image_path),
            width=annotation.width,
            height=annotation.height,
            profile=(payload.get("provenance") or {}).get("profile"),
            tags=list((payload.get("generation") or {}).get("problems") or []),
        ))
    return records


# ---------------------------------------------------------------------------
# Preview sheets (Phase 2.13)
# ---------------------------------------------------------------------------

def make_contact_sheet(
    image_paths: list[str | Path],
    output: str | Path,
    *,
    columns: int = 6,
    thumb_size: int = 220,
    labels: list[str] | None = None,
) -> Path:
    """Build a labelled contact sheet of generated samples for visual review.

    Preview sheets are how a human actually judges whether the synthetic
    distribution looks real - statistics cannot tell you that the glare looks
    like paint.
    """
    from PIL import Image, ImageDraw

    if not image_paths:
        raise GenerationError("No images supplied for the contact sheet")
    columns = max(1, min(columns, len(image_paths)))
    rows = (len(image_paths) + columns - 1) // columns
    label_h = 18 if labels else 0
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_h)),
                      (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for position, path in enumerate(image_paths):
        with Image.open(path) as img:
            thumb = img.convert("RGB").copy()
        thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        col, row = position % columns, position // columns
        x = col * thumb_size + (thumb_size - thumb.width) // 2
        y = row * (thumb_size + label_h) + (thumb_size - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        if labels and position < len(labels):
            draw.text((col * thumb_size + 4, row * (thumb_size + label_h) + thumb_size + 2),
                      labels[position][:34], fill=(220, 220, 220))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out
