"""``vision dataset`` commands: validate, inspect, generate, statistics, split, bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..console import format_table, heading, kv_block, subheading
from ..datasets import (
    DatasetLoader,
    DatasetManifest,
    character_distribution,
    dataset_statistics,
    split_dataset,
    validate_dataset,
)
from ..errors import DatasetError
from ..logging_utils import get_logger, timed
from ..reproducibility import write_json
from .util import add_common_config_args, emit, print_json, resolve_manifest, rows_from_counts

_LOG = get_logger("cli.dataset")


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("dataset", help="dataset management")
    ds = parser.add_subparsers(dest="dataset_command", metavar="ACTION")

    p = ds.add_parser("validate", help="check a dataset for corrupt files, bad annotations "
                                       "and split leakage")
    p.add_argument("--manifest", required=True, help="manifest file or dataset directory")
    p.add_argument("--hashes", action="store_true", help="verify every sha256 (slow)")
    p.add_argument("--duplicates", action="store_true", help="perceptual duplicate detection")
    p.add_argument("--no-images", action="store_true", help="skip reading image files")
    p.add_argument("--require-quad", action="store_true", help="require card boundaries")
    p.add_argument("--limit", type=int, default=None, help="validate at most N samples")
    p.add_argument("--report", default=None, help="write the full report as JSON here")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any error")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_validate)

    p = ds.add_parser("inspect", help="show one sample: fields, boxes and quality")
    p.add_argument("--manifest", required=True)
    p.add_argument("--image-id", default=None, help="specific image id (default: first)")
    p.add_argument("--index", type=int, default=0, help="sample index when no id is given")
    p.add_argument("--crop", default=None, help="write a field crop to this directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_inspect)

    p = ds.add_parser("statistics", help="class/character/resolution distribution report")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default=None, help="write statistics JSON here")
    p.add_argument("--top", type=int, default=12, help="show N characters")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_statistics)

    p = ds.add_parser("split", help="assign leakage-free train/val/test splits")
    p.add_argument("--manifest", required=True)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strategy", default="stratified",
                   choices=["stratified", "random", "hash"])
    p.add_argument("--no-dedupe", action="store_true",
                   help="do not group perceptually similar cards")
    p.add_argument("--out", default=None,
                   help="write a new manifest with splits applied (default: in place)")
    p.add_argument("--dry-run", action="store_true", help="report without writing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_split)

    p = ds.add_parser("generate", help="build a source manifest from a directory of images")
    p.add_argument("--images", required=True, help="directory containing clean card images")
    p.add_argument("--annotations", default=None,
                   help="directory of <name>.json annotations (else sidecars guessed)")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--name", default=None, help="dataset name (default: directory name)")
    p.add_argument("--version", default="0.1.0")
    add_common_config_args(p)
    p.set_defaults(handler=cmd_generate)

    p = ds.add_parser("bootstrap", help="create a demo source dataset with synthetic cards "
                                        "(no external data required)")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--count", type=int, default=40, help="number of cards")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--types", default="POKEMON,MAGIC_THE_GATHERING,YU_GI_OH,SPORTS")
    p.add_argument("--size", type=int, default=630, help="card width in pixels")
    p.add_argument("--version", default="0.1.0", help="dataset version")
    p.add_argument("--frozen", action="store_true", help="mark the dataset immutable")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_bootstrap)


# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = DatasetManifest.load(resolve_manifest(args.manifest))
    with timed(_LOG, f"validate {manifest.name}"):
        report = validate_dataset(
            manifest,
            check_files=not args.no_images,
            check_hashes=args.hashes,
            check_duplicates=args.duplicates,
            require_card_quad=args.require_quad,
            sample_limit=args.limit,
        )
    if args.report:
        report.save(args.report)
    human = "\n".join([
        heading(f"Dataset validation: {manifest.name}"),
        kv_block([
            ("kind", manifest.kind),
            ("samples", len(manifest.samples)),
            ("cards", len(manifest.card_ids)),
            ("checked", report.checked),
            ("ok", report.ok),
            ("errors", len(report.errors)),
            ("warnings", len(report.warnings)),
            ("content_hash", manifest.version.content_hash or "(not sealed)"),
        ]),
        subheading("Findings by code") if report.issues else "",
        format_table(["code", "severity", "count"],
                     [[code, _worst_severity(report, code), n]
                      for code, n in report.codes().items()]) if report.issues else "",
        subheading("First issues") if report.issues else "",
        "\n".join(f"  [{i.severity}] {i.image_id}: {i.code}: {i.message}"
                  for i in report.issues[:15]),
    ])
    emit(args, report.to_dict(), human)
    if args.strict and not report.is_valid:
        return 2
    return 0


def _worst_severity(report, code: str) -> str:
    severities = [i.severity for i in report.issues if i.code == code]
    for level in ("error", "warning", "info"):
        if level in severities:
            return level
    return "info"


def cmd_inspect(args: argparse.Namespace) -> int:
    manifest = DatasetManifest.load(resolve_manifest(args.manifest))
    loader = DatasetLoader(manifest)
    if args.image_id:
        sample = loader.get(args.image_id)
    else:
        if args.index >= len(manifest.samples):
            raise DatasetError(f"index {args.index} is out of range ({len(manifest.samples)} samples)")
        sample = loader.load_record(manifest.samples[args.index])

    fields = []
    for name, item in sample.annotation.fields.items():
        box = item.bbox
        fields.append([name, item.text[:36],
                       f"{box.x0:.0f},{box.y0:.0f} → {box.x1:.0f},{box.y1:.0f}",
                       f"{box.width:.0f}x{box.height:.0f}"])
    payload = {
        "image_id": sample.image_id,
        "card_id": sample.card_id,
        "card_type": sample.annotation.card_type,
        "size": [sample.width, sample.height],
        "path": str(sample.path),
        "fields": {k: v.to_dict() for k, v in sample.annotation.fields.items()},
        "generation": sample.annotation.generation,
    }
    human = "\n".join([
        heading(f"Sample {sample.image_id}"),
        kv_block([
            ("card", sample.card_id),
            ("type", sample.annotation.card_type),
            ("size", f"{sample.width}x{sample.height}"),
            ("profile", sample.profile or "-"),
            ("split", sample.split or "-"),
            ("path", sample.path),
            ("card_quad", "present" if sample.annotation.card_quad else "missing"),
        ]),
        subheading("Fields"),
        format_table(["field", "text", "bbox", "size"], fields),
    ])
    if args.crop:
        from PIL import Image

        target = Path(args.crop)
        target.mkdir(parents=True, exist_ok=True)
        for name, item in sample.annotation.fields.items():
            try:
                crop = sample.crop(item.bbox, pad=2)
            except DatasetError:
                continue
            Image.fromarray(crop).save(target / f"{sample.image_id}__{name}.png")
        print(f"\n  Crops written to {target}")
    return emit(args, payload, human)


def cmd_statistics(args: argparse.Namespace) -> int:
    manifest = DatasetManifest.load(resolve_manifest(args.manifest))
    stats = dataset_statistics(manifest)
    if args.output:
        stats.save(args.output)

    rows = rows_from_counts(stats.card_types)
    characters = stats.characters
    top_chars = characters.counts.most_common(args.top) if characters else []
    human = "\n".join([
        heading(f"Dataset statistics: {manifest.name}"),
        stats.report(),
        subheading("Card types"),
        format_table(["type", "samples", "share"], rows),
        subheading(f"Top {args.top} characters"),
        format_table(["char", "count"],
                     [[repr(ch) if ch == " " else ch, n] for ch, n in top_chars]),
        subheading("Character set"),
        f"  {character_distribution(manifest).charset[:400]}",
    ])
    return emit(args, stats.to_dict(), human)


def cmd_split(args: argparse.Namespace) -> int:
    manifest = DatasetManifest.load(resolve_manifest(args.manifest))
    plan = split_dataset(
        manifest,
        train=args.train, val=args.val, test=args.test,
        seed=args.seed, strategy=args.strategy, dedupe=not args.no_dedupe,
    )
    plan.apply(manifest.samples)
    target = Path(args.out) if args.out else (manifest.path or
                                              (manifest.root / "manifest.json"))
    if args.out and target != manifest.path:
        manifest.root = target.parent
    if not args.dry_run:
        manifest.save(target)
    counts = plan.counts()
    human = "\n".join([
        heading(f"Split plan: {manifest.name}"),
        kv_block([
            ("strategy", plan.strategy),
            ("seed", plan.seed),
            ("ratios", f"train={args.train} val={args.val} test={args.test}"),
            ("cards", sum(counts.values())),
            ("written", "no (dry run)" if args.dry_run else target),
        ]),
        format_table(["split", "cards"], [[k, v] for k, v in counts.items()]),
        *([subheading("Warnings"), "\n".join(f"  ! {w}" for w in plan.warnings)]
          if plan.warnings else []),
    ])
    return emit(args, plan.to_dict(), human)


def cmd_generate(args: argparse.Namespace) -> int:
    from ..datasets.schema import Annotation, CardKnowledge, SampleRecord
    from ..datasets.hashing import file_sha256
    from ..datasets.loader import IMAGE_SUFFIXES, load_image
    from ..datasets.validation import validate_image

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        raise DatasetError(f"--images must be a directory: {images_dir}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = Path(args.annotations) if args.annotations else None

    manifest = DatasetManifest.create(
        name=args.name or images_dir.name, kind="source", root=out_dir,
        version=args.version, description=f"Source cards from {images_dir}",
    )
    problems = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        report = validate_image(path, image_id=path.stem)
        if not report.is_valid:
            problems.append(f"{path.name}: {report.errors[0].message}")
            continue
        annotation_path = None
        if ann_dir is not None:
            for candidate in (ann_dir / f"{path.stem}.json", ann_dir / f"{path.name}.json"):
                if candidate.exists():
                    annotation_path = candidate
                    break
        if annotation_path is not None:
            import json

            annotation = Annotation.from_dict(json.loads(annotation_path.read_text()))
        else:
            image = load_image(path)
            annotation = Annotation(
                image_id=path.stem, card_id=path.stem, card_type="UNKNOWN",
                width=image.shape[1], height=image.shape[0],
                knowledge=CardKnowledge(card_id=path.stem, card_type="UNKNOWN"),
            )
        relative = f"images/{path.name}"
        target = out_dir / relative
        if not target.exists() or target.resolve() != path.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        manifest.add(SampleRecord(
            image_id=path.stem, card_id=annotation.card_id, image_path=relative,
            annotation=annotation, sha256=file_sha256(target),
            width=annotation.width, height=annotation.height,
        ))

    if not manifest.samples:
        raise DatasetError(f"No usable images found in {images_dir}")
    manifest_path = manifest.save()
    payload = {"manifest": str(manifest_path), "samples": len(manifest.samples),
               "rejected": problems}
    human = "\n".join([
        heading(f"Built source manifest: {manifest.name}"),
        kv_block([("manifest", manifest_path), ("samples", len(manifest.samples)),
                  ("rejected", len(problems)),
                  ("content_hash", manifest.version.content_hash)]),
        *([subheading("Rejected files"), "\n".join(f"  ! {p}" for p in problems)]
          if problems else []),
    ])
    return emit(args, payload, human)


def bootstrap_source_dataset(
    out_dir: str | Path,
    *,
    count: int = 40,
    seed: int = 0,
    types: str | None = None,
    size: int = 630,
    version: str = "0.1.0",
    frozen: bool = False,
) -> DatasetManifest:
    """Create a self-contained demo source dataset.

    This is a plain function rather than command logic so tests, scripts and
    ``vision dataset bootstrap`` all produce byte-identical output from the same
    seed. Demo text is invented per game (see :mod:`cardvault_vision.demo`); no
    real card data is involved.
    """
    import numpy as np
    from PIL import Image

    from ..datasets.hashing import file_sha256
    from ..datasets.schema import Annotation, CardKnowledge, FieldAnnotation, BBox, SampleRecord
    from ..demo import random_card_type, random_fields
    from ..synthetic.text import synth_card_image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = out_dir / "images"
    image_dir.mkdir(exist_ok=True)
    annotation_dir = out_dir / "annotations"
    annotation_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)
    type_list = [t.strip() for t in (types or "").split(",") if t.strip()]
    manifest = DatasetManifest.create(
        name=out_dir.name, kind="source", root=out_dir, version=version,
        description="Bootstrap source dataset generated by `vision dataset bootstrap`",
    )
    for index in range(count):
        raw_type = type_list[index % len(type_list)] if type_list else random_card_type(rng)
        # Canonicalise before generating fields: the layout and the demo
        # vocabulary both key off the canonical name, so an un-normalised
        # spelling ("sports") would draw a generic field set.
        from ..datasets.schema import normalize_card_type

        card_type = normalize_card_type(raw_type)
        fields = random_fields(card_type, rng)
        width = size
        height = int(round(width * 88 / 63))
        image, boxes = synth_card_image(fields, size=(width, height), card_type=card_type,
                                        rng=rng)
        card_id = f"{card_type.lower()}_{index:04d}"
        image_id = f"src_{index:04d}"
        relative = f"images/{image_id}.png"
        Image.fromarray(image).save(out_dir / relative)
        annotations = {}
        for name, box in boxes.items():
            annotations[name] = FieldAnnotation(
                text=fields[name],
                bbox=BBox(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                roi=BBox(box[0] / width, box[1] / height, box[2] / width, box[3] / height),
            )
        annotation = Annotation(
            image_id=image_id, card_id=card_id, card_type=card_type,
            fields=annotations, width=width, height=height,
            knowledge=CardKnowledge(
                card_id=card_id, card_type=card_type, name=fields.get("name"),
                number=fields.get("number"), set_name=fields.get("set"),
            ),
        )
        write_json(annotation_dir / f"{image_id}.json", annotation.to_dict())
        manifest.add(SampleRecord(
            image_id=image_id, card_id=card_id, image_path=relative,
            annotation=annotation, sha256=file_sha256(out_dir / relative),
            width=width, height=height,
        ))
    if frozen:
        from datetime import datetime, timezone

        manifest.version.frozen_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest.save()
    return manifest


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """``vision dataset bootstrap``: a demo source dataset to develop against."""
    out_dir = Path(args.out)
    manifest = bootstrap_source_dataset(
        out_dir, count=args.count, seed=args.seed, types=args.types,
        size=args.size, version=args.version, frozen=args.frozen,
    )
    payload = {"manifest": str(out_dir / "manifest.json"), "cards": len(manifest.samples),
               "types": sorted(manifest_samples_types(manifest))}
    human = "\n".join([
        heading("Bootstrap source dataset"),
        kv_block([("output", out_dir), ("cards", len(manifest.samples)),
                  ("manifest", manifest.version and out_dir / "manifest.json"),
                  ("content_hash", manifest.version.content_hash),
                  ("types", ", ".join(payload["types"]))]),
        "",
        "  next: vision synthetic generate --source "
        f"{out_dir} --profile realistic --count 200 --seed 1 --out "
        f"{out_dir.parent / 'synthetic-demo'}",
    ])
    return emit(args, payload, human)


def manifest_samples_types(manifest: DatasetManifest) -> set[str]:
    return {sample.annotation.card_type for sample in manifest.samples}
