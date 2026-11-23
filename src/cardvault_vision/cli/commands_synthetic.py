"""``vision synthetic`` commands: generate, preview, validate, benchmark, profiles."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ..console import format_table, heading, kv_block, progress_bar, subheading
from ..datasets import DatasetManifest, validate_dataset
from ..errors import ConfigError
from ..logging_utils import get_logger
from ..synthetic import (
    MIX_PRESETS,
    PROBLEM_DESCRIPTIONS,
    PROBLEM_TYPES,
    CaptureProfile,
    SyntheticConfig,
    audit_annotations,
    generate_dataset,
    generate_sample,
    make_contact_sheet,
    resolve_profiles,
)
from ..synthetic.annotations import summarize_annotations
from ..synthetic.profiles import PROFILE_NAMES
from ..synthetic.targeting import problems_for_targets
from .util import add_common_config_args, emit, print_json, resolve_manifest, rows_from_counts

_LOG = get_logger("cli.synthetic")


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("synthetic", help="synthetic scan engine")
    syn = parser.add_subparsers(dest="synthetic_command", metavar="ACTION")

    p = syn.add_parser("generate", help="generate a synthetic camera-capture dataset")
    p.add_argument("--source", required=True, help="source manifest or dataset directory")
    p.add_argument("--out", required=True, help="output dataset directory")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--profile", default="phone_normal",
                   help=f"capture profile, mix preset, or 'a:0.7,b:0.3'. "
                        f"profiles: {', '.join(PROFILE_NAMES)}; mixes: {', '.join(MIX_PRESETS)}")
    p.add_argument("--mode", default="normal",
                   choices=["clean", "normal", "poor", "extreme", "targeted"])
    p.add_argument("--problem", action="append", default=[], metavar="NAME[:SEVERITY]",
                   help=f"target a known weakness (repeatable): {', '.join(PROBLEM_TYPES)}")
    p.add_argument("--problem-rate", type=float, default=None,
                   help="fraction of samples that receive a targeted problem")
    p.add_argument("--overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override any profile value, e.g. geometry.max_pitch=30")
    p.add_argument("--render-scale", type=float, default=1.0,
                   help="scale all output resolutions (0.5 = half size, for smoke tests)")
    p.add_argument("--min-visibility", type=float, default=0.45,
                   help="reject samples where less than this fraction of the card is visible")
    p.add_argument("--min-legibility", type=float, default=0.10,
                   help="reject samples whose worst field is more illegible than this")
    p.add_argument("--workers", type=int, default=1, help="parallel worker processes")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--no-resume", action="store_true",
                   help="regenerate indices even if files already exist")
    p.add_argument("--max-images", type=int, default=None, help="cap source images used")
    p.add_argument("--name", default=None, help="dataset name (default: output dir name)")
    p.add_argument("--version", default="0.1.0")
    p.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    p.add_argument("--quiet", action="store_true", help="no progress bar")
    add_common_config_args(p)
    p.set_defaults(handler=cmd_generate)

    p = syn.add_parser("preview", help="render a small contact sheet of samples")
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="preview.png", help="contact sheet path")
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--profile", default="phone_normal")
    p.add_argument("--mode", default="normal",
                   choices=["clean", "normal", "poor", "extreme", "targeted"])
    p.add_argument("--problem", action="append", default=[])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--thumb-size", type=int, default=260)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--keep", default=None, help="also keep individual images in this directory")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_preview)

    p = syn.add_parser("validate", help="audit generated annotations and images")
    p.add_argument("--dataset", required=True, help="synthetic dataset directory")
    p.add_argument("--min-fields", type=int, default=1)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_validate)

    p = syn.add_parser("benchmark", help="measure generation throughput")
    p.add_argument("--source", required=True)
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--profile", default=None, help="benchmark one profile (default: all)")
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_benchmark)

    p = syn.add_parser("profiles", help="list capture profiles and targeted problems")
    p.add_argument("--show", default=None, help="dump one profile's full configuration")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_profiles)


# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> SyntheticConfig:
    overrides: dict = {}
    from ..config import set_dotted

    for item in args.overrides:
        if "=" not in item:
            raise ConfigError(f"--overrides expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        from ..config import parse_value

        set_dotted(overrides, key.strip(), parse_value(raw))
    problems = problems_for_targets(args.problem) if args.problem else []
    problem_rate = args.problem_rate
    if problem_rate is None:
        problem_rate = 1.0 if (problems and args.mode == "targeted") else (
            0.25 if problems else 0.0)
    return SyntheticConfig(
        profile=args.profile,
        mode=args.mode,
        problems=problems,
        problem_rate=float(problem_rate),
        count=args.count,
        seed=args.seed,
        render_scale=args.render_scale,
        min_visibility=args.min_visibility,
        min_legibility=args.min_legibility,
        workers=args.workers,
        shard=args.shard,
        num_shards=args.num_shards,
        resume=not args.no_resume,
        dataset_name=args.name or Path(args.out).name,
        version=args.version,
        image_format=args.image_format,
        overrides=overrides,
    )


def cmd_generate(args: argparse.Namespace) -> int:
    config = _build_config(args)
    manifest_path = resolve_manifest(args.source)
    source = DatasetManifest.load(manifest_path)
    out_dir = Path(args.out)
    # Worker processes reload the source manifest themselves.
    import os

    os.environ["CVV_SOURCE_MANIFEST"] = str(manifest_path)

    started = time.time()
    last = {"t": started}

    def progress(done: int, total: int) -> None:
        if args.quiet or not getattr(sys.stderr, "isatty", lambda: False)():
            return
        now = time.time()
        if now - last["t"] < 0.25 and done != total:
            return
        last["t"] = now
        sys.stderr.write("\r" + progress_bar(done, total, prefix="generating "))
        sys.stderr.flush()

    report = generate_dataset(source, config, output_dir=out_dir, progress=progress)
    if not args.quiet and getattr(sys.stderr, "isatty", lambda: False)():
        sys.stderr.write("\n")

    payload = report.to_dict()
    human = "\n".join([
        heading("Synthetic dataset generated"),
        kv_block([
            ("dataset", report.dataset_name),
            ("output", out_dir),
            ("generated", report.generated),
            ("rejected", report.rejected),
            ("skipped (resume)", report.skipped),
            ("duration", f"{report.duration_s:.1f}s"),
            ("throughput", f"{payload['images_per_second']:.2f} images/s"),
            ("seed", report.seed),
            ("manifest", report.manifest_path),
        ]),
        subheading("Profiles"),
        format_table(["profile", "count", "share"],
                     rows_from_counts(report.profile_counts)) if report.profile_counts else "  -",
        *([subheading("Targeted problems"),
           format_table(["problem", "count", "share"], rows_from_counts(report.problem_counts))]
          if report.problem_counts else []),
        *([subheading("Rejection reasons"),
           format_table(["reason", "count", "share"],
                        rows_from_counts(report.rejection_reasons))]
          if report.rejection_reasons else []),
        "",
        f"  next: vision synthetic preview --source {out_dir} --count 12"
        f" --out {out_dir / 'preview.png'}",
    ])
    return emit(args, payload, human)


def cmd_preview(args: argparse.Namespace) -> int:
    from ..synthetic.pipeline import load_source_pool

    source_manifest = DatasetManifest.load(resolve_manifest(args.source))
    pool = load_source_pool(source_manifest)
    problems = problems_for_targets(args.problem) if args.problem else []
    profile_weights = resolve_profiles(args.profile)
    keep_dir = Path(args.keep) if args.keep else None
    if keep_dir:
        keep_dir.mkdir(parents=True, exist_ok=True)

    from ..synthetic.pipeline import plan_sample

    config = SyntheticConfig(
        profile=args.profile, mode=args.mode, problems=problems,
        problem_rate=1.0 if problems else 0.0, count=args.count, seed=args.seed,
        render_scale=args.render_scale, min_visibility=-1.0, min_legibility=-1.0,
    )
    paths = []
    labels = []
    failures = 0
    for index in range(args.count):
        plan = plan_sample(index=index, seed=args.seed, pool=pool, config=config,
                           profile_weights=profile_weights)
        try:
            sample = generate_sample(plan.source_card, plan.config, seed=args.seed,
                                     index=plan.index, image_id=plan.image_id)
        except Exception as exc:  # previews must not die on one bad sample
            failures += 1
            _LOG.warning("Sample %d failed: %s", index, exc)
            continue
        target = (keep_dir / f"{plan.image_id}.png") if keep_dir else \
            Path("/tmp") / f"cvv_preview_{index:03d}.png"
        sample.to_pil().save(target)
        paths.append(target)
        labels.append(f"{plan.profile.name} | {plan.source_card.card_id}"
                      f"{' | ' + plan.problems[0] if plan.problems else ''}")

    if not paths:
        raise ConfigError("No preview images were generated")
    sheet = make_contact_sheet(paths, args.out, columns=args.columns,
                               thumb_size=args.thumb_size, labels=labels)
    if not keep_dir:
        for path in paths:
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                pass
    payload = {"sheet": str(sheet), "images": len(paths), "failures": failures,
               "labels": labels}
    human = "\n".join([
        heading("Preview sheet"),
        kv_block([("sheet", sheet), ("samples", len(paths)), ("failures", failures),
                  ("profiles", ", ".join(profile_weights)),
                  ("problems", ", ".join(problems) or "-")]),
    ])
    return emit(args, payload, human)


def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.dataset)
    if root.is_dir() and (root / "manifest.json").exists():
        manifest = DatasetManifest.load(root / "manifest.json")
        report = validate_dataset(manifest, check_annotations=True, check_leakage=False)
        audit = audit_annotations(root / "annotations", image_dir=root / "images",
                                  min_fields=args.min_fields)
        payload = {"manifest": report.to_dict(), "annotations": audit.to_dict()}
        human = "\n".join([
            heading(f"Synthetic dataset audit: {manifest.name}"),
            kv_block([
                ("samples", len(manifest.samples)),
                ("validation", report.summary_line()),
                ("annotations checked", audit.checked),
                ("annotation problems", len(audit.problems)),
                ("schema versions", audit.schema_versions),
            ]),
            *([subheading("Problems"),
               "\n".join(f"  ! {i}: {m}" for i, m in audit.problems[:15])]
              if audit.problems else []),
            subheading("Annotation summary"),
            _format_summary(summarize_annotations(root / "annotations")),
        ])
        return emit(args, payload, human)

    audit = audit_annotations(root, min_fields=args.min_fields)
    payload = audit.to_dict()
    human = "\n".join([
        heading("Annotation audit"),
        kv_block([("checked", audit.checked), ("problems", len(audit.problems)),
                  ("ok", audit.ok)]),
        *([subheading("Problems"),
           "\n".join(f"  ! {i}: {m}" for i, m in audit.problems[:20])]
          if audit.problems else []),
    ])
    if not args.json:
        print(human)
    else:
        print_json(payload)
    return 0 if audit.ok else 2


def _format_summary(summary: dict) -> str:
    lines = [
        f"  annotations   : {summary['annotations']}",
        f"  fields        : {summary['fields']}",
        f"  field counts  : {summary['field_counts']}",
    ]
    if summary["problems"]:
        lines.append(f"  targeted      : {summary['problems']}")
    return "\n".join(lines)


def cmd_benchmark(args: argparse.Namespace) -> int:
    from ..synthetic.pipeline import load_source_pool

    source = DatasetManifest.load(resolve_manifest(args.source))
    pool = load_source_pool(source)
    profiles = [args.profile] if args.profile else [
        "clean", "phone_normal", "phone_bad_lighting", "phone_low_end",
        "child_camera", "scanner", "extreme",
    ]
    rows = []
    payload = {"source": str(args.source), "count": args.count, "results": {}}
    for name in profiles:
        profile = CaptureProfile.get(name)
        times = []
        shapes = set()
        rejects = 0
        from ..synthetic.generator import GenerationConfig

        for index in range(args.count):
            config = GenerationConfig(profile=profile.to_dict(),
                                      render_scale=args.render_scale,
                                      min_visibility=0.0, min_legibility=0.0)
            card = pool.choose(__import__("numpy").random.default_rng(index))
            started = time.perf_counter()
            try:
                sample = generate_sample(card, config, seed=args.seed, index=index)
                shapes.add(sample.image.shape[:2])
            except Exception:
                rejects += 1
            times.append(time.perf_counter() - started)
        mean = sum(times) / len(times)
        p95 = sorted(times)[int(len(times) * 0.95) - 1] if len(times) > 1 else mean
        payload["results"][name] = {
            "mean_s": round(mean, 4), "p95_s": round(p95, 4),
            "images_per_second": round(1.0 / mean, 2) if mean else 0.0,
            "rejected": rejects, "sizes": sorted({f"{w}x{h}" for h, w in shapes}),
        }
        rows.append([name, f"{mean * 1000:.0f} ms", f"{p95 * 1000:.0f} ms",
                     f"{1.0 / mean:.2f}" if mean else "-", rejects])
    human = "\n".join([
        heading("Synthetic generation benchmark"),
        kv_block([("source", args.source), ("samples per profile", args.count),
                  ("render_scale", args.render_scale)]),
        format_table(["profile", "mean", "p95", "img/s", "rejected"], rows),
        "",
        "  note: single-process CPU numbers; use --workers on generate for throughput.",
    ])
    return emit(args, payload, human)


def cmd_profiles(args: argparse.Namespace) -> int:
    if args.show:
        profile = CaptureProfile.get(args.show)
        import json

        payload = profile.to_dict()
        if args.json:
            print_json(payload)
        else:
            print(heading(f"Profile: {profile.name}"))
            print(json.dumps(payload, indent=2))
        return 0

    rows = []
    for name in PROFILE_NAMES:
        if name == "custom":
            rows.append([name, "user-defined", "-", "-", "-"])
            continue
        profile = CaptureProfile.get(name)
        geometry = profile.section("geometry")
        rows.append([
            name,
            profile.mode,
            f"{profile.target_size[0]}x{profile.target_size[1]}",
            f"±{geometry.get('max_roll', 0):.0f}° roll / {geometry.get('max_pitch', 0):.0f}° pitch",
            f"{profile.section('camera').get('base_iso', '-')} ISO",
        ])
    problem_rows = [[name, PROBLEM_DESCRIPTIONS[name]] for name in PROBLEM_TYPES]
    mixes = [
        [name, ", ".join(f"{k}={v:.2f}" for k, v in weights.items())]
        for name, weights in MIX_PRESETS.items()
    ]
    payload = {
        "profiles": {name: CaptureProfile.get(name).to_dict() for name in PROFILE_NAMES
                     if name != "custom"},
        "mixes": MIX_PRESETS,
        "problems": PROBLEM_DESCRIPTIONS,
    }
    human = "\n".join([
        heading("Capture profiles"),
        format_table(["profile", "mode", "resolution", "geometry", "noise"], rows),
        subheading("Mix presets"),
        format_table(["mix", "weights"], [[m[0], m[1]] for m in mixes]),
        subheading("Targeted problems (--problem)"),
        format_table(["problem", "what it simulates"], problem_rows),
        "",
        "  vision synthetic generate --profile realistic --count 1000 --seed 1 --out out/",
        "  vision synthetic generate --profile phone --problem glare:3 --count 500 --out out/",
    ])
    return emit(args, payload, human)
