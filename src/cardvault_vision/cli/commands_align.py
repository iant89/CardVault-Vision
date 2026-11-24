"""``vision align`` - grade the CardVault Image Engine against synthetic truth.

The synthetic engine records the exact card quad for every image it renders, so
this command scores the TypeScript engine's detector against known-correct
corners instead of against another heuristic's opinion.

It reports two numbers on purpose. ``detection rate`` is the precision of the
answer the engine actually returns; ``oracle detection rate`` is how often the
true card was among the candidates it proposed, before ranking. A large gap
between them is a ranking problem, not a detection problem, and the two need
different fixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..console import heading, kv_block, subheading
from ..errors import ConfigError
from ..logging_utils import get_logger
from .util import emit, resolve_manifest

_LOG = get_logger("cli.align")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "align",
        help="grade the image engine's card alignment against synthetic ground truth",
    )
    p.add_argument("--dataset", required=True,
                   help="synthetic dataset (or manifest) with recorded card quads")
    p.add_argument("--profile", default=None,
                   help="card profile for ranking: standard-trading-card, pokemon, "
                        "magic-the-gathering, yu-gi-oh, sports")
    p.add_argument("--engine", default=None,
                   help="path to the built engine dist directory "
                        "(default: $CVV_IMAGE_ENGINE)")
    p.add_argument("--limit", type=int, default=None, help="grade at most N samples")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="corner error, as a fraction of the card diagonal, that "
                        "counts as correct (default: 0.02)")
    p.add_argument("--edge-threshold", type=int, default=None,
                   help="override the engine's Sobel threshold for this run")
    p.add_argument("--output", default=None, help="write the full report as JSON here")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(handler=cmd_align)


def cmd_align(args: argparse.Namespace) -> int:
    from ..datasets import DatasetLoader, DatasetManifest
    from ..integrations.fixtures import distribution, grade, worst
    from ..integrations.image_engine import engine_available

    if not engine_available(args.engine):
        raise ConfigError(
            "The image engine is not available. Build it with "
            "`npm run build --workspace @cardvault/image-engine` and pass "
            "--engine <dist directory> or set CVV_IMAGE_ENGINE."
        )

    manifest = DatasetManifest.load(resolve_manifest(args.dataset))
    if not any(sample.annotation.card_quad for sample in manifest.samples):
        raise ConfigError(
            f"{args.dataset} has no recorded card quads, so there is nothing to "
            "grade against. Synthetic datasets record them; imported ones may not."
        )

    options = {}
    if args.edge_threshold is not None:
        options["edgeThreshold"] = args.edge_threshold

    loader = DatasetLoader(manifest)
    result = grade(loader, profile=args.profile, tolerance=args.tolerance,
                   limit=args.limit, options=options or None, engine=args.engine)
    report = result.report
    summary = report.summary()
    spread = distribution(result)

    payload = {
        "dataset": str(args.dataset),
        "profile": args.profile,
        "tolerance": args.tolerance,
        **summary,
        "error_distribution": spread,
        "worst": [
            {"image_id": case.image_id, "status": case.status,
             "corner_error": case.corner_error, "iou": case.iou}
            for case in worst(result, limit=10)
        ],
    }
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.json:
        emit(payload, as_json=True)
        return 0

    heading("Card alignment against synthetic ground truth")
    print(kv_block([
        ("samples graded", summary["count"]),
        ("aligned (any)", summary["detected"]),
        ("within tolerance", summary["within_tolerance"]),
        ("tolerance", f"{args.tolerance:.1%} of card diagonal"),
    ]))
    if summary["count"]:
        print()
        print(f"  detection rate        : {summary['detection_rate']:.1%}")
        print(f"  ORACLE detection rate : {summary['oracle_detection_rate']:.1%}"
              "   (true card among the candidates, ignoring ranking)")
        print(f"  ranked below a worse  : {summary['ranked_wrong']} sample(s)")
        print(f"  median corner error   : {summary['median_corner_error']:.4f} of diagonal")
        print(f"  best-candidate error  : {summary['median_oracle_corner_error']:.4f}")
        print(f"  median IoU            : {summary['median_iou']:.4f}")
        print()
        subheading("error distribution")
        for key, value in spread.items():
            print(f"  {key:<5} {value:.4f}")
        if summary["status_counts"]:
            print()
            subheading("engine status")
            for status, count in sorted(summary["status_counts"].items()):
                print(f"  {status:<16} {count}")
        if summary["rotation_shifts"].get("0", 0) != summary["detected"]:
            print("\n  WARNING: some quads only matched after rotating - corner "
                  "order conventions differ.")
    if result.failures:
        print()
        subheading(f"worst {min(5, len(result.failures))} of {len(result.failures)} failing")
        for case in worst(result, limit=5):
            error = "n/a" if case.corner_error is None else f"{case.corner_error:.3f}"
            print(f"  {case.image_id}: status={case.status} error={error}")
    print("\nThis measures performance on *synthetic* captures, not real photographs.")
    return 0
