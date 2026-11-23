"""``vision export`` commands (Phases 14-17)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..console import format_table, heading, kv_block, subheading
from ..errors import ExportError
from ..export import (
    artifact_manifest,
    export_onnx,
    list_versions,
    load_registry,
    production_model,
    promote,
    register_export,
    spec_from_checkpoint,
    validate_export,
)
from ..logging_utils import get_logger
from .util import emit

_LOG = get_logger("cli.export")


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("export", help="export, validate and promote models")
    ex = parser.add_subparsers(dest="export_command", metavar="ACTION")

    p = ex.add_parser("export", help="export a trained checkpoint to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--version", required=True, help="semantic version, e.g. 1.0.0")
    p.add_argument("--task", default=None,
                   help="name-ocr | number-ocr | card-classifier (inferred from checkpoint)")
    p.add_argument("--out", default="exports", help="exports root directory")
    p.add_argument("--metrics", default=None, help="JSON file of evaluation metrics")
    p.add_argument("--no-validate", action="store_true",
                   help="skip post-export validation (not recommended)")
    p.add_argument("--register", action="store_true", help="add to registry.json")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_export)

    p = ex.add_parser("validate", help="independently validate an exported artifact")
    p.add_argument("--path", default=None, help="artifact directory (or use --all)")
    p.add_argument("--all", action="store_true", help="validate every artifact under exports/")
    p.add_argument("--checkpoint", default=None,
                   help="reference checkpoint for numerical comparison")
    p.add_argument("--exports", default="exports")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_validate)

    p = ex.add_parser("inspect-model", help="show an artifact's metadata and contract")
    p.add_argument("--path", required=True)
    p.add_argument("--manifest", action="store_true",
                   help="print the full consumer manifest instead")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_inspect_model)

    p = ex.add_parser("validate-export", help="alias for `export validate`")
    p.add_argument("--path", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--exports", default="exports")
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_validate)

    p = ex.add_parser("registry", help="list registered versions and production pointers")
    p.add_argument("--exports", default="exports")
    p.add_argument("--task", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_registry)

    p = ex.add_parser("register", help="add an existing artifact to the registry")
    p.add_argument("--path", required=True)
    p.add_argument("--exports", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_register)

    p = ex.add_parser("promote", help="promote a version to production")
    p.add_argument("--task", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--exports", default="exports")
    p.add_argument("--gate", default=None,
                   help="regression gate report JSON; blocks promotion if it failed")
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_promote)


# ---------------------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    from ..training.trainer import build_from_checkpoint

    model, checkpoint = build_from_checkpoint(args.checkpoint, device="cpu")
    metrics = {}
    if args.metrics:
        metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    spec = spec_from_checkpoint(checkpoint, version=args.version, task=args.task,
                                metrics=metrics)
    out_dir = export_onnx(model, spec, args.out, validate=not args.no_validate)
    entry = None
    if args.register:
        entry = register_export(out_dir, exports_root=args.out, metrics=metrics)

    payload = {
        "export_dir": str(out_dir),
        "task": spec.task,
        "version": spec.version,
        "architecture": spec.architecture,
        "num_classes": spec.num_classes,
        "registered": bool(entry),
        "metrics": metrics,
    }
    human = "\n".join([
        heading("Model exported"),
        kv_block([
            ("task", spec.task),
            ("version", spec.version),
            ("architecture", spec.architecture),
            ("num_classes", spec.num_classes),
            ("output", out_dir),
            ("registered", "yes" if entry else "no"),
        ]),
        subheading("Files"),
        format_table(["file", "purpose"], [
            ["model.onnx", "the inference graph"],
            ["metadata.json", "provenance, specs and checksum"],
            ["labels.json", "label list for decoding"],
            ["preprocessing.json", "input contract for the consumer"],
            ["contract.json", "output contract and decoding rules"],
            ["checksum.sha256", "artifact integrity"],
            ["validation.json", "independent validation result"],
        ]),
        "",
        f"  next: vision export promote --task {spec.task} --version {spec.version}",
    ])
    return emit(args, payload, human)


def cmd_validate(args: argparse.Namespace) -> int:
    from ..training.trainer import build_from_checkpoint

    reference = None
    if args.checkpoint:
        reference, _ = build_from_checkpoint(args.checkpoint, device="cpu")

    if args.all:
        reports = []
        for metadata_path in sorted(Path(args.exports).glob("*/*/metadata.json")):
            reports.append(validate_export(metadata_path.parent,
                                           reference_model=reference,
                                           tolerance=args.tolerance))
        if not reports:
            raise ExportError(f"No artifacts found under {args.exports}")
        rows = [[Path(r.path).parent.name + "/" + Path(r.path).name,
                 "APPROVED" if r.passed else "REJECTED",
                 f"{r.max_abs_diff:.2e}" if r.max_abs_diff is not None else "-",
                 "; ".join(r.problems)[:60] or "-"] for r in reports]
        payload = {"reports": [r.to_dict() for r in reports],
                   "approved": sum(1 for r in reports if r.passed),
                   "rejected": sum(1 for r in reports if not r.passed)}
        human = "\n".join([
            heading("Export validation"),
            format_table(["artifact", "status", "max |diff|", "problems"], rows),
        ])
        emit(args, payload, human)
        return 0 if payload["rejected"] == 0 else 2

    if not args.path:
        raise ExportError("Provide --path <artifact dir> or --all")
    report = validate_export(args.path, reference_model=reference,
                             tolerance=args.tolerance)
    human = heading("Export validation") + "\n" + report.summary()
    emit(args, report.to_dict(), human)
    return 0 if report.passed else 2


def cmd_inspect_model(args: argparse.Namespace) -> int:
    directory = Path(args.path)
    if args.manifest:
        parts = directory.parts
        task, version = parts[-2], parts[-1]
        payload = artifact_manifest(task, version, exports_root=directory.parent.parent)
        return emit(args, payload, json.dumps(payload, indent=2))

    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise ExportError(f"No metadata.json in {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = directory / "validation.json"
    status = "-"
    if validation.exists():
        status = json.loads(validation.read_text(encoding="utf-8")).get("status", "-")
    human = "\n".join([
        heading(f"Artifact: {metadata.get('task')} {metadata.get('version')}"),
        kv_block([
            ("architecture", metadata.get("architecture")),
            ("format / opset", f"{metadata.get('format')} / {metadata.get('opset')}"),
            ("num_classes", metadata.get("num_classes")),
            ("input", metadata.get("input_spec", {}).get("shape")),
            ("output", metadata.get("output_spec", {}).get("shape")),
            ("checksum", str(metadata.get("checksum_sha256"))[:24] + "..."),
            ("git commit", metadata.get("git_commit")),
            ("dataset", metadata.get("dataset")),
            ("validated", status),
            ("exported", metadata.get("exported_at")),
        ]),
        subheading("Metrics"),
        json.dumps(metadata.get("metrics") or {}, indent=2),
    ])
    return emit(args, metadata, human)


def cmd_registry(args: argparse.Namespace) -> int:
    registry = load_registry(args.exports)
    models = list_versions(args.task, args.exports)
    productions = registry.get("productions", {})
    rows = [[item["task"], item["version"], item["architecture"],
             "yes" if item.get("validated") else "no",
             (productions.get(item["task"]) or {}).get("version", "-"),
             str(item.get("git_commit"))[:8]] for item in models]
    payload = {"models": models, "productions": productions}
    human = "\n".join([
        heading("Model registry"),
        format_table(["task", "version", "architecture", "validated",
                      "production", "commit"], rows) if rows else "  (no models registered)",
        *([subheading("Production pointers"),
           json.dumps(productions, indent=2)] if productions else []),
    ])
    return emit(args, payload, human)


def cmd_register(args: argparse.Namespace) -> int:
    entry = register_export(args.path, exports_root=args.exports, notes=args.notes)
    human = "\n".join([
        heading("Model registered"),
        kv_block([("task", entry.task), ("version", entry.version),
                  ("validated", entry.validated), ("checksum", entry.checksum_sha256[:16])]),
    ])
    return emit(args, entry.to_dict(), human)


def cmd_promote(args: argparse.Namespace) -> int:
    gate = None
    if args.gate:
        gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    pointer = promote(args.task, args.version, exports_root=args.exports,
                      gate_report=gate, force=args.force)
    human = "\n".join([
        heading("Production pointer updated"),
        kv_block([("task", args.task), ("version", pointer["version"]),
                  ("previous", pointer.get("previous_version") or "none"),
                  ("promoted_at", pointer["promoted_at"])]),
        "",
        "  CardVault should now load this version.",
    ])
    return emit(args, pointer, human)


def cmd_production(args: argparse.Namespace) -> int:
    pointer = production_model(args.task, args.exports)
    if pointer is None:
        raise ExportError(f"No production model for {args.task}")
    return emit(args, pointer, json.dumps(pointer, indent=2))
