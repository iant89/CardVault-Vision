"""``vision test`` and ``vision evaluate`` commands (Phases 4, 9, 18)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..console import format_table, heading, kv_block, subheading
from ..datasets import DatasetLoader, DatasetManifest
from ..errors import ConfigError, NotTrainedError
from ..evaluation.metrics import (
    RegressionGate,
    evaluate_classification,
    evaluate_ocr,
)
from ..logging_utils import get_logger
from ..reproducibility import write_json
from ..synthetic.text import number_charset, universal_charset
from .util import emit, resolve_manifest

_LOG = get_logger("cli.eval")


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("test", help="evaluate a trained model on a held-out split")
    test = parser.add_subparsers(dest="test_task", metavar="TASK")
    for task in ("classifier", "name-ocr", "number-ocr"):
        p = test.add_parser(task, help=f"evaluate the {task} model")
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--dataset", required=True)
        p.add_argument("--split", default="test", help="manifest split to evaluate")
        p.add_argument("--field", default=None)
        p.add_argument("--device", default="cpu")
        p.add_argument("--max-samples", type=int, default=None)
        p.add_argument("--output", default=None, help="write the report as JSON")
        p.add_argument("--predictions", default=None,
                       help="write per-sample predictions as JSONL")
        p.add_argument("--baseline", default=None,
                       help="baseline report JSON for a regression gate (Phase 18)")
        p.add_argument("--json", action="store_true")
        p.set_defaults(handler=cmd_test, task=task)

    p = sub.add_parser("evaluate", help="evaluation utilities")
    ev = p.add_subparsers(dest="eval_command", metavar="ACTION")

    p2 = ev.add_parser("confusion", help="confusion matrix from a predictions file")
    p2.add_argument("--predictions", required=True,
                    help="JSONL of {image_id, prediction, reference}")
    p2.add_argument("--classes", default=None, help="comma-separated class list")
    p2.add_argument("--output", default=None)
    p2.add_argument("--json", action="store_true")
    p2.set_defaults(handler=cmd_confusion)

    p2 = ev.add_parser("hard-examples", help="rank failures by error type (Phase 9)")
    p2.add_argument("--predictions", required=True)
    p2.add_argument("--dataset", default=None, help="dataset for generation metadata")
    p2.add_argument("--output", default=None, help="write a hard-example manifest")
    p2.add_argument("--top", type=int, default=20)
    p2.add_argument("--json", action="store_true")
    p2.set_defaults(handler=cmd_hard_examples)

    p2 = ev.add_parser("gate", help="compare a candidate report against a baseline")
    p2.add_argument("--candidate", required=True)
    p2.add_argument("--baseline", required=True)
    p2.add_argument("--max-cer-delta", type=float, default=0.01)
    p2.add_argument("--max-latency-ratio", type=float, default=1.25)
    p2.add_argument("--json", action="store_true")
    p2.set_defaults(handler=cmd_gate)


# ---------------------------------------------------------------------------


def cmd_test(args: argparse.Namespace) -> int:
    from ..training.trainer import build_from_checkpoint

    manifest = DatasetManifest.load(resolve_manifest(args.dataset))
    model, checkpoint = build_from_checkpoint(args.checkpoint, device=args.device)
    task_key = {"classifier": "card-classifier", "name-ocr": "name-ocr",
                "number-ocr": "number-ocr"}[args.task]

    loader = DatasetLoader(manifest, split=args.split, limit=args.max_samples)
    samples = list(loader)
    if not samples:
        raise ConfigError(
            f"Split {args.split!r} is empty in {manifest.name}; "
            f"run `vision dataset split` first"
        )

    if task_key == "card-classifier":
        return _test_classifier(args, model, checkpoint, samples, manifest)
    return _test_ocr(args, model, checkpoint, samples, manifest, task_key)


def _kept_probabilities(probabilities, best, *, blank: int = 1, pad: int = 0):
    """Probabilities of the characters that survive CTC greedy decoding.

    Greedy decoding collapses repeats and drops blanks/padding. Scoring every
    time step instead would report near-certainty for a blank prediction, so the
    confidence has to be measured on exactly the tokens the caller will see.
    """
    indices = best[:, 0].tolist()
    kept_rows, previous = [], None
    for row, index in enumerate(indices):
        if index in (blank, pad):
            previous = index
            continue
        if index != previous:
            kept_rows.append(row)
        previous = index
    if not kept_rows:
        return probabilities.new_zeros(0)
    return probabilities[kept_rows, 0, [indices[row] for row in kept_rows]]


def _test_ocr(args, model, checkpoint, samples, manifest, task_key: str) -> int:
    import numpy as np
    import torch

    from ..training.data import OCRLineTransform
    from ..synthetic.text import decode_indices, normalize_text

    charset = checkpoint.get("charset") or (
        universal_charset() if task_key == "name-ocr" else number_charset())
    field = args.field or ("name" if task_key == "name-ocr" else "number")
    transform = OCRLineTransform(input_height=int(checkpoint.get("config", {}).get(
        "input_height", 32)))
    model.eval()

    predictions, references, ids, confidences, records = [], [], [], [], []
    with torch.no_grad():
        for sample in samples:
            annotation = sample.annotation.fields.get(field)
            if annotation is None or not annotation.text:
                continue
            try:
                crop = sample.crop(annotation.bbox, pad=2)
            except Exception:
                continue
            tensor = torch.from_numpy(transform(crop))[None, None]
            logits = model(tensor)                        # (T, 1, C)
            best = torch.argmax(logits, dim=2)            # (T, 1)
            sequence = best[:, 0].tolist()
            predicted = decode_indices(sequence, charset)
            # Confidence of the *decoded* sequence: mean probability of the
            # non-blank, post-collapse characters that were actually kept. A mean
            # over blank frames would report ~1.0 for an empty prediction.
            probabilities = torch.softmax(logits, dim=2)
            kept = _kept_probabilities(probabilities, best, blank=1, pad=0)
            confidence = float(kept.mean().item()) if kept.numel() else 0.0
            reference = normalize_text(annotation.text)
            predictions.append(predicted)
            references.append(reference)
            ids.append(sample.image_id)
            confidences.append(confidence)
            records.append({"image_id": sample.image_id,
                            "prediction": predicted, "reference": reference,
                            "confidence": round(confidence, 5),
                            "card_type": sample.annotation.card_type,
                            "profile": sample.profile,
                            "problems": list(sample.annotation.generation.get("problems") or [])})

    if not predictions:
        raise ConfigError(f"No samples with a {field!r} field in split {args.split!r}")

    report = evaluate_ocr(predictions, references, image_ids=ids,
                          confidences=confidences)
    payload = {"task": task_key, "field": field, "split": args.split,
               "dataset": manifest.name, "checkpoint": str(args.checkpoint),
               **report.to_dict()}
    if args.output:
        report.save(args.output)
    if args.predictions:
        _write_predictions(args.predictions, records)

    gate_line = _apply_gate(args, payload)
    human = "\n".join([
        heading(f"{task_key} evaluation ({field})"),
        kv_block([("dataset", manifest.name), ("split", args.split),
                  ("checkpoint", args.checkpoint), ("samples", report.count)]),
        "",
        report.report(),
        *([subheading("Regression gate"), gate_line] if gate_line else []),
    ])
    status = emit(args, payload, human)
    if gate_line and gate_line.startswith("FAILED"):
        return 2
    return status


def _test_classifier(args, model, checkpoint, samples, manifest) -> int:
    import torch

    class_names = checkpoint.get("class_names") or []
    if not class_names:
        raise NotTrainedError(
            "Checkpoint has no class_names; retrain the classifier with this version"
        )
    input_size = int(checkpoint.get("input_size", 160))
    from ..synthetic.imaging import resize, to_float

    model.eval()
    predictions, references, ids, records = [], [], [], []
    with torch.no_grad():
        for sample in samples:
            image = to_float(sample.image)
            resized = resize(image, (input_size, input_size), resample="bilinear")
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)[None].astype("float32"))
            logits = model(tensor)
            index = int(torch.argmax(logits, dim=1).item())
            predicted = class_names[index] if index < len(class_names) else "UNKNOWN"
            predictions.append(predicted)
            references.append(sample.annotation.card_type)
            ids.append(sample.image_id)
            records.append({"image_id": sample.image_id, "prediction": predicted,
                            "reference": sample.annotation.card_type,
                            "confidence": round(float(
                                torch.softmax(logits, dim=1).max().item()), 5)})

    report = evaluate_classification(predictions, references, class_names)
    payload = {"task": "card-classifier", "split": args.split, "dataset": manifest.name,
               "checkpoint": str(args.checkpoint), **report.to_dict()}
    if args.output:
        report.save(args.output)
    if args.predictions:
        _write_predictions(args.predictions, records)
    gate_line = _apply_gate(args, payload)
    human = "\n".join([
        heading("card-classifier evaluation"),
        kv_block([("dataset", manifest.name), ("split", args.split),
                  ("checkpoint", args.checkpoint)]),
        "",
        report.report(),
        *([subheading("Regression gate"), gate_line] if gate_line else []),
    ])
    status = emit(args, payload, human)
    if gate_line and gate_line.startswith("FAILED"):
        return 2
    return status


def _apply_gate(args: argparse.Namespace, payload: dict) -> str | None:
    if not getattr(args, "baseline", None):
        return None
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    passed, reasons = RegressionGate().check(payload, baseline)
    if passed:
        return "PASSED - candidate is not a regression"
    return "FAILED - " + "; ".join(reasons)


def _write_predictions(path: str | Path, records: list[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return target


# ---------------------------------------------------------------------------
# Evaluate subcommands
# ---------------------------------------------------------------------------

def _read_predictions(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Predictions file not found: {p}")
    records = []
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def cmd_confusion(args: argparse.Namespace) -> int:
    records = _read_predictions(args.predictions)
    if not records:
        raise ConfigError("Predictions file is empty")
    classes = ([c.strip() for c in args.classes.split(",")] if args.classes
               else sorted({r["reference"] for r in records} | {r["prediction"] for r in records}))
    report = evaluate_classification([r["prediction"] for r in records],
                                     [r["reference"] for r in records], classes)
    if args.output:
        report.save(args.output)
    human = heading("Confusion matrix") + "\n" + report.report()
    return emit(args, report.to_dict(), human)


def cmd_hard_examples(args: argparse.Namespace) -> int:
    """Rank failures and classify *why* they failed (Phase 9 pipeline)."""
    from ..evaluation.hard_examples import rank_hard_examples

    records = _read_predictions(args.predictions)
    if not records:
        raise ConfigError("Predictions file is empty")
    generation = {}
    if args.dataset:
        manifest = DatasetManifest.load(resolve_manifest(args.dataset))
        generation = {sample.image_id: sample.annotation.generation
                      for sample in manifest.samples}
    payload = rank_hard_examples(records, generation=generation, top=args.top)
    if args.output:
        write_json(args.output, payload)
    rows = [[item["image_id"], f"{item['cer']:.3f}", item["primary_error"],
             item["reference"][:22], item["prediction"][:22]]
            for item in payload["examples"]]
    human = "\n".join([
        heading("Hard examples"),
        kv_block([
            ("predictions", len(records)),
            ("failures", payload["summary"]["failures"]),
            ("overall CER", f"{payload['summary']['cer']:.4f}"),
        ]),
        subheading("Error-type distribution"),
        format_table(["error type", "count"],
                     [[k, v] for k, v in payload["summary"]["by_error_type"].items()]),
        subheading(f"Top {args.top} hardest"),
        format_table(["image", "CER", "primary error", "reference", "prediction"], rows),
        "",
        "  next: vision synthetic generate --problem <error type> to target these.",
    ])
    return emit(args, payload, human)


def cmd_gate(args: argparse.Namespace) -> int:
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    gate = RegressionGate(max_cer_delta=args.max_cer_delta,
                          max_latency_ratio=args.max_latency_ratio)
    passed, reasons = gate.check(candidate, baseline)
    payload = {
        "passed": passed,
        "reasons": reasons,
        "candidate": {k: candidate.get(k) for k in ("cer", "wer", "exact_match",
                                                    "accuracy", "macro_f1")},
        "baseline": {k: baseline.get(k) for k in ("cer", "wer", "exact_match",
                                                  "accuracy", "macro_f1")},
    }
    human = "\n".join([
        heading("Regression gate"),
        kv_block([("result", "PASSED" if passed else "FAILED"),
                  ("baseline CER", f"{baseline.get('cer', float('nan')):.4f}"),
                  ("candidate CER", f"{candidate.get('cer', float('nan')):.4f}")]),
        *([subheading("Blocking reasons"), "\n".join(f"  ! {r}" for r in reasons)]
          if reasons else []),
    ])
    emit(args, payload, human)
    return 0 if passed else 2
