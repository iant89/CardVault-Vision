"""``vision train`` and ``vision test`` commands (Phases 3-6)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..console import format_table, heading, kv_block, subheading
from ..datasets import DatasetLoader, DatasetManifest
from ..errors import ConfigError
from ..logging_utils import get_logger
from ..models import MODEL_REGISTRY, describe_models
from ..synthetic.text import number_charset, universal_charset
from .util import add_common_config_args, emit, resolve_manifest

_LOG = get_logger("cli.train")

TASKS = {"name-ocr": "name", "number-ocr": "number", "card-classifier": None}


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("train", help="train a model")
    train = parser.add_subparsers(dest="train_task", metavar="TASK")

    for task in ("classifier", "name-ocr", "number-ocr"):
        p = train.add_parser(task, help=f"train the {task} model")
        p.add_argument("--dataset", required=True, help="dataset directory or manifest")
        p.add_argument("--out", default=None,
                       help="experiment directory (default: experiments/<task>)")
        p.add_argument("--run-name", default=None,
                       help="label for this run (default: <architecture>-<timestamp>)")
        p.add_argument("--architecture", default=None,
                       help=f"one of {sorted(MODEL_REGISTRY[_task_key(task)])}")
        p.add_argument("--epochs", type=int, default=20)
        p.add_argument("--batch-size", type=_batch_size, default=32,
                       help="int, or 'auto'")
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--weight-decay", type=float, default=1e-4)
        p.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
        p.add_argument("--scheduler", default="cosine",
                       choices=["cosine", "linear", "none", "plateau"])
        p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
        p.add_argument("--mixed-precision", action="store_true", help="AMP (CUDA only)")
        p.add_argument("--workers", type=int, default=0, help="dataloader processes")
        p.add_argument("--threads", type=int, default=None, help="CPU numerical threads")
        p.add_argument("--accumulation", type=int, default=1,
                       help="gradient accumulation micro-batches")
        p.add_argument("--patience", type=int, default=5, help="early-stopping patience")
        p.add_argument("--field", default=None,
                       help="annotation field to read (default: name / number)")
        p.add_argument("--train-split", default="train")
        p.add_argument("--val-split", default="val")
        p.add_argument("--max-train", type=int, default=None, help="cap training samples")
        p.add_argument("--max-val", type=int, default=None, help="cap validation samples")
        p.add_argument("--limit-steps", type=int, default=None,
                       help="steps per epoch (smoke tests)")
        p.add_argument("--seed", type=int, default=0,
                       help="seed for init, shuffling and augmentation")
        p.add_argument("--resume", default=None, help="checkpoint to resume from")
        p.add_argument("--config", action="append", default=[], metavar="FILE")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--json", action="store_true")
        p.set_defaults(handler=cmd_train, task=task)

    p = train.add_parser("compare", help="train every architecture on the same split "
                                        "and compare")
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", default="name-ocr",
                   choices=["name-ocr", "number-ocr", "card-classifier"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=_batch_size, default=32)
    p.add_argument("--max-train", type=int, default=400)
    p.add_argument("--max-val", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="experiments/compare")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_compare)


def _batch_size(value: str) -> int | str:
    """argparse type for ``--batch-size``: an integer, or the literal ``auto``.

    Without this, ``--batch-size 4`` arrives as the string ``"4"`` and the
    auto-sizing helper rejects it - an easy mistake to make, silently plausible
    looking, and only caught when a run refuses to start.
    """
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        size = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer or 'auto', got {value!r}"
        ) from None
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be >= 1")
    return size


def _task_key(task: str) -> str:
    return {"classifier": "card-classifier", "name-ocr": "name-ocr",
            "number-ocr": "number-ocr"}[task]


# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    from ..training.trainer import TrainConfig, train_classifier, train_ocr

    from ..config import Config

    manifest = DatasetManifest.load(resolve_manifest(args.dataset))
    task_key = _task_key(args.task)
    architecture = args.architecture or next(iter(MODEL_REGISTRY[task_key]))
    out_dir = Path(args.out or f"experiments/{args.task}")

    field = args.field or TASKS.get(task_key)
    if task_key != "card-classifier":
        field = field or ("name" if task_key == "name-ocr" else "number")
        charset = universal_charset() if task_key == "name-ocr" else number_charset()
    else:
        charset = ""

    overrides = _config_overrides(args)
    train_config = TrainConfig(
        epochs=int(overrides.get("epochs", args.epochs)),
        batch_size=overrides.get("batch_size", args.batch_size),
        learning_rate=float(overrides.get("learning_rate", args.lr)),
        weight_decay=float(overrides.get("weight_decay", args.weight_decay)),
        optimizer=str(overrides.get("optimizer", args.optimizer)),
        scheduler=str(overrides.get("scheduler", args.scheduler)),
        device=str(overrides.get("device", args.device)),
        mixed_precision=bool(overrides.get("mixed_precision", args.mixed_precision)),
        workers=int(overrides.get("workers", args.workers)),
        threads=overrides.get("threads", args.threads),
        accumulation_steps=int(overrides.get("accumulation_steps", args.accumulation)),
        patience=int(overrides.get("patience", args.patience)),
        seed=int(overrides.get("seed", args.seed or 0)),
        limit_steps=args.limit_steps,
        input_height=int(overrides.get("input_height", 32)),
        input_size=int(overrides.get("input_size", 160)),
        max_width=int(overrides.get("max_width", 1600)),
    )
    train_config.validate()

    train_samples, val_samples = _load_split(manifest, args.train_split, args.val_split,
                                             args.max_train, args.max_val, train_config)
    if not train_samples:
        raise ConfigError(f"No '{args.train_split}' samples in {manifest.name}")

    if task_key == "card-classifier":
        class_names = overrides.get("class_names") or _observed_classes(train_samples)
        result = train_classifier(
            train_samples, val_samples or train_samples, class_names=class_names,
            config=train_config, out_dir=out_dir,
            channels=tuple(overrides.get("channels", (32, 64, 128, 192))),
            dropout=float(overrides.get("dropout", 0.2)),
        )
    else:
        result = train_ocr(
            train_samples, val_samples or train_samples, field=field, charset=charset,
            architecture=architecture, config=train_config, out_dir=out_dir,
            model_kwargs=overrides.get("model_kwargs") or {},
        )
    payload = {**result.to_dict(), "field": field, "dataset": manifest.name}
    return emit(args, payload, result.summary())


def _config_overrides(args: argparse.Namespace) -> dict:
    """Parse ``--config`` / ``--set`` into a flat override dict."""
    from ..config import Config

    config = Config.load(*args.config, sets=args.set)
    return config.to_dict()


def _load_split(manifest: DatasetManifest, train_split: str, val_split: str,
                max_train: int | None, max_val: int | None, config) -> tuple[list, list]:
    splits = {sample.split for sample in manifest.samples if sample.split}
    if not splits:
        # No split information: fall back to a deterministic card-level split so
        # a quick run still cannot leak cards across the two sets.
        from ..datasets.split import split_dataset

        plan = split_dataset(manifest, train=0.8, val=0.2, test=0.0, seed=config.seed)
        plan.apply(manifest.samples)
        _LOG.warning("Dataset has no splits; applied a card-level 80/20 split (seed=%d)",
                     config.seed)

    loader = DatasetLoader(manifest, split=train_split,
                           max_side=None, limit=max_train)
    train_samples = list(loader)
    val_loader = DatasetLoader(manifest, split=val_split, limit=max_val)
    val_samples = list(val_loader)
    if not val_samples and train_split != val_split:
        _LOG.warning("Validation split %r is empty", val_split)
    return train_samples, val_samples


def _observed_classes(samples) -> list[str]:
    from ..datasets.schema import CARD_TYPES

    observed = {sample.annotation.card_type for sample in samples}
    ordered = [name for name in CARD_TYPES if name in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def cmd_compare(args: argparse.Namespace) -> int:
    """Train each architecture under identical conditions and compare."""
    from ..training.trainer import TrainConfig, train_classifier, train_ocr

    manifest = DatasetManifest.load(resolve_manifest(args.dataset))
    task_key = _task_key(args.task)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                         device=args.device, seed=args.seed, patience=0)
    train_samples, val_samples = _load_split(manifest, "train", "val", args.max_train,
                                             args.max_val, config)
    field = "name" if task_key == "name-ocr" else "number"
    charset = universal_charset() if task_key == "name-ocr" else number_charset()

    results = {}
    rows = []
    for architecture in MODEL_REGISTRY[task_key]:
        _LOG.info("Training %s ...", architecture)
        try:
            if task_key == "card-classifier":
                result = train_classifier(
                    train_samples, val_samples or train_samples,
                    class_names=_observed_classes(train_samples), config=config,
                    out_dir=Path(args.out) / architecture,
                )
                metric = result.best_metric
                metric_name = "macro_f1"
            else:
                result = train_ocr(
                    train_samples, val_samples or train_samples, field=field,
                    charset=charset, architecture=architecture, config=config,
                    out_dir=Path(args.out) / architecture,
                )
                metric = result.best_metric
                metric_name = "cer"
        except Exception as exc:  # keep comparing the rest
            _LOG.warning("%s failed: %s", architecture, exc)
            results[architecture] = {"error": str(exc)}
            rows.append([architecture, "failed", "-", str(exc)[:40]])
            continue
        payload = result.to_dict()
        payload["metric_name"] = metric_name
        results[architecture] = payload
        rows.append([architecture, f"{metric:.4f}", f"{result.duration_s:.1f}s",
                     str(result.best_checkpoint)])

    best = min(results.items(),
               key=lambda kv: kv[1].get("best_metric", 1e9)
               if kv[1].get("metric_name") == "cer"
               else -kv[1].get("best_metric", -1e9)) if results else None
    human = "\n".join([
        heading(f"Architecture comparison: {args.task}"),
        kv_block([("dataset", manifest.name), ("epochs", args.epochs),
                  ("train samples", len(train_samples)), ("val samples", len(val_samples)),
                  ("seed", args.seed)]),
        subheading("Results"),
        format_table(["architecture", "metric", "time", "checkpoint"], rows),
        "",
        f"  best: {best[0]}" if best and best[1].get("best_metric") is not None else "",
        "  note: identical split, seed, budget per architecture (roadmap Phase 3 rule).",
    ])
    payload = {"task": args.task, "results": results,
               "best": best[0] if best else None}
    return emit(args, payload, human)
