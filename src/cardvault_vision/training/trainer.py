"""Training loops for the OCR heads and the card classifier (Phases 3-6, 12).

Both loops share the same skeleton so experiments are comparable:

    data -> model -> loss -> optimiser/scheduler -> validation -> checkpoint

Design decisions that matter:

* **Validation picks the checkpoint.** The ``best`` checkpoint is chosen by the
  validation metric (CER for OCR, macro-F1 for the classifier), never by training
  loss - training loss is not correlated with the metric you ship on.
* **Mixed precision is opt-in and recorded.** AMP changes numerics, so it is part
  of the run's provenance, not an invisible optimisation.
* **Every run writes a summary**: config, hardware, per-epoch metrics, best
  checkpoint and the exact dataset content hash. That is what makes Phase 17
  (model registry) and Phase 18 (regression testing) possible.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConfigError, NotTrainedError
from ..logging_utils import RunRecorder, get_logger
from ..models import require_torch
from ..reproducibility import library_versions, seed_everything, write_json
from .data import (
    ClassifierDataset,
    DataLoaderConfig,
    OCRDataset,
    build_loader,
    collate_classifier,
    collate_ocr,
    configure_threads,
    resolve_batch_size,
)
from .losses import classification_loss, compute_class_weights, ctc_loss_from_widths

_LOG = get_logger("training")


@dataclass
class TrainConfig:
    """Resolved training hyper-parameters."""

    epochs: int = 20
    batch_size: int | str = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: float = 0.5
    grad_clip: float = 5.0
    accumulation_steps: int = 1
    mixed_precision: bool = False
    device: str = "auto"
    seed: int = 0
    patience: int = 5
    workers: int = 0
    threads: int | None = None
    label_smoothing: float = 0.0
    class_weighting: str = "balanced"
    augment: bool = True
    max_width: int = 1600
    input_height: int = 32
    input_size: int = 160
    limit_steps: int | None = None
    val_every: int = 1

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs, "batch_size": self.batch_size,
            "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
            "optimizer": self.optimizer, "scheduler": self.scheduler,
            "grad_clip": self.grad_clip, "accumulation_steps": self.accumulation_steps,
            "mixed_precision": self.mixed_precision, "seed": self.seed,
            "patience": self.patience, "workers": self.workers, "threads": self.threads,
            "label_smoothing": self.label_smoothing,
            "class_weighting": self.class_weighting,
            "max_width": self.max_width, "input_height": self.input_height,
            "input_size": self.input_size,
        }

    def validate(self) -> None:
        if self.epochs < 1:
            raise ConfigError("epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be > 0")
        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ConfigError(f"Unknown optimizer {self.optimizer!r}")
        if self.scheduler not in {"cosine", "linear", "none", "plateau"}:
            raise ConfigError(f"Unknown scheduler {self.scheduler!r}")
        if self.accumulation_steps < 1:
            raise ConfigError("accumulation_steps must be >= 1")


def resolve_device(requested: str | None) -> str:
    """Resolve ``auto``/``cpu``/``cuda`` without silently downgrading."""
    from ..hardware import resolve_device as _resolve

    return _resolve(requested or "auto")


def _build_optimizer(model, config: TrainConfig):
    import torch

    parameters = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    return torch.optim.SGD(parameters, lr=config.learning_rate, momentum=0.9,
                           weight_decay=config.weight_decay, nesterov=True)


def _build_scheduler(optimizer, config: TrainConfig, total_steps: int):
    import torch

    if config.scheduler == "none":
        return None
    if config.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                          factor=0.5, patience=1)
    warmup = int(max(0, config.warmup_epochs / max(config.epochs, 1) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup and warmup > 0:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(max(progress, 0.0), 1.0)
        if config.scheduler == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None = None
    cer: float | None = None
    exact_match: float | None = None
    accuracy: float | None = None
    macro_f1: float | None = None
    learning_rate: float | None = None
    duration_s: float = 0.0
    samples_per_second: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TrainResult:
    """Everything a completed training run produced."""

    task: str
    architecture: str
    best_checkpoint: Path
    last_checkpoint: Path
    best_metric: float
    metric_name: str
    history: list[EpochMetrics] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    device: str = "cpu"
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "architecture": self.architecture,
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint),
            "metric_name": self.metric_name,
            "best_metric": self.best_metric,
            "history": [epoch.to_dict() for epoch in self.history],
            "config": self.config,
            "device": self.device,
            "duration_s": round(self.duration_s, 2),
        }

    def summary(self) -> str:
        from ..console import format_table, heading, kv_block

        rows = []
        for epoch in self.history:
            rows.append([
                epoch.epoch,
                f"{epoch.train_loss:.4f}",
                f"{epoch.val_loss:.4f}" if epoch.val_loss is not None else "-",
                f"{epoch.cer:.4f}" if epoch.cer is not None else
                (f"{epoch.macro_f1:.4f}" if epoch.macro_f1 is not None else "-"),
                f"{epoch.samples_per_second:.0f}",
            ])
        return "\n".join([
            heading(f"Training finished: {self.task} / {self.architecture}"),
            kv_block([
                ("epochs", len(self.history)),
                ("best", f"{self.metric_name}={self.best_metric:.4f}"),
                ("checkpoint", self.best_checkpoint),
                ("device", self.device),
                ("duration", f"{self.duration_s:.1f}s"),
            ]),
            "",
            format_table(["epoch", "train loss", "val loss",
                          "CER" if self.metric_name == "cer" else "macro F1", "samples/s"],
                         rows),
        ])


# ---------------------------------------------------------------------------
# OCR training
# ---------------------------------------------------------------------------

def train_ocr(
    train_samples: Sequence,
    val_samples: Sequence,
    *,
    field: str,
    charset: str,
    architecture: str = "crnn",
    config: TrainConfig | None = None,
    out_dir: str | Path = "experiments/ocr",
    model_kwargs: dict | None = None,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Train an OCR head with CTC.

    ``train_samples`` / ``val_samples`` are :class:`~cardvault_vision.datasets.Sample`
    objects - the loop never loads images itself, so the caller controls which
    split and which fields are used.
    """
    require_torch("OCR training")
    import torch

    from ..models import build_ocr
    from ..synthetic.text import decode_indices

    config = config or TrainConfig()
    config.validate()
    configure_threads(config.threads)
    seed = seed_everything(config.seed)
    device = resolve_device(config.device)
    batch_size = resolve_batch_size(config.batch_size, device=device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(out_path / "run", quiet=True)

    num_classes = len(charset) + 2
    model = build_ocr(architecture, num_classes=num_classes, **(model_kwargs or {}))
    model.to(device)

    train_set = OCRDataset(train_samples, field, charset, input_height=config.input_height,
                           augment=config.augment, max_width=config.max_width)
    val_set = OCRDataset(val_samples, field, charset, input_height=config.input_height,
                         augment=False, max_width=config.max_width)
    loader_config = DataLoaderConfig(batch_size=batch_size, workers=config.workers,
                                     threads=config.threads)
    optimizer = _build_optimizer(model, config)
    steps_per_epoch = max(1, math.ceil(len(train_set) / batch_size))
    total_steps = steps_per_epoch * config.epochs
    scheduler = _build_scheduler(optimizer, config, total_steps)
    amp_enabled = bool(config.mixed_precision and device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    recorder.event("train_start", task="name-ocr", field=field, architecture=architecture,
                   samples=len(train_set), val_samples=len(val_set), batch_size=batch_size,
                   device=device, seed=seed, mixed_precision=amp_enabled)
    _LOG.info("Training %s on field %r: %d train / %d val samples, device=%s, bs=%d",
              architecture, field, len(train_set), len(val_set), device, batch_size)

    start_epoch = 0
    best_metric = float("inf")   # OCR: lower CER is better
    history: list[EpochMetrics] = []
    started = time.perf_counter()

    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
        history = [EpochMetrics(**m) for m in checkpoint.get("history", [])]
        _LOG.info("Resumed from %s at epoch %d", resume_from, start_epoch)

    best_checkpoint = out_path / "best.pt"
    last_checkpoint = out_path / "last.pt"
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, indices in enumerate(train_set.length_batches(
                batch_size, shuffle=True, seed=seed + epoch, buckets=4)):
            batch = collate_ocr([train_set.items[i] for i in indices])
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            target_lengths = batch["target_lengths"]
            with torch.autocast(device_type=device.split(":")[0], enabled=amp_enabled):
                logits = model(images)
                loss = ctc_loss_from_widths(logits, batch["widths"], targets,
                                            target_lengths, model=model)
                loss = loss / max(1, config.accumulation_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_index + 1) % max(1, config.accumulation_steps) == 0:
                if config.grad_clip:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and config.scheduler != "plateau":
                    scheduler.step()
            total_loss += float(loss.item()) * len(indices)
            seen += len(indices)
            if config.limit_steps and batch_index + 1 >= config.limit_steps:
                break
        train_loss = total_loss / max(seen, 1)

        val_loss = cer = em = None
        if (epoch + 1) % max(1, config.val_every) == 0:
            val_loss, cer, em, _ = _validate_ocr(
                model, val_set, charset, batch_size, device, decode_indices,
                loader_config, amp_enabled, config,
            )
        duration = time.perf_counter() - epoch_started
        metrics = EpochMetrics(
            epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, cer=cer,
            exact_match=em, duration_s=duration,
            samples_per_second=seen / duration if duration else 0.0,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        recorder.metric("train_loss", train_loss, step=epoch + 1)
        if cer is not None:
            recorder.metric("cer", cer, step=epoch + 1)
            recorder.metric("exact_match", em or 0.0, step=epoch + 1)
        _LOG.info("epoch %d/%d  loss=%.4f  %s", epoch + 1, config.epochs, train_loss,
                  f"cer={cer:.4f} em={em:.2%}" if cer is not None else "no validation")

        if config.scheduler == "plateau" and scheduler is not None and cer is not None:
            scheduler.step(cer)

        candidate = cer if cer is not None else float("inf")
        improved = candidate < best_metric - 1e-6
        if improved:
            best_metric = candidate
            epochs_without_improvement = 0
            _save_checkpoint(best_checkpoint, model, optimizer, epoch + 1, best_metric,
                             history, architecture, charset, field, config)
            recorder.event("best_checkpoint", epoch=epoch + 1, cer=best_metric)
        else:
            epochs_without_improvement += 1
        _save_checkpoint(last_checkpoint, model, optimizer, epoch + 1, best_metric, history,
                         architecture, charset, field, config)

        if config.patience and epochs_without_improvement >= config.patience:
            _LOG.info("Early stopping: no improvement for %d epoch(s)",
                      epochs_without_improvement)
            break

    result = TrainResult(
        task="name-ocr", architecture=architecture, best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint, best_metric=best_metric, metric_name="cer",
        history=history, config=config.to_dict(), device=device,
        duration_s=time.perf_counter() - started,
    )
    write_json(out_path / "training.json", {
        **result.to_dict(),
        "field": field,
        "charset": charset,
        "num_classes": num_classes,
        "libraries": library_versions(),
    })
    recorder.write_summary(result.to_dict())
    if not best_checkpoint.exists():
        best_checkpoint = last_checkpoint
    result.best_checkpoint = best_checkpoint
    return result


def _validate_ocr(model, val_set: OCRDataset, charset: str, batch_size: int,
                  device: str, decode_indices, loader_config: DataLoaderConfig,
                  amp_enabled: bool, config: TrainConfig):
    """Run the validation split, returning ``(loss, cer, exact_match, report)``."""
    import torch

    from ..evaluation.metrics import evaluate_ocr

    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for start in range(0, len(val_set), batch_size):
            batch_items = val_set.items[start:start + batch_size]
            batch = collate_ocr(batch_items)
            images = batch["images"].to(device)
            with torch.autocast(device_type=device.split(":")[0], enabled=amp_enabled):
                logits = model(images)
                loss = ctc_loss_from_widths(logits, batch["widths"], batch["targets"],
                                            batch["target_lengths"], model=model)
            total_loss += float(loss.item()) * len(batch_items)
            seen += len(batch_items)
            best_path = torch.argmax(logits, dim=2)          # (T, N)
            for column in range(best_path.shape[1]):
                sequence = best_path[:, column].tolist()
                predictions.append(decode_indices(sequence, charset))
            references.extend(batch["texts"])
    loss_value = total_loss / max(seen, 1)
    report = evaluate_ocr(predictions, references)
    return loss_value, report.cer, report.exact_match, report


# ---------------------------------------------------------------------------
# Classifier training
# ---------------------------------------------------------------------------

def train_classifier(
    train_samples: Sequence,
    val_samples: Sequence,
    *,
    class_names: Sequence[str],
    config: TrainConfig | None = None,
    out_dir: str | Path = "experiments/classifier",
    channels: Sequence[int] = (32, 64, 128, 192),
    dropout: float = 0.2,
    resume_from: str | Path | None = None,
) -> TrainResult:
    """Train the card-type classifier."""
    require_torch("classifier training")
    import torch

    from ..models import build_classifier

    config = config or TrainConfig()
    config.validate()
    configure_threads(config.threads)
    seed = seed_everything(config.seed)
    device = resolve_device(config.device)
    batch_size = resolve_batch_size(config.batch_size, device=device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    recorder = RunRecorder(out_path / "run", quiet=True)

    model = build_classifier(num_classes=len(class_names), channels=tuple(channels),
                             dropout=dropout, input_size=config.input_size)
    model.to(device)

    train_set = ClassifierDataset(train_samples, class_names,
                                  input_size=config.input_size, augment=config.augment)
    val_set = ClassifierDataset(val_samples, class_names,
                                input_size=config.input_size, augment=False)

    counts = [0] * len(class_names)
    for _, label in train_set.items:
        counts[label] += 1
    class_weights = compute_class_weights(counts, mode=config.class_weighting)

    train_loader = build_loader(train_set.as_dataset(), DataLoaderConfig(
        batch_size=batch_size, workers=config.workers), shuffle=True,
        collate_fn=collate_classifier)
    val_loader = build_loader(val_set.as_dataset(), DataLoaderConfig(
        batch_size=batch_size, workers=config.workers), shuffle=False,
        collate_fn=collate_classifier)

    optimizer = _build_optimizer(model, config)
    total_steps = max(1, len(train_loader)) * config.epochs
    scheduler = _build_scheduler(optimizer, config, total_steps)
    amp_enabled = bool(config.mixed_precision and device.startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    recorder.event("train_start", task="card-classifier", samples=len(train_set),
                   val_samples=len(val_set), batch_size=batch_size, device=device,
                   seed=seed, classes=list(class_names))
    _LOG.info("Training card classifier: %d train / %d val, %d classes, device=%s",
              len(train_set), len(val_set), len(class_names), device)

    best_f1 = -1.0
    history: list[EpochMetrics] = []
    best_checkpoint = out_path / "best.pt"
    last_checkpoint = out_path / "last.pt"
    started = time.perf_counter()
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_f1 = float(checkpoint.get("best_metric", best_f1))
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)
            with torch.autocast(device_type=device.split(":")[0], enabled=amp_enabled):
                logits = model(images)
                loss = classification_loss(logits, labels, class_weights=class_weights,
                                           label_smoothing=config.label_smoothing)
                loss = loss / max(1, config.accumulation_steps)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % max(1, config.accumulation_steps) == 0:
                if config.grad_clip:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and config.scheduler != "plateau":
                    scheduler.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
            if config.limit_steps and step + 1 >= config.limit_steps:
                break
        train_loss = total_loss / max(seen, 1)

        from ..evaluation.metrics import evaluate_classification

        model.eval()
        val_loss = 0.0
        predicted: list[int] = []
        actual: list[int] = []
        val_seen = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                loss = classification_loss(logits, labels.to(device),
                                           class_weights=class_weights)
                val_loss += float(loss.item()) * len(labels)
                val_seen += len(labels)
                predicted.extend(torch.argmax(logits, dim=1).tolist())
                actual.extend(labels.tolist())
        report = evaluate_classification(predicted, [class_names[i] for i in actual],
                                        class_names)
        val_loss /= max(val_seen, 1)

        duration = time.perf_counter() - epoch_started
        metrics = EpochMetrics(
            epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
            accuracy=report.accuracy, macro_f1=report.macro_f1, duration_s=duration,
            samples_per_second=seen / duration if duration else 0.0,
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        history.append(metrics)
        recorder.metric("train_loss", train_loss, step=epoch + 1)
        recorder.metric("val_loss", val_loss, step=epoch + 1)
        recorder.metric("macro_f1", report.macro_f1, step=epoch + 1)
        recorder.metric("accuracy", report.accuracy, step=epoch + 1)
        _LOG.info("epoch %d/%d  loss=%.4f  val_loss=%.4f  acc=%.2f%%  macroF1=%.4f",
                  epoch + 1, config.epochs, train_loss, val_loss,
                  report.accuracy * 100, report.macro_f1)

        if config.scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_loss)

        improved = report.macro_f1 > best_f1 + 1e-6
        if improved:
            best_f1 = report.macro_f1
            epochs_without_improvement = 0
            _save_checkpoint(best_checkpoint, model, optimizer, epoch + 1, best_f1,
                             history, "card-type-cnn", "", "", config,
                             extra={"class_names": list(class_names),
                                    "input_size": config.input_size})
        else:
            epochs_without_improvement += 1
        _save_checkpoint(last_checkpoint, model, optimizer, epoch + 1, best_f1, history,
                         "card-type-cnn", "", "", config,
                         extra={"class_names": list(class_names),
                                "input_size": config.input_size})
        if config.patience and epochs_without_improvement >= config.patience:
            _LOG.info("Early stopping: no improvement for %d epoch(s)",
                      epochs_without_improvement)
            break

    result = TrainResult(
        task="card-classifier", architecture="card-type-cnn",
        best_checkpoint=best_checkpoint, last_checkpoint=last_checkpoint,
        best_metric=best_f1, metric_name="macro_f1", history=history,
        config=config.to_dict(), device=device,
        duration_s=time.perf_counter() - started,
    )
    write_json(out_path / "training.json", {
        **result.to_dict(), "class_names": list(class_names),
        "libraries": library_versions(),
    })
    recorder.write_summary(result.to_dict())
    return result


def _save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float,
                     history: list[EpochMetrics], architecture: str, charset: str,
                     field: str, config: TrainConfig, extra: dict | None = None) -> None:
    import torch

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "history": [m.__dict__ for m in history],
        "architecture": architecture,
        "charset": charset,
        "field": field,
        "config": config.to_dict(),
        **(extra or {}),
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> dict:
    """Load a checkpoint, raising a precise error when it is missing."""
    require_torch("model loading")
    import torch

    p = Path(path)
    if not p.exists():
        raise NotTrainedError(
            f"Checkpoint not found: {p}\n"
            "Train a model first, e.g.: vision train name-ocr --dataset <synthetic dataset>"
        )
    return torch.load(p, map_location=device)


def build_from_checkpoint(path: str | Path, *, device: str = "cpu"):
    """Rebuild a model from a checkpoint (used by eval/export)."""
    require_torch("model loading")
    from ..models import build_classifier, build_ocr

    checkpoint = load_checkpoint(path, device=device)
    architecture = checkpoint.get("architecture", "crnn")
    if architecture == "card-type-cnn":
        class_names = checkpoint.get("class_names") or []
        model = build_classifier(num_classes=len(class_names),
                                 input_size=int(checkpoint.get("input_size", 160)))
    else:
        charset = checkpoint.get("charset", "")
        model = build_ocr(architecture, num_classes=len(charset) + 2)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint
