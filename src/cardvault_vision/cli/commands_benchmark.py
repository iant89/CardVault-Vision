"""``vision benchmark`` command (Phase 11)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..benchmarking import (
    benchmark_classifier,
    benchmark_ocr_model,
    benchmark_synthetic_engine,
    write_benchmark,
)
from ..console import format_table, heading, kv_block, subheading
from ..errors import ConfigError
from ..logging_utils import get_logger
from .util import emit

_LOG = get_logger("cli.benchmark")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("benchmark", help="benchmark models and the data generator")
    p.add_argument("--checkpoint", action="append", default=[],
                   help="checkpoint to benchmark (repeatable)")
    p.add_argument("--all", action="store_true",
                   help="benchmark every architecture at its default size")
    p.add_argument("--synthetic", default=None, metavar="SOURCE_MANIFEST",
                   help="also benchmark synthetic generation from this source")
    p.add_argument("--synthetic-count", type=int, default=10)
    p.add_argument("--devices", default="cpu", help="comma-separated: cpu,cuda")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--width", type=int, default=256, help="OCR input width for timing")
    p.add_argument("--output", default="benchmark.json")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_benchmark)


def cmd_benchmark(args: argparse.Namespace) -> int:
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    results = []
    rows = []
    extra: dict = {}

    checkpoints = list(args.checkpoint)
    if args.all and not checkpoints:
        results.extend(_benchmark_architectures(args, devices, rows))
    for checkpoint in checkpoints:
        from ..training.trainer import build_from_checkpoint

        model, meta = build_from_checkpoint(checkpoint, device="cpu")
        architecture = meta.get("architecture", "crnn")
        if architecture == "card-type-cnn":
            benchmark = benchmark_classifier(model, name=Path(checkpoint).parent.name,
                                             devices=devices,
                                             input_size=int(meta.get("input_size", 160)))
        else:
            benchmark = benchmark_ocr_model(
                model, name=f"{architecture}:{Path(checkpoint).parent.name}",
                devices=devices, input_width=args.width,
                input_height=int((meta.get("config") or {}).get("input_height", 32)),
                iterations=args.iterations)
        results.append(benchmark)
        cpu = benchmark.latency.get("cpu", {})
        rows.append([benchmark.name, str(benchmark.metrics.get("parameters", "-")),
                     f"{benchmark.metrics.get('size_mb', 0):.2f}",
                     f"{cpu.get('mean_ms', 0):.1f}",
                     f"{cpu.get('p95_ms', 0):.1f}"])

    if args.synthetic:
        extra["synthetic"] = benchmark_synthetic_engine(
            args.synthetic, count=args.synthetic_count)
        extra["synthetic"]["source"] = str(args.synthetic)

    if not results and "synthetic" not in extra:
        raise ConfigError(
            "Nothing to benchmark. Pass --checkpoint <file> or --all or --synthetic <manifest>"
        )

    if results:
        write_benchmark(args.output, models=results, extra=extra)
    payload = {"models": [r.to_dict() for r in results], **extra,
               "output": args.output if results else None}

    sections = [heading("Benchmark")]
    if results:
        sections += [
            subheading("Models"),
            format_table(["model", "params", "size MB", "CPU mean ms", "CPU p95 ms"], rows),
        ]
        for benchmark in results:
            if benchmark.notes:
                sections.append("\n".join(f"  ! {n}" for n in benchmark.notes))
    if "synthetic" in extra:
        synth_rows = [[name, f"{v['mean_ms']:.0f}", f"{v['throughput_per_s']:.2f}", v["rejected"]]
                      for name, v in extra["synthetic"].items() if isinstance(v, dict)]
        sections += [
            subheading("Synthetic generation (single process)"),
            format_table(["profile", "mean ms", "images/s", "rejected"], synth_rows),
        ]
    sections += ["", f"  written: {args.output}"]
    return emit(args, payload, "\n".join(sections))


def _benchmark_architectures(args, devices, rows):
    """Benchmark every OCR architecture at its default size."""
    from ..models import MODEL_REGISTRY, build_model

    results = []
    for task, architectures in MODEL_REGISTRY.items():
        for architecture in architectures:
            try:
                model = build_model(task, architecture)
            except Exception as exc:  # pragma: no cover
                _LOG.warning("Could not build %s/%s: %s", task, architecture, exc)
                continue
            if task == "card-classifier":
                benchmark = benchmark_classifier(model, name=f"{task}:{architecture}",
                                                 devices=devices,
                                                 iterations=args.iterations)
            else:
                benchmark = benchmark_ocr_model(
                    model, name=f"{task}:{architecture}", task=task, devices=devices,
                    input_width=args.width, iterations=args.iterations)
            results.append(benchmark)
            cpu = benchmark.latency.get("cpu", {})
            rows.append([benchmark.name, str(benchmark.metrics.get("parameters", "-")),
                         f"{benchmark.metrics.get('size_mb', 0):.2f}",
                         f"{cpu.get('mean_ms', 0):.1f}",
                         f"{cpu.get('p95_ms', 0):.1f}"])
    return results
