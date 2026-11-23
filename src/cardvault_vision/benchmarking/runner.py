"""Benchmarking framework (Phase 11).

Measures what the roadmap asks for, and writes machine-readable ``benchmark.json``:

    parameters, model size, training time, samples/sec,
    CPU latency, GPU latency, RAM, VRAM,
    CER, WER, exact match, character accuracy

Measurement rules, because benchmarks are only useful if they are honest:

* **Warm up before timing.** The first inference pays lazy allocation and kernel
  autotuning costs; including it overstates latency by 2-10x.
* **Report p50/p95, not just the mean.** A model with a good mean and a bad p95
  will feel bad on a real device.
* **Synchronise CUDA before stopping the clock.** Otherwise you time the enqueue,
  not the work.
* **Record the environment.** A latency number without the machine it came from
  is not a benchmark.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from ..logging_utils import get_logger
from ..reproducibility import executable_versions, write_json

_LOG = get_logger("benchmarking")


@dataclass
class LatencyStats:
    """Latency distribution for one configuration."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_per_s: float

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "throughput_per_s": round(self.throughput_per_s, 2),
        }

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float], batch_size: int = 1) -> "LatencyStats":
        values = sorted(samples_ms)
        if not values:
            return cls(0, 0, 0, 0, 0, 0, 0, 0)
        mean = statistics.fmean(values)
        return cls(
            count=len(values), mean_ms=mean,
            p50_ms=values[int(len(values) * 0.50)],
            p95_ms=values[min(len(values) - 1, int(len(values) * 0.95))],
            p99_ms=values[min(len(values) - 1, int(len(values) * 0.99))],
            min_ms=values[0], max_ms=values[-1],
            throughput_per_s=(1000.0 / mean) * batch_size if mean else 0.0,
        )


def measure_latency(
    call: Callable[[], object],
    *,
    warmup: int = 3,
    iterations: int = 20,
    batch_size: int = 1,
    sync: Callable[[], None] | None = None,
) -> LatencyStats:
    """Time ``call`` and return the distribution."""
    for _ in range(max(0, warmup)):
        call()
    if sync:
        sync()
    timings: list[float] = []
    for _ in range(max(1, iterations)):
        started = time.perf_counter()
        call()
        if sync:
            sync()
        timings.append((time.perf_counter() - started) * 1000.0)
    return LatencyStats.from_samples(timings, batch_size=batch_size)


def measure_memory() -> dict:
    """Process RSS and CUDA allocated/reserved memory."""
    result: dict = {}
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())
        result["process_rss_mb"] = round(process.memory_info().rss / 1024**2, 1)
    except ImportError:
        try:
            with open("/proc/self/status", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        result["process_rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                        break
        except OSError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            result["cuda_allocated_mb"] = round(
                torch.cuda.memory_allocated() / 1024**2, 1)
            result["cuda_reserved_mb"] = round(
                torch.cuda.memory_reserved() / 1024**2, 1)
            result["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    except Exception:  # pragma: no cover
        pass
    return result


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "parameters": int(total),
        "trainable_parameters": int(trainable),
        "size_mb": round(total * 4 / 1024**2, 3),
    }


def measure_model_size(model) -> dict:
    """Serialised state-dict size (what actually ships)."""
    try:
        import io

        import torch

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        state_bytes = buffer.tell()
        buffer = io.BytesIO()
        torch.save(model, buffer)
        full_bytes = buffer.tell()
        return {
            "state_dict_mb": round(state_bytes / 1024**2, 3),
            "torchscript_mb": round(full_bytes / 1024**2, 3),
        }
    except Exception:  # pragma: no cover
        return {}


@dataclass
class BenchmarkResult:
    """One benchmarked model."""

    name: str
    task: str
    metrics: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    size: dict = field(default_factory=dict)
    accuracy: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "task": self.task,
            "metrics": self.metrics,
            "latency": self.latency,
            "memory": self.memory,
            "size": self.size,
            "accuracy": self.accuracy,
            "notes": self.notes,
        }


def benchmark_ocr_model(
    model,
    *,
    name: str,
    task: str = "name-ocr",
    devices: Sequence[str] = ("cpu",),
    input_width: int = 256,
    input_height: int = 32,
    iterations: int = 20,
    batch_size: int = 1,
    accuracy: dict | None = None,
) -> BenchmarkResult:
    """Benchmark an OCR model's latency, size and memory on each device."""
    import torch

    result = BenchmarkResult(name=name, task=task, accuracy=accuracy or {})
    result.metrics.update({
        **count_parameters(model),
        "input_shape": [1, 1, input_height, input_width],
        "batch_size": batch_size,
    })
    result.size.update(measure_model_size(model))
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            result.notes.append(f"skipped {device}: CUDA not available")
            continue
        model.to(device).eval()
        tensor = torch.randn(batch_size, 1, input_height, input_width, device=device)
        sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else None
        def call():
            with torch.no_grad():
                return model(tensor)
        stats = measure_latency(call, warmup=3, iterations=iterations,
                                batch_size=batch_size, sync=sync)
        key = "cpu" if device == "cpu" else "gpu"
        result.latency[key] = stats.to_dict()
    result.memory.update(measure_memory())
    return result


def benchmark_classifier(
    model,
    *,
    name: str = "card-classifier",
    devices: Sequence[str] = ("cpu",),
    input_size: int = 160,
    iterations: int = 20,
    accuracy: dict | None = None,
) -> BenchmarkResult:
    import torch

    result = BenchmarkResult(name=name, task="card-classifier", accuracy=accuracy or {})
    result.metrics.update({**count_parameters(model), "input_size": input_size})
    result.size.update(measure_model_size(model))
    for device in devices:
        if device.startswith("cuda") and not torch.cuda.is_available():
            result.notes.append(f"skipped {device}: CUDA not available")
            continue
        model.to(device).eval()
        tensor = torch.randn(1, 3, input_size, input_size, device=device)
        sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else None
        def call():
            with torch.no_grad():
                return model(tensor)
        stats = measure_latency(call, warmup=3, iterations=iterations, sync=sync)
        result.latency["cpu" if device == "cpu" else "gpu"] = stats.to_dict()
    result.memory.update(measure_memory())
    return result


def benchmark_synthetic_engine(
    source: str | Path,
    *,
    profiles: Sequence[str] | None = None,
    count: int = 10,
    render_scale: float = 1.0,
    seed: int = 7,
) -> dict:
    """Throughput of the data generator - often the real bottleneck, not the model."""
    from ..datasets.manifest import DatasetManifest
    from ..synthetic.generator import GenerationConfig, generate_sample
    from ..synthetic.pipeline import load_source_pool
    from ..synthetic.profiles import CaptureProfile

    manifest = DatasetManifest.load(source)
    pool = load_source_pool(manifest)
    profiles = list(profiles or ["clean", "phone_normal", "scanner"])
    results = {}
    for profile_name in profiles:
        profile = CaptureProfile.get(profile_name)
        timings = []
        rejects = 0
        for index in range(count):
            config = GenerationConfig(profile=profile.to_dict(),
                                      render_scale=render_scale,
                                      min_visibility=0.0, min_legibility=0.0)
            card = pool.choose(np.random.default_rng(index))
            started = time.perf_counter()
            try:
                generate_sample(card, config, seed=seed, index=index)
            except Exception:  # pragma: no cover
                rejects += 1
            timings.append((time.perf_counter() - started) * 1000.0)
        stats = LatencyStats.from_samples(timings)
        results[profile_name] = {**stats.to_dict(), "rejected": rejects}
    return results


def write_benchmark(path: str | Path, *, models: Sequence[BenchmarkResult] = (),
                    extra: dict | None = None) -> Path:
    """Write ``benchmark.json`` with full provenance."""
    payload = {
        "schema_version": "1.0",
        "environment": executable_versions(),
        "models": [model.to_dict() for model in models],
        **(extra or {}),
    }
    return write_json(path, payload)
