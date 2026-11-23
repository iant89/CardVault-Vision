"""Benchmarking framework (Phase 11)."""

from __future__ import annotations

from .runner import (
    BenchmarkResult,
    LatencyStats,
    benchmark_classifier,
    benchmark_ocr_model,
    benchmark_synthetic_engine,
    count_parameters,
    measure_latency,
    measure_memory,
    write_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "LatencyStats",
    "benchmark_classifier",
    "benchmark_ocr_model",
    "benchmark_synthetic_engine",
    "count_parameters",
    "measure_latency",
    "measure_memory",
    "write_benchmark",
]
