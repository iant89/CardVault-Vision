"""Shared fixtures.

The guiding rule: tests must be fast and deterministic by default. Anything that
needs torch, a GPU or a long render is marked (`torch`, `gpu`, `slow`) and
skipped when unavailable, so ``pytest`` on a bare interpreter still covers the
dataset, synthetic and export-contract layers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: requires PyTorch")
    config.addinivalue_line("markers", "gpu: requires a CUDA device")
    config.addinivalue_line("markers", "slow: slow integration test")


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        return False


def pytest_collection_modifyitems(config, items):
    """Skip torch/gpu-marked tests when the dependency is missing.

    Registering the markers alone did nothing: the docstring in this file claimed
    those tests were "skipped when unavailable", but a bare interpreter without
    torch failed the suite outright with ModuleNotFoundError. A CI job on a slim
    image hits that on its first run, so the marker is now acted on rather than
    merely declared.
    """
    reasons = {}
    if not _module_available("torch"):
        reasons["torch"] = "PyTorch is not installed"
    if not _module_available("torch") or not _cuda_available():
        reasons["gpu"] = "no CUDA device available"

    for item in items:
        for marker, reason in reasons.items():
            if item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
                break


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - unusual torch install
        return False


@pytest.fixture(scope="session")
def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture
def require_torch(has_torch):
    if not has_torch:
        pytest.skip("PyTorch is not installed")


@pytest.fixture(scope="session")
def source_dataset(tmp_path_factory) -> Path:
    """A tiny bootstrap source dataset, generated once per test session.

    Small on purpose (8 cards): every downstream test needs *a* dataset, none of
    them needs a big one, and this keeps the whole suite under a minute.
    """
    from cardvault_vision.cli.commands_dataset import bootstrap_source_dataset

    out = tmp_path_factory.mktemp("source")
    bootstrap_source_dataset(out, count=8, seed=1)
    return out


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory, source_dataset) -> Path:
    """A small synthetic dataset rendered at reduced scale for speed."""
    from cardvault_vision.synthetic import SyntheticConfig
    from cardvault_vision.synthetic.pipeline import generate_dataset

    out = tmp_path_factory.mktemp("synthetic")
    config = SyntheticConfig(
        profile="realistic", count=6, seed=7, render_scale=0.35,
        min_visibility=0.0, min_legibility=0.0,
    )
    generate_dataset(source_dataset, config, output_dir=out)
    return out
