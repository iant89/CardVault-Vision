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
