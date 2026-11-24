# CardVault Vision

A standalone model factory for trading-card recognition.

CardVault Vision owns everything behind the model: dataset management, card
image processing, synthetic scan generation, training, testing, benchmarking,
hard-example mining, optimisation, export and production validation.

**CardVault — the application — consumes only exported, versioned artifacts.** It
never needs Python, PyTorch, training code or dataset code. A release is a
directory of files it can load with any ONNX runtime:

```
exports/name-ocr/1.2.0/
  model.onnx          the inference graph (self-contained, no sidecar weights)
  metadata.json       provenance: version, opset, dataset hash, metrics, checksum
  labels.json         label list plus the index offset to apply
  preprocessing.json  the input contract, implementable without Python
  contract.json       output shape and decoding rules
  checksum.sha256     artifact integrity
  validation.json     independent validation verdict
```

## Why this repository exists

Training a card OCR model on a laptop with clean, evenly-lit scans produces a
model that fails on the first phone photo taken under a kitchen light. The fix is
not augmentation — it is generating data that looks like the deployments the
model will actually meet.

So the centrepiece of this repository is a **synthetic scan engine that simulates
camera captures**, not one that perturbs existing images. It composes geometry,
optics, sensor noise, illumination, materials, backgrounds, occluders and JPEG
artefacts, then auto-generates ground truth for every image.

## Quick start

No installation is required to try the pipeline; the dataset and synthetic layers
need only `numpy` and `pillow`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,export,dev]"

# 1. a small demo source dataset (invented text; no real card data)
vision dataset bootstrap --count 40 --out datasets/source/demo

# 2. simulate 5000 phone captures of those cards
vision synthetic generate --source datasets/source/demo \
    --profile realistic --count 5000 --seed 12345 --out datasets/synthetic/v1

# 3. look before you train - always look before you train
vision synthetic preview --source datasets/source/demo --count 12 \
    --out datasets/synthetic/preview.png

# 4. card-level train/val/test split (a card can never span two splits)
vision dataset split --manifest datasets/synthetic/v1 --seed 7

# 5. train, evaluate, export
vision train name-ocr --dataset datasets/synthetic/v1 --config train-ocr
vision test  name-ocr --checkpoint experiments/name-ocr/best.pt \
                    --dataset datasets/synthetic/v1 --split test
vision export export --checkpoint experiments/name-ocr/best.pt --version 1.0.0
vision export validate --path exports/name-ocr/1.0.0
vision export register --path exports/name-ocr/1.0.0
vision export promote  --task name-ocr --version 1.0.0
```

`vision --help` lists every command; `vision doctor` reports what this machine
can actually do (CPU, CUDA, optional dependencies).

## The three guarantees

These are the properties the rest of the system is allowed to rely on. Each one
is enforced by code and covered by a test.

### 1. The same inputs produce the same dataset

Every sample's pixels depend only on `(seed, index, source card, resolved
profile)`. Random streams are derived per purpose via BLAKE2b
(`RngStreams(seed, index)`), not drawn from one global generator, so re-ordering
work — or resuming an interrupted run — cannot change any individual image.

Every generated image records the exact parameters and stream seeds that made it.
The config fingerprint is written into the manifest, so two runs that differ can
be diffed down to the config key that changed.

*`tests/unit/test_synthetic_determinism.py`*

### 2. A physical card never spans two splits

Splits are assigned by **card id**, not by file. Two photographs of one card
cannot end up one in train and one in test — the classic way a card model
reports a flattering score that collapses in production. Near-duplicate images
are detected and grouped for the same reason.

*`tests/unit/test_datasets.py::test_no_card_leaks_between_train_and_test`*

### 3. The shipped artifact is verifiable and self-contained

Exports are a single `model.onnx` with no external weight sidecar, covered by a
sha256 checksum. `vision export validate` re-runs the exported graph against the
PyTorch model on inputs the exporter never saw (including widths it did not
trace) and records the verdict in `validation.json`. Promotion to production
requires a passing validation report.

*`tests/integration/test_end_to_end.py`*

## Layout

```
configs/                    reviewable YAML: synthetic, training
src/cardvault_vision/
  datasets/                 manifest, hashing, validation, analysis, splitting
  synthetic/                the capture simulator (16 modules)
  layouts/                  card layout definitions and field ROIs
  models/                   OCR heads + card classifier
  training/                 data, losses, trainer
  evaluation/               metrics, hard-example mining
  benchmarking/             latency, memory, throughput
  export/                   ONNX export, validation, versioned registry
  cli/                      the `vision` command
tests/{unit,integration,golden}
scripts/                    reproducible experiment drivers
docs/                       model card, dataset card, decisions
```

## Separating `--workers` from `--threads`

These are different resources and conflating them is a common cause of
"parallelism made it slower":

- `--workers` — data parallelism: how many processes decode and transform images.
- `--threads` — CPU numerical parallelism: how many threads torch/BLAS use.

Setting both to the core count oversubscribes the machine by a factor of the
core count. Default is `--workers 0` (in-process) and `--threads` unset (let the
library decide).

## Optional dependencies

The core (dataset, synthetic, export validation) runs on `numpy` + `pillow`.
Extras unlock the rest:

| Extra | Installs | Unlocks |
|---|---|---|
| `train` | torch | training, checkpoint loading |
| `export` | onnx, onnxruntime | ONNX export, numerical validation |
| `perf` | psutil | memory measurement in benchmarks |
| `dev` | pytest, pytest-cov | the test suite |

A command that needs a missing extra fails with an actionable message naming the
extra to install, rather than an `ImportError` from deep in a stack trace.

## Documentation

- [`docs/model-card.md`](docs/model-card.md) — what a shipped model is, and its limits
- [`docs/dataset-card.md`](docs/dataset-card.md) — what the synthetic data is and is not
- [`docs/decisions.md`](docs/decisions.md) — what was chosen, what was rejected, and why

## Continuous integration

`.github/workflows/ci.yml` runs two jobs on every push to `main` and every PR:

- **Core (no torch)** — installs only `numpy`, `pillow`, `pyyaml`, `pytest` and
  `ruff`, asserts torch is genuinely absent, then lints and runs the suite. This
  is what keeps the "core needs no torch" promise honest; without it, an
  accidental top-level `import torch` would only surface when a user without the
  `train` extra tried to validate a dataset.
- **Full (torch + onnx)** — the whole suite including training, export and the
  torch-free consumer check, plus a determinism smoke test that renders the same
  seed at `--workers 1` and `--workers 2` and compares image bytes.

Tests that need torch are marked `@pytest.mark.torch` and are skipped (not
failed) when it is missing. The marker was registered but never acted on until
now, so a bare interpreter failed with `ModuleNotFoundError` instead of
skipping.

## Status

Phase 0 (foundation), Phase 1 (dataset engine) and Phase 2 (synthetic scan
engine) are complete. Phases 3–19 are runnable infrastructure — training,
evaluation, benchmarking, export and registry all work end to end — with model
quality still to be built up over subsequent iterations.
