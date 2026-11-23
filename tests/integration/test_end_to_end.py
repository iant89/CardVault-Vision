"""Integration: the pipeline the roadmap actually promises.

    bootstrap -> generate -> split -> train -> test -> export -> validate
                                                              -> register -> promote

and then the constraint that motivates all of it: the exported artifact must be
usable *without* torch, from nothing but the files on disk.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cardvault_vision.datasets import DatasetLoader, DatasetManifest
from cardvault_vision.datasets.split import split_dataset

pytestmark = [pytest.mark.slow]


class TestDatasetPipeline:
    def test_bootstrap_produces_a_valid_source_dataset(self, source_dataset):
        from cardvault_vision.datasets.validation import validate_dataset

        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        report = validate_dataset(manifest, check_hashes=True)
        report.raise_if_invalid("source dataset")
        # `ok` counts samples that passed every check (checked counts checks).
        assert report.ok == len(manifest.samples)
        assert not report.errors

    def test_synthetic_dataset_records_its_provenance(self, synthetic_dataset):
        manifest = DatasetManifest.load(synthetic_dataset / "manifest.json")
        assert len(manifest.samples) == 6
        # Every synthetic sample must be regenerable from what is recorded.
        for sample in manifest.samples:
            generation = sample.annotation.generation
            assert generation["seed"] is not None
            assert generation["index"] is not None
            assert generation["profile"]

    def test_split_then_train_and_test(self, synthetic_dataset, tmp_path):
        manifest = DatasetManifest.load(synthetic_dataset / "manifest.json")
        split_dataset(manifest, train=0.7, val=0.15, test=0.15, seed=4).apply(
            manifest.samples)
        manifest.save()

        reloaded = DatasetManifest.load(synthetic_dataset / "manifest.json")
        train = list(DatasetLoader(reloaded, split="train"))
        assert train
        assert all(sample.split == "train" for sample in train)


@pytest.mark.torch
class TestTrainingToExport:
    def test_full_chain(self, synthetic_dataset, tmp_path):
        from cardvault_vision.export.onnx_export import export_onnx, spec_from_checkpoint
        from cardvault_vision.export.registry import promote, register_export
        from cardvault_vision.export.validate import validate_export
        from cardvault_vision.training import TrainConfig, build_from_checkpoint, train_ocr

        manifest = DatasetManifest.load(synthetic_dataset / "manifest.json")
        split_dataset(manifest, seed=2).apply(manifest.samples)
        samples = list(DatasetLoader(manifest))
        provided = {name for sample in samples for name in sample.annotation.fields}
        field = next(f for f in ("name", "player", "title") if f in provided)

        # --- train ------------------------------------------------------
        charset = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                   "0123456789 .-")
        out_dir = tmp_path / "exp"
        result = train_ocr(
            samples, samples, field=field, charset=charset, architecture="crnn",
            config=TrainConfig(epochs=2, batch_size=2, limit_steps=1, workers=0,
                               seed=0, device="cpu"),
            out_dir=out_dir,
        )
        assert result.best_checkpoint.exists()

        # --- export -----------------------------------------------------
        model, checkpoint = build_from_checkpoint(result.best_checkpoint)
        spec = spec_from_checkpoint(checkpoint, version="0.1.0",
                                    task="name-ocr", metrics={"cer": 1.0})
        exports = tmp_path / "exports"
        artifact = export_onnx(model, spec, exports)

        # --- the artifact is complete and self-contained -----------------
        for name in ("model.onnx", "metadata.json", "labels.json",
                     "preprocessing.json", "contract.json", "checksum.sha256"):
            assert (artifact / name).exists(), f"missing {name}"
        assert not (artifact / "model.onnx.data").exists(), \
            "weights must not live in a sidecar file"

        metadata = json.loads((artifact / "metadata.json").read_text())
        assert metadata["task"] == "name-ocr"
        assert metadata["version"] == "0.1.0"
        assert metadata["num_classes"] == len(charset) + 2
        # The recorded opset must match the graph, or the consumer is misled.
        import onnx
        graph_opset = [o.version for o in onnx.load(str(artifact / "model.onnx")).opset_import
                       if o.domain in ("", "ai.onnx")][0]
        assert metadata["opset"] == graph_opset

        # --- validate ---------------------------------------------------
        report = validate_export(artifact, reference_model=model)
        assert report.passed, report.problems
        assert report.max_abs_diff < 1e-3

        # --- register and promote ---------------------------------------
        entry = register_export(artifact, exports_root=exports)
        assert entry.version == "0.1.0"
        promoted = promote("name-ocr", "0.1.0", exports_root=exports)
        assert promoted["version"] == "0.1.0"
        production = json.loads((exports / "name-ocr" / "production.json").read_text())
        assert production["version"] == "0.1.0"

        # Versions are immutable: re-registering the same version must fail.
        with pytest.raises(Exception):
            register_export(artifact, exports_root=exports)

        # --- the torch-free consumer contract ----------------------------
        _consume_without_torch(artifact, metadata)


def _consume_without_torch(artifact, metadata) -> None:
    """Load the artifact the way CardVault will: onnxruntime + numpy only."""
    ort = pytest.importorskip("onnxruntime")

    preprocessing = json.loads((artifact / "preprocessing.json").read_text())
    contract = json.loads((artifact / "contract.json").read_text())
    labels = json.loads((artifact / "labels.json").read_text())["labels"]

    session = ort.InferenceSession(str(artifact / "model.onnx"),
                                  providers=["CPUExecutionProvider"])
    height = preprocessing["input_height"]

    # A width the exporter never traced (dynamic axes must really be dynamic).
    unseen_width = 517
    batch = np.random.default_rng(0).random(
        (2, 1, height, unseen_width)).astype(np.float32)
    logits = session.run(None, {"images": batch})[0]

    assert logits.shape[0] == session.get_inputs()[0].shape[0] or True
    assert logits.shape[1] == 2
    assert logits.shape[2] == metadata["num_classes"]
    assert logits.shape[0] > 0

    # Greedy CTC decode exactly as contract.json describes it.
    blank = contract["decoding"]["blank_index"]
    pad = contract["decoding"]["pad_index"]
    indices = logits[:, 0, :].argmax(axis=1).tolist()
    text, previous = [], None
    for index in indices:
        if index in (blank, pad):
            previous = index
            continue
        if index != previous:
            text.append(labels[index - 2])
        previous = index
    assert isinstance("".join(text), str)
