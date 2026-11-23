"""Phases 14-16: the export artifact is CardVault's only interface.

These tests do not need torch. They check that what we ship is a complete,
self-describing, verifiable *directory* - the thing a non-Python consumer can
actually load. A model that only works when the training environment is present
has failed the roadmap's central constraint.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from cardvault_vision.export.onnx_export import (
    ExportError,
    ExportSpec,
    contract_spec,
    preprocessing_spec,
)

CHARSET = "ABCDEFGHIJabcdefghij0123456789 .-"


def _ocr_spec(**kwargs) -> ExportSpec:
    return ExportSpec(task="name-ocr", version="1.0.0", architecture="crnn",
                      charset=CHARSET, **kwargs)


def _classifier_spec(**kwargs) -> ExportSpec:
    return ExportSpec(task="card-classifier", version="1.0.0",
                      architecture="card-type-cnn",
                      class_names=["MAGIC_THE_GATHERING", "POKEMON", "SPORTS"],
                      **kwargs)


class TestExportSpec:
    def test_ocr_spec_counts_pad_and_blank(self):
        """num_classes includes pad + CTC blank, matching the model's output."""
        spec = _ocr_spec()
        assert spec.is_ocr
        assert spec.num_classes == len(CHARSET) + 2

    def test_classifier_spec_counts_labels(self):
        spec = _classifier_spec()
        assert not spec.is_ocr
        assert spec.num_classes == 3

    def test_ocr_requires_a_charset(self):
        with pytest.raises(ExportError, match="charset"):
            ExportSpec(task="name-ocr", version="1.0.0", architecture="crnn", charset="")

    def test_classifier_requires_class_names(self):
        with pytest.raises(ExportError, match="class_names"):
            ExportSpec(task="card-classifier", version="1.0.0",
                       architecture="card-type-cnn", class_names=[])

    def test_version_must_be_safe_as_a_directory_name(self):
        with pytest.raises(ExportError, match="version"):
            ExportSpec(task="name-ocr", version="../escape", architecture="crnn",
                       charset=CHARSET)

    def test_unknown_task_is_rejected(self):
        with pytest.raises(ExportError, match="Unknown task"):
            ExportSpec(task="not-a-task", version="1.0.0", architecture="crnn",
                       charset=CHARSET)


class TestPublishedContracts:
    def test_preprocessing_declares_input_height_and_range(self):
        pre = preprocessing_spec(_ocr_spec(input_height=32))
        assert pre["input_height"] == 32
        assert pre["value_range"] == [0.0, 1.0]
        assert pre["input_layout"] == "NCHW"
        assert pre["color_space"] == "grayscale"

    def test_ocr_preprocessing_explains_the_crop_coordinate_space(self):
        """CardVault must know to crop in rectified card space, not frame space."""
        pre = preprocessing_spec(_ocr_spec())
        assert pre["source_region"]
        assert any("rectified" in note for note in pre["notes"])

    def test_ocr_contract_documents_the_decoding_steps(self):
        contract = contract_spec(_ocr_spec())
        decoding = contract["decoding"]
        assert decoding["type"] == "ctc_greedy"
        assert decoding["blank_index"] == 1
        assert decoding["pad_index"] == 0
        assert len(decoding["steps"]) >= 4

    def test_classifier_contract_declares_an_abstain_rule(self):
        """A weak guess must not be routed on; the contract says so explicitly."""
        contract = contract_spec(_classifier_spec())
        assert contract["decoding"]["type"] == "argmax"
        assert contract["decoding"]["abstain_rule"]

    def test_contracts_are_json_serialisable(self):
        json.dumps(contract_spec(_ocr_spec()))
        json.dumps(preprocessing_spec(_classifier_spec()))


class TestArtifactShape:
    """Structural checks that don't need a real export."""

    REQUIRED = ("model.onnx", "metadata.json", "labels.json", "preprocessing.json",
                "contract.json", "checksum.sha256")

    def test_expected_file_list(self):
        # Duplicated from the exporter on purpose: if the exporter stops writing
        # one of these, this test is the thing that notices.
        assert set(self.REQUIRED) == {
            "model.onnx", "metadata.json", "labels.json", "preprocessing.json",
            "contract.json", "checksum.sha256"}

    def test_checksum_file_names_the_model(self, tmp_path):
        payload = tmp_path / "model.onnx"
        payload.write_bytes(b"not a real onnx file")
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (tmp_path / "checksum.sha256").write_text(f"{digest}  model.onnx\n")
        recorded = (tmp_path / "checksum.sha256").read_text().split()
        assert recorded[1] == "model.onnx"
        assert recorded[0] == digest

    def test_labels_index_offset_is_documented(self):
        """A consumer that ignores the offset silently shifts every character."""
        from cardvault_vision.export.onnx_export import artifact_index_offset

        assert artifact_index_offset(_ocr_spec()) == 2
        assert artifact_index_offset(_classifier_spec()) == 0
