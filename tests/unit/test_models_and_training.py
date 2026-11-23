"""Phases 3-6, 12: model contracts and the training loop.

Marked ``torch``; skipped cleanly when PyTorch is absent so the rest of the suite
still runs on a bare interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardvault_vision.models import MODEL_REGISTRY, describe_models

pytestmark = pytest.mark.torch


class TestModelRegistry:
    def test_registry_covers_every_task(self):
        for task in ("name-ocr", "number-ocr", "card-classifier"):
            assert task in MODEL_REGISTRY, f"no architectures registered for {task}"
            assert MODEL_REGISTRY[task]

    def test_ocr_architectures_are_comparable(self):
        """The roadmap requires several architectures to compare on equal terms."""
        assert len(MODEL_REGISTRY["name-ocr"]) >= 3

    def test_describe_models_is_json_safe(self):
        import json

        json.dumps(describe_models())

    def test_unknown_architecture_lists_the_alternatives(self):
        from cardvault_vision.models import build_model

        with pytest.raises(Exception) as excinfo:
            build_model("name-ocr", "not-an-architecture")
        assert "crnn" in str(excinfo.value)


class TestOcrContract:
    """input (N,1,32,W) -> output (T,N,C), C = len(charset)+2, blank at index 1."""

    @pytest.mark.parametrize("architecture", ["crnn", "tcn", "gru", "transformer"])
    def test_shapes_and_time_steps_agree(self, architecture, require_torch):
        import torch

        from cardvault_vision.models import build_ocr

        num_classes = 330
        model = build_ocr(architecture, num_classes=num_classes).eval()
        with torch.no_grad():
            logits = model(torch.zeros(2, 1, 32, 320))
        assert logits.shape[1] == 2
        assert logits.shape[2] == num_classes
        # output_length() is part of the contract: the loss and the export
        # metadata both rely on it rather than guessing.
        assert int(model.output_length(320)) == logits.shape[0]

    @pytest.mark.parametrize("architecture", ["crnn", "tcn", "gru", "transformer"])
    def test_dynamic_width_scales_time_steps(self, architecture, require_torch):
        import torch

        from cardvault_vision.models import build_ocr

        model = build_ocr(architecture, num_classes=330).eval()
        with torch.no_grad():
            narrow = model(torch.zeros(1, 1, 32, 160))
            wide = model(torch.zeros(1, 1, 32, 640))
        assert wide.shape[0] > narrow.shape[0] * 3
        assert int(model.output_length(160)) == narrow.shape[0]
        assert int(model.output_length(640)) == wide.shape[0]

    def test_output_length_returns_a_plain_int(self, require_torch):
        """A tensor here breaks the CTC length arithmetic in subtle ways."""
        from cardvault_vision.models import build_ocr

        value = build_ocr("crnn", num_classes=10).output_length(256)
        assert isinstance(value, int)

    def test_long_names_still_have_enough_time_steps(self, require_torch):
        """CTC needs at least one step per character; this is the regression
        test for the 16x width-reduction bug that made CTC unalignable."""
        from cardvault_vision.models import build_ocr

        model = build_ocr("crnn", num_classes=330)
        # A 19-character name at height 32 is roughly 200 px wide. CTC needs at
        # least one time step per character (and realistically ~2-4x that), so
        # this asserts the hard lower bound, not a comfort margin.
        name = "Crystal Beast Tamer"
        assert int(model.output_length(200)) >= len(name)
        # With the old 16x reduction this was 12 steps for 19 characters.
        assert int(model.output_length(200)) >= 4 * len(name) // 2


class TestClassifierContract:
    def test_shape_and_class_count(self, require_torch):
        import torch

        from cardvault_vision.models import build_classifier

        model = build_classifier(num_classes=9, input_size=160).eval()
        with torch.no_grad():
            logits = model(torch.zeros(3, 3, 160, 160))
        assert logits.shape == (3, 9)

    def test_single_image_batch(self, require_torch):
        import torch

        from cardvault_vision.models import build_classifier

        model = build_classifier(num_classes=4).eval()
        with torch.no_grad():
            assert model(torch.zeros(1, 3, 160, 160)).shape[0] == 1


class TestCtcLoss:
    def test_impossible_alignment_raises_instead_of_scoring_zero(self, require_torch):
        """The regression test for the fake-perfect-loss bug.

        With ``zero_infinity=True`` torch returns 0.0 when targets cannot align,
        which reads as a perfect score. Training must fail loudly instead.
        """
        import torch

        from cardvault_vision.training.losses import ctc_loss_from_widths

        logits = torch.zeros(4, 1, 10)          # only 4 time steps
        targets = torch.arange(0, 20, dtype=torch.long)   # 20 characters
        with pytest.raises(ValueError) as excinfo:
            ctc_loss_from_widths(logits, widths=torch.tensor([16]),
                                 targets=targets,
                                 target_lengths=torch.tensor([20]))
        assert "time step" in str(excinfo.value)

    def test_valid_alignment_returns_a_positive_loss(self, require_torch):
        import torch

        from cardvault_vision.training.losses import ctc_loss_from_widths

        logits = torch.randn(50, 1, 10)
        loss = ctc_loss_from_widths(logits, widths=torch.tensor([200]),
                                    targets=torch.tensor([2, 3, 4]),
                                    target_lengths=torch.tensor([3]))
        assert float(loss) > 0.0


class TestTrainerSmoke:
    def test_training_reduces_loss_and_writes_checkpoints(
            self, synthetic_dataset, tmp_path, require_torch):
        """A tiny real training run: the loop, the scheduler, checkpointing and
        the run record all have to work together."""
        from cardvault_vision.datasets import DatasetLoader, DatasetManifest
        from cardvault_vision.training import TrainConfig, train_ocr

        manifest = DatasetManifest.load(synthetic_dataset / "manifest.json")
        samples = list(DatasetLoader(manifest))
        assert samples
        # Field names are per game ("name" for TCG, "player" for sports), so
        # pick whichever title field this dataset actually provides.
        provided = {name for sample in samples for name in sample.annotation.fields}
        field = next((f for f in ("name", "player", "title") if f in provided), None)
        assert field, f"no usable text field in {sorted(provided)}"

        out = tmp_path / "run"
        result = train_ocr(
            samples, samples, field=field,
            charset="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ",
            architecture="crnn",
            # Three epochs, not one: a single epoch cannot show that the loss is
            # trending down, which is the whole point of this test.
            config=TrainConfig(epochs=3, batch_size=2, limit_steps=2, workers=0,
                               seed=0, device="cpu"),
            out_dir=out,
        )
        assert (out / "best.pt").exists()
        assert (out / "last.pt").exists()
        assert result.history
        # Learning must actually happen: an unchanged loss means gradients are
        # not flowing (frozen weights, a detached graph, or a zeroed LR).
        losses = [epoch.train_loss for epoch in result.history]
        assert losses[-1] < losses[0]
