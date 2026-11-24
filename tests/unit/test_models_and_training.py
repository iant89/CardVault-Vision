"""Phases 3-6, 12: model contracts and the training loop.

Marked ``torch``; skipped cleanly when PyTorch is absent so the rest of the suite
still runs on a bare interpreter.
"""

from __future__ import annotations

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


class TestVocabularySplit:
    """CER must be separable into "reads glyphs" and "recalls a list".

    With a fixed 12-name vocabulary a model can score well by memorising it, and
    a single CER number cannot tell that apart from real reading.
    """

    def test_seen_and_unseen_are_reported_separately(self):
        from cardvault_vision.evaluation.metrics import evaluate_ocr

        report = evaluate_ocr(
            ["Diego Ortiz", "Diego Ortiz", "Treno bekiz"],
            ["Diego Ortiz", "Julio Ortiz", "Trevor Mbeki"],
            known_references=["Diego Ortiz", "Julio Ortiz"],
        )
        assert set(report.by_vocabulary) == {"seen", "unseen"}
        assert report.by_vocabulary["seen"]["count"] == 2
        assert report.by_vocabulary["unseen"]["count"] == 1
        # Memorised names score better than the genuinely unseen one.
        assert (report.by_vocabulary["seen"]["cer"]
                < report.by_vocabulary["unseen"]["cer"])

    def test_split_is_omitted_without_a_known_vocabulary(self):
        """Callers that do not know the training vocabulary must not get a
        misleading empty or all-zero block."""
        from cardvault_vision.evaluation.metrics import evaluate_ocr

        report = evaluate_ocr(["a"], ["a"])
        assert report.by_vocabulary == {}
        assert "by training vocabulary" not in report.report()

    def test_a_perfect_memoriser_is_exposed(self):
        """The failure mode this exists to catch: exact on seen, hopeless on
        unseen. Overall CER alone would look respectable."""
        from cardvault_vision.evaluation.metrics import evaluate_ocr

        report = evaluate_ocr(
            ["Diego Ortiz", "Diego Ortiz", "XXXXX"],
            ["Diego Ortiz", "Diego Ortiz", "Trevor Mbeki"],
            known_references=["Diego Ortiz"],
        )
        assert report.by_vocabulary["seen"]["exact_match"] == 1.0
        assert report.by_vocabulary["unseen"]["exact_match"] == 0.0


class TestOcrBatchBucketing:
    """Batches must group similar widths; cost scales with the widest crop."""

    def _dataset(self, widths, require_torch):
        from cardvault_vision.training.data import OCRDataset

        dataset = OCRDataset.__new__(OCRDataset)
        import numpy as np

        dataset.items = [(np.zeros((32, w), dtype=np.float32), [1], "x") for w in widths]
        dataset.widths = list(widths)
        return dataset

    def test_batches_do_not_pair_short_crops_with_long_ones(self, require_torch):
        """Measured on real data, one 1166 px crop in a batch of 8 costs 685 ms
        versus 78 ms for a 166 px batch, so a short crop padded to a long batch
        is billed at the long crop's price. Shuffling within a wide window made
        that the common case (23% padding); within a narrow window it is 10%.
        """
        # A realistic mixture: most crops are short, with a long tail.
        widths = [90] * 120 + [150] * 120 + [260] * 60 + [1166] * 4
        dataset = self._dataset(widths, require_torch)
        padding = 0
        real = 0
        for batch in dataset.length_batches(8, shuffle=True, seed=3, buckets=4):
            widest = max(widths[i] for i in batch)
            real += sum(widths[i] for i in batch)
            padding += widest * len(batch)
        assert padding, "no batches produced"
        assert (padding - real) / padding < 0.15

    def test_every_item_appears_exactly_once_per_epoch(self, require_torch):
        widths = [90 + 7 * i for i in range(53)]
        dataset = self._dataset(widths, require_torch)
        seen = [int(i) for batch in dataset.length_batches(8, shuffle=True, seed=1)
                for i in batch]
        assert sorted(seen) == list(range(len(widths)))


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


class TestFieldResolutionPrefersVocabulary:
    """The OCR field must be chosen for *diversity*, not sample count.

    A 700-sample run resolved ``name`` (204 samples, 104 distinct strings) over
    ``player`` (333 samples, 40 distinct, one string repeated 22 times). That
    looked like a count bug, but the diversity ranking agreed with the counts.
    The case that matters is the opposite one: a field with far more samples but
    a nearly fixed vocabulary teaches the head to memorise, so the resolver must
    reject it. Ranking by raw count silently picks it.
    """

    def _manifest(self, name_texts, player_texts):
        from cardvault_vision.datasets.manifest import DatasetManifest
        from cardvault_vision.datasets.schema import Annotation, BBox, FieldAnnotation, SampleRecord

        samples = []

        def add(image_id, field, text, split):
            fields = {field: FieldAnnotation(text=text,
                                             bbox=BBox(0, 0, 10, 10))}
            samples.append(SampleRecord(
                image_id=image_id, card_id=image_id,
                image_path=f"images/{image_id}.jpg",
                annotation=Annotation(image_id=image_id, card_id=image_id,
                                      card_type="MAGIC", fields=fields),
                split=split))

        for i, text in enumerate(name_texts):
            add(f"n{i}", "name", text, "train" if i % 5 else "val")
        for i, text in enumerate(player_texts):
            add(f"p{i}", "player", text, "train" if i % 5 else "val")
        return DatasetManifest(name="t", kind="synthetic", samples=samples)

    def test_abundant_but_repetitive_field_loses_to_a_varied_one(self):
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        # 200 `player` samples drawing on 4 strings, 60 `name` samples drawing on
        # 60 distinct strings. The count-based rule picks `player`.
        player = [f"Star Player {i % 4}" for i in range(200)]
        names = [f"Unique Card {i}" for i in range(60)]
        manifest = self._manifest(names, player)

        assert _resolve_ocr_field(manifest, "name-ocr") == "name"

    def test_canonical_field_wins_when_diversity_is_comparable(self):
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        player = [f"Star Player {i}" for i in range(200)]
        names = [f"Unique Card {i}" for i in range(150)]
        assert _resolve_ocr_field(self._manifest(names, player), "name-ocr") == "name"
