"""Phase 1: dataset manifests, hashing, validation and splitting."""

from __future__ import annotations

import json

import pytest

from cardvault_vision.datasets import DatasetLoader, DatasetManifest
from cardvault_vision.datasets.hashing import dataset_content_hash, file_sha256
from cardvault_vision.datasets.split import split_dataset
from cardvault_vision.errors import DatasetError


class TestManifest:
    def test_bootstrap_manifest_round_trips(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        assert len(manifest.samples) == 8
        assert manifest.kind == "source"
        # Every sample must carry its annotation and image checksum.
        for sample in manifest.samples:
            assert sample.annotation.fields
            assert len(sample.sha256) == 64

    def test_content_hash_is_stable_across_reloads(self, source_dataset):
        first = DatasetManifest.load(source_dataset / "manifest.json")
        second = DatasetManifest.load(source_dataset / "manifest.json")
        assert first.content_hash() == second.content_hash()

    def test_manifest_is_valid_json(self, source_dataset):
        payload = json.loads((source_dataset / "manifest.json").read_text())
        assert payload["samples"]

    def test_saved_manifest_does_not_drift(self, source_dataset, tmp_path):
        """Load -> save -> load must not change the content hash.

        A hash that changes on a no-op round trip would make every run look like
        a different dataset and destroy provenance.
        """
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        before = manifest.content_hash()
        manifest.save()
        reloaded = DatasetManifest.load(source_dataset / "manifest.json")
        assert reloaded.content_hash() == before


class TestHashing:
    def test_file_hash_matches_direct_hash(self, tmp_path):
        import hashlib

        path = tmp_path / "x.bin"
        payload = b"cardvault" * 1000
        path.write_bytes(payload)
        assert file_sha256(path) == hashlib.sha256(payload).hexdigest()

    def test_content_hash_depends_on_order(self):
        # Two files with identical bytes must still hash differently when their
        # names differ, because the name is part of the dataset's identity.
        assert (dataset_content_hash([("a.png", "x"), ("b.png", "y")])
                != dataset_content_hash([("b.png", "x"), ("a.png", "y")]))

    def test_content_hash_is_deterministic(self):
        entries = [("a.png", "deadbeef")]
        assert dataset_content_hash(entries) == dataset_content_hash(entries)

    def test_content_hash_tracks_bytes(self):
        assert (dataset_content_hash([("a.png", "deadbeef")])
                != dataset_content_hash([("a.png", "feedface")]))


class TestOcrFieldResolution:
    """``train name-ocr`` must pick a field the split actually contains."""

    def _manifest(self, cards):
        from cardvault_vision.datasets import DatasetManifest
        from cardvault_vision.datasets.schema import Annotation, BBox, FieldAnnotation, SampleRecord

        samples = []
        for index, (card_type, fields) in enumerate(cards):
            annotation = Annotation(
                image_id=f"{index:04d}", card_id=f"card{index}", card_type=card_type,
                width=100, height=100,
                fields={name: FieldAnnotation(
                    text=text, bbox=BBox(1, 1, 40, 20)) for name, text in fields.items()},
            )
            samples.append(SampleRecord(
                image_id=f"{index:04d}", card_id=f"card{index}",
                image_path=f"images/{index:04d}.jpg", annotation=annotation,
                split="train" if index % 4 else "val",
            ))
        return DatasetManifest(name="fixture", samples=samples)

    def test_title_field_falls_back_to_player_when_name_is_rare(self):
        """Sports layouts put the title in ``player``; name-ocr used to hardcode
        ``name`` and silently skip 92% of a mixed dataset."""
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        cards = [("POKEMON", {"name": "Pikachu"})]
        cards += [("BASEBALL", {"player": "Ken Griffey"})] * 8
        # Enough ``name`` samples in every split to be usable, but sports dominate.
        assert _resolve_ocr_field(self._manifest(cards), "name-ocr") == "player"

    def test_canonical_field_is_kept_when_it_is_usable(self):
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        cards = [("POKEMON", {"name": "Pikachu"})] * 8
        assert _resolve_ocr_field(self._manifest(cards), "name-ocr") == "name"

    def test_test_split_absence_only_warns(self):
        """Test is often a handful of samples. A field with 598 usable training
        samples must not be rejected because 5 test samples lack it -- warn and
        keep going instead."""
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        manifest = self._manifest([("BASEBALL", {"player": "A"})] * 8)
        for sample in manifest.samples:
            if sample.split == "test":
                sample.annotation.fields.pop("player", None)
        assert _resolve_ocr_field(manifest, "name-ocr") == "player"

    def test_field_missing_from_a_split_is_not_chosen(self):
        """A field that covers train but no validation is worse than useless:
        it yields an empty val set and a confusing training failure."""
        from cardvault_vision.cli.commands_models import _resolve_ocr_field

        cards = [("BASEBALL", {"player": "A"})] * 3 + [("POKEMON", {"name": "B"})] * 3
        # Drop ``player`` from every val sample.
        manifest = self._manifest(cards)
        for sample in manifest.samples:
            if sample.split == "val":
                sample.annotation.fields.pop("player", None)
        assert _resolve_ocr_field(manifest, "name-ocr") == "name"


class TestSplitting:
    def test_splits_are_disjoint_and_complete(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        plan = split_dataset(manifest, train=0.6, val=0.2, test=0.2, seed=3)
        assigned = list(plan.assignments)
        assert len(assigned) == len(set(assigned)), "a card appears in two splits"
        assert set(assigned) == set(manifest.card_ids)

    def test_each_stratum_is_split_proportionally(self):
        """Allocation must happen per stratum, not over one global sequence.

        The old code interleaved every stratum into one list and sliced a prefix
        per split. With a dominant stratum first, val and test were filled
        entirely from it: a 4:1 split of a mixed dataset sent all 621 sports
        cards to train and left val holding only TCG types, so training a field
        that only sports cards carry had an empty validation set.

        Four equal strata make the property unambiguous: each must contribute to
        all three splits, and every card must be assigned exactly once.
        """
        from collections import Counter

        from cardvault_vision.datasets.split import _allocate

        strata = [[[f"t{i}c{j}"] for j in range(25)] for i in range(4)]
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        assignments: dict[str, str] = {}
        for index, groups in enumerate(strata):
            assignments.update(
                _allocate(groups, ratios, warning_prefix=f"s{index}"))

        assert len(assignments) == sum(len(group) for stratum in strata
                                       for group in stratum)
        for index, groups in enumerate(strata):
            counts = Counter(assignments[card] for group in groups for card in group)
            assert counts["val"] > 0 and counts["test"] > 0, (
                f"stratum s{index} did not reach every split: {dict(counts)}")

    def test_no_card_leaks_between_train_and_test(self, source_dataset):
        """The central Phase 1 guarantee.

        Splitting by *card* rather than by image is what stops one physical card
        photographed twice from being both a training example and a test
        example - the classic way a card model reports an unrealistically high
        score.
        """
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        plan = split_dataset(manifest, train=0.5, val=0.25, test=0.25, seed=11)
        assert not (set(plan.cards("train")) & set(plan.cards("test")))
        assert not (set(plan.cards("train")) & set(plan.cards("val")))
        assert not (set(plan.cards("val")) & set(plan.cards("test")))

    def test_card_ids_map_to_samples_not_files(self, source_dataset):
        """Two images of the same card must land in the same split."""
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        plan = split_dataset(manifest, train=0.5, val=0.25, test=0.25, seed=5)
        plan.apply(manifest.samples)
        by_card: dict[str, set[str]] = {}
        for sample in manifest.samples:
            by_card.setdefault(sample.card_id, set()).add(sample.split)
        for card_id, splits in by_card.items():
            assert len(splits) == 1, f"card {card_id} spans splits {splits}"

    def test_split_is_reproducible_for_a_seed(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        first = split_dataset(manifest, seed=99)
        second = split_dataset(manifest, seed=99)
        assert first.assignments == second.assignments

    def test_different_seeds_produce_different_splits(self, source_dataset):
        """Seeds must actually change the assignment.

        Checked over several seed pairs: with the 8-card fixture a split is one
        val card and one test card, so some seed pairs legitimately land on the
        same card and a single pair was flaky (it broke purely because the demo
        vocabulary changed how many RNG draws bootstrap makes).
        """
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        base = split_dataset(manifest, seed=1).assignments
        differing = 0
        for seed in (2, 3, 4, 5):
            other = split_dataset(manifest, seed=seed).assignments
            if other != base:
                differing += 1
        assert differing, "no seed changed the split, so the seed is not used"


class TestLoader:
    def test_loader_yields_samples_for_a_split(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        split_dataset(manifest, seed=3).apply(manifest.samples)
        loader = DatasetLoader(manifest, split="train")
        samples = list(loader)
        assert samples
        assert all(sample.split == "train" for sample in samples)

    def test_limit_is_respected(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        samples = list(DatasetLoader(manifest, limit=3))
        assert len(samples) == 3

    def test_bad_root_is_reported(self, source_dataset, tmp_path):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        manifest.root = tmp_path / "does-not-exist"
        # This used to assert only that *something* raised, and what actually
        # raised was AttributeError: DatasetLoader has no .validate(). The real
        # contract is that loading from a bad root names the missing image.
        with pytest.raises(DatasetError, match="Missing image"):
            list(DatasetLoader(manifest, limit=1))


class TestCardTypeTaxonomy:
    """The taxonomy decides which layout (and so which fields) a card gets."""

    def test_sports_is_a_first_class_type(self):
        """Collapsing "sports" into OTHER silently dropped the player field.

        Every sports card became OTHER, the catch-all bucket looked common, and
        the classifier was trained to confuse "unknown game" with "sport".
        """
        from cardvault_vision.datasets.schema import CARD_TYPES, normalize_card_type

        assert "SPORTS" in CARD_TYPES
        assert normalize_card_type("SPORTS") == "SPORTS"
        assert normalize_card_type("sports") == "SPORTS"

    def test_sports_subtypes_share_the_sports_layout(self):
        from cardvault_vision.layouts import resolve_layout

        for card_type in ("SPORTS", "BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"):
            layout = resolve_layout(card_type)
            assert layout.card_type == "SPORTS", card_type
            assert "player" in layout.fields

    def test_unknown_game_gets_a_generic_layout(self):
        """An unknown game must not fail closed - it still has a name and number."""
        from cardvault_vision.layouts import resolve_layout

        layout = resolve_layout("OTHER")
        assert layout.card_type == "GENERIC"
        assert "name" in layout.fields and "number" in layout.fields

    def test_aliases_are_case_insensitive(self):
        from cardvault_vision.datasets.schema import normalize_card_type

        assert normalize_card_type("pokémon") == "POKEMON"
        assert normalize_card_type("MTG") == "MAGIC_THE_GATHERING"
        assert normalize_card_type("yu-gi-oh") == "YU_GI_OH"

    def test_unknown_type_is_rejected(self):
        from cardvault_vision.datasets.schema import normalize_card_type

        with pytest.raises(ValueError):
            normalize_card_type("not-a-real-game")

    def test_bootstrap_cards_keep_their_real_fields(self, source_dataset):
        """End-to-end: a sports card must still carry a player, not just a number."""
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        types = {sample.annotation.card_type for sample in manifest.samples}
        assert "SPORTS" not in types or any(
            sample.annotation.card_type == "SPORTS"
            and "player" in sample.annotation.fields
            for sample in manifest.samples
        )
        for sample in manifest.samples:
            assert sample.annotation.fields, sample.image_id


class TestDemoVocabulary:
    """Demo cards must exercise the real layout, or the pipeline is tested on
    an easier problem than it claims to solve."""

    def test_sports_cards_get_sports_fields(self):
        import numpy as np

        from cardvault_vision.demo import random_fields

        rng = np.random.default_rng(0)
        for card_type in ("SPORTS", "BASEBALL", "FOOTBALL", "BASKETBALL", "HOCKEY"):
            fields = random_fields(card_type, rng)
            assert set(fields) == {"player", "number", "team", "year"}, card_type
            assert all(value.strip() for value in fields.values())

    def test_trading_card_games_get_their_own_fields(self):
        import numpy as np

        from cardvault_vision.demo import random_fields

        rng = np.random.default_rng(0)
        assert "name" in random_fields("POKEMON", rng)
        assert "type_line" in random_fields("MAGIC_THE_GATHERING", rng)

    def test_title_vocabulary_is_large_enough_to_measure_reading(self):
        """Duplicate titles make CER measure memorisation, not OCR.

        The old pools were fixed lists (12 Magic names, 6 Yu-Gi-Oh names), so a
        400-card bootstrap was 84% and 91% duplicates respectively and a model
        could score well by memorising them. Composed names must keep the
        space far larger than any realistic bootstrap.
        """
        import numpy as np

        from cardvault_vision.demo import random_name

        rng = np.random.default_rng(0)
        for card_type, floor in (("MAGIC_THE_GATHERING", 500), ("YU_GI_OH", 200),
                                 ("POKEMON", 500)):
            seen = {random_name(card_type, rng) for _ in range(3000)}
            assert len(seen) >= floor, (
                f"{card_type} produced only {len(seen)} distinct names in 3000 "
                f"draws (expected at least {floor})")

    def test_composed_names_stay_well_formed(self):
        """Longer names must remain plausible single-line card text."""
        import re

        import numpy as np

        from cardvault_vision.demo import random_name

        rng = np.random.default_rng(1)
        for card_type in ("MAGIC_THE_GATHERING", "YU_GI_OH", "BASEBALL", "POKEMON"):
            for _ in range(200):
                name = random_name(card_type, rng)
                assert name.strip() == name, f"leading/trailing space: {name!r}"
                assert "  " not in name, f"double space: {name!r}"
                assert len(name) <= 48, f"implausibly long name: {name!r}"
                assert re.search(r"[^\W\d_]", name), f"no letters: {name!r}"

    def test_random_card_type_returns_only_canonical_types(self):
        import numpy as np

        from cardvault_vision.datasets.schema import CARD_TYPES
        from cardvault_vision.demo import random_card_type

        rng = np.random.default_rng(0)
        for _ in range(200):
            assert random_card_type(rng) in CARD_TYPES

    def test_bootstrap_accepts_aliases_and_canonicalises(self, tmp_path):
        """`--types sports,pokemon` must produce complete sports cards."""
        from cardvault_vision.cli.commands_dataset import bootstrap_source_dataset

        manifest = bootstrap_source_dataset(tmp_path / "aliased", count=4, seed=1,
                                            types="sports,pokemon")
        types = {sample.annotation.card_type for sample in manifest.samples}
        assert types == {"SPORTS", "POKEMON"}
        for sample in manifest.samples:
            if sample.annotation.card_type == "SPORTS":
                assert "player" in sample.annotation.fields
