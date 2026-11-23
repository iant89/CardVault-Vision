"""Phase 1: dataset manifests, hashing, validation and splitting."""

from __future__ import annotations

import json

import pytest

from cardvault_vision.datasets import DatasetLoader, DatasetManifest
from cardvault_vision.datasets.hashing import dataset_content_hash, file_sha256
from cardvault_vision.datasets.split import split_dataset


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


class TestSplitting:
    def test_splits_are_disjoint_and_complete(self, source_dataset):
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        plan = split_dataset(manifest, train=0.6, val=0.2, test=0.2, seed=3)
        assigned = list(plan.assignments)
        assert len(assigned) == len(set(assigned)), "a card appears in two splits"
        assert set(assigned) == set(manifest.card_ids)

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
        manifest = DatasetManifest.load(source_dataset / "manifest.json")
        first = split_dataset(manifest, seed=1)
        second = split_dataset(manifest, seed=2)
        assert first.assignments != second.assignments


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
        with pytest.raises(Exception):
            dataset = DatasetLoader(manifest)
            dataset.validate()


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
        from cardvault_vision.datasets.schema import CARD_TYPES

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

    def test_random_card_type_returns_only_canonical_types(self):
        import numpy as np

        from cardvault_vision.demo import random_card_type
        from cardvault_vision.datasets.schema import CARD_TYPES

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
