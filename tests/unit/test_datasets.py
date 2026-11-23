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
