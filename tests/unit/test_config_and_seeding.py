"""Phase 0: configuration, seeding, hashing and hardware detection."""

from __future__ import annotations

import json

import pytest

from cardvault_vision.config import Config, deep_merge, parse_value, resolve_config_path
from cardvault_vision.errors import ConfigError
from cardvault_vision.rng import MAX_SEED, RngStreams, normalize_seed
from cardvault_vision import hardware


class TestConfigLayering:
    def test_later_files_win(self, tmp_path):
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text("training:\n  epochs: 5\n  lr: 0.1\n")
        second.write_text("training:\n  epochs: 10\n")
        config = Config.load(first, second)
        assert config.get("training.epochs") == 10
        # Deep merge must not drop keys the later file did not mention.
        assert config.get("training.lr") == 0.1

    def test_set_overrides_files(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("training:\n  epochs: 5\n")
        config = Config.load(path, sets=["training.epochs=99"])
        assert config.get("training.epochs") == 99

    def test_bare_names_resolve_against_repo_configs(self):
        """`--config train-ocr` is the documented form, not just a file path."""
        path = resolve_config_path("train-ocr")
        assert path.exists()
        assert path.parent.name == "configs"

    def test_repository_configs_all_load(self):
        for name in ("base", "synthetic", "train-ocr", "train-classifier"):
            config = Config.load(name)
            assert config.to_dict(), f"{name} resolved to an empty config"

    def test_fingerprint_is_stable_and_sensitive(self):
        a = Config({"x": 1, "y": [1, 2]})
        b = Config({"y": [1, 2], "x": 1})       # same content, different order
        c = Config({"x": 2, "y": [1, 2]})
        assert a.fingerprint() == b.fingerprint()
        assert a.fingerprint() != c.fingerprint()

    def test_missing_key_raises(self):
        with pytest.raises(ConfigError):
            Config({"a": 1}).require("b")

    def test_bad_set_syntax_raises(self):
        with pytest.raises(ConfigError):
            Config.load(sets=["no-equals-sign"])


class TestDeepMerge:
    def test_lists_are_replaced_not_concatenated(self):
        merged = deep_merge({"a": [1, 2]}, {"a": [3]})
        assert merged["a"] == [3]

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base["a"]["b"] == 1


class TestParseValue:
    @pytest.mark.parametrize("text,expected", [
        ("true", True), ("false", False), ("null", None), ("3", 3),
        ("0.5", 0.5), ('{"a": 1}', {"a": 1}), ("hello", "hello"),
        ('["x", "y"]', ["x", "y"]),
    ])
    def test_values(self, text, expected):
        assert parse_value(text) == expected


class TestRngStreams:
    def test_same_inputs_same_stream(self):
        a = RngStreams(master_seed=42, sample_index=7).get("geometry")
        b = RngStreams(master_seed=42, sample_index=7).get("geometry")
        assert a.random() == b.random()

    def test_streams_are_independent(self):
        streams = RngStreams(master_seed=42, sample_index=7)
        assert streams.get("geometry").random() != streams.get("lighting").random()

    def test_index_changes_stream(self):
        assert (RngStreams(master_seed=42, sample_index=1).get("a").random()
                != RngStreams(master_seed=42, sample_index=2).get("a").random())

    def test_repeated_get_is_not_a_new_stream(self):
        """Calling get() twice must continue the same stream, not restart it.

        If it restarted, two consumers of the "geometry" stream would draw
        identical numbers and the rendered sample would be self-correlated.
        """
        streams = RngStreams(master_seed=1, sample_index=0)
        first = streams.get("geometry").random()
        second = streams.get("geometry").random()
        assert first != second

    def test_friendly_text_seeds_are_hashed_deterministically(self):
        """``--seed cardvault`` must work and must mean the same thing twice."""
        assert normalize_seed("cardvault") == normalize_seed("cardvault")
        assert normalize_seed("cardvault") != normalize_seed("cardvault2")
        # Must land in the same accepted range as an integer seed. (Comparing
        # against normalize_seed(None) would be comparing against fresh entropy.)
        assert 0 <= normalize_seed("cardvault") <= MAX_SEED

    def test_negative_seed_is_rejected(self):
        """A negative seed is a bug in the caller, not something to re-map."""
        with pytest.raises(ConfigError):
            normalize_seed(-1)


class TestHardware:
    def test_cpu_is_always_available(self):
        info = hardware.detect_hardware()
        assert info.cpu_count >= 1
        assert isinstance(info.cuda_available, bool)

    def test_auto_resolves_to_a_usable_device(self):
        assert hardware.resolve_device("auto") in {"cpu", "cuda"} or \
            hardware.resolve_device("auto").startswith("cuda")

    def test_explicit_cpu_is_honoured(self):
        assert hardware.resolve_device("cpu") == "cpu"

    def test_requesting_cuda_without_one_is_not_silent(self):
        """A requested GPU must never degrade to CPU without telling the caller."""
        if hardware.detect_hardware().cuda_available:
            pytest.skip("CUDA present")
        with pytest.raises(Exception):
            hardware.resolve_device("cuda")

    def test_info_is_json_serialisable(self):
        json.dumps(hardware.detect_hardware().to_dict())
