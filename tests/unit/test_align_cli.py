"""``vision align`` argument handling and report shape.

The engine itself is mocked: these tests cover the command's contract - which
configuration it grades, and that its output is well-formed - not the detector.
The ``--json`` path is covered explicitly because it used to raise
``TypeError: emit() got an unexpected keyword argument 'as_json'`` while the
human-readable path worked, which is exactly the kind of bug a CLI test catches
and a screenshot does not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cardvault_vision.cli import commands_align
from cardvault_vision.datasets.schema import Quad
from cardvault_vision.integrations.image_engine import AlignmentResult
from cardvault_vision.integrations.selection import EDGE_PRESETS


def quad(x0, y0, x1, y1):
    return [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]


#: Truth and a candidate 20 px off it: comfortably inside a 2% tolerance on a
#: card ~1400 px on the diagonal, so the assertions are about plumbing, not luck.
TRUTH = Quad(points=[(100, 100), (800, 100), (800, 1100), (100, 1100)])
NEAR = quad(110, 110, 810, 1110)


class _Sample:
    def __init__(self, image_id="00000000"):
        self.image_id = image_id
        self.image = np.zeros((1200, 900, 3), np.uint8)
        self.profile = "pokemon"
        self.annotation = type("A", (), {"card_quad": TRUTH})()


class _Manifest:
    def __init__(self, samples):
        self.samples = samples

    @classmethod
    def load(cls, path):        # noqa: ARG003 - signature parity with the real one
        return cls([_Sample()])


def _args(monkeypatch, capsys, argv_extra, samples=None):
    from cardvault_vision.cli.main import build_parser

    monkeypatch.setattr("cardvault_vision.integrations.image_engine.engine_available",
                        lambda *a, **k: True)
    monkeypatch.setattr("cardvault_vision.datasets.DatasetManifest",
                        type("M", (_Manifest,), {"load": classmethod(
                            lambda cls, path: cls(samples or [_Sample()]))}))
    monkeypatch.setattr("cardvault_vision.datasets.DatasetLoader",
                        lambda manifest: iter(manifest.samples))
    monkeypatch.setattr(
        "cardvault_vision.integrations.image_engine.align",
        lambda image, *, profile=None, options=None, **kw: AlignmentResult(
            status="ALIGNED", corners=[(110.0, 110.0), (810.0, 110.0),
                                       (810.0, 1110.0), (110.0, 1110.0)],
            confidence=0.9, candidates=[{"corners": NEAR, "confidence": 0.9}],
            total_ms=1.0, engine_path="/e"),
    )
    parser = build_parser()
    return parser.parse_args(["align", "--dataset", "x", *argv_extra])


@pytest.fixture(autouse=True)
def _tmp_manifest_dir(monkeypatch):
    """``--dataset`` is only used for its manifest, which is stubbed out."""
    monkeypatch.setattr(commands_align, "resolve_manifest", lambda path: path)


class TestJsonOutput:
    def test_json_mode_prints_one_parseable_document(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, ["--json"])
        assert args.handler(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["count"] == 1
        assert payload["reranked"] is True
        assert payload["single_pass"] is False
        assert "error_distribution" in payload

    def test_edge_threshold_forces_a_single_pass(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, ["--json", "--edge-threshold", "48"])
        assert args.handler(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["single_pass"] is True

    def test_no_rerank_is_reported(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, ["--json", "--no-rerank"])
        assert args.handler(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["reranked"] is False
        assert payload["single_pass"] is False

    def test_json_output_can_also_be_written_to_a_file(self, monkeypatch, capsys, tmp_path):
        target = tmp_path / "report.json"
        args = _args(monkeypatch, capsys, ["--json", "--output", str(target)])
        assert args.handler(args) == 0
        assert json.loads(target.read_text())["count"] == 1


class TestHumanOutput:
    def test_the_default_path_names_the_configuration(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, [])
        assert args.handler(args) == 0
        out = capsys.readouterr().out
        assert "pooled edge presets, re-ranked" in out
        assert "detection rate" in out

    def test_single_pass_mode_says_so(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, ["--edge-threshold", "48"])
        args.handler(args)
        assert "single engine pass" in capsys.readouterr().out

    def test_no_rerank_mode_says_so(self, monkeypatch, capsys):
        args = _args(monkeypatch, capsys, ["--no-rerank"])
        args.handler(args)
        assert "engine ranking" in capsys.readouterr().out


class TestArgumentContract:
    def test_reranking_is_on_by_default(self):
        from cardvault_vision.cli.main import build_parser

        args = build_parser().parse_args(["align", "--dataset", "x"])
        assert args.rerank is True
        assert args.edge_threshold is None
        assert args.tolerance == pytest.approx(0.02)

    def test_it_refuses_a_dataset_without_card_quads(self, monkeypatch, capsys):
        from cardvault_vision.cli.main import build_parser
        from cardvault_vision.errors import ConfigError

        sample = _Sample()
        sample.annotation = type("A", (), {"card_quad": None})()
        monkeypatch.setattr("cardvault_vision.integrations.image_engine.engine_available",
                            lambda *a, **k: True)
        monkeypatch.setattr("cardvault_vision.datasets.DatasetManifest",
                            type("M", (_Manifest,), {"load": classmethod(
                                lambda cls, path: cls([sample]))}))
        monkeypatch.setattr(commands_align, "resolve_manifest", lambda path: path)
        args = build_parser().parse_args(["align", "--dataset", "x"])
        with pytest.raises(ConfigError, match="no recorded card quads"):
            args.handler(args)

    def test_it_says_what_to_do_when_the_engine_is_missing(self, monkeypatch):
        from cardvault_vision.cli.main import build_parser
        from cardvault_vision.errors import ConfigError

        monkeypatch.setattr("cardvault_vision.integrations.image_engine.engine_available",
                            lambda *a, **k: False)
        args = build_parser().parse_args(["align", "--dataset", "x"])
        with pytest.raises(ConfigError, match="npm run build"):
            args.handler(args)


def test_the_presets_used_by_the_command_are_the_shipped_ones():
    """The CLI grades the default configuration, so it must not carry its own."""
    assert len(EDGE_PRESETS) >= 2
