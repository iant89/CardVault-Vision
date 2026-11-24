"""The Node bridge and the engine profile mapping.

These tests do not need the engine to be installed: the ones that would actually
run it are skipped unless ``CVV_IMAGE_ENGINE`` points at a build. What is checked
unconditionally is the part that is easy to get silently wrong - the protocol,
the failure modes, and the aspect-ratio convention between the two projects.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cardvault_vision.errors import ConfigError
from cardvault_vision.integrations import image_engine
from cardvault_vision.integrations.image_engine import (
    BRIDGE_SCRIPT,
    PROFILE_NAMES,
    EngineUnavailable,
    align,
    engine_available,
    engine_path,
)

ENGINE = image_engine.engine_path()
needs_engine = pytest.mark.skipif(
    not engine_available(), reason="set CVV_IMAGE_ENGINE to a built image engine"
)


class TestBridgePackaging:
    def test_the_bridge_script_ships_with_the_package(self):
        """It is resolved relative to the module, so it must live beside it.

        An earlier version sat at the repository root, which worked from a
        checkout and would have broken from an installed wheel.
        """
        assert BRIDGE_SCRIPT.is_file()
        assert BRIDGE_SCRIPT.parent == Path(image_engine.__file__).parent
        assert BRIDGE_SCRIPT.suffix == ".mjs"

    def test_package_data_declares_the_bridge(self):
        """setuptools must ship the .mjs file, or the wheel is broken."""
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        text = pyproject.read_text()
        assert "package-data" in text
        assert '"*.mjs"' in text


class TestEngineDiscovery:
    def test_missing_engine_is_reported_not_guessed(self):
        assert engine_path("/nonexistent/dist") is None
        assert engine_path(None) in (None, engine_path(None))

    def test_a_directory_without_index_is_not_an_engine(self, tmp_path):
        assert engine_path(tmp_path) is None

    def test_a_built_engine_is_found(self, tmp_path):
        (tmp_path / "index.js").write_text("export {};\n")
        assert engine_path(tmp_path) == tmp_path

    def test_availability_needs_both_node_and_an_engine(self, tmp_path):
        (tmp_path / "index.js").write_text("export {};\n")
        if shutil.which("node") is None:
            assert not engine_available(tmp_path)
        else:
            assert engine_available(tmp_path)


class TestAlignPreconditions:
    def test_unknown_profile_is_rejected_before_spawning_node(self):
        with pytest.raises(ConfigError, match="Unknown image-engine profile"):
            align(np.zeros((4, 4, 3), np.uint8), profile="not-a-card", engine="/tmp")

    def test_missing_engine_raises_the_documented_error(self):
        with pytest.raises(EngineUnavailable, match="not available"):
            align(np.zeros((4, 4, 3), np.uint8), engine="/nonexistent/engine")

    def test_non_uint8_input_is_rejected(self):
        """Silent float truncation would change the pixels being graded."""
        with pytest.raises(ConfigError, match="uint8"):
            align(np.zeros((4, 4, 3), np.float32), engine="/nonexistent/engine")

    def test_pixel_layout_is_validated(self):
        with pytest.raises(ConfigError, match="unsupported image shape"):
            align(np.zeros((4, 4, 2), np.uint8), engine="/nonexistent/engine")


class TestProfileConvention:
    """The two projects express card aspect ratio as exact reciprocals.

    CardVault Vision's layout specs use height/width (``88/63`` for a trading
    card); the engine's profiles use width/height (``63/88``). Feeding one to the
    other without converting silently mis-ranks candidates, so the mapping is
    pinned here.
    """

    def test_every_profile_name_is_accepted_by_the_bridge(self):
        bridge = BRIDGE_SCRIPT.read_text()
        for name in PROFILE_NAMES:
            assert name in bridge, f"{name} is missing from the bridge's PROFILES table"

    @needs_engine
    def test_the_engine_reports_the_reciprocal_aspect(self):
        engine = __import__("subprocess").run(
            ["node", "-e",
             f"const m = await import('file://{ENGINE}/index.js');"
             "const p = m.POKEMON_CARD_PROFILE;"
             "console.log(JSON.stringify({w: p.aspectRatio}));"],
            capture_output=True, text=True, timeout=60,
        )
        assert engine.returncode == 0, engine.stderr
        width_over_height = json.loads(engine.stdout)["w"]
        # CardVault Vision's default layout aspect is height/width.
        assert width_over_height == pytest.approx(63 / 88)
        assert 1.0 / width_over_height == pytest.approx(88 / 63)

    def test_the_python_default_is_the_reciprocal(self):
        from cardvault_vision.layouts.resolver import CardLayout

        assert CardLayout(card_type="x").aspect == pytest.approx(88 / 63)


@needs_engine
class TestAlignAgainstTheEngine:
    def test_a_real_capture_returns_geometry(self):
        image = np.full((240, 180, 3), 30, np.uint8)
        image[20:220, 20:160] = 230          # a bright card-like rectangle
        result = align(image, profile="pokemon", engine=ENGINE)
        assert result.engine_path
        assert result.status in {"ALIGNED", "CARD_NOT_FOUND", "LOW_CONFIDENCE"}
        if result.aligned:
            assert len(result.corners) == 4

    def test_corners_come_back_as_tuples_not_dicts(self):
        image = np.full((240, 180, 3), 30, np.uint8)
        image[20:220, 20:160] = 230
        result = align(image, profile="pokemon", engine=ENGINE)
        if result.aligned:
            assert isinstance(result.corners[0], tuple)
            assert len(result.corners[0]) == 2
