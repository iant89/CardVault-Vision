"""Phase 2: the synthetic engine's reproducibility contract.

The roadmap requires that a sample's pixels depend only on
``(seed, index, source card, resolved profile)``. These tests pin that down
precisely, because a generator that is *almost* deterministic silently breaks
every later claim about dataset versioning and model comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardvault_vision.errors import ConfigError, GenerationError
from cardvault_vision.synthetic import CaptureProfile, SyntheticConfig, generate_sample
from cardvault_vision.synthetic.pipeline import (
    load_source_pool,
    plan_sample,
    resolve_profiles,
)

# Small and forgiving: these tests are about determinism, not image quality, so
# they must not be flaky when a particular seed happens to draw low legibility.
CONFIG = dict(render_scale=0.25, min_visibility=0.0, min_legibility=0.0)


@pytest.fixture(scope="module")
def pool(source_dataset):
    return load_source_pool(source_dataset)


def _generate(pool, index: int, seed: int, profile: str = "phone_normal"):
    config = SyntheticConfig(profile=profile, count=1, seed=seed, **CONFIG)
    weights = resolve_profiles(profile)
    plan = plan_sample(index=index, seed=seed, pool=pool, config=config,
                       profile_weights=weights)
    return generate_sample(plan.source_card, plan.config, seed=seed, index=index)


class TestDeterminism:
    def test_same_seed_and_index_reproduce_pixels(self, pool):
        first = _generate(pool, index=3, seed=1234)
        second = _generate(pool, index=3, seed=1234)
        assert np.array_equal(first.image, second.image)

    def test_same_seed_and_index_reproduce_annotation(self, pool):
        first = _generate(pool, index=3, seed=1234)
        second = _generate(pool, index=3, seed=1234)
        assert first.annotation.to_dict() == second.annotation.to_dict()

    def test_different_index_changes_the_sample(self, pool):
        a = _generate(pool, index=1, seed=99)
        b = _generate(pool, index=2, seed=99)
        assert a.image.shape != b.image.shape or not np.array_equal(a.image, b.image)

    def test_different_seed_changes_the_sample(self, pool):
        a = _generate(pool, index=1, seed=99)
        b = _generate(pool, index=1, seed=100)
        assert a.image.shape != b.image.shape or not np.array_equal(a.image, b.image)

    def test_pixels_do_not_depend_on_generation_order(self, pool):
        """Sample 5 must be identical whether generated first or later.

        This is what makes parallel generation and ``--resume`` safe.
        """
        forward = [_generate(pool, index=i, seed=7) for i in range(3)]
        backward = [_generate(pool, index=i, seed=7) for i in (2, 1, 0)][::-1]
        for a, b in zip(forward, backward, strict=True):
            assert np.array_equal(a.image, b.image)


class TestGenerationRecord:
    def test_every_sample_records_its_provenance(self, pool):
        """Every image must be traceable to the exact parameters that made it."""
        sample = _generate(pool, index=2, seed=555)
        generation = sample.generation
        assert generation["seed"] == 555
        assert generation["index"] == 2
        assert generation["profile"]
        assert generation["engine"]
        # Per-stream seeds are what make an individual image regenerable without
        # regenerating the whole dataset in the same order.
        assert generation["stream_seeds"]
        for section in ("geometry", "camera", "lighting"):
            assert section in generation

    def test_annotation_carries_the_source_text(self, pool):
        """Ground truth must be the source card's text, never resampled noise."""
        sample = _generate(pool, index=0, seed=11)
        fields = sample.annotation.fields
        assert fields, "a generated sample must have field annotations"
        for name, field in fields.items():
            assert field.text, f"field {name} has empty ground-truth text"

    def test_annotation_rois_are_within_the_image(self, pool):
        sample = _generate(pool, index=4, seed=8)
        height, width = sample.image.shape[:2]
        for name, field in sample.annotation.fields.items():
            box = field.bbox
            # Boxes may sit slightly outside the frame when the card is cropped,
            # but never by more than a pixel of rounding.
            assert box.x0 >= -1 and box.y0 >= -1, name
            assert box.x1 <= width + 1, name
            assert box.y1 <= height + 1, name

    def test_quality_report_is_present(self, pool):
        sample = _generate(pool, index=6, seed=3)
        quality = sample.quality
        for key in ("sharpness", "card_area_fraction", "is_usable"):
            assert key in quality


class TestProfiles:
    def test_profiles_keep_their_identity(self):
        """Overrides must not rename a profile to 'custom'.

        If they did, per-profile dataset statistics and the "did the dataset
        match the target distribution" check would both be meaningless.
        """
        profile = CaptureProfile.get("phone_normal")
        assert profile.name == "phone_normal"
        assert profile.to_dict()["name"] == "phone_normal"

    def test_profile_mixture_sums_to_one(self):
        weights = resolve_profiles("realistic")
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_unknown_profile_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown profile"):
            CaptureProfile.get("no-such-profile")

    def test_each_profile_has_a_distinct_character(self):
        phone = CaptureProfile.get("phone_normal")
        scanner = CaptureProfile.get("scanner")
        assert phone.target_size != scanner.target_size


class TestSourcePoolBalance:
    """Type balancing must not let one card dominate the whole run."""

    def _pool(self, type_counts):
        from cardvault_vision.synthetic.generator import SourceCard
        from cardvault_vision.synthetic.pipeline import SourcePool

        cards = []
        for type_name, count in type_counts.items():
            for index in range(count):
                cards.append(SourceCard(
                    card_id=f"{type_name}_{index}", card_type=type_name,
                    image=np.zeros((8, 8, 3), dtype=np.uint8), fields={},
                ))
        return SourcePool(cards)

    def test_every_card_is_reachable(self):
        """A type with one card used to take 43% of all draws (290 of 700
        samples carried the same name), which leaves OCR nothing to learn and
        a classifier nothing to learn *from*."""
        pool = self._pool({"BASKETBALL": 1, "FOOTBALL": 2, "POKEMON": 26})
        probabilities = pool.card_probabilities()
        assert len(probabilities) == len(pool.cards)
        assert probabilities.min() > 0, "some cards can never be selected"
        assert probabilities.sum() == pytest.approx(1.0)

    def test_dominance_is_capped(self):
        pool = self._pool({"BASKETBALL": 1, "FOOTBALL": 2, "POKEMON": 26})
        probabilities = pool.card_probabilities()
        uniform = 1.0 / len(pool.cards)
        assert probabilities.max() <= pool.MAX_CARD_DOMINANCE * uniform + 1e-9

    def test_small_types_are_still_represented(self):
        """Capping must not undo the type balancing it exists to temper.

        The lone basketball card must still be drawn far more often than any
        single Pokemon card (which is what prevents a 26-card type from burying
        a 1-card type), while the whole Pokemon type keeps most of the mass.
        """
        pool = self._pool({"BASKETBALL": 1, "POKEMON": 26})
        probabilities = pool.card_probabilities()
        basketball = probabilities[pool.by_type["BASKETBALL"][0]]
        any_pokemon = probabilities[pool.by_type["POKEMON"]][0]
        pokemon_total = probabilities[pool.by_type["POKEMON"]].sum()
        assert basketball > any_pokemon, (
            "the lone basketball card should outrank any individual Pokemon card")
        assert pokemon_total > basketball


class TestQualityControl:
    def test_unusable_samples_are_rejected_not_written(self, pool):
        """A sample nobody can read is a bad label, so it must be refused."""
        config = SyntheticConfig(profile="phone_normal", count=1, seed=1,
                                 render_scale=0.25, min_visibility=0.99,
                                 min_legibility=0.99)
        weights = resolve_profiles("phone_normal")
        plan = plan_sample(index=0, seed=1, pool=pool, config=config,
                           profile_weights=weights)
        with pytest.raises(GenerationError, match="not usable"):
            generate_sample(plan.source_card, plan.config, seed=1, index=0)


class TestImagingPrimitives:
    def test_blur_reduces_high_frequency_energy(self):
        from cardvault_vision.synthetic.imaging import gaussian_blur

        rng = np.random.default_rng(0)
        image = rng.random((64, 64, 3)).astype(np.float32)
        blurred = gaussian_blur(image, 2.0)
        assert blurred.std() < image.std()

    def test_blur_preserves_shape_and_range(self):
        from cardvault_vision.synthetic.imaging import gaussian_blur

        image = np.random.default_rng(1).random((40, 50, 3)).astype(np.float32)
        assert gaussian_blur(image, 1.5).shape == image.shape
        assert gaussian_blur(image, 1.5).max() <= 1.0

    def test_zero_parameter_optics_are_identity(self):
        """A no-op distortion must be exactly a no-op, not approximately."""
        from cardvault_vision.synthetic.imaging import chromatic_aberration, lens_distortion

        image = np.random.default_rng(2).random((30, 40, 3)).astype(np.float32)
        assert np.allclose(chromatic_aberration(image, 0.0), image)
        assert np.allclose(lens_distortion(image, 0.0, 0.0), image)

    def test_motion_blur_is_directional(self):
        """Motion blur must stay anisotropic; a symmetric approximation would
        erase the very cue that makes the 'blur' category learnable."""
        from cardvault_vision.synthetic.imaging import (
            convolve,
            motion_blur_kernel,
        )

        image = np.zeros((61, 61, 3), dtype=np.float32)
        image[30, 30] = 1.0
        spread = convolve(image, motion_blur_kernel(15, 0.3))
        row = spread.sum(axis=(1, 2))
        column = spread.sum(axis=(0, 2))
        assert row.max() > column.max() * 1.5

    def test_radial_mask_peaks_at_the_centre(self):
        from cardvault_vision.synthetic.imaging import radial_mask

        mask = radial_mask(120, 90, 60, 45, 30, softness=0.3)
        assert mask[45, 60] == pytest.approx(1.0)
        assert mask[0, 0] == 0.0


def _fake_handle(plan, other):
    """Stand-in for the real renderer: returns a minimal valid record."""
    from cardvault_vision.datasets.schema import Annotation, SampleRecord

    return SampleRecord(
        image_id=plan.image_id, card_id=plan.source_card.card_id,
        image_path=f"images/{plan.image_id}.jpg",
        annotation=Annotation(image_id=plan.image_id,
                              card_id=plan.source_card.card_id,
                              card_type=plan.source_card.card_type),
    )


class TestWorkerCrashesAreNotRejections:
    """A dead worker must not be reported as a low-quality sample.

    A 2800-sample run returned 1810 images with 990 "rejections", all of them
    ``worker_error: A process in the process pool was terminated abruptly``. The
    kernel log showed the OOM killer terminating a worker at 1.47 GB RSS; the
    pool converted that death into a per-sample rejection, so the dataset shrank
    by a third while the run reported success. Data loss must be loud.
    """

    def test_broken_pool_is_retried_not_counted_as_rejection(self, monkeypatch, pool):
        from concurrent.futures.process import BrokenProcessPool

        from cardvault_vision.synthetic import pipeline

        calls = []

        class _ExplodingExecutor:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def submit(self, fn, index, *a, **k):
                from concurrent.futures import Future

                calls.append(index)
                future: Future = Future()
                future.set_exception(
                    BrokenProcessPool("worker terminated abruptly"))
                return future

        monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _ExplodingExecutor)

        config = SyntheticConfig(count=3, seed=5, workers=2, **CONFIG)
        report = pipeline.GenerationReport(
            dataset_name="t", generated=0, rejected=0, skipped=0,
            duration_s=0.0, manifest_path=None,
        )
        records = pipeline._generate_parallel(
            pending=[0, 1, 2], seed=5, pool=pool, config=config,
            profile_weights=resolve_profiles("phone_normal"), out_dir=None,
            report=report, handle=_fake_handle, progress=None, total=3,
        )

        # The retry path ran in-process, so the crash was never billed as a
        # rejection.
        assert not any("worker_error" in key for key in report.rejection_reasons), \
            report.rejection_reasons
        assert report.rejected == 0
        assert len(records) == 3

    def test_worker_pool_is_capped_in_size(self):
        """Workers downscale the source pool; the parent must not."""
        from cardvault_vision.synthetic.pipeline import _WORKER_SOURCE_MAX_SIDE

        assert 256 <= _WORKER_SOURCE_MAX_SIDE <= 2000

class TestAutoExposure:
    """The camera must meter the scene, not just multiply it darker.

    Every term in the chain used to darken: the illumination field averages at
    its normalised mean while ``soft_light`` scales by 0.6..1.4, ``adjust_exposure``
    multiplies by ``2**brightness`` with a negative mean, and the surface shading
    multiplies again. Renders came out 2.03x darker than the source cards with
    only 0.36 correlation between source and rendered luminance, and 30% of name
    crops had a maximum luminance below 0.35 - text no reader could recover.
    """

    def _config(self, **kwargs):
        from cardvault_vision.synthetic.lighting import LightingConfig
        from cardvault_vision.synthetic.materials import LightSource

        base = dict(brightness=0.0, contrast=1.0, gamma=1.0, saturation=1.0,
                    kelvin=5500, ambient=0.35, fill=None,
                    key=LightSource(azimuth=45, elevation=40, intensity=0.8,
                                    hardness=0.3, kelvin=5500))
        base.update(kwargs)
        return LightingConfig(**base)

    def test_a_dark_scene_is_metered_up(self):
        """Two inputs of the same scene brightness land at the same exposure.

        This is the whole point of metering: the output must depend on the
        rendered scene, not on how many multiplying stages happened to darken it.
        """
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        scene = np.full((32, 64, 3), 0.55, dtype=np.float32)
        underexposed = self._config(brightness=-1.2)
        out = apply_lighting(scene, underexposed, np.random.default_rng(0))
        assert underexposed.auto_exposure > 1.0
        assert out.card.mean() > 0.25, out.card.mean()

    def test_a_bright_scene_is_metered_down(self):
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        scene = np.full((32, 64, 3), 0.85, dtype=np.float32)
        config = self._config()
        out = apply_lighting(scene, config, np.random.default_rng(0))
        assert config.auto_exposure < 1.0
        assert out.card.mean() < 0.85, out.card.mean()

    def test_ordinary_scenes_land_near_the_meter_target(self):
        """Without metering the same scene landed at 0.35-0.42 by accident."""
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        scene = np.full((32, 64, 3), 0.60, dtype=np.float32)
        out = apply_lighting(scene, self._config(), np.random.default_rng(0))
        luma = float(np.dot(out.card, [0.299, 0.587, 0.114]).mean())
        assert 0.30 < luma < 0.55, luma

    def test_the_meter_leaves_highlights_unclipped(self):
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        scene = np.full((32, 64, 3), 0.30, dtype=np.float32)
        scene[:, 32:] = 0.95
        out = apply_lighting(scene, self._config(), np.random.default_rng(0))
        clipped = float((out.card >= 0.999).mean())
        assert clipped < 0.02, clipped

    def test_a_deliberately_dark_scene_stays_dark(self):
        """`low_light` must not be metered back to a normal exposure.

        Auto-exposure has finite gain. Its first implementation brightened the
        low-light problem until its median luminance (0.304) was identical to an
        ordinary frame's (0.304) - the defect vanished from the dataset.
        """
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        scene = np.full((32, 64, 3), 0.55, dtype=np.float32)
        dark = self._config(brightness=-1.2)
        normal = self._config(brightness=0.0)
        dark_out = apply_lighting(scene, dark, np.random.default_rng(0))
        normal_out = apply_lighting(scene, normal, np.random.default_rng(0))
        # The dark frame pins at its own (lower) gain ceiling rather than
        # metering its way back up to a normal exposure. It does end up close to
        # the target - that is what a meter at its limit looks like - but it is
        # measurably darker and, crucially, the loop cannot reach the general
        # ceiling while doing it.
        assert dark.auto_exposure <= 1.6 + 1e-6, dark.auto_exposure
        assert dark.auto_exposure > normal.auto_exposure
        assert dark_out.card.mean() < normal_out.card.mean(), (
            dark_out.card.mean(), normal_out.card.mean())

    def test_the_gain_is_reported_for_reproducibility(self):
        import numpy as np

        from cardvault_vision.synthetic.lighting import apply_lighting

        config = self._config()
        apply_lighting(np.full((16, 32, 3), 0.5, dtype=np.float32), config,
                       np.random.default_rng(0))
        payload = config.to_dict()
        assert "auto_exposure" in payload
        assert "exposure_bias" in payload
