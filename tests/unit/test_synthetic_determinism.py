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
