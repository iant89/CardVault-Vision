"""Synthetic Scan Engine (Phase 2).

The engine is a *simulated camera environment*, not an augmentation utility. It
takes clean card images and produces the images a real CardVault user would
actually submit: tilted, glary, blurry, badly lit, partially occluded, shot on a
cheap phone and shared through a chat app.

Typical use::

    from cardvault_vision.synthetic import SyntheticConfig, generate_dataset

    report = generate_dataset(
        "datasets/manifests/source.json",
        SyntheticConfig(profile="realistic", count=5000, seed=12345),
        output_dir="datasets/synthetic/realistic-001",
    )

Or one sample at a time::

    from cardvault_vision.synthetic import SourceCard, CaptureProfile, GenerationConfig
    from cardvault_vision.synthetic.generator import generate_sample

    card = SourceCard(card_id="c1", image=rgb_array, card_type="POKEMON",
                      fields={"name": "Pikachu", "number": "58/102"})
    sample = generate_sample(
        card,
        GenerationConfig(profile=CaptureProfile.get("phone_bad_lighting").to_dict()),
        seed=1, index=0,
    )

Stage modules:

===================  ==========================================================
``geometry``         card placement, perspective, bending, cropping
``camera``           lens, focus, sensor noise, resolution, JPEG
``lighting``         exposure, colour temperature, shadows
``materials``        finishes (holo/foil/chrome), specular glare
``background``       environment materials
``occlusion``        fingers, hands, sleeves, other cards
``artifacts``        moire, banding, double compression, quality metrics
``profiles``         capture distributions (phone, scanner, child, extreme)
``targeting``        problem-targeted generation
``annotations``      automatic ground truth + audits
``pipeline``         dataset-scale generation with sharding and resume
===================  ==========================================================
"""

from __future__ import annotations

from .annotations import (
    AnnotationAudit,
    audit_annotations,
    load_annotation,
    summarize_annotations,
)
from .generator import GeneratedSample, GenerationConfig, SourceCard, generate_sample
from .pipeline import (
    GenerationReport,
    SourcePool,
    SyntheticConfig,
    generate_dataset,
    load_source_pool,
    make_contact_sheet,
    plan_sample,
)
from .profiles import (
    BUILTIN_PROFILES,
    MIX_PRESETS,
    PROFILE_NAMES,
    CaptureProfile,
    resolve_profiles,
)
from .targeting import PROBLEM_DESCRIPTIONS, PROBLEM_TYPES, apply_problem

__all__ = [
    "AnnotationAudit",
    "BUILTIN_PROFILES",
    "CaptureProfile",
    "GeneratedSample",
    "GenerationConfig",
    "GenerationReport",
    "MIX_PRESETS",
    "PROBLEM_DESCRIPTIONS",
    "PROBLEM_TYPES",
    "PROFILE_NAMES",
    "SourceCard",
    "SourcePool",
    "SyntheticConfig",
    "apply_problem",
    "audit_annotations",
    "generate_dataset",
    "generate_sample",
    "load_annotation",
    "load_source_pool",
    "make_contact_sheet",
    "plan_sample",
    "resolve_profiles",
    "summarize_annotations",
]
