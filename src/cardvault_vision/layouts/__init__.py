"""Card layout system (Phase 7).

A layout says *where the fields are on a card*, in normalised card coordinates
(0..1 of card width/height), keyed by card type. The synthetic engine uses it to
place text and derive field boxes; the pipeline uses it to cut ROIs before OCR;
the export ships it as preprocessing metadata.

Layouts are data (``configs/cards/*.yaml``) with a Python fallback, so a new card
game can be added without touching code.
"""

from __future__ import annotations

from .resolver import (
    CardLayout,
    LayoutField,
    available_layouts,
    load_layouts,
    resolve_layout,
    validate_layouts,
)

__all__ = [
    "CardLayout",
    "LayoutField",
    "available_layouts",
    "load_layouts",
    "resolve_layout",
    "validate_layouts",
]
