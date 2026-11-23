"""Card text rendering and the universal character set (Phase 4 groundwork).

Two jobs:

1. **Draw** field text onto a clean card image when the source card does not
   already contain it (useful for synthesising source cards from scratch, and for
   ``vision dataset bootstrap``).
2. **Normalise** text so that the same name scanned from different games maps to
   one canonical form for OCR labels.

The character set is deliberately *universal*: card names contain accented
Latin, Japanese kana/kanji for Japanese-language cards, and typographic
punctuation (—, ', é, ♀). A model trained on ASCII only silently fails on all of
it, so the charset here is explicit and wide, and the analysis module reports
coverage against it.

Font discovery falls back gracefully: DejaVu is bundled with most Linux images;
on other systems we search common locations and, failing that, use PIL's default
bitmap font so generation never hard-fails.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger

_LOG = get_logger("synthetic.text")

#: Printable Latin + common typographic punctuation used on cards.
_LATIN = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    " .,:;!?'\"-_/&()[]+*#%@"          # ASCII punctuation seen on cards
    "éèêëáàâäíìîïóòôöúùûüñçßæøå"      # accented Latin (Pokémon/Magic/localised)
    "ÉÈÊËÁÀÂÄÍÌÎÏÓÒÔÖÚÙÛÜÑÇ"
    "×÷±°§†‡•…–—‘’“”«»€£¥¢♥★☆"          # symbols used on cards and numbers
)

#: Hiragana, katakana and a small CJK block cover Japanese card names.
_KANA = (
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとど"
    "なにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん"
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトド"
    "ナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲン"
    "ー・「」『』"
)

#: Marker tokens used by the CTC encoder.
PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"


@lru_cache(maxsize=1)
def universal_charset() -> str:
    """The full character set, sorted and de-duplicated.

    Ordering is deterministic (codepoint order), which matters: the CTC label
    index of a character must be identical across every training and inference
    process, or exported models silently mis-decode.
    """
    return "".join(sorted(set(_LATIN + _KANA)))


@lru_cache(maxsize=1)
def number_charset() -> str:
    """Constrained charset for the collector-number head (Phase 5)."""
    return "".join(sorted(set("0123456789/--- ")))


def charset_labels(charset: str | None = None) -> list[str]:
    """CTC label list: ``[pad, blank] + charset`` (index 0 reserved)."""
    text = charset if charset is not None else universal_charset()
    return [PAD_TOKEN, BLANK_TOKEN, *list(text)]


def encode_text(text: str, charset: str | None = None) -> list[int]:
    """Encode text to CTC label indices; unknown characters are dropped."""
    text = text or ""
    labels = charset if charset is not None else universal_charset()
    table = {ch: i + 2 for i, ch in enumerate(labels)}   # 0=pad, 1=blank
    return [table[ch] for ch in text if ch in table]


def decode_indices(indices, charset: str | None = None) -> str:
    """Greedy CTC decode (collapse repeats, drop blanks)."""
    labels = charset if charset is not None else universal_charset()
    out = []
    previous = None
    for index in indices:
        index = int(index)
        if index == previous:
            continue
        previous = index
        if index >= 2 and index - 2 < len(labels):
            out.append(labels[index - 2])
    return "".join(out)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

#: Punctuation that OCR models consistently confuse; normalising labels is
#: better than teaching the model that two glyphs are different.
_DASHES = {"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
           "\u2014": "-", "\u2212": "-"}
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
           "\u201c": '"', "\u201d": '"', "\u201e": '"'}
_SPACES = {"\u00a0": " ", "\u2007": " ", "\u202f": " ", "\t": " "}


def normalize_text(text: str, *, keep_case: bool = True, unicode_form: str = "NFKC") -> str:
    """Canonicalise text for use as an OCR label.

    NFKC folds full-width Latin (common in Japanese card numbers, e.g. ``１２３``)
    onto ASCII, which is essential: otherwise the label set doubles for no
    benefit.
    """
    if not text:
        return ""
    out = unicodedata.normalize(unicode_form, str(text))
    for source, target in {**_DASHES, **_QUOTES, **_SPACES}.items():
        out = out.replace(source, target)
    out = " ".join(out.split())
    return out if keep_case else out.upper()


def normalize_number(text: str) -> str:
    """Normalise a collector number while preserving its printed format.

    ``4/102``, ``004/189``, ``RC-001`` and ``TG05/TG30`` are all valid and must
    keep their structure. We only fold characters and spacing, never the digits
    or the leading zeros - the leading zeros are information.
    """
    text = normalize_text(text).strip()
    return text.replace(" ", "")


def validate_number_format(text: str, pattern: str | None = None) -> bool:
    """Check a collector number against a game's format rules.

    Without an explicit pattern, the accepted generic shapes are:
    ``N``, ``N/M``, ``XX-NNN``, ``XXN/YYY``. Anything else is suspicious but not
    necessarily wrong (sports cards vary widely), so callers use this as a
    confidence signal rather than a hard filter.
    """
    import re

    if pattern:
        return bool(re.fullmatch(pattern, text))
    generic = (
        r"\d+"
        r"|\d+/\d+"
        r"|[A-Z]{1,3}-?\d+"
        r"|[A-Z]{1,3}\d+/?[A-Z]{0,3}\d*"
        r"|\d+/[A-Z]{1,3}\d*"
    )
    return bool(re.fullmatch(generic, text))


# ---------------------------------------------------------------------------
# Fonts and drawing
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@lru_cache(maxsize=8)
def find_font(preferred: str | None = None):
    """Return a loaded PIL font, or ``None`` when no scalable font exists."""
    from PIL import ImageFont

    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(_FONT_CANDIDATES)
    # Also look inside the package (fonts shipped with the repo, if any).
    package_fonts = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    if package_fonts.exists():
        candidates.extend(str(p) for p in sorted(package_fonts.glob("*.ttf")))
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, 32)
            except OSError:  # pragma: no cover
                continue
    _LOG.warning("No scalable font found; falling back to PIL's default bitmap font")
    return None


def _load_font(size: int, preferred: str | None = None):
    from PIL import ImageFont

    base = find_font(preferred)
    if base is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(base.path, size)
    except (AttributeError, OSError):  # pragma: no cover
        return ImageFont.load_default()


@dataclass
class DrawnField:
    name: str
    text: str
    bbox: tuple[int, int, int, int]


def draw_card_text(
    image: np.ndarray,
    fields: dict[str, str],
    *,
    rois: dict[str, tuple[float, float, float, float]] | None = None,
    rng: np.random.Generator | None = None,
    ink_colour: tuple[int, int, int] = (16, 16, 20),
    font_path: str | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Render field text onto a card image at the layout's ROIs.

    Returns ``(image, boxes)`` where boxes are the actual pixel boxes the glyphs
    occupy. Those are the *true* annotations - drawing text and then guessing its
    box would introduce exactly the label noise the engine exists to avoid.

    Text is auto-fitted to the ROI width and the box is centred, which mirrors how
    cards actually print their fields.
    """
    from PIL import Image, ImageDraw

    rng = rng or np.random.default_rng(0)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    base = image if image.dtype == np.uint8 else np.clip(image * 255, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(base)
    draw = ImageDraw.Draw(canvas)
    height, width = base.shape[:2]
    boxes: dict[str, tuple[int, int, int, int]] = {}

    if rois is None:
        from ..layouts import resolve_layout

        rois = resolve_layout("POKEMON").rois_in_pixels((width, height))

    for name, text in fields.items():
        roi = rois.get(name)
        if roi is None or not text:
            continue
        x0, y0, x1, y1 = roi
        if max(roi) <= 1.5:  # normalised ROIs
            x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height
        roi_w = max(4, int(x1 - x0))
        roi_h = max(4, int(y1 - y0))
        text = normalize_text(text)
        size = _fit_font_size(draw, text, roi_w, roi_h)
        font = _load_font(size, font_path)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        except Exception:  # pragma: no cover - bitmap font fallback
            left, top, right, bottom = 0, 0, draw.textlength(text), size
        text_w = right - left
        text_h = bottom - top
        # Small random jitter: real cards are not perfectly aligned, and the
        # ink is not always dead-centre from the layout's nominal box.
        jitter_x = int(rng.normal(0, max(1.0, roi_w * 0.02)))
        jitter_y = int(rng.normal(0, max(1.0, roi_h * 0.04)))
        px = int(x0 + max(0, (roi_w - text_w) / 2) + jitter_x)
        py = int(y0 + max(0, (roi_h - text_h) / 2) + jitter_y)
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        draw.text((px - left, py - top), text, font=font, fill=ink_colour)
        boxes[name] = (
            max(0, px),
            max(0, py),
            min(width, px + int(text_w)),
            min(height, py + int(text_h)),
        )
    return np.asarray(canvas, dtype=np.uint8), boxes


def _fit_font_size(draw, text: str, max_width: int, max_height: int,
                   max_size: int = 400) -> int:
    """Binary search the largest font size that fits the box."""
    low, high = 6, min(max_size, max_height * 2)
    best = 6
    while low <= high:
        mid = (low + high) // 2
        try:
            length = draw.textlength(text, font=_load_font(mid))
        except Exception:  # pragma: no cover
            return best
        if length <= max_width * 0.96 and mid <= max_height:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def synth_card_image(
    fields: dict[str, str],
    size: tuple[int, int] = (630, 880),
    *,
    card_type: str = "POKEMON",
    card_id: str | None = None,
    background_colour: tuple[int, int, int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Create a complete synthetic clean card image from field text.

    This is what makes the repo self-contained: ``vision dataset bootstrap``
    can produce a source dataset with no copyrighted artwork, purely for
    pipeline testing and for exercising the synthetic engine end to end.
    """
    import colorsys

    from PIL import Image, ImageDraw, ImageFilter

    from ..layouts import resolve_layout

    rng = rng or np.random.default_rng(0)
    width, height = size
    if background_colour is None:
        hue = float(rng.random())
        saturation = float(rng.uniform(0.15, 0.6))
        value = float(rng.uniform(0.55, 0.92))
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        background_colour = tuple(int(c * 255) for c in rgb)

    canvas = Image.new("RGB", (width, height), background_colour)
    draw = ImageDraw.Draw(canvas)

    # A simple frame + art window, so the card reads as a card (the model must
    # not learn "plain rectangle = card").
    margin = int(min(width, height) * 0.05)
    draw.rectangle([margin, margin, width - margin, height - margin],
                   outline=(20, 20, 20), width=max(2, min(width, height) // 120))
    art_top = int(height * 0.16)
    art_bottom = int(height * 0.62)
    art_colour = tuple(int(c * 0.8 + rng.integers(0, 60)) for c in background_colour)
    art_colour = tuple(int(np.clip(c, 0, 255)) for c in art_colour)
    draw.rectangle([margin * 2, art_top, width - margin * 2, art_bottom], fill=art_colour)

    layout = resolve_layout(card_type)
    rois = layout.rois_in_pixels((width, height))
    image, boxes = draw_card_text(np.asarray(canvas), fields, rois=rois, rng=rng)
    # A touch of texture so the card is not a perfectly flat vector graphic.
    blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.4))
    return np.asarray(blurred, dtype=np.uint8), boxes
