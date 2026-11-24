# Upstream patch: two detection/ranking fixes for `CardVault-Image-Engine`

This is a hand-off document, not code in this repository. It records two changes
that belong in the engine rather than in CardVault Vision's bridge, the
measurements behind them, and the exact places they touch. Nothing here has been
pushed: this repository has no write access to
[`CardVault-Image-Engine`](https://github.com/iant89/CardVault-Image-Engine), and
its `main` is the source of truth for that project.

Everything below was measured with `vision align` against 80 synthetic captures
with exact recorded quads (two independent sets of 40). The full comparison lives
in [`image-engine-integration.md`](image-engine-integration.md).

## Summary

| change | effect on 40-sample dev set | effect on 40-sample holdout |
|---|---|---|
| lower the default `edgeThreshold` from 48 to 16, `morphologyRadius` 1 to 3 | 15 -> 40 samples produce a candidate; < 0.05 goes 0 -> 9 | 22 -> 40; 2 -> 7 |
| add a frame-border term to `computeConfidence` | among fixed candidate lists, picks under 0.05 go 9 -> 13 | 7 -> 17 |
| both together | 0 -> 13 under 0.05, 0 -> 24 under 0.10; median error 0.472 -> 0.066 | 2 -> 17 under 0.05, 5 -> 24 under 0.10; median 0.381 -> 0.053 |

Neither change is project-specific: both make the detector better at "a card that
does not fill the frame and has a textured face", which is what a card detector
sees in every real photograph. Nothing here references CardVault Vision, its
datasets, or its profiles.

## Patch 1: the default edge threshold is too high for real material

**File:** `packages/core/src/alignment/automaticCardAlignment.ts` (the stage's
default options).

```diff
-  const { edgeThreshold = 48, morphologyRadius = 1, ... } = options;
+  const { edgeThreshold = 16, morphologyRadius = 3, ... } = options;
```

**Why.** At 48 the Canny/Sobel stage emits only the strongest boundaries. On a
card photographed at an angle, the outline is fragmented by shadow, gloss and the
card's own artwork, so no closed contour survives and the card is never proposed
at all - `CARD_NOT_FOUND` on 17 of 40 captures that each contain exactly one
card. Raising the threshold to suppress background texture makes this worse, not
better: 0 of 6 detected at 96, 160 or 220. Lowering it is the fix. The measured
usable band is 12-24 with radius 3; below 8, noise edges dominate.

**Note for the engine.** 16/3 is a better *default* than 48/1 for this kind of
material, but no single threshold is best for every image: a holo card that
vanishes at 12 survives at 24 and vice versa. CardVault Vision works around that
by running 12/3, 16/3 and 24/3 and pooling the candidate lists. If the engine
preferred to own that, a "multi-scale mask" option - run the mask at several
thresholds and merge the contours before filtering - would make the single-pass
API good enough on its own and remove a full round-trip per extra threshold.

**Risk.** Lower thresholds admit more spurious contours. They are filtered later
by `minAreaRatio` and by confidence; the measured effect on samples that already
worked at 48 was neutral (none regressed from aligned to not-found), but that is
40 samples, not a corpus.

## Patch 2: confidence rewards the wrong large quad

**File:** `packages/core/src/alignment/cardDetector.ts`, `computeConfidence`.

Current formula:

```
confidence = 0.25 * areaScore + 0.45 * shapeScore + 0.30 * fillScore
areaScore  = min(1, sqrt(areaRatio / 0.35))
shapeScore = max(0, 1 - abs(ratioError) / 0.12)
fillScore  = min(1, rectangularity / 0.85)
```

Two problems, one much bigger than the other.

### 2a. The image border wins (dominant)

A quad equal to the whole frame is maximal on all three terms: area ratio 1.0,
aspect exactly the frame's, fill 1.0. It is returned as the top candidate on 25 of
40 captures. **A detector whose answer is "the entire image" has not detected
anything**, and the frame's aspect ratio is an artefact of the camera, not of a
card. Suggested term:

```diff
+  // Corners sitting on the frame border are background, not card. Penalise
+  // them: a frame-sized quad is maximal on every other term, and returning
+  // the whole image is not a detection.
+  const borderScore = fractionOfCornersOnFrameBorder(corners, width, height, margin);
   return (
     0.25 * areaScore +
     0.45 * shapeScore +
-    0.30 * fillScore
+    0.30 * fillScore -
+    0.25 * borderScore
   );
```

where `margin` is `max(1, 0.01 * min(width, height))` and `borderScore` is the
fraction of corners within `margin` of the frame edge. A real card may graze the
border (tight crops exist), so this is a penalty, not a rejection. Weight 0.25 is
what CardVault Vision ships; weights from 0.15 to 0.30 measured the same, and it
degrades gracefully below that.

Measured on otherwise-identical candidate lists: picks under 0.05 go 9 -> 13
(dev) and 7 -> 17 (holdout); median error 0.251 -> 0.066 and 0.236 -> 0.053.

### 2b. `shapeScore` punishes correct cards with fragmented masks (minor)

`shapeScore` is aspect-ratio agreement, weighted 0.45 - the largest single term -
but it is not a shape measure: it only says the quad is card-proportioned, which
a great many wrong quads also are. Meanwhile `fillScore` (rectangularity) is the
term a genuinely fragmented edge mask damages, and it carries the smallest weight.
The observed consequence is a 3%-area sliver outranking the true card on a handful
of samples (confidences 0.375 vs 0.265 on sample `00000001`).

Two safer options, in order of preference:

1. **Report the terms.** Add `confidenceBreakdown: { areaScore, shapeScore,
   fillScore, borderScore }` to the candidate record. Then a consumer can judge
   the ranking without re-deriving the geometry from corners, which is what
   CardVault Vision currently has to do.
2. **Rebalance towards geometry you cannot fake.** Moving weight from
   `shapeScore` to `fillScore` (e.g. 0.35/0.30) helped on our data, but by less
   than the border term and with a smaller margin over noise. It is not worth
   changing without a broader corpus; the border term does not have that problem,
   because a frame-sized quad is unambiguously not a card.

## What we are *not* asking for

- No project-specific logic in the engine: no layout names, no vocabulary, no
  CardVault Vision profiles. Both patches are expressed purely in terms of the
  frame and the candidate quads the engine already computes.
- No change to the `StandardizedCardImage` or `CardGeometry` contracts. The
  candidates array already carries everything needed; the only additive change
  proposed is `confidenceBreakdown`.

## How to verify

The harness is this repository's, but the measurements do not depend on it:

```bash
vision dataset bootstrap --out /tmp/src --count 60 --seed 11
vision synthetic generate --source /tmp/src --out /tmp/ds --count 40 --seed 21
CVV_IMAGE_ENGINE=/path/to/engine/packages/core/dist \
  vision align --dataset /tmp/ds --profile pokemon --json
```

`vision align --edge-threshold 48` reproduces the baseline, `--no-rerank` isolates
the effect of pooling, and the default grades the full bridge. Once the engine
adopts Patch 1, CardVault Vision can drop the multi-threshold loop, which removes
two of the three Node round-trips per image.
