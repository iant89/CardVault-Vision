# Integrating the CardVault Image Engine

[`CardVault-Image-Engine`](https://github.com/iant89/CardVault-Image-Engine) is a
TypeScript library for card-boundary detection, perspective rectification,
thresholding, CLAHE and capture-quality scoring. This repository is Python. This
document records how the two meet, what has actually been measured, and where the
seam still hurts.

## Why the two projects meet where they do

The exported artifacts already describe a dependency on an image layer without
naming one. `preprocessing.json` says, in as many words:

> Crop the field ROI in *rectified card space*, not in the raw frame.
> Perspective-normalise the card to a rectangle first, then cut the ROI.

Nothing in this repository does that to a real photograph. The synthetic engine
*knows* the card pose because it chose it; a consumer holding a phone does not.
The image engine's `AutomaticCardAlignmentStage` is the missing front half:
frame in, `CardGeometry{ corners, homography, confidence }` out.

That geometry contract is also unusually convenient for measurement. The
synthetic pipeline records the exact `card_quad` for every image it renders, so
the engine can be graded against ground truth rather than against another
heuristic's opinion.

## The bridge

The engine is TypeScript and this repository is Python, so the two talk through a
single Node process rather than a second embedded runtime:

```
Python (cardvault_vision.integrations.image_engine.align)
    -> one JSON job on stdin  { operation, width, height, format, pixels(base64) }
    -> node src/cardvault_vision/integrations/align_bridge.mjs
    -> the engine's AutomaticCardAlignmentStage
    -> one JSON result on stdout { status, geometry, candidates, timings }
```

Design notes worth keeping:

- **The bridge script lives inside the package**, next to `image_engine.py`,
  because it is resolved relative to that module. It is declared in
  `[tool.setuptools.package-data]` so a wheel ships it. An earlier version sat at
  the repository root: it worked from a checkout and would have broken from an
  installed package.
- **Absence is reported, not guessed.** `engine_available()` is False unless both
  a `node` binary and a built engine are present, and `vision align` exits with an
  actionable message instead of a traceback.
- **Input validation happens before any process is spawned**, so a float image or
  an unknown profile raises a programming error rather than "Node not found".
- **The engine is not on npm** (404 on `@cardvault/image-engine`), so it is
  consumed from a build. Point `CVV_IMAGE_ENGINE` at the `dist` directory that
  contains `index.js`.

### Verified environment

Node 22.22.3, npm 10.9.8. Building the engine (`npm install`, then
`npm run build --workspace @cardvault/image-engine`) and running its suite gives
**28 test files / 65 tests passing** in ~6 s on this machine.

## A convention trap that had to be pinned

The two projects express card aspect ratio as **exact reciprocals**:

| Project | Field | Trading card |
|---|---|---|
| CardVault Vision `CardLayout.aspect` | height / width | `88 / 63` ≈ 1.397 |
| Image engine `CardProfile.aspectRatio` | width / height | `63 / 88` ≈ 0.714 |

Feeding one convention to the other does not raise — it silently mis-ranks
candidates, because the engine's `isAspectRatioCompatible` normalises with
`Math.min(r, 1/r)` and therefore *accepts* a reciprocal. The conversion lives in
exactly one place (the `PROFILES` table in `align_bridge.mjs`) and
`tests/unit/test_image_engine_bridge.py` pins both numbers.

Corner **order**, by contrast, already agrees: both sides use
`TL, TR, BR, BL`. The grader still reports a rotation shift rather than silently
correcting for one, so a future divergence becomes visible instead of becoming a
permanently wrong crop.

## What the first measurement showed

`vision align --dataset <synthetic dataset> --profile pokemon`, on 12 full-scale
`phone_normal` captures:

| Metric | Value |
|---|---|
| within 2% of the card diagonal | **0 / 12** |
| detection rate (top candidate close) | 33.3% |
| **oracle** detection rate (best candidate close) | 0.0% |
| ranked below a worse candidate | 5 / 12 |
| median corner error | **0.474** of the diagonal |
| best-candidate median error | 0.302 |
| median IoU | 0.017 |
| engine status | 4 ALIGNED, 6 CARD_NOT_FOUND, 2 LOW_CONFIDENCE |

**The automatic aligner does not currently locate cards in this simulator's
output.** That is the honest reading, and it is stated here rather than buried.

Two distinct failures are visible in the numbers, and the distinction matters
because they need different fixes:

1. **Ranking.** On sample `00000001` the true card *is* found — IoU 0.764,
   corner error 0.061 — but it ranks **second**, below a 3%-area sliver
   (confidence 0.375 vs 0.265). The engine's confidence weights rectangularity
   (`fillScore`, weight 0.30) and area saturate quickly, while `shapeScore` —
   aspect-ratio accuracy, weight 0.45 — collapses for a *correct* card whose edge
   mask is fragmented by perspective and shadow. A perfect axis-aligned rectangle
   scores ≈ 0.998; this real card scores 0.562; the sliver scores 0.340. The
   ranking is doing what its formula says; the formula rewards a clean mask over a
   correct card.
2. **Detection.** 6 of 12 samples produced no candidate at all, and raising the
   Sobel threshold above 48 made this worse (0/6 detected at 96, 160 or 220), so
   the edge stage cannot be tuned to a working point on this data as-is.

Neither is a bug in this repository's bridge: the bridge reports what the engine
returns. Both are findings for the engine.

## Reproducing

```bash
# build the engine somewhere
git clone https://github.com/iant89/CardVault-Image-Engine
cd CardVault-Image-Engine && npm install && npm run build --workspace @cardvault/image-engine

# generate captures with known card quads
vision dataset bootstrap --out /tmp/src --count 40 --seed 11
vision synthetic generate --source /tmp/src --out /tmp/ds --count 12 --seed 21

# grade
export CVV_IMAGE_ENGINE=/path/to/CardVault-Image-Engine/packages/core/dist
vision align --dataset /tmp/ds --profile pokemon --output alignment.json
```

## What is deliberately not built yet

- **No Python re-implementation of the engine's stages.** Two implementations of
  the same filter are two sources of truth. The bridge calls the real one.
- **No WASM or in-process embedding.** The Node subprocess keeps the Python
  dependency list unchanged, at the cost of requiring Node for this one workflow.
  Every consumer of an *exported artifact* still needs only an ONNX runtime.
- **No rectification in the ingest path.** Rectifying a real photo, and cropping
  field ROIs in rectified card space, waits until detection works on synthetic
  data — otherwise it would be a pipeline built on a component that returns the
  wrong rectangle.
