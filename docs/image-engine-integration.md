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

## What the measurements showed

The first run of `vision align` used the engine's defaults and found almost
nothing. Chasing that produced two findings, and the second one only became
visible after the first was fixed - so they are recorded here in the order they
were understood, not the order they appear in the pipeline.

### Finding 1: the edge threshold is the dominant defect

The engine's default `edgeThreshold` is 48. On these captures that is far too
aggressive: the card's outline is broken by perspective, shadow, gloss and its
own interior artwork, so at 48 the Sobel mask stops closing a boundary around
the card and the card is never proposed as a candidate at all. On 40 synthetic
captures, **17 produced zero candidates**, and on the 23 that did produce one,
the best candidate still had a median corner error of 0.360.

Raising the threshold made it worse (0/6 detected at 96, 160 and 220): with a
higher threshold there are fewer edges, not cleaner ones. *Lowering* it is what
works. A sweep over threshold x morphology radius, 40 samples each, all measured
as the **oracle** error - how close the best proposed candidate was to truth,
ignoring ranking - so that detection is isolated from ordering:

| `edgeThreshold` | `morphologyRadius` | candidates | oracle < 0.05 | oracle < 0.15 | median oracle error |
|---|---|---|---|---|---|
| 16 | 1 | 126 | 8 | 21 | 0.125 |
| 16 | 3 | 105 | 10 | 20 | 0.157 |
| 24 | 3 | 95 | 4 | 16 | 0.184 |
| 32 | 3 | 83 | 4 | 16 | 0.219 |
| 48 | 3 | 46 | 0 | 4 | 0.300 |

A second sweep narrowed it further: 12/3 gave 9 samples under 0.05 with a median
of 0.165; 8/3 was much worse (0 under 0.05, median 0.273, i.e. noise edges come
back); 12/5 gave 6. The usable band is roughly **12-24 with radius 3**, and no
single value is best for every image - a holo card that vanishes at 12 survives
at 24 and vice versa.

So the bridge does not pick one. It runs three presets (12/3, 16/3, 24/3), pools
the candidate lists, and lets selection choose from the union
(`integrations/selection.py:EDGE_PRESETS`).

Pooling on its own buys nothing - it only *widens* the list, and the engine's own
ranking still puts the whole frame on top, so pooled-with-engine-ranking is
slightly worse than a single tuned pass (median 0.286 vs 0.251). Pooling and
re-ranking have to land together: the pool supplies the right quad, the re-ranker
has to be able to pick it.

### Finding 2: ranking is real but second-order

With edges tuned, the card is proposed far more often, and now the ordering
matters. The engine's confidence is

```
conf = 0.25 * areaScore + 0.45 * shapeScore + 0.30 * fillScore
```

`shapeScore` collapses for a *correct* card whose edge mask is fragmented, while
`fillScore` rewards a clean compact mask - so a 3%-area sliver can outrank the
real card. The same formula has a second failure that matters more here: a quad
covering the **whole frame** is large, perfectly rectangular and correctly
proportioned, so it scores near 1.0 and wins. It was the top pick on 25 of 40
samples. No real card looks like that; the image border does.

`integrations/selection.py` re-ranks the pooled list with a policy that penalises
corners lying on the frame border and, deliberately, does not let area dominate.

### The configurations, measured

Two independent sets of 40 synthetic captures, profile `pokemon`, corner error
normalised by the truth card's diagonal. "returned" is how many samples produced
an answer at all.

Development set (dataset seed 11, capture seed 21):

| configuration | returned | < 0.02 | < 0.05 | < 0.10 | median error |
|---|---|---|---|---|---|
| engine defaults (thr 48), engine top-1 | 15 | 0 | 0 | 0 | 0.472 |
| engine defaults (thr 48), re-ranked | 9 | 0 | 0 | 2 | 0.157 |
| single tuned pass (thr 12/3), engine top-1 | 40 | 1 | 9 | 13 | 0.251 |
| single tuned pass (thr 12/3), re-ranked | 39 | 1 | 9 | 14 | 0.160 |
| pooled presets, engine top-1 | 40 | 2 | 8 | 10 | 0.286 |
| **pooled presets + re-ranked (`vision align` default)** | 39 | 2 | **13** | **24** | **0.066** |
| *ceiling: oracle over the pooled candidates* | 40 | 4 | 17 | 24 | 0.069 |

Holdout set (dataset seed 77, capture seed 93 - different cards and backgrounds):

| configuration | returned | < 0.02 | < 0.05 | < 0.10 | median error |
|---|---|---|---|---|---|
| engine defaults (thr 48), engine top-1 | 22 | 1 | 2 | 5 | 0.381 |
| engine defaults (thr 48), re-ranked | 16 | 1 | 2 | 6 | 0.158 |
| single tuned pass (thr 12/3), engine top-1 | 40 | 0 | 7 | 13 | 0.236 |
| single tuned pass (thr 12/3), re-ranked | 39 | 0 | 7 | 12 | 0.172 |
| pooled presets, engine top-1 | 40 | 1 | 11 | 17 | 0.271 |
| **pooled presets + re-ranked** | 39 | 0 | **17** | **24** | **0.053** |
| *ceiling: oracle over the pooled candidates* | 40 | 2 | 18 | 27 | 0.052 |

Reading the two tables together, the split of responsibility is:

- **edge tuning** is worth roughly 7-9 samples crossing 0.05 and cuts median
  error by about a third (0.472 -> 0.251 dev, 0.381 -> 0.236 holdout);
- **re-ranking on top** is worth 4-6 more samples crossing 0.05 and cuts median
  error by another three quarters (0.251 -> 0.066 dev, 0.236 -> 0.053 holdout);
- and the two only pay together, for the reason above.

Two honest qualifications:

- **The 2% tolerance is still not met.** The shipped configuration lands 2 of 40
  inside it on the development set and 0 of 40 on the holdout. This is a large
  improvement on 0, not a solved problem - and the development set is where the
  policy was tuned, so the holdout number is the one to trust.
- **What remains is ranking, not detection.** 13 (dev) and 17 (holdout) samples
  are under 0.05, but the oracle ceiling on the *same candidate list* is 17 and
  18. A small number of picks are the right quad a few pixels off; most of the
  remaining failures are a larger quad that swallowed the shadow, or the whole
  frame. Closing those needs a signal this bridge cannot compute: the engine is
  the only side that can tell "one edge is wrong" from "the geometry is right but
  the contour was drawn outside the print".

### Residual error: what is left and why

Reading the failing quads rather than the aggregate is what produced the
border term, and the same exercise shows the next limit. Typical remaining
failures are:

- **The whole frame, still.** About 4-6 per set are frame-sized quads that the
  border penalty reduces but does not eliminate, because with `border_weight`
  0.25 a 1.0-border quad still scores ~0.70 versus ~0.85 for a card.
- **A quad one edge too large**, covering the card plus its shadow, or one edge
  too small, cutting inside the printed border. Both are visually plausible and
  differ from truth by 0.10-0.15.
- **Holo and foil cards.** Sample `00000002` is a foil card whose diffraction
  stripes are strong interior edges; at any threshold the contour stage prefers
  the stripes. This is a detector limitation, not a ranking one.

## What is deliberately not built yet

- **No Python re-implementation of the engine's stages.** Two implementations of
  the same filter are two sources of truth. The bridge calls the real one.
- **No WASM or in-process embedding.** The Node subprocess keeps the Python
  dependency list unchanged, at the cost of requiring Node for this one workflow.
- **No rectification in the ingest path.** Rectifying a real photo, and cropping
  field ROIs in rectified card space, waits until detection clears 2% on
  synthetic data.
- **No engine change from here.** The two fixes that belong upstream - a lower
  default `edgeThreshold`, and a border-aware confidence - are written up as a
  patch description in `docs/image-engine-upstream-patch.md` rather than pushed,
  because this repository cannot publish to the engine.

## Reproducing

```bash
# build the engine somewhere
git clone https://github.com/iant89/CardVault-Image-Engine
cd CardVault-Image-Engine && npm install && npm run build --workspace @cardvault/image-engine

# generate captures with known card quads
vision dataset bootstrap --out /tmp/src --count 60 --seed 11
vision synthetic generate --source /tmp/src --out /tmp/ds --count 40 --seed 21

export CVV_IMAGE_ENGINE=/path/to/CardVault-Image-Engine/packages/core/dist

# what ships: pooled edge presets, re-ranked
vision align --dataset /tmp/ds --profile pokemon --json

# the same pool with the engine's own ranking: isolates pooling
vision align --dataset /tmp/ds --profile pokemon --no-rerank --json

# the baseline: one engine pass at its default threshold, engine ranking
vision align --dataset /tmp/ds --profile pokemon --edge-threshold 48 --json
```

The defaults were tuned on the first set, so a second set with different seeds is
the honest check - the numbers above were produced from `--seed 77` (dataset) and
`--seed 93` (captures) as well.
