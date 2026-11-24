# Model card — name OCR and card classifier

## Intended use

CardVault uses these models to read a card's identifying fields from a user's
photo, after the card boundary has been located and perspective-normalised.

| Task | Input | Output | Consumer |
|---|---|---|---|
| `name-ocr` | grayscale line crop, `(1, 32, W)` dynamic width | CTC logits `(T, C)` | reads card names |
| `number-ocr` | same contract | CTC logits `(T, C)` | reads collector numbers |
| `card-classifier` | RGB card image `(3, 160, 160)` | logits `(n_classes,)` | picks the game-specific layout |

## Out of scope

- **Grading or condition assessment.** No model here evaluates card condition.
- **Price or authenticity.** Nothing here estimates value or detects fakes.
- **Arbitrary text.** The charset is fixed at export time; characters outside it
  are dropped, and the model has never seen scripts outside it.
- **Raw frames.** OCR expects a rectified line crop, not a full photo. See
  `preprocessing.json` in the artifact.

## The contract

Both OCR architectures share one contract, which is what makes them comparable:

```
input:  (N, 1, 32, W)   float32, grayscale, values 0..1, height fixed, width dynamic
output: (T, N, C)       float32 raw logits
C    =  2 + len(charset)
```

Index 0 is padding and index 1 is the CTC blank, so character *i* of
`labels.json` lives at output index `i + 2`. `contract.json` states this, and a
consumer that ignores it shifts every character.

Time steps are `W / 4` for the CNN backbones. This is 4× rather than 16× on
purpose: a 19-character name is roughly 200 px wide, and a 16× reduction leaves
12 time steps — fewer than the characters to align, which makes CTC
mathematically unsolvable. That bug was found by refusing to let a zero loss
pass as success; see `docs/decisions.md`.

## Architectures

| Name | Character | Parameters |
|---|---|---|
| `crnn` | CNN + BiGRU + CTC; the baseline | 3.76 M |
| `crnn_small` | same, narrower | 0.59 M |
| `gru` | single-layer GRU head | 0.55 M |
| `tcn` | dilated temporal convolutions; cheapest to run | 2.10 M |
| `transformer` | self-attention over time | 2.41 M |

Comparisons use the same dataset, split, seed, budget and metric — enforced by
`vision train compare`. An architecture is only adopted if it wins on the metric
that the product cares about, not on training loss.

The classifier uses a small CNN over the whole card. There is deliberately **no
shared multi-task backbone**: the roadmap only permits one if a benchmark proves
it wins, and that benchmark has not been run.

## Evaluation

- **OCR:** character error rate (CER), word error rate, exact-match rate, plus
  per-length and per-confidence breakdowns. CER is the selection metric.
- **Classifier:** macro-F1 and accuracy, with a declared abstain threshold. A
  weak guess is reported as `UNKNOWN` rather than routed on.
- **Hard-example mining** attributes each failure to a likely cause (glare,
  blur, low light, perspective, occlusion, tiny text, compression, character
  confusion, layout/language) and emits a targeted generation command to cover
  the gap.

## Measured baseline (CPU, synthetic)

A `crnn` name-OCR run on a 700-sample synthetic set at full render scale
(`phone_normal`, 250 source cards producing 215 distinct titles), split 80/10/10
by card into 560/83/57 samples. Field `name`, 104 distinct strings in train,
40 epochs, batch 8, lr 1e-3, no early stopping.

| Metric | Value |
|---|---|
| held-out CER (val, 31 samples) | **0.845** |
| held-out CER (test, 24 samples) | **0.824** |
| exact match | 0.0 |
| character accuracy | 0.16 / 0.18 |
| train loss (epoch 40) | 3.26 |

**This number is a floor, not a ceiling, and the run that produced it was
under-trained.** A follow-up budget sweep on the same data showed the loss
sitting on a plateau of ~3.3 until roughly epoch 40, then collapsing: on a
24-sample subset, CER went from 1.000 at epoch 10 to 0.75 at epoch 100 to a
**best of 0.119** at epoch ~380, with exact match reaching 42% on the samples it
trained on. Twenty to forty epochs of this configuration is simply before the
model has started to read; the plateau is not a data-vocabulary ceiling.

### Does it read, or does it recall?

That was the question this baseline existed to answer, and the 40-epoch run
cannot answer it: at CER 0.85 both halves are equally bad. The
`evaluate_ocr(..., known_references=...)` split reports CER separately for
held-out strings that appeared in training ("seen") and strings that did not
("unseen"):

| Split | Seen | Unseen |
|---|---|---|
| val | CER 0.864 (n=6) | CER 0.841 (n=25) |
| test | CER 0.828 (n=5) | CER 0.824 (n=19) |

The halves are indistinguishable, which rules out *simple* memorisation but
proves nothing about reading — a model that has collapsed to a near-constant
output scores the same on anything. The predictions show the actual failure
mode: `'Sable Reaver of Nine Winters'` decodes as `'ShoPa Pas Paergs'` and
`'Müller's Wager'` as `'Polene'`. Outputs cluster on a handful of strings
(`Sho Paene`, `Pol Bene`, `Sno Werg`) that are too short — the model emits 6-10
characters for references 14-38 characters long.

The honest reading is that **the study is inconclusive at this budget**, and the
next measurement is a properly-trained run, not a bigger dataset. The instrument
for the real question (`known_references`) is now in place and covered by tests.

### Data-quality caveat found while measuring

Two generator defects distorted this baseline and are now fixed:

* **A worker OOM killed 990 of 2,800 samples and the run still reported
  success.** The pool turned `BrokenProcessPool` into a per-sample
  "rejection", so a third of the dataset silently vanished. Workers now load a
  downscaled source pool, and a dead worker is a loud retry rather than a
  quality statistic.
* **34% of name crops were near-illegible — fixed.** On the 700-sample set, 34%
  of `name` crops had maximum luminance below 0.35 and 11% had contrast below
  0.15; source cards themselves were fine (median ROI contrast 0.52), so the loss
  happened in the camera simulation. `low_light` was not the cause
  (`phone_normal` has a 5% underexposure chance). The cause was that **no stage
  of the chain could ever brighten a frame**: the illumination field averages
  below 1, `soft_light` scales by 0.6..1.4, `adjust_exposure` multiplies by
  `2**brightness` with a negative mean, and the surface shading multiplies
  again. Measured over 24 renders the output was 2.03x darker than the source
  card with only 0.36 correlation between source and rendered luminance — the
  simulator was discarding the scene's brightness. A real camera meters the
  scene and settles on an exposure; `apply_lighting` now does the same, with a
  gain limit. On 220 freshly generated samples the share of crops with max
  luminance under 0.35 fell from 34% to 11% and the share with contrast under
  0.15 from 11% to 1%; median crop contrast rose from 0.410 to 0.485 and the
  10th percentile from 0.146 to 0.272. Training a head to read crops a human
  cannot read sets a ceiling no dataset size can lift, so this had to be fixed
  before any accuracy number means anything.

  The gain limit is not incidental. Metering unbounded erased the `low_light`
  defect entirely — those frames came out at the same median luminance as
  ordinary ones — so a deliberately underexposed scene meters against a lower
  ceiling and stays dark (median 0.141 vs 0.289, 5 of 16 under 0.12).

  **Better crops did not, on their own, produce a better model.** A paired run —
  same architecture, batch 8, lr 3e-3, 160 epochs, no early stopping — gives
  best val CER **0.819 on the dark crops** and **0.881 on the metered crops**. A
  second "after" dataset, generated later, reached only CER 1.0 before collapsing
  to constant output.

  That result is reported as measured, and it does not support "fixing exposure
  fixes the model". The two datasets are not a clean controlled comparison: they
  use different source-card draws (11 vs 31) and different splits, so they have
  different held-out sets — the before run's 31 validation samples contain
  duplicates of only a handful of distinct titles. The honest reading is that
  **held-out CER is not separable from sampling noise at this dataset size**, in
  either direction. Fixing legibility is still correct on its own terms — the
  model cannot read text that is not in the image — but the accuracy claim needs
  a controlled experiment, not this one. See "What to measure next".

Getting even this far required fixing four earlier data bugs, each of which
produced CER exactly 1.0 and looked like a broken model: stratified splitting
that confined whole card types to one split, `name-ocr` hardcoding the `name`
field when sports layouts use `player`, a source pool that let one card take 43%
of all draws, and rendering at `render_scale` 0.5, which left name crops ~10 px
tall. See `docs/decisions.md`.

## What to measure next

The measurements above repeatedly ran into the same wall: with 100-200 training
crops over ~100 distinct strings, held-out CER cannot distinguish a model that
reads from one that has memorised, and a factor-of-two change in data quality is
inside the noise. Before any accuracy figure is worth publishing:

1. **Scale the training set past the memorisation regime.** The generator can
   now produce it (worker crashes fixed), but a run large enough to separate
   reading from recall needs roughly 3,000 samples over the current 162-card
   source pool, and more source cards than that.
2. **Fix the run length.** Best CER on a 24-sample subset arrives near epoch 380;
   most runs here stopped at 40-160. A budget sweep should set the default, not
   habit.
3. **Report seen and unseen separately, always.** `known_references` makes this
   free. A single aggregate CER hides the only distinction that matters.

## Limitations

1. Metrics reported so far are measured on **synthetic** data. They describe the
   simulator's learnability, not real-world accuracy.
2. The models have no notion of uncertainty beyond the softmax confidence, which
   is known to be over-confident on out-of-distribution inputs.
3. Names outside the training distribution's languages and lengths will degrade
   silently unless the abstain threshold catches them.
4. Holo and foil cards under direct glare remain the hardest case: the specular
   highlight can cover characters completely.

## Artifact and reproducibility

Every artifact records the training config, dataset content hash, hardware,
library versions and metrics. Validation re-runs the exported graph against
PyTorch on inputs the exporter never traced (including unseen widths) and
records the result. Promotion to production requires that validation to pass.
