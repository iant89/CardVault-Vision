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
