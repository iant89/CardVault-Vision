# Design decisions

Short records of choices that are easy to get wrong and expensive to reverse.
Each entry says what was decided, what was rejected, and why.

---

## Synthesise captures, not augmentations

**Decision.** The synthetic engine builds a scene and runs a simulated camera
over it: geometry → materials → lighting → optics → sensor → scene → JPEG.

**Rejected.** Applying random affine transforms and colour jitter to existing
images. That produces data along the axes you thought of, and none of the axes
that break real deployments — occlusion by a thumb, a specular highlight over the
one character that matters, a card half out of frame.

**Consequence.** Generation is slower (order 1 s per image at full resolution),
so throughput is a first-class concern and was measured and optimised rather than
assumed.

---

## Card-level splitting, not image-level

**Decision.** Splits assign *card ids*; every image of a card lands in the same
split. Near-duplicate images are grouped too.

**Rejected.** Splitting by file path. If the same physical card appears twice,
image-level splitting leaks it into the test set and inflates the score. The test
suite asserts this invariant directly.

---

## `num_classes = len(charset) + 2`, blank at index 1

**Decision.** Index 0 is padding, index 1 is the CTC blank, characters start at 2.

**Rejected.** Blank at index 0. It works with torch's CTC, but padding and blank
then collide, and every decoding path has to special-case it. Keeping them
separate makes the contract expressible in `contract.json` for non-Python
consumers.

---

## 4× width reduction, not 16×

**Decision.** Only the first two backbone blocks pool across width, so time steps
are `W/4`.

**Rejected.** Symmetric 2×2 pooling in every block. With four blocks that is a
16× reduction: a 200 px name crop yields 12 time steps for a 19-character
target. CTC then has no valid alignment, and with `zero_infinity=True` the loss
silently becomes `0.0` — indistinguishable from a perfectly trained model.

**Consequence.** Deep backbones keep a coarse sequence axis. `output_length()` is
the single source of truth, and the loss raises with the actual numbers when a
target cannot fit.

---

## Losses fail loudly

**Decision.** `zero_infinity=False`, plus an explicit pre-check that names the
widths and lengths involved.

**Why.** A silent zero loss is worse than a crash: it looks like convergence and
it survives review. This exact failure cost real debugging time here.

---

## Exports are one self-contained file

**Decision.** `external_data=False`. An export that tries to write a
`model.onnx.data` sidecar fails.

**Rejected.** Sidecar weights. The handoff to CardVault is a directory of files
that may be copied, cached or archived individually; a checksum covering only
`model.onnx` would not detect a missing weight file, and the failure would appear
in production rather than at export time.

---

## Metadata records what was produced, not what was requested

**Decision.** The exporter reads the opset back out of the written graph and
records that. Default target is opset 18.

**Why.** Current torch.exporter silently upgrades a requested opset when it lacks
implementations for the older one. Recording the request produced a metadata file
claiming 17 while the graph said 18 — a contract that lies to every consumer.

---

## Validation derives the task from metadata, not tensor rank

**Decision.** `metadata["task"]` decides the probe shapes.

**Why.** Both an OCR line and a classifier image are rank-4 tensors. Inferring
"OCR" from rank fed a `(1, 1, 32, W)` grayscale probe to a 3-channel classifier
and rejected a correct model.

---

## Validation probes widths the exporter never traced

**Decision.** The numerical check runs at several widths, including odd ones.

**Why.** "Dynamic axes" is a claim, not a fact. Only running an unseen width
proves the graph is actually dynamic rather than accidentally fixed at the trace
size.

---

## `--workers` and `--threads` stay separate

**Decision.** Two independent knobs: worker processes for data loading, threads
for CPU numerical work.

**Why.** Setting both to the core count oversubscribes the machine by roughly the
core count, and "I added parallelism and it got slower" is a recurring and
avoidable failure.

---

## CUDA requests never silently fall back

**Decision.** `--device cuda` on a machine without CUDA raises.

**Rejected.** Falling back to CPU with a warning. A GPU run that quietly becomes a
CPU run produces a result that is attributed to the wrong hardware and takes an
unexpected amount of time.

---

## Config resolution flattens sectioned YAML

**Decision.** `Config.load` resolves bare names, and the CLI flattens the
`training:` section so `--config train-ocr --set epochs=40` overrides both.

**Why.** Config files are grouped for readability but the CLI reads one
namespace. Without flattening, a config file loaded successfully, reported a
fingerprint, and silently changed nothing — the worst kind of bug.

---

## Torch is optional everywhere except training

**Decision.** Import torch lazily; raise `MissingDependencyError` naming the extra
to install.

**Why.** The dataset, synthetic and export-validation layers must run in a slim
environment. A `vision dataset validate` call should never cost a 2 GB download.

---

## Versions are immutable

**Decision.** Re-registering an existing `task + version` fails; promotion is the
only mutation.

**Why.** A version that can change underneath a consumer is not a version. The
registry stores a checksum so what was promoted can be verified afterwards.
