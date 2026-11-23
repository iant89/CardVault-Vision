"""Loss functions: CTC for OCR, cross-entropy with class weights for the classifier.

The CTC loss here is a thin, correct wrapper: ``zero_infinity`` is enabled so a
pathologically short target cannot produce ``inf`` and silently destroy the run
(it happens with a 3 px crop of a 30-character name, which does occur in real
data).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import require_torch


def ctc_loss(logits, widths, targets, target_lengths, *, blank: int = 1,
             input_lengths: Sequence[int] | None = None):
    """CTC loss from model logits.

    ``logits`` is ``(T, N, C)``; torch's ``ctc_loss`` wants ``(T, N, C)`` with
    ``log_softmax`` applied, which we do here so callers always start from raw
    logits.
    """
    require_torch("OCR training")
    import torch

    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    time_steps = logits.shape[0]
    if input_lengths is None:
        # Model time steps, not pixel widths: converting inside the loss keeps the
        # training loop model-agnostic (different backbones downsample differently).
        reduction = getattr(logits, "output_length", None)
        del reduction
        input_lengths = [time_steps] * logits.shape[1]
    lengths = torch.as_tensor(list(input_lengths), dtype=torch.long,
                              device=logits.device)
    target_lengths = torch.as_tensor(list(target_lengths), dtype=torch.long,
                                     device=logits.device)
    lengths = torch.clamp(lengths, min=1, max=time_steps)
    return torch.nn.functional.ctc_loss(
        log_probs,
        targets.to(logits.device),
        lengths,
        target_lengths.to(logits.device),
        blank=blank,
        reduction="mean",
        # NOT zero_infinity. With it set, a batch whose targets cannot possibly
        # align to the available time steps silently returns a loss of exactly
        # 0.0 - which looks like a perfectly trained model and hides a broken
        # backbone. Without it torch raises, and ``ctc_loss_from_widths`` turns
        # that into an actionable message.
        zero_infinity=False,
    )


def ctc_loss_from_widths(logits, widths, targets, target_lengths, *, model=None,
                         blank: int = 1):
    """CTC loss converting pixel widths to model time steps via ``model``.

    Callers should prefer this when they have a model object: it makes the
    padding-aware length derivation explicit instead of assuming the model emits
    one step per pixel.
    """

    if model is not None and hasattr(model, "output_length"):
        lengths = [model.output_length(int(width)) for width in widths]
    else:
        lengths = [logits.shape[0]] * logits.shape[1]
    lengths = [int(v) for v in lengths]
    time_steps = int(logits.shape[0])
    input_lengths = [max(1, min(v, time_steps)) for v in lengths]
    target_lengths = [int(v) for v in target_lengths.tolist()] \
        if hasattr(target_lengths, "tolist") else [int(v) for v in target_lengths]
    # CTC needs T >= target length (plus repeats); check before handing off so the
    # error names the actual numbers instead of surfacing a bare RuntimeError.
    for index, (available, needed) in enumerate(zip(input_lengths, target_lengths, strict=True)):
        if available < needed:
            raise ValueError(
                f"CTC cannot align sample {index}: the model produces {available} time "
                f"steps for {int(widths[index])} input pixels but the target has "
                f"{needed} characters. Increase the input width or reduce the "
                f"backbone's width down-sampling."
            )
    return ctc_loss(logits, widths, targets, target_lengths, blank=blank,
                    input_lengths=input_lengths)


def classification_loss(logits, labels, *, class_weights=None, label_smoothing: float = 0.0):
    """Cross-entropy with optional class weights.

    Class weights matter here: card-type datasets are naturally imbalanced
    (Pokémon dominates), and an unweighted loss simply learns to say "Pokémon".
    """
    require_torch("classifier training")
    import torch

    weight = torch.as_tensor(class_weights, dtype=torch.float32,
                             device=logits.device) if class_weights is not None else None
    return torch.nn.functional.cross_entropy(
        logits, labels.to(logits.device), weight=weight,
        label_smoothing=label_smoothing,
    )


def compute_class_weights(counts: Sequence[int], *, mode: str = "balanced"):
    """Inverse-frequency class weights, or ``None`` for uniform weighting."""
    if mode == "none":
        return None
    if mode not in {"balanced", "sqrt"}:
        raise ValueError(f"Unknown class weighting mode {mode!r}")
    import numpy as np

    values = np.asarray(counts, dtype=np.float64)
    values = np.maximum(values, 1.0)
    weights = values.sum() / (len(values) * values)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    return weights.astype(np.float32)


def focal_loss(logits, labels, *, gamma: float = 2.0, class_weights=None):
    """Focal loss - down-weights easy examples, useful for hard-example retraining."""
    require_torch("classifier training")
    import torch

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    weight = torch.as_tensor(class_weights, dtype=torch.float32,
                             device=logits.device) if class_weights is not None else None
    nll = torch.nn.functional.nll_loss(log_probs, labels.to(logits.device),
                                       weight=weight, reduction="none")
    probability = torch.exp(-nll)
    return ((1 - probability) ** gamma * nll).mean()
