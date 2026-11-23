"""Model export to portable inference artifacts (Phases 14-16).

The export structure follows the roadmap, because CardVault's loader is written
against it::

    exports/
      name-ocr/
        1.0.0/
          model.onnx
          metadata.json          # architecture, dataset, metrics, git commit
          labels.json            # CTC label list (index 0 = pad, 1 = blank)
          preprocessing.json     # input spec: shape, normalisation, resize rule
          checksum.sha256        # of model.onnx
          contract.json          # output spec + decoding rules for the consumer
          validation.json        # written by Phase 15 export validation

Design constraints that shaped this:

* **CardVault never imports Python from this repo.** The ``contract.json`` and
  ``preprocessing.json`` files describe everything the consumer needs in data, so
  a TypeScript/Go/.NET loader can implement it.
* **The ONNX graph exports the *raw logits*, not the decode.** Decoding is CTC
  best-path, which is 20 lines in any language; exporting it as a graph makes the
  model harder to consume and impossible to inspect.
* **Dynamic width.** The OCR graph takes a dynamic time dimension so a long name
  does not need padding to a fixed 1600 px, but it is exported with a fixed named
  dimension for the axis that must stay fixed (height).
* **Checksums are mandatory.** A model artifact without a hash cannot be pinned.

Torch/ONNX are optional; exporting raises a precise error with an install hint.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..errors import ExportError
from ..logging_utils import get_logger
from ..models import require_torch
from ..reproducibility import git_commit, library_versions, write_json
from ..version import SCHEMA_VERSION

_LOG = get_logger("export")

#: Target ONNX opset. 18 is the lowest opset every maintained ONNX Runtime
#: accepts *and* the lowest the current torch exporter implements natively -
#: asking for 17 makes torch warn and silently write 18 anyway.
OPSET_VERSION = 18


#: Tasks that may be exported and registered.
KNOWN_TASKS = frozenset({"name-ocr", "number-ocr", "card-classifier"})

#: Version strings are used as directory names, so keep them boring.
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class ExportSpec:
    """How to export one model."""

    task: str                     # name-ocr | number-ocr | card-classifier
    version: str
    architecture: str
    input_height: int = 32
    input_width: int = 320        # dynamic-width reference; export uses this trace
    input_size: int = 160         # classifier square input
    dynamic_width: bool = True
    charset: str = ""
    class_names: tuple[str, ...] = ()
    opset: int = OPSET_VERSION
    dataset_version: str | None = None
    dataset_hash: str | None = None
    metrics: dict = field(default_factory=dict)
    training_config: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate at construction so an impossible spec cannot be carried around
        # and only explode midway through an export.
        self.validate()

    @property
    def is_ocr(self) -> bool:
        return self.task in {"name-ocr", "number-ocr"}

    @property
    def num_classes(self) -> int:
        """Model output width: 2 CTC specials + charset, or the class count."""
        return len(self.charset) + 2 if self.is_ocr else len(self.class_names)

    def validate(self) -> None:
        """Fail before touching the filesystem or torch.

        ``version`` becomes a directory name and the task becomes a registry key,
        so both are validated strictly - a version of ``../production`` would
        otherwise write outside the exports tree.
        """
        if self.task not in KNOWN_TASKS:
            raise ExportError(
                f"Unknown task {self.task!r}; expected one of {sorted(KNOWN_TASKS)}"
            )
        if not self.version or not _SAFE_VERSION.match(self.version):
            raise ExportError(
                f"Invalid version {self.version!r}; use digits and dots "
                f"(for example 1.0.0 or 2026.11)"
            )
        if self.is_ocr and not self.charset:
            raise ExportError(f"{self.task} export needs a charset")
        if not self.is_ocr and not self.class_names:
            raise ExportError("Classifier export needs class_names")
        if self.input_height < 8 or self.input_width < 8 or self.input_size < 8:
            raise ExportError("input dimensions are implausibly small")
        if self.is_ocr and len(set(self.charset)) != len(self.charset):
            raise ExportError("charset contains duplicate entries")


def _graph_opset(model_path: Path) -> int:
    """The default-domain opset actually recorded in the graph."""
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx is required to export at all
        return 0
    model = onnx.load(str(model_path), load_external_data=False)
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return 0


def _verify_dynamic_axes(model_path: Path, spec: ExportSpec) -> None:
    """Check the written graph really has the symbolic axes metadata promises.

    ``metadata.json`` tells CardVault it may send any batch size and (for OCR) any
    width. An exporter that quietly bakes one of those in as a constant produces a
    model that works until the first batch of two arrives. Fail the export here
    instead.
    """
    try:
        import onnx
    except ImportError:  # pragma: no cover
        return
    graph = onnx.load(str(model_path), load_external_data=False).graph
    inputs = {value.name: value for value in graph.input}
    image_input = inputs.get("images")
    if image_input is None:
        raise ExportError("Exported graph has no input named 'images'")
    axes = [(dim.dim_param or dim.dim_value)
            for dim in image_input.type.tensor_type.shape.dim]
    expected = {0: "batch"}
    if spec.is_ocr:
        expected[3] = "width"
    for index, name in expected.items():
        if axes[index] != name:
            raise ExportError(
                f"Exported graph axis {index} is {axes[index]!r}, expected symbolic "
                f"{name!r}. metadata.json would advertise a dynamic axis the model "
                f"does not have. Check the exporter version or the dynamic_axes call."
            )


def _assert_self_contained(out_dir: Path, model_path: Path) -> None:
    """Fail the export if the graph references weights stored beside it.

    An external-data model is only loadable while its ``.data`` sibling is present
    with the exact expected name. CardVault's handoff is a directory of files, so
    a silent sidecar is a latent production breakage; catch it at export time.
    """
    if not (out_dir / "model.onnx.data").exists():
        return
    size = (out_dir / "model.onnx.data").stat().st_size
    raise ExportError(
        f"ONNX export produced an external-weights sidecar (model.onnx.data, "
        f"{size / 1024**2:.1f} MiB) for {model_path.parent.name}. The artifact "
        f"must be a single self-contained model.onnx so a consumer can verify it "
        f"with checksum.sha256. Remove the sidecar and re-export with "
        f"external_data=False, or reduce the model size."
    )


def _artifact_checksum(model_path: Path) -> str:
    """sha256 of the complete artifact.

    Hashing ``model.onnx`` alone is not enough when weights live beside it, so the
    sidecar (checked absent above) is folded in defensively for older exports.
    """
    digest = hashlib.sha256()
    digest.update(model_path.read_bytes())
    sidecar = model_path.with_suffix(model_path.suffix + ".data")
    if sidecar.exists():
        digest.update(sidecar.read_bytes())
    return digest.hexdigest()


def artifact_index_offset(spec: ExportSpec) -> int:
    """How far ``labels.json`` is shifted from the model's output index.

    OCR outputs reserve index 0 for padding and index 1 for the CTC blank, so
    label *i* lives at output index ``i + 2``. A consumer that ignores this
    silently shifts every decoded character. The classifier has no reserved
    indices, hence 0.
    """
    return 2 if spec.is_ocr else 0


def preprocessing_spec(spec: ExportSpec) -> dict:
    """The input contract, expressed so a non-Python consumer can implement it."""
    if spec.is_ocr:
        return {
            "type": "ocr-line",
            "color_space": "grayscale",
            "channel_order": "none",
            "input_layout": "NCHW",
            "input_height": spec.input_height,
            "width": "dynamic" if spec.dynamic_width else spec.input_width,
            "resize_rule": "scale to input_height preserving aspect ratio; "
                           "round to nearest integer width",
            "value_range": [0.0, 1.0],
            "normalization": {"mean": [0.0], "std": [1.0]},
            "source_region": "field_roi from layout, expanded by 2 px, "
                             "clipped to the card boundary",
            "notes": [
                "Crop the field ROI in *rectified card space*, not in the raw frame.",
                "Perspective-normalise the card to a rectangle first, then cut the ROI.",
            ],
        }
    return {
        "type": "whole-card-classification",
        "color_space": "rgb",
        "channel_order": "RGB",
        "input_layout": "NCHW",
        "input_size": spec.input_size,
        "resize_rule": f"resize the perspective-normalised card to "
                       f"{spec.input_size}x{spec.input_size} (bilinear)",
        "value_range": [0.0, 1.0],
        "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "notes": [
            "Normalise the card boundary to a rectangle before classification.",
            "Aspect ratio distortion is acceptable and was present during training.",
        ],
    }


def contract_spec(spec: ExportSpec) -> dict:
    """The output contract plus decoding rules."""
    if spec.is_ocr:
        return {
            "outputs": [
                {
                    "name": "logits",
                    "shape": ["T", "N", spec.num_classes],
                    "dtype": "float32",
                    "description": "unnormalised logits over the CTC label set",
                }
            ],
            "decoding": {
                "type": "ctc_greedy",
                "blank_index": 1,
                "pad_index": 0,
                "steps": [
                    "softmax over the last axis if probabilities are needed",
                    "argmax over the last axis for each time step",
                    "collapse consecutive equal indices",
                    "drop blank (1) and pad (0)",
                    "map remaining indices to labels.json (index i-2)",
                ],
                "confidence": "mean of the max softmax probability over the "
                              "collapsed steps",
            },
            "labels": "labels.json",
        }
    return {
        "outputs": [
            {
                "name": "logits",
                "shape": ["N", spec.num_classes],
                "dtype": "float32",
                "description": "unnormalised class logits",
            }
        ],
        "decoding": {
            "type": "argmax",
            "steps": [
                "softmax over the last axis",
                "argmax",
                "map index to labels.json",
            ],
            "confidence": "max softmax probability",
            "abstain_rule": "treat confidence below 0.5 as UNKNOWN rather than "
                            "routing the card on a weak guess",
        },
        "labels": "labels.json",
    }


def export_onnx(
    model,
    spec: ExportSpec,
    out_root: str | Path = "exports",
    *,
    validate: bool = True,
) -> Path:
    """Export ``model`` to ``<out_root>/<task>/<version>/`` and return that path."""
    require_torch("model export")
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        from ..errors import MissingDependencyError

        raise MissingDependencyError("ONNX export", "onnx", "export") from exc

    import torch

    spec.validate()
    out_dir = Path(out_root) / spec.task / spec.version
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.onnx"

    model.eval()
    model.cpu()
    if spec.is_ocr:
        dummy = torch.zeros(1, 1, spec.input_height, spec.input_width)
        dynamic_axes = {"images": {0: "batch", 3: "width"}, "logits": {0: "time", 1: "batch"}}
        input_names = ["images"]
    else:
        dummy = torch.zeros(1, 3, spec.input_size, spec.input_size)
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}}
        input_names = ["images"]

    # Use ``dynamic_axes`` rather than the newer ``dynamic_shapes``.
    #
    # The dynamo exporter prefers ``dynamic_shapes`` and warns about this call,
    # but in this torch version it only applies the *last* axis of a rank-4
    # input, silently exporting batch=1 as a constant. Verified by reading the
    # written graph: the width stayed symbolic while batch did not. A graph that
    # only accepts batch 1 while metadata advertises a dynamic batch is a worse
    # outcome than a warning, so we keep the API that produces the right graph.
    # ``_verify_dynamic_axes`` below re-checks the graph and fails the export if
    # this behaviour ever changes.
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*dynamic_axes.*")
            torch.onnx.export(
                model,
                dummy,
                str(model_path),
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=spec.opset,
                do_constant_folding=True,
                export_params=True,
                # Force a single self-contained file. torch.onnx now defaults to
                # spilling weights over 2 GB into a sidecar "<model>.data" file,
                # and even smaller models can end up externalised depending on
                # version. A producer/consumer handoff through a file drop cannot
                # rely on two files arriving together, and a checksum covering
                # only the graph would not detect a missing weight file.
                external_data=False,
            )
    except Exception as exc:
        raise ExportError(f"ONNX export failed for {spec.task}: {exc}") from exc

    labels = (list(spec.charset) if spec.is_ocr else list(spec.class_names))
    write_json(out_dir / "labels.json", {
        "task": spec.task,
        "count": len(labels),
        "labels": labels,
        "index_offset": 2 if spec.is_ocr else 0,
        "note": ("index 0 is <pad> and index 1 is the CTC blank; character i maps "
                 "to labels[i] at model index i+2") if spec.is_ocr else
                "index i maps to labels[i]",
    })
    write_json(out_dir / "preprocessing.json", preprocessing_spec(spec))
    write_json(out_dir / "contract.json", contract_spec(spec))

    _assert_self_contained(out_dir, model_path)
    _verify_dynamic_axes(model_path, spec)
    actual_opset = _graph_opset(model_path)
    if actual_opset != spec.opset:
        # Newer torch.onnx can upgrade the requested opset. The graph is what
        # CardVault loads, so metadata (and the contract CardVault keys off) must
        # record the opset that was actually written, not the one we asked for.
        _LOG.warning(
            "Export requested ONNX opset %s but the graph was written at opset %s; "
            "recording %s in metadata.json", spec.opset, actual_opset, actual_opset,
        )
    checksum = _artifact_checksum(model_path)
    (out_dir / "checksum.sha256").write_text(f"{checksum}  model.onnx\n", encoding="utf-8")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "task": spec.task,
        "version": spec.version,
        "architecture": spec.architecture,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "onnx",
        "opset": actual_opset,
        "input_spec": {
            "name": "images",
            "shape": (["batch", 1, spec.input_height, "width"] if spec.is_ocr
                      else ["batch", 3, spec.input_size, spec.input_size]),
            "dtype": "float32",
        },
        "output_spec": {
            "name": "logits",
            "shape": (["time", "batch", spec.num_classes] if spec.is_ocr
                      else ["batch", spec.num_classes]),
            "dtype": "float32",
        },
        "num_classes": spec.num_classes,
        "checksum_sha256": checksum,
        "git_commit": git_commit(),
        "dataset": {"version": spec.dataset_version, "content_hash": spec.dataset_hash},
        "metrics": spec.metrics,
        "training_config": spec.training_config,
        "libraries": library_versions(),
        "files": ["model.onnx", "metadata.json", "labels.json", "preprocessing.json",
                  "contract.json", "checksum.sha256"],
        **spec.extra,
    }
    write_json(out_dir / "metadata.json", metadata)
    _LOG.info("Exported %s %s -> %s (%s, %.1f KB)", spec.task, spec.version, out_dir,
              checksum[:12], model_path.stat().st_size / 1024)

    if validate:
        from .validate import validate_export

        report = validate_export(out_dir, reference_model=model, spec=spec)
        if not report.passed:
            raise ExportError(
                f"Exported model failed validation: {'; '.join(report.problems)}"
            )
    return out_dir


def spec_from_checkpoint(checkpoint: dict, *, version: str, task: str | None = None,
                         metrics: dict | None = None) -> ExportSpec:
    """Build an :class:`ExportSpec` from a trained checkpoint."""
    architecture = checkpoint.get("architecture", "crnn")
    is_ocr = architecture != "card-type-cnn"
    task = task or ("name-ocr" if is_ocr else "card-classifier")
    config = checkpoint.get("config") or {}
    return ExportSpec(
        task=task,
        version=version,
        architecture=architecture,
        input_height=int(config.get("input_height", 32)),
        input_size=int(checkpoint.get("input_size", 160)),
        charset=checkpoint.get("charset", ""),
        class_names=tuple(checkpoint.get("class_names") or ()),
        metrics=metrics or {},
        training_config=config,
        extra={"epoch": checkpoint.get("epoch"), "field": checkpoint.get("field")},
    )
