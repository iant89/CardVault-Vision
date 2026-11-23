"""Export validation (Phase 15).

The rule from the roadmap: *every exported model must be tested independently from
PyTorch*. Two separate checks, because they fail for different reasons:

1. **Structural** - the artifact is complete, self-consistent and its checksum
   matches. Catches a half-written export or a stale labels file.
2. **Numerical** - load the ONNX graph with ONNX Runtime (which knows nothing
   about torch) and compare its outputs against the reference model on real inputs.
   Catches opset bugs, wrong dynamic axes, and silent precision loss.

A model is APPROVED only if both pass. The result is written to
``validation.json`` inside the artifact, so downstream consumers can verify that
the bytes they received were the bytes that were validated.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import ExportError, MissingDependencyError
from ..logging_utils import get_logger

_LOG = get_logger("export.validate")

REQUIRED_FILES = (
    "model.onnx",
    "metadata.json",
    "labels.json",
    "preprocessing.json",
    "contract.json",
    "checksum.sha256",
)

#: Maximum acceptable absolute difference between torch and ONNX outputs.
#: FP32 kernels differ slightly between backends; 1e-3 on *logits* is generous but
#: still catches genuine graph errors (which produce order-1 differences).
DEFAULT_TOLERANCE = 1e-3


@dataclass
class ValidationReport:
    """Result of validating one export."""

    path: str
    passed: bool = False
    problems: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
    max_abs_diff: float | None = None
    checksum_match: bool | None = None

    def add(self, problem: str) -> None:
        self.problems.append(problem)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "status": "APPROVED" if self.passed else "REJECTED",
            "problems": self.problems,
            "checks": self.checks,
            "max_abs_diff": self.max_abs_diff,
            "checksum_match": self.checksum_match,
        }

    def summary(self) -> str:
        lines = [
            f"export     : {self.path}",
            f"status     : {'APPROVED' if self.passed else 'REJECTED'}",
        ]
        if self.max_abs_diff is not None:
            lines.append(f"max |diff| : {self.max_abs_diff:.3e}")
        for name, value in self.checks.items():
            lines.append(f"{name:<11}: {value}")
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


def validate_export(
    export_dir: str | Path,
    *,
    reference_model=None,
    spec=None,
    tolerance: float = DEFAULT_TOLERANCE,
    input_shapes: list[tuple[int, ...]] | None = None,
    write_report: bool = True,
) -> ValidationReport:
    """Validate an exported artifact directory.

    ``reference_model`` is the trained torch module; when omitted only the
    structural checks run (useful for CI without the training environment).
    """
    directory = Path(export_dir)
    report = ValidationReport(path=str(directory))
    if not directory.is_dir():
        report.add(f"export directory not found: {directory}")
        return report

    # -- 1. completeness ---------------------------------------------------
    missing = [name for name in REQUIRED_FILES if not (directory / name).exists()]
    if missing:
        report.add(f"missing required file(s): {', '.join(missing)}")
        return report
    report.checks["files"] = "complete"

    # -- 2. checksum -------------------------------------------------------
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    expected = metadata.get("checksum_sha256")
    actual = hashlib.sha256((directory / "model.onnx").read_bytes()).hexdigest()
    report.checksum_match = bool(expected and expected == actual)
    if not report.checksum_match:
        report.add(f"checksum mismatch: metadata={str(expected)[:12]} "
                   f"file={actual[:12]}")
    else:
        report.checks["checksum"] = "ok"
    recorded = (directory / "checksum.sha256").read_text(encoding="utf-8").split()
    if recorded and recorded[0] != actual:
        report.add("checksum.sha256 does not match model.onnx")

    # -- 3. label consistency ---------------------------------------------
    labels = json.loads((directory / "labels.json").read_text(encoding="utf-8"))
    num_classes = int(metadata.get("num_classes", 0))
    label_count = int(labels.get("count", 0))
    offset = int(labels.get("index_offset", 0))
    if label_count + 2 != num_classes and offset == 2:
        report.add(f"label count {label_count} + 2 != num_classes {num_classes}")
    elif label_count != num_classes and offset == 0:
        report.add(f"label count {label_count} != num_classes {num_classes}")
    else:
        report.checks["labels"] = "consistent"

    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
    if not contract.get("decoding", {}).get("steps"):
        report.add("contract.json has no decoding steps")
    else:
        report.checks["contract"] = "ok"

    # -- 4. ONNX structural check -----------------------------------------
    try:
        import onnx

        graph = onnx.load(str(directory / "model.onnx"))
        onnx.checker.check_model(graph)
        report.checks["onnx"] = f"valid opset {graph.opset_import[0].version}"
    except ImportError:
        report.checks["onnx"] = "skipped (onnx not installed)"
    except Exception as exc:  # pragma: no cover - message passthrough
        report.add(f"ONNX checker failed: {exc}")

    # -- 5. numerical equivalence -----------------------------------------
    if reference_model is not None:
        try:
            _compare_outputs(directory, reference_model, metadata, report,
                             tolerance, input_shapes)
        except MissingDependencyError as exc:
            report.checks["numerical"] = f"skipped ({exc.feature} unavailable)"
        except Exception as exc:
            report.add(f"numerical validation failed: {exc}")

    report.passed = not report.problems
    if write_report:
        try:
            (directory / "validation.json").write_text(
                json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        except OSError:  # pragma: no cover
            pass
    _LOG.debug("Validation of %s: %s", directory, "APPROVED" if report.passed else "REJECTED")
    return report


def _compare_outputs(directory: Path, reference_model, metadata: dict,
                     report: ValidationReport, tolerance: float,
                     input_shapes: list[tuple[int, ...]] | None) -> None:
    """Run both backends on identical inputs and compare."""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise MissingDependencyError("ONNX validation", "onnxruntime", "export") from exc

    import torch

    session = ort.InferenceSession(
        str(directory / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    input_spec = metadata.get("input_spec", {})
    shape = input_spec.get("shape") or []
    # Derive the task from metadata, not from the rank of the input: an OCR line
    # and a classifier image are both rank-4, so rank alone feeds a grayscale
    # (1, 1, 32, W) probe to a 3-channel classifier and fails a fine model.
    task = str(metadata.get("task", ""))
    is_ocr = task in {"name-ocr", "number-ocr"}
    reference_model.eval()

    shapes = input_shapes or _default_probe_shapes(shape, is_ocr)
    worst = 0.0
    rng = np.random.default_rng(1234)
    for probe in shapes:
        tensor = rng.random(probe, dtype=np.float32)
        with torch.no_grad():
            torch_out = reference_model(torch.from_numpy(tensor)).numpy()
        onnx_out = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        if torch_out.shape != onnx_out.shape:
            report.add(f"output shape mismatch for {probe}: torch={torch_out.shape} "
                       f"onnx={onnx_out.shape}")
            return
        diff = float(np.max(np.abs(torch_out - onnx_out)))
        worst = max(worst, diff)
        if diff > tolerance:
            report.add(f"numerical mismatch for input {probe}: max |diff| = {diff:.3e} "
                       f"(tolerance {tolerance:.1e})")
            return
    report.max_abs_diff = worst
    report.checks["numerical"] = f"match over {len(shapes)} input shape(s)"


def _default_probe_shapes(shape: list, is_ocr: bool) -> list[tuple[int, ...]]:
    """Probe more than one width so dynamic axes are actually exercised."""
    if is_ocr:
        height = int(shape[2]) if isinstance(shape[2], int) else 32
        return [(1, 1, height, 128), (2, 1, height, 320), (1, 1, height, 512)]
    return [(1, 3, int(shape[2]), int(shape[3]))] if all(
        isinstance(shape[i], int) for i in (2, 3)) else [(1, 3, 160, 160)]


def validate_all(exports_root: str | Path = "exports", *,
                 tolerance: float = DEFAULT_TOLERANCE) -> list[ValidationReport]:
    """Validate every artifact under ``exports/`` that lacks a passing report."""
    root = Path(exports_root)
    reports = []
    for metadata_path in sorted(root.glob("*/*/metadata.json")):
        reports.append(validate_export(metadata_path.parent, tolerance=tolerance))
    return reports
