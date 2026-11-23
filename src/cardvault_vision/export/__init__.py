"""Model export, validation and the production registry (Phases 14-17).

    export_onnx(model, spec)        -> exports/<task>/<version>/model.onnx + metadata
    validate_export(dir, model)     -> APPROVED / REJECTED (torch-free checks + ORT)
    register_export(dir)            -> records the version in registry.json
    promote(task, version)          -> points production at it

CardVault consumes only the exported directory; it never imports this package.
"""

from __future__ import annotations

from .onnx_export import (
    OPSET_VERSION,
    ExportSpec,
    contract_spec,
    export_onnx,
    preprocessing_spec,
    spec_from_checkpoint,
)
from .registry import (
    ModelEntry,
    artifact_manifest,
    list_versions,
    load_registry,
    production_model,
    promote,
    register_export,
)
from .validate import ValidationReport, validate_all, validate_export

__all__ = [
    "ExportSpec",
    "ModelEntry",
    "OPSET_VERSION",
    "ValidationReport",
    "artifact_manifest",
    "contract_spec",
    "export_onnx",
    "list_versions",
    "load_registry",
    "preprocessing_spec",
    "production_model",
    "promote",
    "register_export",
    "spec_from_checkpoint",
    "validate_all",
    "validate_export",
]
