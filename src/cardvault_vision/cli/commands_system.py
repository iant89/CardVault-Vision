"""``vision doctor`` and ``vision layouts`` commands."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys

from ..console import format_table, heading, kv_block, subheading
from ..layouts import available_layouts, load_layouts, validate_layouts
from ..reproducibility import git_commit
from ..version import SCHEMA_VERSION, __version__
from .util import emit


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("doctor", help="check the environment and report capabilities")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_doctor)

    p = sub.add_parser("layouts", help="show card layouts and validate their ROIs")
    p.add_argument("--show", default=None, help="dump one layout")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_layouts)

    p = sub.add_parser("info", help="version, paths and a summary of the repo")
    p.add_argument("--json", action="store_true")
    p.set_defaults(handler=cmd_info)


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except ImportError:
        return None


def cmd_doctor(args: argparse.Namespace) -> int:
    from ..hardware import detect_hardware

    hardware = detect_hardware()
    optional = {name: _module_version(name) for name in
                ("torch", "onnx", "onnxruntime", "cv2", "psutil", "scipy")}
    required = {name: _module_version(name) for name in ("numpy", "PIL", "yaml")}
    missing_required = [name for name, version in required.items() if version is None]

    capabilities = {
        "synthetic generation": True,
        "dataset management": not missing_required,
        "training": optional.get("torch") is not None,
        "onnx export": optional.get("onnx") is not None,
        "export validation": optional.get("onnxruntime") is not None,
        "gpu training": hardware.cuda_available,
        "hard-example mining": True,
    }
    payload = {
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "hardware": hardware.to_dict(),
        "required": required,
        "optional": optional,
        "capabilities": capabilities,
        "missing_required": missing_required,
    }
    install_hints = []
    if optional.get("torch") is None:
        install_hints.append("pip install 'cardvault-vision[train]'    # training")
    if optional.get("onnx") is None or optional.get("onnxruntime") is None:
        install_hints.append("pip install 'cardvault-vision[export]'   # export + validation")

    human = "\n".join([
        heading("CardVault Vision doctor"),
        kv_block([
            ("version", f"{__version__} (schema {SCHEMA_VERSION})"),
            ("git commit", git_commit()),
            ("python", f"{sys.version.split()[0]} ({platform.machine()})"),
            ("cpu cores", hardware.cpu_count),
            ("ram", f"{hardware.total_ram_gb} GB" if hardware.total_ram_gb else "-"),
            ("cuda", "yes" if hardware.cuda_available else "no"),
            ("gpu", ", ".join(d.get("name", "?") for d in hardware.cuda_devices) or "-"),
            ("torch", optional.get("torch") or "NOT INSTALLED"),
            ("onnx", optional.get("onnx") or "NOT INSTALLED"),
            ("onnxruntime", optional.get("onnxruntime") or "NOT INSTALLED"),
        ]),
        subheading("Capabilities"),
        format_table(["capability", "available"],
                     [[name, "yes" if value else "no"] for name, value in
                      capabilities.items()]),
        *([subheading("To unlock more"), "\n".join(f"  {h}" for h in install_hints)]
          if install_hints else []),
        *([subheading("Problems"),
           "\n".join(f"  ! missing required module: {m}" for m in missing_required)]
          if missing_required else []),
    ])
    emit(args, payload, human)
    return 0 if not missing_required else 2


def cmd_layouts(args: argparse.Namespace) -> int:
    if args.show:
        layouts = load_layouts()
        key = args.show.upper()
        if key not in layouts:
            from ..errors import ConfigError

            raise ConfigError(f"Unknown layout {args.show!r}; available: {sorted(layouts)}")
        layout = layouts[key]
        if args.json:
            import json

            print(json.dumps(layout.to_dict(), indent=2))
        else:
            rows = [[name, f"{f.x0:.3f}, {f.y0:.3f} -> {f.x1:.3f}, {f.y1:.3f}",
                     f.group or "-", str(f.max_chars or "-")]
                    for name, f in layout.fields.items()]
            print(heading(f"Layout: {layout.card_type}"))
            print(format_table(["field", "normalised ROI", "group", "max chars"], rows))
        return 0

    layouts = load_layouts()
    problems = validate_layouts()
    rows = []
    for name in available_layouts():
        layout = layouts[name]
        rows.append([name, len(layout.fields), ", ".join(layout.required),
                     f"{layout.aspect:.3f}", layout.source.split("/")[-1]])
    payload = {"layouts": {name: layout.to_dict() for name, layout in layouts.items()},
               "problems": problems}
    human = "\n".join([
        heading("Card layouts"),
        format_table(["layout", "fields", "required", "aspect", "source"], rows),
        *([subheading("Problems"), "\n".join(f"  ! {p}" for p in problems)]
          if problems else ["", "  all layouts valid"]),
        "",
        "  override or add a layout in configs/cards/<name>.yaml",
    ])
    emit(args, payload, human)
    return 0 if not problems else 2


def cmd_info(args: argparse.Namespace) -> int:
    from ..config import CONFIG_DIR, REPO_ROOT

    payload = {
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "config_dir": str(CONFIG_DIR),
        "python": sys.version,
        "git_commit": git_commit(),
        "disk_free_gb": round(shutil.disk_usage(REPO_ROOT).free / 1024**3, 2),
    }
    human = "\n".join([
        heading("CardVault Vision"),
        kv_block([(key, value) for key, value in payload.items()
                  if key not in {"python"}]),
    ])
    return emit(args, payload, human)
