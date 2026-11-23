"""Hardware detection.

Answers the questions the training/eval/benchmark code needs to ask *before*
importing torch: is there a CUDA device, how many CPU cores may I use, how much
memory is there, and what is the machine actually called (for benchmark
provenance).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class HardwareInfo:
    """A JSON-serialisable snapshot of the machine."""

    python: str
    platform: str
    machine: str
    cpu_count: int
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_devices: list[dict] = field(default_factory=list)
    torch: str | None = None
    torch_version: str | None = None
    total_ram_gb: float | None = None
    disk_free_gb: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def device(self) -> str:
        """Conventional default device string (``cuda`` when available)."""
        return "cuda" if self.cuda_available else "cpu"


def _total_ram_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
    except OSError:
        pass
    try:  # pragma: no cover - non-Linux fallback
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        return None
    return None


def _disk_free_gb(path: str = ".") -> float | None:
    try:
        return round(shutil.disk_usage(path).free / 1024**3, 2)
    except OSError:
        return None


def _nvidia_smi_devices() -> list[dict]:
    """Best-effort GPU enumeration without torch (used for provenance)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    devices = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                vram = float(parts[1]) / 1024
            except ValueError:
                vram = None
            devices.append({"name": parts[0], "vram_gb": round(vram, 1) if vram else None,
                            "driver": parts[2] if len(parts) > 2 else None})
    return devices


def detect_hardware(probe_torch: bool = True) -> HardwareInfo:
    """Inspect the machine, importing torch only when it is installed."""
    info = HardwareInfo(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count() or 1,
        total_ram_gb=_total_ram_gb(),
        disk_free_gb=_disk_free_gb(),
    )
    try:
        import torch

        info.torch = "pytorch"
        info.torch_version = torch.__version__
        info.cuda_available = bool(torch.cuda.is_available())
        if info.cuda_available:
            info.cuda_device_count = torch.cuda.device_count()
            for i in range(info.cuda_device_count):
                props = torch.cuda.get_device_properties(i)
                info.cuda_devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "vram_gb": round(props.total_memory / 1024**3, 1),
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
    except Exception:
        # torch absent (or broken CUDA build): fall back to nvidia-smi only.
        devices = _nvidia_smi_devices()
        info.cuda_devices = devices
        info.cuda_device_count = len(devices)
        info.cuda_available = bool(devices) and probe_torch is False
    return info


def resolve_device(requested: str | None = None) -> str:
    """Map a user request (``auto``/``cpu``/``cuda``/``cuda:1``) to a device.

    We never silently downgrade an explicit ``cuda`` request to CPU: a training
    run that thinks it is on a GPU but is not wastes hours.
    """
    from .errors import ConfigError

    hw = detect_hardware()
    requested = (requested or "auto").lower()
    if requested == "auto":
        return hw.device
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not hw.cuda_available:
            raise ConfigError(
                "CUDA was requested but no CUDA device is available. "
                "Use --device cpu or install a CUDA-enabled PyTorch build."
            )
        if ":" in requested:
            idx = requested.split(":", 1)[1]
            if not idx.isdigit() or int(idx) >= hw.cuda_device_count:
                raise ConfigError(f"Invalid CUDA device index in {requested!r}")
            if not 0 <= int(idx) < hw.cuda_device_count:
                raise ConfigError(f"{requested} not present; found {hw.cuda_device_count} device(s)")
        return requested
    raise ConfigError(f"Unknown device {requested!r}; expected auto, cpu, cuda or cuda:N")
