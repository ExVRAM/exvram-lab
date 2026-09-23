"""Best-effort hardware detection. Missing CUDA is a supported environment state."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HardwareInfo:
    os: str
    python: str
    cuda_available: bool = False
    cuda_runtime_available: bool = False
    nvidia_smi_available: bool = False
    device_name: str | None = None
    vram_bytes: int | None = None
    compute_capability: str | None = None
    cuda_runtime: str | None = None
    torch_version: str | None = None
    source: str = "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def vram_gib(self) -> float | None:
        if self.vram_bytes is None:
            return None
        return self.vram_bytes / (1024**3)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["vram_gib"] = self.vram_gib
        return result


def _detect_with_torch(info: HardwareInfo) -> None:
    if importlib.util.find_spec("torch") is None:
        return
    try:
        torch = importlib.import_module("torch")
        info.torch_version = getattr(torch, "__version__", None)
        info.cuda_available = bool(torch.cuda.is_available())
        if not info.cuda_available:
            info.warnings.append("PyTorch is installed but torch.cuda.is_available() is false")
            return
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        info.device_name = str(props.name)
        info.vram_bytes = int(props.total_memory)
        info.compute_capability = f"{props.major}.{props.minor}"
        info.cuda_runtime_available = True
        info.cuda_runtime = getattr(torch.version, "cuda", None)
        info.source = "torch"
    except Exception as exc:  # optional dependency probing must not break the CLI
        info.warnings.append(f"PyTorch probe failed: {type(exc).__name__}: {exc}")


def _detect_with_nvidia_smi(info: HardwareInfo) -> None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return
    info.nvidia_smi_available = True
    if info.cuda_available:
        return
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        line = next((item.strip() for item in completed.stdout.splitlines() if item.strip()), "")
        if completed.returncode != 0 or not line:
            info.warnings.append("nvidia-smi was found but returned no GPU telemetry")
            return
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 3:
            info.device_name = fields[0]
            info.vram_bytes = int(float(fields[1]) * 1024**2)
            info.compute_capability = fields[2]
            info.cuda_available = True
            info.source = "nvidia-smi"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        info.warnings.append(f"nvidia-smi probe failed: {type(exc).__name__}: {exc}")


def detect_hardware() -> HardwareInfo:
    info = HardwareInfo(os=platform.platform(), python=platform.python_version())
    _detect_with_torch(info)
    _detect_with_nvidia_smi(info)
    if info.cuda_available and not info.cuda_runtime_available:
        info.warnings.append(
            "nvidia-smi sees a CUDA-capable GPU, but a CUDA-enabled Python runtime was not verified"
        )
    if not info.cuda_available:
        info.warnings.append(
            "CUDA GPU telemetry unavailable; CPU/synthetic workflows remain supported"
        )
    return info
