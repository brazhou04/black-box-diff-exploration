from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


PACKAGES = ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes")


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "gpu_model": None,
        "gpu_count": 0,
        "total_vram_bytes": None,
        "cuda_version": None,
        "pytorch_cuda_version": None,
        "cuda_compute_capability": None,
        "fp16_available": False,
        "bf16_available": False,
    }
    try:
        import torch

        info["pytorch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            properties = torch.cuda.get_device_properties(0)
            info.update(
                gpu_model=properties.name,
                total_vram_bytes=properties.total_memory,
                cuda_compute_capability=f"{properties.major}.{properties.minor}",
                fp16_available=True,
                bf16_available=bool(torch.cuda.is_bf16_supported()),
            )
    except ImportError:
        pass
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if completed.returncode == 0:
            info["cuda_driver"] = completed.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["cuda_driver"] = None
    return info


def git_commit(repo_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def disk_info(path: str | Path) -> dict[str, int]:
    usage = shutil.disk_usage(Path(path))
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def assert_t4_feasible(config: dict[str, Any], smoke_test: bool = False) -> list[str]:
    warnings: list[str] = []
    hw = hardware_info()
    if hw["gpu_count"] == 0:
        if smoke_test:
            warnings.append("No CUDA GPU detected; smoke test will be slow or may be rejected by the trainer")
        else:
            raise RuntimeError("No CUDA GPU detected. Enable a Kaggle GPU accelerator before training.")
    vram = hw.get("total_vram_bytes")
    max_length = int(config["training"]["max_seq_length"])
    batch_size = int(config["training"]["per_device_batch_size"])
    rank = int(config["peft"]["r"])
    quantized = bool(config["quantization"].get("enabled") and config["quantization"].get("load_in_4bit"))
    if vram and vram <= 17 * 1024**3 and (not quantized or batch_size > 2 or max_length > 2048 or rank > 64):
        raise RuntimeError(
            "Requested configuration is risky for ~16 GB VRAM. Apply the same change to M1/M2/M3: "
            "enable 4-bit quantization, reduce per_device_batch_size, max_seq_length, or LoRA rank."
        )
    if hw.get("gpu_model") and "T4" in hw["gpu_model"] and config["quantization"].get("compute_dtype") == "bfloat16":
        raise RuntimeError("T4 does not support BF16 adequately; use quantization.compute_dtype=float16 consistently")
    return warnings

