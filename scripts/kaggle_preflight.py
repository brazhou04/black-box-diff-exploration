from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import REPO_ROOT, load_yaml, resolve_path
from safety_training.runtime import disk_info, hardware_info, package_versions


def _gib(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1024**3:.1f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle/T4 readiness diagnostics")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    args = parser.parse_args()
    config = load_yaml(args.config)
    hw = hardware_info()
    versions = package_versions()
    output_root = resolve_path(args.output_root or config["paths"]["output_root"])
    cache = Path(args.hf_home or config["paths"]["hf_home"])
    data_paths = [resolve_path(path) for path in config["evaluation"]["datasets"].values()]
    data_paths += [
        REPO_ROOT / "data" / "benign_control" / "train.jsonl",
        REPO_ROOT / "data" / "safety_shared" / "prompts.jsonl",
        REPO_ROOT / "data" / "safety_direct" / "train.jsonl",
        REPO_ROOT / "data" / "safety_constitutional" / "train.jsonl",
        REPO_ROOT / "data" / "safety_constitutional" / "generation_manifest.json",
    ]
    writable = False
    write_error = None
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_root, delete=True):
            writable = True
    except OSError as exc:
        write_error = str(exc)
    gpu = hw["gpu_model"] or "none"
    print(f"Detected GPU: {gpu}")
    print(f"GPU count: {hw['gpu_count']}")
    print(f"VRAM: {_gib(hw['total_vram_bytes'])}")
    print(f"CUDA compute capability: {hw['cuda_compute_capability']}")
    print(f"CUDA/driver: {hw.get('cuda_driver')}")
    print(f"PyTorch CUDA version: {hw['pytorch_cuda_version']}")
    print(f"FP16 availability: {hw['fp16_available']}")
    print(f"BF16 availability: {hw['bf16_available']}")
    print(f"bitsandbytes status: {'installed ' + versions['bitsandbytes'] if versions['bitsandbytes'] else 'not installed'}")
    for package in ("transformers", "peft", "trl"):
        print(f"{package} version: {versions[package] or 'not installed'}")
    print(f"Available disk: {_gib(disk_info(output_root)['free_bytes'])}" if writable else "Available disk: unavailable")
    available_data = sum(path.exists() for path in data_paths)
    print(f"Training/eval data availability: {available_data}/{len(data_paths)} required MVP files")
    for path in data_paths:
        if not path.exists():
            print(f"  missing: {path}")
    snapshots = cache / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots"
    print(f"Base model availability in shared cache: {'yes' if snapshots.exists() else 'not cached (will download once)'}")
    print(f"Output directory write access: {'yes' if writable else 'no: ' + str(write_error)}")
    quantization_ready = bool(versions["bitsandbytes"]) if config["quantization"].get("enabled") else True
    ready = bool(
        hw["gpu_count"]
        and hw["fp16_available"]
        and writable
        and versions["transformers"]
        and versions["peft"]
        and quantization_ready
        and available_data == len(data_paths)
    )
    if hw["total_vram_bytes"] and hw["total_vram_bytes"] < 14 * 1024**3:
        ready = False
        print("Memory diagnostic: less than 14 GiB; reduce batch size/sequence length/rank or use consistent 4-bit loading.")
    print("Target environment: single NVIDIA T4, approximately 16 GB VRAM")
    print(f"Status: {'READY' if ready else 'NOT READY'}")
    if not ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
