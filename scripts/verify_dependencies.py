from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from packaging.specifiers import SpecifierSet
from packaging.version import Version


REQUIRED = {
    "transformers": "==4.57.3",
    "huggingface-hub": ">=0.34.0,<1.0",
    "datasets": ">=3.2,<4",
    "accelerate": ">=1.4,<2",
    "peft": "==0.17.1",
    "trl": "==0.22.2",
    "bitsandbytes": ">=0.45,<0.49",
}


def main() -> None:
    failures: list[str] = []
    for distribution, constraint in REQUIRED.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{distribution}: not installed")
            continue
        print(f"{distribution}=={version} (required {constraint})")
        if Version(version) not in SpecifierSet(constraint):
            failures.append(f"{distribution}=={version} violates {constraint}")
    probes = {
        "transformers": ("AutoTokenizer",),
        "huggingface_hub": ("HfApi",),
        "peft": ("LoraConfig",),
        "trl": ("DPOConfig", "DPOTrainer"),
    }
    for module_name, names in probes.items():
        try:
            module = importlib.import_module(module_name)
            for name in names:
                getattr(module, name)
        except Exception as exc:
            failures.append(f"{module_name} import probe failed: {type(exc).__name__}: {exc}")
    if failures:
        print("DEPENDENCIES BROKEN")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("DEPENDENCIES READY")


if __name__ == "__main__":
    main()
