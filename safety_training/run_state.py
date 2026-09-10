from __future__ import annotations

from pathlib import Path
from typing import Literal


RunStatus = Literal["COMPLETE", "PARTIAL", "NOT_STARTED"]


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    root = Path(run_dir)
    checkpoints = []
    for path in root.glob("checkpoints/checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    return max(checkpoints, default=(0, None))[1]


def run_status(run_dir: str | Path) -> RunStatus:
    root = Path(run_dir)
    if (root / "manifest.json").exists() and (root / "adapter").is_dir():
        return "COMPLETE"
    if latest_checkpoint(root) is not None:
        return "PARTIAL"
    return "NOT_STARTED"

