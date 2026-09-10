from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import REPO_ROOT
from safety_training.public_data import finalize_reviewed_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate human approvals and create experimental JSONL files")
    parser.add_argument("--review-dir", default="data/review")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = finalize_reviewed_candidates(args.review_dir, REPO_ROOT, args.overwrite)
    print(f"Approved data manifest: {path}")
    print("NEXT: generate M3 targets, freeze evaluation data, then validate experimental balance.")


if __name__ == "__main__":
    main()
