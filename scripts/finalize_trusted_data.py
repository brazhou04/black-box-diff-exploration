from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.config import REPO_ROOT
from safety_training.public_data import finalize_trusted_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create binary study data by trusting documented public-source labels without human review"
    )
    parser.add_argument("--candidate-dir", default="data/review")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = finalize_trusted_candidates(args.candidate_dir, REPO_ROOT, args.overwrite)
    print(f"Source-trust data manifest: {path}")
    print("Human review: NOT PERFORMED")
    print("Taxonomy: binary safe/unsafe; no dual-use category or dual-use claims")
    print("Caveat: ambiguous examples are not independently identified without review")
    print("NEXT: generate M3 targets, freeze evaluation data, then validate experimental balance.")


if __name__ == "__main__":
    main()
