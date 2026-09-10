from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_training.public_data import review_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize public-data candidate review progress")
    parser.add_argument("--review-dir", default="data/review")
    args = parser.parse_args()
    print(json.dumps(review_summary(args.review_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
