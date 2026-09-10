from __future__ import annotations

import argparse

from safety_training.baseline import create_m0_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the untouched M0 conversational checkpoint")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    args = parser.parse_args()
    print(create_m0_manifest(args.config, args.output_root, args.hf_home))


if __name__ == "__main__":
    main()

