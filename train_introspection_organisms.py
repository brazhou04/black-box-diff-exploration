from __future__ import annotations

import argparse

from safety_training.introspection_config import load_introspection_config
from safety_training.introspection_organisms import train_introspection_organisms


def main() -> None:
    parser = argparse.ArgumentParser(description="Train auxiliary behavior LoRAs used only to teach the IA")
    parser.add_argument("--config", default="configs/introspection_sft.yaml")
    parser.add_argument("--specs", default="data/introspection/organism_specs.jsonl")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--hf-home")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_introspection_config(args.config, args.seed)
    path = train_introspection_organisms(
        config,
        specs_path=args.specs,
        output_root=args.output_root,
        hf_home=args.hf_home,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(path)


if __name__ == "__main__":
    main()

