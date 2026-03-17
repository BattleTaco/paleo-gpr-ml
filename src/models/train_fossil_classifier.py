"""Training script for the fossil image classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fossil image classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fossil_classifier.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def train(config: dict) -> None:
    """Run the training loop defined by *config*.

    Parameters
    ----------
    config:
        Parsed YAML configuration dictionary.
    """
    print(
        f"[train_fossil_classifier] Starting experiment: {config['experiment']['name']}"
    )
    # TODO: implement data loading, model instantiation, and training loop.
    raise NotImplementedError("Training loop not yet implemented.")


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    train(config)


if __name__ == "__main__":
    main()
