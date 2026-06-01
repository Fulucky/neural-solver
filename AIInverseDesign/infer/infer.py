"""Unified inference entry point for heatsink inverse-design models."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


METHODS = {
    "cvae": ("AIInverseDesign.infer.cvae_inferencer", "Threshold-free CVAE with latent optimization."),
    "threshold-cvae": ("AIInverseDesign.infer.guided_cvae_inferencer", "Threshold-conditioned CVAE conditioned on temp_threshold."),
    "diffusion": ("AIInverseDesign.infer.diffusion_inferencer", "Conditional diffusion with surrogate guidance."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified inferencer. Pick one heatsink inverse-design path with --method."
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS.keys()))
    parser.add_argument("inferencer_args", nargs=argparse.REMAINDER)
    return parser


def main(argv=None) -> None:
    configure_logging()
    args = build_parser().parse_args(argv)
    module_name, description = METHODS[args.method]
    inferencer_args = args.inferencer_args
    if inferencer_args and inferencer_args[0] == "--":
        inferencer_args = inferencer_args[1:]

    LOGGER.info("Selected inference method: %s", args.method)
    LOGGER.info("Dispatching to: %s (%s)", module_name, description)

    module = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    try:
        sys.argv = [f"{module_name}.py", *inferencer_args]
        module.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
