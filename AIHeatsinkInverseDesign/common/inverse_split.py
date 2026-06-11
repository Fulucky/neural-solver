"""Dataset split helpers for inverse-design training and evaluation."""

from __future__ import annotations

import argparse
import logging
import random
from typing import Dict, List

from AIHeatsinkInverseDesign.common.data_adapter import extract_heatsink_ids, load_json_samples
from AIHeatsinkInverseDesign.common.experiment_config import TEST_HEATSINKS

LOGGER = logging.getLogger(__name__)


def split_fixed_heatsinks(samples: list) -> tuple[list, list]:
    test_set = {str(x) for x in TEST_HEATSINKS}
    train_samples = [s for s in samples if str(s["heatsink"]) not in test_set]
    test_samples = [s for s in samples if str(s["heatsink"]) in test_set]
    return train_samples, test_samples


def parse_heatsink_id_list(raw: str) -> List[str]:
    if not raw:
        return []
    normalized = raw.replace(",", " ").replace(";", " ").replace("\n", " ").replace("\t", " ")
    return [item.strip() for item in normalized.split(" ") if item.strip()]


def split_grouped_random(samples: list, test_fraction: float, split_seed: int) -> tuple[list, list, List[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")
    heatsinks = sorted(set(extract_heatsink_ids(samples)))
    rng = random.Random(split_seed)
    rng.shuffle(heatsinks)
    test_count = max(1, round(len(heatsinks) * test_fraction))
    test_count = min(test_count, max(1, len(heatsinks) - 1))
    test_ids = set(heatsinks[:test_count])
    train_samples = [sample for sample in samples if str(sample["heatsink"]) not in test_ids]
    test_samples = [sample for sample in samples if str(sample["heatsink"]) in test_ids]
    return train_samples, test_samples, sorted(test_ids)


def split_specified_heatsinks(samples: list, requested: List[str]) -> tuple[list, list, List[str]]:
    if not requested:
        raise ValueError("--test-heatsinks is required when --test-mode specified.")
    available = set(extract_heatsink_ids(samples))
    selected = sorted(str(item) for item in requested if str(item) in available)
    missing = sorted(set(str(item) for item in requested) - available)
    if missing:
        LOGGER.warning("Requested test heatsinks not found: %s", missing)
    if not selected:
        raise ValueError("No requested test heatsinks were found in the dataset.")
    test_ids = set(selected)
    train_samples = [sample for sample in samples if str(sample["heatsink"]) not in test_ids]
    test_samples = [sample for sample in samples if str(sample["heatsink"]) in test_ids]
    return train_samples, test_samples, selected


def load_and_split(args: argparse.Namespace) -> tuple[list, list, Dict]:
    samples = load_json_samples(args.data)
    if args.test_mode == "fixed-config":
        train_samples, test_samples = split_fixed_heatsinks(samples)
        test_heatsinks = sorted(str(x) for x in TEST_HEATSINKS)
    elif args.test_mode == "specified":
        train_samples, test_samples, test_heatsinks = split_specified_heatsinks(
            samples,
            parse_heatsink_id_list(args.test_heatsinks),
        )
    else:
        train_samples, test_samples, test_heatsinks = split_grouped_random(
            samples,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
        )
    if not train_samples:
        raise ValueError("No training samples remain after split.")
    if not test_samples:
        raise ValueError("No test samples selected.")
    split = {
        "test_mode": args.test_mode,
        "test_fraction": args.test_fraction if args.test_mode == "grouped-random" else None,
        "split_seed": args.split_seed if args.test_mode == "grouped-random" else None,
        "test_heatsinks": test_heatsinks,
        "train_heatsinks": sorted(set(extract_heatsink_ids(train_samples))),
    }
    return train_samples, test_samples, split


def make_heatsink_split(train_samples: list, split: Dict | None = None) -> Dict:
    if split is not None:
        return split
    return {"train_heatsinks": sorted(set(extract_heatsink_ids(train_samples))), "test_heatsinks": []}
