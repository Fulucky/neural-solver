"""Shared helpers for heatsink inverse-design training and inference."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from AIInverseDesign.common.data_adapter import (
    CONDITION_KEYS,
    RECOMMEND_KEYS,
    InverseDataset,
    StandardScaler,
    build_inference_condition_tensor,
    build_inverse_condition_inputs,
    build_recommend_targets,
    build_threshold_augmented_training_tensors,
    build_threshold_condition_inputs,
)
from AIInverseDesign.common.checkpoints import (
    CheckpointPayloadConfig,
    base_checkpoint_payload,
    load_checkpoint,
    load_cvae_from_payload,
    load_diffusion_from_payload,
    save_checkpoint,
)
from AIInverseDesign.common.inverse_scoring import (
    candidate_pool_summary,
    diversity_rerank_candidates,
    geometry_unique_count,
    normalized_geometry_metrics,
    raw_geometry_metrics,
    score_candidate_pool,
    score_candidates,
    score_geometry,
    select_candidates_from_pool,
    write_candidates,
    write_pool_summary,
)
from AIInverseDesign.common.inverse_split import (
    load_and_split,
    make_heatsink_split,
    parse_heatsink_id_list,
    split_fixed_heatsinks,
    split_grouped_random,
    split_specified_heatsinks,
)
from AIInverseDesign.common.surrogate import (
    ForwardInputScaler,
    SurrogateTrainConfig,
    build_forward_input_from_parts,
    evaluate_condition_baseline_scaled_loss,
    evaluate_forward_model,
    evaluate_forward_model_by_heatsink,
    evaluate_real_temperature_metrics,
    evaluate_scaled_loss,
    forward_residual_rows,
    get_forward_surrogate,
    load_surrogate_checkpoint,
    pairwise_temperature_difference_loss,
    predict_temperature_tensor,
    save_surrogate_checkpoint,
    split_sample_validation,
    split_validation_samples,
    surrogate_checkpoint_payload,
    surrogate_scaled_loss,
    train_condition_baseline_teacher,
    train_forward_surrogate,
)

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the default command-line logging format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=str, default="D:/AI_Heatsink_Generation/dataset/training_data_filtered.json")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--test-mode",
        choices=["grouped-random", "fixed-config", "specified"],
        default="grouped-random",
        help="How to choose held-out heatsinks for model-capacity evaluation.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--test-heatsinks",
        type=str,
        default="",
        help="Comma/space separated heatsink IDs used when --test-mode specified.",
    )
    parser.add_argument("--surrogate-epochs", type=int, default=80)
    parser.add_argument(
        "--surrogate-val-mode",
        choices=["sample-random", "grouped-random"],
        default="grouped-random",
        help="How to choose validation samples inside the surrogate training split.",
    )
    parser.add_argument("--surrogate-val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--condition-transform",
        choices=["none", "log1p", "boxcox"],
        default="boxcox",
        help="Transform the first five condition features before StandardScaler.",
    )
    parser.add_argument("--boxcox-constant", type=float, default=1.0)
    parser.add_argument(
        "--surrogate-scheduler",
        choices=["none", "onecycle"],
        default="onecycle",
        help="Learning-rate scheduler for the ForwardMLP surrogate.",
    )
    parser.add_argument("--surrogate-max-lr", type=float, default=1e-2)
    parser.add_argument(
        "--surrogate-loss",
        choices=["mse", "huber"],
        default="mse",
        help="Loss used to train the ForwardMLP surrogate in scaled-temperature space.",
    )
    parser.add_argument(
        "--surrogate-best-metric",
        choices=["rmse", "mae"],
        default="rmse",
        help="Real-temperature validation metric used to select the best ForwardMLP checkpoint.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=0.2,
        help="Huber delta in scaled-temperature units when --surrogate-loss huber.",
    )
    parser.add_argument(
        "--surrogate-architecture",
        choices=["flat", "two_branch_concat", "residual", "two_branch_residual_concat"],
        default="flat",
        help=(
            "Forward surrogate architecture. Use residual or two_branch_residual_concat "
            "with --residual-teacher for residual supervision."
        ),
    )
    parser.add_argument(
        "--residual-teacher",
        action="store_true",
        help="Train a condition+bbox-only teacher and supervise the residual branch.",
    )
    parser.add_argument(
        "--residual-loss-weight",
        type=float,
        default=0.5,
        help="Weight for residual branch loss when --residual-teacher is enabled.",
    )
    parser.add_argument(
        "--residual-teacher-epochs",
        type=int,
        default=40,
        help="Epochs for the condition+bbox-only teacher used by --residual-teacher.",
    )
    parser.add_argument(
        "--pairwise-loss-weight",
        type=float,
        default=0.0,
        help="Weight for near-condition pairwise temperature-difference loss. Disabled when 0.",
    )
    parser.add_argument(
        "--pairwise-condition-quantile",
        type=float,
        default=0.10,
        help="Condition-distance quantile used to select near-condition pairs within each batch.",
    )
    parser.add_argument(
        "--pairwise-max-pairs",
        type=int,
        default=4096,
        help="Maximum selected pairs per batch for pairwise loss.",
    )
    parser.add_argument(
        "--pairwise-geometry-epsilon",
        type=float,
        default=1e-6,
        help="Minimum standardized geometry distance for pairwise pairs.",
    )
    parser.add_argument(
        "--surrogate-checkpoint",
        type=str,
        default="",
        help="Optional ForwardMLP surrogate checkpoint to reuse instead of retraining.",
    )
    parser.add_argument(
        "--generator-train-all",
        action="store_true",
        help="Train inverse generators on train+test historical samples; surrogate test metrics still use the held-out split.",
    )
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--threshold-samples-per-layout",
        type=int,
        default=3,
        help="Threshold-CVAE augmentation rows per observed layout.",
    )
    parser.add_argument(
        "--threshold-upper-strategy",
        choices=["global_max", "heatsink_max"],
        default="global_max",
        help="Upper bound used when sampling augmented temp_threshold values.",
    )


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-path", "--checkpoint", dest="checkpoint_path", required=True)
    parser.add_argument(
        "--surrogate-checkpoint",
        type=str,
        default="",
        help="Optional ForwardMLP surrogate checkpoint used to override the generator checkpoint's embedded surrogate.",
    )
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--candidate-pool-size", dest="candidate_pool_size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--temp-threshold", "--temp-limit", dest="temp_threshold", type=float, required=True)
    parser.add_argument("--chip-length", type=float, required=True)
    parser.add_argument("--rjc", type=float, required=True)
    parser.add_argument("--rjb", type=float, required=True)
    parser.add_argument("--power", type=float, required=True)
    parser.add_argument("--wind-speed", type=float, required=True)
    parser.add_argument("--base-width", type=float, required=True)
    parser.add_argument("--base-depth", type=float, required=True)
    parser.add_argument("--total-height", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--diversity-rerank-weight",
        type=float,
        default=0.15,
        help="Weight for geometry diversity during Top-K reranking. Use 0 to keep pure temperature ranking.",
    )
    parser.add_argument(
        "--diversity-temp-tolerance",
        type=float,
        default=2.0,
        help="Preferred predicted-temperature window, in degC, for diversity reranking within each feasibility group.",
    )
    parser.add_argument("--pool-summary-json", type=str, default="")
    parser.add_argument("--engineering-variant-mode", choices=["off", "auto", "on"], default="off")
    parser.add_argument("--engineering-variant-count-per-candidate", type=int, default=2)
    parser.add_argument("--engineering-variant-max-trials", type=int, default=20)
    parser.add_argument("--engineering-variant-scale", type=float, default=0.08)
    parser.add_argument("--engineering-variant-required-temp-margin", type=float, default=1.0)
    parser.add_argument("--engineering-variant-min-unique-ratio", type=float, default=0.8)
    parser.add_argument("--engineering-variant-min-norm-mean-dist", type=float, default=1.0)
    parser.add_argument("--engineering-variant-min-norm-min-dist", type=float, default=0.3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def default_output_dir(method: str) -> Path:
    mapping = {
        "cvae": "outputs_thresholdfree_cvae/heatsink",
        "threshold-cvae": "outputs_guided_cvae/heatsink",
        "diffusion": "outputs_conditional_diffusion/heatsink",
    }
    return Path(mapping[method])


def request_from_args(args: argparse.Namespace) -> tuple[Dict[str, float], Dict[str, float], float]:
    condition = {
        "chip_length": args.chip_length,
        "Rjc": args.rjc,
        "Rjb": args.rjb,
        "power": args.power,
        "wind_speed": args.wind_speed,
    }
    bbox = {
        "base_width": args.base_width,
        "base_depth": args.base_depth,
        "total_height": args.total_height,
    }
    return condition, bbox, float(args.temp_threshold)


def build_train_tensors(train_samples: list, guided: bool, device: torch.device) -> tuple:
    cond_raw = (
        build_threshold_condition_inputs(train_samples)
        if guided
        else build_inverse_condition_inputs(train_samples)
    )
    geom_raw = build_recommend_targets(train_samples)
    cond_scaler = StandardScaler().fit(cond_raw)
    recommend_scaler = StandardScaler().fit(geom_raw)
    ds = InverseDataset(cond_scaler.transform(cond_raw), recommend_scaler.transform(geom_raw))
    return cond_raw.to(device), geom_raw.to(device), cond_scaler, recommend_scaler, ds


def build_guided_train_tensors(
    train_samples: list,
    device: torch.device,
    threshold_samples_per_layout: int,
    seed: int,
    threshold_upper_strategy: str,
) -> tuple:
    cond_raw, geom_raw, _observed_temp_raw, stats = build_threshold_augmented_training_tensors(
        train_samples,
        n_threshold_samples=threshold_samples_per_layout,
        seed=seed,
        upper_strategy=threshold_upper_strategy,
    )
    cond_scaler = StandardScaler().fit(cond_raw)
    recommend_scaler = StandardScaler().fit(geom_raw)
    ds = InverseDataset(cond_scaler.transform(cond_raw), recommend_scaler.transform(geom_raw))
    return cond_raw.to(device), geom_raw.to(device), cond_scaler, recommend_scaler, ds, stats


def make_inference_cond(
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_threshold: float,
    guided: bool,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    raw = build_inference_condition_tensor(
        condition,
        bbox,
        temp_limit=temp_threshold if guided else None,
    ).to(device)
    return payload["cond_scaler"].transform(raw).repeat(n, 1)
