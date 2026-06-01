"""Shared helpers for heatsink inverse-design training and inference."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scipy.stats import boxcox

from AIInverseDesign.common.data_adapter import (
    CONDITION_KEYS,
    RECOMMEND_KEYS,
    ForwardDataset,
    InverseDataset,
    StandardScaler,
    build_forward_inputs,
    build_full_geometry_dict,
    build_inference_condition_tensor,
    build_inverse_condition_inputs,
    build_recommend_targets,
    build_threshold_augmented_training_tensors,
    build_threshold_condition_inputs,
    extract_heatsink_ids,
    load_json_samples,
    tensorize_target,
)
from AIInverseDesign.common.experiment_config import TEST_HEATSINKS
from AIInverseDesign.common.inverse_scoring import score_candidates as score_candidates_with_engineering
from AIInverseDesign.common.models import CVAE, ConditionBaselineMLP, DiffusionDenoiser, ForwardMLP

LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the default command-line logging format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@dataclass(frozen=True)
class SurrogateTrainConfig:
    """ForwardMLP 代理模型训练配置，集中管理训练超参数。"""

    device: torch.device
    batch_size: int
    epochs: int
    lr: float
    hidden_dim: int
    val_fraction: float = 0.15
    val_mode: str = "grouped-random"
    seed: int = 42
    condition_transform: str = "boxcox"
    boxcox_constant: float = 1.0
    scheduler_name: str = "onecycle"
    max_lr: float = 1e-2
    loss_name: str = "mse"
    best_metric_name: str = "rmse"
    huber_delta: float = 0.2
    architecture: str = "flat"
    residual_teacher: bool = False
    residual_loss_weight: float = 0.5
    residual_teacher_epochs: int = 40


@dataclass(frozen=True)
class CheckpointPayloadConfig:
    """逆向生成模型 checkpoint 的公共字段配置。"""

    method: str
    forward_model: ForwardMLP
    forward_input_scaler: StandardScaler
    target_scaler: StandardScaler
    cond_scaler: StandardScaler
    recommend_scaler: StandardScaler
    train_samples: list
    summary: Dict
    split: Dict | None = None


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
    parser.add_argument("--num-samples", "--num-generate", dest="num_samples", type=int, default=1024)
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


def split_sample_validation(samples: list, val_fraction: float, seed: int) -> tuple[list, list]:
    if not 0.0 < val_fraction < 1.0 or len(samples) < 2:
        return samples, []
    indices = list(range(len(samples)))
    rng = random.Random(seed + 10_000)
    rng.shuffle(indices)
    val_count = max(1, round(len(indices) * val_fraction))
    val_count = min(val_count, len(indices) - 1)
    val_indices = set(indices[:val_count])
    train_core = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val_samples = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train_core, val_samples


def split_validation_samples(
    samples: list,
    val_mode: str,
    val_fraction: float,
    seed: int,
) -> tuple[list, list, List[str]]:
    if val_mode == "sample-random":
        train_core, val_samples = split_sample_validation(samples, val_fraction, seed)
        return train_core, val_samples, sorted(set(extract_heatsink_ids(val_samples)))
    if val_mode == "grouped-random":
        train_core, val_samples, val_heatsinks = split_grouped_random(
            samples,
            test_fraction=val_fraction,
            split_seed=seed + 20_000,
        )
        return train_core, val_samples, val_heatsinks
    raise ValueError("--surrogate-val-mode must be one of: sample-random, grouped-random.")


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


class ForwardInputScaler:
    """Condition transform + StandardScaler pipeline for surrogate inputs."""

    def __init__(self, condition_transform: str = "none", boxcox_constant: float = 1.0) -> None:
        self.condition_transform = condition_transform
        self.boxcox_constant = boxcox_constant
        self.boxcox_lambdas: List[float] = []
        self.scaler = StandardScaler()

    def _transform_condition_fit(self, values: torch.Tensor) -> torch.Tensor:
        if self.condition_transform == "none":
            return values
        if self.condition_transform == "log1p":
            shifted = values + self.boxcox_constant
            if torch.any(shifted <= 0):
                raise ValueError("log1p condition transform requires positive shifted values.")
            return torch.log(shifted)
        if self.condition_transform != "boxcox":
            raise ValueError(f"Unknown condition transform: {self.condition_transform}")
        transformed = values.detach().cpu().numpy().copy()
        self.boxcox_lambdas = []
        for idx in range(transformed.shape[1]):
            shifted = transformed[:, idx] + self.boxcox_constant
            if np.any(shifted <= 0):
                raise ValueError(
                    "Box-Cox requires positive shifted condition values; "
                    f"condition_index={idx}, min={float(transformed[:, idx].min())}."
                )
            if float(np.max(shifted) - np.min(shifted)) <= 1e-12:
                lambda_value = 1.0
                transformed[:, idx] = shifted - 1.0
            else:
                transformed[:, idx], lambda_value = boxcox(shifted)
            self.boxcox_lambdas.append(float(lambda_value))
        return torch.tensor(transformed, dtype=torch.float32)

    def _transform_condition(self, values: torch.Tensor) -> torch.Tensor:
        if self.condition_transform == "none":
            return values
        if self.condition_transform == "log1p":
            shifted = values + self.boxcox_constant
            if torch.any(shifted <= 0):
                raise ValueError("log1p condition transform requires positive shifted values.")
            return torch.log(shifted)
        if self.condition_transform != "boxcox":
            raise ValueError(f"Unknown condition transform: {self.condition_transform}")
        if not self.boxcox_lambdas:
            raise ValueError("ForwardInputScaler has not been fitted.")

        transformed = values.detach().cpu().numpy().copy()
        for idx, lambda_value in enumerate(self.boxcox_lambdas):
            shifted = transformed[:, idx] + self.boxcox_constant
            if np.any(shifted <= 0):
                raise ValueError(
                    "Box-Cox requires positive shifted condition values; "
                    f"condition_index={idx}, min={float(transformed[:, idx].min())}."
                )
            if abs(lambda_value) <= 1e-12:
                transformed[:, idx] = np.log(shifted)
            else:
                transformed[:, idx] = (np.power(shifted, lambda_value) - 1.0) / lambda_value
        return torch.tensor(transformed, dtype=torch.float32, device=values.device)

    def transform_raw(self, values: torch.Tensor, fit: bool = False) -> torch.Tensor:
        cond = values[:, : len(CONDITION_KEYS)]
        rest = values[:, len(CONDITION_KEYS) :]
        cond_transformed = self._transform_condition_fit(cond) if fit else self._transform_condition(cond)
        cond_transformed = cond_transformed.to(values.device)
        return torch.cat([cond_transformed, rest], dim=1)

    def fit(self, values: torch.Tensor) -> "ForwardInputScaler":
        self.scaler.fit(self.transform_raw(values, fit=True))
        return self

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return self.scaler.transform(self.transform_raw(values, fit=False))

    def state_dict(self) -> Dict:
        return {
            "type": "forward_input_scaler_v2",
            "condition_transform": self.condition_transform,
            "boxcox_constant": self.boxcox_constant,
            "boxcox_lambdas": self.boxcox_lambdas,
            "scaler": self.scaler.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "ForwardInputScaler":
        if state.get("type") != "forward_input_scaler_v2":
            scaler = cls(condition_transform="none")
            scaler.scaler = StandardScaler.from_state_dict(state)
            return scaler
        scaler = cls(
            condition_transform=state.get("condition_transform", "none"),
            boxcox_constant=float(state.get("boxcox_constant", 1.0)),
        )
        scaler.boxcox_lambdas = [float(item) for item in state.get("boxcox_lambdas", [])]
        scaler.scaler = StandardScaler.from_state_dict(state["scaler"])
        return scaler


def train_forward_surrogate(
    train_samples: list,
    config: SurrogateTrainConfig,
) -> tuple[ForwardMLP, ForwardInputScaler, StandardScaler, Dict]:
    if config.loss_name not in {"mse", "huber"}:
        raise ValueError("--surrogate-loss must be one of: mse, huber.")
    if config.best_metric_name not in {"rmse", "mae"}:
        raise ValueError("--surrogate-best-metric must be one of: rmse, mae.")
    if config.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive.")
    if config.residual_loss_weight < 0.0:
        raise ValueError("--residual-loss-weight must be non-negative.")
    if config.residual_teacher and config.architecture not in {"residual", "two_branch_residual_concat"}:
        raise ValueError(
            "--residual-teacher requires --surrogate-architecture residual "
            "or two_branch_residual_concat."
        )
    if config.residual_teacher_epochs <= 0:
        raise ValueError("--residual-teacher-epochs must be positive.")

    train_core_samples, val_samples, val_heatsinks = split_validation_samples(
        train_samples,
        val_mode=config.val_mode,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    x_raw = build_forward_inputs(train_core_samples)
    y_raw = tensorize_target(train_core_samples)
    x_scaler = ForwardInputScaler(config.condition_transform, config.boxcox_constant).fit(x_raw)
    y_scaler = StandardScaler().fit(y_raw)
    ds = ForwardDataset(x_scaler.transform(x_raw), y_scaler.transform(y_raw))
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    teacher = None
    teacher_train_loss = None
    teacher_val_loss = None
    if config.residual_teacher:
        teacher, teacher_train_loss, teacher_val_loss = train_condition_baseline_teacher(
            train_loader=loader,
            val_samples=val_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            config=config,
        )

    val_loader = None
    if val_samples:
        val_x = x_scaler.transform(build_forward_inputs(val_samples))
        val_y = y_scaler.transform(tensorize_target(val_samples))
        val_loader = DataLoader(ForwardDataset(val_x, val_y), batch_size=config.batch_size, shuffle=False)

    model = ForwardMLP(
        in_dim=x_raw.shape[1],
        hidden_dim=config.hidden_dim,
        architecture=config.architecture,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = None
    if config.scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=max(1, len(loader)),
        )

    best_state = None
    best_val_metric = float("inf")
    best_val_scaled_loss = None
    best_val_real_metrics = None
    final_train_loss = 0.0
    final_train_temp_loss = 0.0
    final_train_residual_loss = None
    final_val_scaled_loss = None
    final_val_real_metrics = None
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        running_temp = 0.0
        running_residual = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            pred = model(x_batch)
            temp_loss = surrogate_scaled_loss(pred, y_batch, config.loss_name, config.huber_delta)
            loss = temp_loss
            residual_loss = None
            if teacher is not None:
                _, residual_pred = model.forward_parts(x_batch)
                with torch.no_grad():
                    baseline_teacher = teacher(x_batch[:, :8])
                    residual_target = y_batch - baseline_teacher
                residual_loss = surrogate_scaled_loss(
                    residual_pred,
                    residual_target,
                    config.loss_name,
                    config.huber_delta,
                )
                loss = loss + config.residual_loss_weight * residual_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
            running_temp += temp_loss.item()
            if residual_loss is not None:
                running_residual += residual_loss.item()
        final_train_loss = running / len(loader)
        final_train_temp_loss = running_temp / len(loader)
        final_train_residual_loss = running_residual / len(loader) if teacher is not None else None
        if val_loader is not None:
            val_loss = evaluate_scaled_loss(model, val_loader, config.device, config.loss_name, config.huber_delta)
            val_real_metrics = evaluate_real_temperature_metrics(model, val_loader, y_scaler, config.device)
            val_metric = val_real_metrics[config.best_metric_name]
            final_val_scaled_loss = val_loss
            final_val_real_metrics = val_real_metrics
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_val_scaled_loss = val_loss
                best_val_real_metrics = dict(val_real_metrics)
                best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 20 == 0:
            loss_label = "mse" if config.loss_name == "mse" else "huber"
            val_text = ""
            if val_loader is not None:
                val_text = (
                    f" val_{loss_label}={val_loss:.6f}"
                    f" val_mae={val_real_metrics['mae']:.6f}"
                    f" val_rmse={val_real_metrics['rmse']:.6f}"
                )
            LOGGER.info(
                "[ForwardMLP] epoch=%03d train_%s=%.6f%s",
                epoch + 1,
                loss_label,
                final_train_loss,
                val_text,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    train_info = {
        "surrogate_train_core_samples": len(train_core_samples),
        "surrogate_val_samples": len(val_samples),
        "surrogate_val_mode": config.val_mode,
        "surrogate_val_fraction": config.val_fraction if config.val_mode in {"sample-random", "grouped-random"} else None,
        "surrogate_val_heatsinks": val_heatsinks,
        "surrogate_train_core_heatsinks": sorted(set(extract_heatsink_ids(train_core_samples))),
        "surrogate_best_metric": config.best_metric_name,
        "best_val_metric": None if best_state is None else float(best_val_metric),
        "best_val_real_metrics": best_val_real_metrics,
        "final_val_real_metrics": final_val_real_metrics,
        "best_val_scaled_loss": best_val_scaled_loss,
        "final_val_scaled_loss": final_val_scaled_loss,
        "final_train_scaled_loss": float(final_train_loss),
        "best_val_mse": best_val_scaled_loss if config.loss_name == "mse" else None,
        "final_train_mse": None if config.loss_name != "mse" else float(final_train_loss),
        "condition_transform": config.condition_transform,
        "boxcox_constant": config.boxcox_constant,
        "surrogate_scheduler": config.scheduler_name,
        "surrogate_max_lr": config.max_lr if config.scheduler_name == "onecycle" else None,
        "surrogate_loss": config.loss_name,
        "huber_delta": config.huber_delta if config.loss_name == "huber" else None,
        "surrogate_architecture": config.architecture,
        "residual_teacher": config.residual_teacher,
        "residual_loss_weight": config.residual_loss_weight if config.residual_teacher else None,
        "residual_teacher_epochs": config.residual_teacher_epochs if config.residual_teacher else None,
        "residual_teacher_train_scaled_loss": teacher_train_loss,
        "residual_teacher_val_scaled_loss": teacher_val_loss,
        "final_train_temp_scaled_loss": float(final_train_temp_loss),
        "final_train_residual_scaled_loss": final_train_residual_loss,
    }
    return model, x_scaler, y_scaler, train_info


def train_condition_baseline_teacher(
    train_loader: DataLoader,
    val_samples: list,
    x_scaler: ForwardInputScaler,
    y_scaler: StandardScaler,
    config: SurrogateTrainConfig,
) -> tuple[ConditionBaselineMLP, float, float | None]:
    teacher = ConditionBaselineMLP(in_dim=8, hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=config.lr)
    scheduler = None
    if config.scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.max_lr,
            epochs=config.residual_teacher_epochs,
            steps_per_epoch=max(1, len(train_loader)),
        )

    val_loader = None
    if val_samples:
        val_x = x_scaler.transform(build_forward_inputs(val_samples))[:, :8]
        val_y = y_scaler.transform(tensorize_target(val_samples))
        val_loader = DataLoader(ForwardDataset(val_x, val_y), batch_size=config.batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    final_train_loss = 0.0
    final_val_loss = None
    for epoch in range(config.residual_teacher_epochs):
        teacher.train()
        running = 0.0
        for x_batch, y_batch in train_loader:
            context_batch = x_batch[:, :8].to(config.device)
            y_batch = y_batch.to(config.device)
            loss = surrogate_scaled_loss(teacher(context_batch), y_batch, config.loss_name, config.huber_delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
        final_train_loss = running / len(train_loader)
        if val_loader is not None:
            val_loss = evaluate_condition_baseline_scaled_loss(
                teacher,
                val_loader,
                config.device,
                config.loss_name,
                config.huber_delta,
            )
            final_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(teacher.state_dict())
        if (epoch + 1) % 20 == 0:
            val_text = "" if final_val_loss is None else f" val_loss={final_val_loss:.6f}"
            LOGGER.info(
                "[ConditionTeacher] epoch=%03d train_loss=%.6f%s",
                epoch + 1,
                final_train_loss,
                val_text,
            )
    if best_state is not None:
        teacher.load_state_dict(best_state)
    teacher.eval()
    return teacher, float(final_train_loss), None if final_val_loss is None else float(final_val_loss)


def evaluate_condition_baseline_scaled_loss(
    model: ConditionBaselineMLP,
    loader: DataLoader,
    device: torch.device,
    loss_name: str,
    huber_delta: float,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            total += surrogate_scaled_loss(model(x_batch), y_batch, loss_name, huber_delta).item()
    return total / max(1, len(loader))


def surrogate_scaled_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "huber":
        return F.huber_loss(pred, target, delta=huber_delta)
    raise ValueError(f"Unknown surrogate loss: {loss_name}")


def evaluate_scaled_loss(
    model: ForwardMLP,
    loader: DataLoader,
    device: torch.device,
    loss_name: str = "mse",
    huber_delta: float = 0.2,
) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            total += surrogate_scaled_loss(model(x_batch), y_batch, loss_name, huber_delta).item()
    return total / max(1, len(loader))


def evaluate_real_temperature_metrics(
    model: ForwardMLP,
    loader: DataLoader,
    y_scaler: StandardScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = y_scaler.inverse_transform(model(x_batch))
            target = y_scaler.inverse_transform(y_batch)
            preds.append(pred.detach().cpu())
            targets.append(target.detach().cpu())
    pred_all = torch.cat(preds, dim=0)
    target_all = torch.cat(targets, dim=0)
    diff = pred_all - target_all
    abs_diff = torch.abs(diff)
    mse = torch.mean(diff.pow(2))
    ss_tot = torch.sum((target_all - target_all.mean()).pow(2))
    r2 = 0.0 if float(ss_tot.item()) <= 1e-12 else 1.0 - float((torch.sum(diff.pow(2)) / ss_tot).item())
    return {
        "mae": float(torch.mean(abs_diff).item()),
        "rmse": float(torch.sqrt(mse).item()),
        "r2": r2,
        "p90_abs_error": float(torch.quantile(abs_diff.flatten(), 0.90).item()),
        "p95_abs_error": float(torch.quantile(abs_diff.flatten(), 0.95).item()),
        "max_abs_error": float(torch.max(abs_diff).item()),
    }


def surrogate_checkpoint_payload(
    forward_model: ForwardMLP,
    forward_input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    summary: Dict,
    split: Dict,
) -> Dict:
    return {
        "method": "forward-surrogate",
        "forward_model_state": forward_model.state_dict(),
        "forward_model_config": {
            "in_dim": forward_model.in_dim,
            "hidden_dim": forward_model.hidden_dim,
            "architecture": getattr(forward_model, "architecture", "flat"),
        },
        "scalers": {
            "forward_input_scaler": forward_input_scaler.state_dict(),
            "target_scaler": target_scaler.state_dict(),
        },
        "heatsink_split": split,
        "summary": summary,
    }


def save_surrogate_checkpoint(payload: Dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "surrogate.pt"
    torch.save(payload, path)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload.get("summary", {}), f, ensure_ascii=False, indent=2)
    return path


def load_surrogate_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ForwardMLP, ForwardInputScaler, StandardScaler, Dict]:
    payload = torch.load(Path(path), map_location=device)
    forward_model = ForwardMLP(**payload["forward_model_config"]).to(device)
    forward_model.load_state_dict(payload["forward_model_state"])
    forward_model.eval()
    forward_input_scaler = ForwardInputScaler.from_state_dict(payload["scalers"]["forward_input_scaler"])
    target_scaler = StandardScaler.from_state_dict(payload["scalers"]["target_scaler"])
    return forward_model, forward_input_scaler, target_scaler, payload


def get_forward_surrogate(
    args: argparse.Namespace,
    train_samples: list,
    test_samples: list,
    split: Dict,
    device: torch.device,
) -> tuple[ForwardMLP, StandardScaler, StandardScaler, Dict[str, float], Dict]:
    if getattr(args, "surrogate_checkpoint", ""):
        LOGGER.info("Loading ForwardMLP surrogate: %s", args.surrogate_checkpoint)
        forward_model, forward_input_scaler, target_scaler, payload = load_surrogate_checkpoint(
            args.surrogate_checkpoint,
            device,
        )
        forward_metrics = evaluate_forward_model(
            forward_model,
            forward_input_scaler,
            target_scaler,
            test_samples,
            device,
        )
        per_heatsink_metrics = evaluate_forward_model_by_heatsink(
            forward_model,
            forward_input_scaler,
            target_scaler,
            test_samples,
            device,
        )
        payload.setdefault("summary", {})
        payload["summary"]["forward_test_metrics"] = forward_metrics
        payload["summary"]["forward_test_metrics_by_heatsink"] = per_heatsink_metrics
        return forward_model, forward_input_scaler, target_scaler, forward_metrics, payload

    forward_model, forward_input_scaler, target_scaler, surrogate_train_info = train_forward_surrogate(
        train_samples=train_samples,
        config=SurrogateTrainConfig(
            device=device,
            batch_size=args.batch_size,
            epochs=args.surrogate_epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            val_fraction=args.surrogate_val_fraction,
            val_mode=args.surrogate_val_mode,
            seed=args.seed,
            condition_transform=args.condition_transform,
            boxcox_constant=args.boxcox_constant,
            scheduler_name=args.surrogate_scheduler,
            max_lr=args.surrogate_max_lr,
            loss_name=args.surrogate_loss,
            best_metric_name=args.surrogate_best_metric,
            huber_delta=args.huber_delta,
            architecture=args.surrogate_architecture,
            residual_teacher=args.residual_teacher,
            residual_loss_weight=args.residual_loss_weight,
            residual_teacher_epochs=args.residual_teacher_epochs,
        ),
    )
    forward_metrics = evaluate_forward_model(
        forward_model,
        forward_input_scaler,
        target_scaler,
        test_samples,
        device,
    )
    per_heatsink_metrics = evaluate_forward_model_by_heatsink(
        forward_model,
        forward_input_scaler,
        target_scaler,
        test_samples,
        device,
    )
    payload = surrogate_checkpoint_payload(
        forward_model,
        forward_input_scaler,
        target_scaler,
        {
            "method": "forward-surrogate",
            "data": args.data,
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "split": split,
            "forward_test_metrics": forward_metrics,
            "forward_test_metrics_by_heatsink": per_heatsink_metrics,
            "surrogate_epochs": args.surrogate_epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "surrogate_best_metric": args.surrogate_best_metric,
            **surrogate_train_info,
        },
        split,
    )
    return forward_model, forward_input_scaler, target_scaler, forward_metrics, payload


def predict_temperature_tensor(
    forward_model: ForwardMLP,
    forward_input_scaler,
    target_scaler: StandardScaler,
    raw_forward_inputs: torch.Tensor,
) -> torch.Tensor:
    scaled_x = forward_input_scaler.transform(raw_forward_inputs)
    scaled_pred = forward_model(scaled_x)
    return target_scaler.inverse_transform(scaled_pred)


def build_forward_input_from_parts(
    condition_raw: torch.Tensor,
    geom_raw: torch.Tensor,
) -> torch.Tensor:
    """Build raw surrogate inputs from raw inverse condition and raw geometry."""

    return torch.cat([condition_raw[:, :8], geom_raw], dim=1)


def evaluate_forward_model(
    model: ForwardMLP,
    x_scaler,
    y_scaler: StandardScaler,
    samples: list,
    device: torch.device,
) -> Dict[str, float]:
    if not samples:
        return {
            "sample_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "max_abs_error": 0.0,
            "p90_abs_error": 0.0,
            "p95_abs_error": 0.0,
            "pass_rate_1deg": 0.0,
            "pass_rate_2deg": 0.0,
            "pass_rate_5deg": 0.0,
        }
    x = build_forward_inputs(samples).to(device)
    y = tensorize_target(samples).to(device)
    model.eval()
    with torch.no_grad():
        pred = predict_temperature_tensor(model, x_scaler, y_scaler, x)
    diff = pred - y
    abs_diff = torch.abs(diff)
    mse = torch.mean(diff.pow(2))
    ss_tot = torch.sum((y - y.mean()).pow(2))
    r2 = 0.0 if float(ss_tot.item()) <= 1e-12 else 1.0 - float((torch.sum(diff.pow(2)) / ss_tot).item())
    return {
        "sample_count": len(samples),
        "mae": float(torch.mean(abs_diff).item()),
        "rmse": float(torch.sqrt(mse).item()),
        "r2": r2,
        "max_abs_error": float(torch.max(abs_diff).item()),
        "p90_abs_error": float(torch.quantile(abs_diff.flatten(), 0.90).item()),
        "p95_abs_error": float(torch.quantile(abs_diff.flatten(), 0.95).item()),
        "pass_rate_1deg": float(torch.mean((abs_diff <= 1.0).float()).item()),
        "pass_rate_2deg": float(torch.mean((abs_diff <= 2.0).float()).item()),
        "pass_rate_5deg": float(torch.mean((abs_diff <= 5.0).float()).item()),
    }


def evaluate_forward_model_by_heatsink(
    model: ForwardMLP,
    x_scaler,
    y_scaler: StandardScaler,
    samples: list,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, list] = {}
    for sample in samples:
        grouped.setdefault(str(sample["heatsink"]), []).append(sample)
    return {
        heatsink: evaluate_forward_model(model, x_scaler, y_scaler, group_samples, device)
        for heatsink, group_samples in sorted(grouped.items())
    }


def forward_residual_rows(
    model: ForwardMLP,
    x_scaler,
    y_scaler: StandardScaler,
    samples: list,
    device: torch.device,
) -> List[Dict]:
    if not samples:
        return []
    x = build_forward_inputs(samples).to(device)
    y = tensorize_target(samples).to(device)
    model.eval()
    with torch.no_grad():
        pred = predict_temperature_tensor(model, x_scaler, y_scaler, x)
    rows = []
    for idx, sample in enumerate(samples):
        true_temp = float(y[idx].detach().cpu().item())
        pred_temp = float(pred[idx].detach().cpu().item())
        condition = sample.get("condition", {})
        geometry = sample.get("geometry", {})
        derived = sample.get("derived", {})
        row = {
            "index": idx,
            "heatsink": str(sample.get("heatsink", "")),
            "true_temp": true_temp,
            "pred_temp": pred_temp,
            "error": pred_temp - true_temp,
            "abs_error": abs(pred_temp - true_temp),
        }
        row.update({f"condition_{key}": condition.get(key, "") for key in CONDITION_KEYS})
        row.update({f"geometry_{key}": geometry.get(key, "") for key in RECOMMEND_KEYS})
        row["bbox_base_width"] = geometry.get("base_width", "")
        row["bbox_base_depth"] = geometry.get("base_depth", "")
        row["bbox_total_height"] = derived.get("total_height", "")
        rows.append(row)
    return rows


def make_heatsink_split(train_samples: list, split: Dict | None = None) -> Dict:
    if split is not None:
        return split
    return {"train_heatsinks": sorted(set(extract_heatsink_ids(train_samples))), "test_heatsinks": []}


def save_checkpoint(payload: Dict, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "best_model.pt"
    torch.save(payload, path)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload.get("summary", {}), f, ensure_ascii=False, indent=2)
    return path


def base_checkpoint_payload(config: CheckpointPayloadConfig) -> Dict:
    return {
        "method": config.method,
        "forward_model_state": config.forward_model.state_dict(),
        "forward_model_config": {
            "in_dim": config.forward_model.in_dim,
            "hidden_dim": config.forward_model.hidden_dim,
            "architecture": getattr(config.forward_model, "architecture", "flat"),
        },
        "scalers": {
            "forward_input_scaler": config.forward_input_scaler.state_dict(),
            "target_scaler": config.target_scaler.state_dict(),
            "cond_scaler": config.cond_scaler.state_dict(),
            "recommend_scaler": config.recommend_scaler.state_dict(),
        },
        "heatsink_split": make_heatsink_split(config.train_samples, config.split),
        "summary": config.summary,
    }


def load_checkpoint(path: str | Path, device: torch.device, surrogate_checkpoint: str | Path = "") -> Dict:
    payload = torch.load(Path(path), map_location=device)
    forward_model = ForwardMLP(**payload["forward_model_config"]).to(device)
    forward_model.load_state_dict(payload["forward_model_state"])
    forward_model.eval()
    payload["forward_model"] = forward_model
    payload["forward_input_scaler"] = ForwardInputScaler.from_state_dict(payload["scalers"]["forward_input_scaler"])
    payload["target_scaler"] = StandardScaler.from_state_dict(payload["scalers"]["target_scaler"])
    payload["cond_scaler"] = StandardScaler.from_state_dict(payload["scalers"]["cond_scaler"])
    payload["recommend_scaler"] = StandardScaler.from_state_dict(payload["scalers"]["recommend_scaler"])
    if surrogate_checkpoint:
        surrogate_model, surrogate_input_scaler, surrogate_target_scaler, surrogate_payload = load_surrogate_checkpoint(
            surrogate_checkpoint,
            device,
        )
        payload["forward_model"] = surrogate_model
        payload["forward_input_scaler"] = surrogate_input_scaler
        payload["target_scaler"] = surrogate_target_scaler
        payload["surrogate_checkpoint_override"] = str(surrogate_checkpoint)
        payload["surrogate_checkpoint_summary"] = surrogate_payload.get("summary", {})
    return payload


def load_cvae_from_payload(payload: Dict, device: torch.device) -> CVAE:
    model = CVAE(**payload["cvae_model_config"]).to(device)
    model.load_state_dict(payload["cvae_model_state"])
    model.eval()
    return model


def load_diffusion_from_payload(payload: Dict, device: torch.device) -> DiffusionDenoiser:
    model = DiffusionDenoiser(**payload["diffusion_model_config"]).to(device)
    model.load_state_dict(payload["diffusion_model_state"])
    model.eval()
    return model


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


def score_candidates(
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    geom_rows: Iterable[Iterable[float]],
    temp_threshold: float,
    top_k: int,
    diversity_rerank_weight: float = 0.15,
    diversity_temp_tolerance: float = 2.0,
    engineering_variant_mode: str = "off",
    engineering_variant_count_per_candidate: int = 2,
    engineering_variant_max_trials: int = 20,
    engineering_variant_scale: float = 0.08,
    engineering_variant_required_temp_margin: float = 1.0,
    engineering_variant_min_unique_ratio: float = 0.8,
    engineering_variant_min_norm_mean_dist: float = 1.0,
    engineering_variant_min_norm_min_dist: float = 0.3,
) -> List[Dict[str, float]]:
    return score_candidates_with_engineering(
        payload=payload,
        condition=condition,
        bbox=bbox,
        geom_rows=geom_rows,
        temp_threshold=temp_threshold,
        top_k=top_k,
        diversity_rerank_weight=diversity_rerank_weight,
        diversity_temp_tolerance=diversity_temp_tolerance,
        engineering_variant_mode=engineering_variant_mode,
        engineering_variant_count_per_candidate=engineering_variant_count_per_candidate,
        engineering_variant_max_trials=engineering_variant_max_trials,
        engineering_variant_scale=engineering_variant_scale,
        engineering_variant_required_temp_margin=engineering_variant_required_temp_margin,
        engineering_variant_min_unique_ratio=engineering_variant_min_unique_ratio,
        engineering_variant_min_norm_mean_dist=engineering_variant_min_norm_mean_dist,
        engineering_variant_min_norm_min_dist=engineering_variant_min_norm_min_dist,
    )


def diversity_rerank_candidates(
    rows: List[Dict[str, float]],
    payload: Dict,
    top_k: int,
    diversity_rerank_weight: float = 0.15,
    diversity_temp_tolerance: float = 2.0,
) -> List[Dict[str, float]]:
    if top_k <= 0 or not rows:
        return []
    if diversity_rerank_weight <= 0.0 or top_k == 1:
        return rows[:top_k]

    ok_rows = [row for row in rows if row["threshold_ok"]]
    fallback_rows = [row for row in rows if not row["threshold_ok"]]
    selected = _select_diverse_group(
        ok_rows,
        payload,
        min(top_k, len(ok_rows)),
        diversity_rerank_weight,
        diversity_temp_tolerance,
    )
    if len(selected) < top_k:
        selected.extend(
            _select_diverse_group(
                fallback_rows,
                payload,
                top_k - len(selected),
                diversity_rerank_weight,
                diversity_temp_tolerance,
            )
        )
    return selected[:top_k]


def _select_diverse_group(
    rows: List[Dict[str, float]],
    payload: Dict,
    top_k: int,
    diversity_rerank_weight: float,
    diversity_temp_tolerance: float,
) -> List[Dict[str, float]]:
    if top_k <= 0 or not rows:
        return []
    if len(rows) <= top_k:
        return rows[:]

    vectors = _normalized_geometry_vectors(rows, payload)
    selected_indices = [0]
    remaining = set(range(1, len(rows)))
    best_temp = float(rows[0]["pred_cpu_temp"])
    temp_span = max(
        max(float(row["pred_cpu_temp"]) for row in rows) - best_temp,
        abs(float(diversity_temp_tolerance)),
        1e-6,
    )

    while len(selected_indices) < top_k and remaining:
        preferred = [
            idx
            for idx in remaining
            if diversity_temp_tolerance <= 0.0
            or float(rows[idx]["pred_cpu_temp"]) <= best_temp + diversity_temp_tolerance
        ]
        pool = preferred or list(remaining)
        best_idx = min(
            pool,
            key=lambda idx: _diverse_candidate_cost(
                rows,
                vectors,
                idx,
                selected_indices,
                best_temp,
                temp_span,
                diversity_rerank_weight,
            ),
        )
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [rows[idx] for idx in selected_indices]


def _normalized_geometry_vectors(rows: List[Dict[str, float]], payload: Dict) -> torch.Tensor:
    device = next(payload["forward_model"].parameters()).device
    raw = torch.tensor(
        [[float(row[name]) for name in RECOMMEND_KEYS] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    return payload["recommend_scaler"].transform(raw)


def _diverse_candidate_cost(
    rows: List[Dict[str, float]],
    vectors: torch.Tensor,
    idx: int,
    selected_indices: List[int],
    best_temp: float,
    temp_span: float,
    diversity_rerank_weight: float,
) -> float:
    temp_cost = (float(rows[idx]["pred_cpu_temp"]) - best_temp) / temp_span
    rank_cost = idx / max(len(rows) - 1, 1)
    selected_vectors = vectors[selected_indices]
    min_distance = torch.cdist(vectors[idx : idx + 1], selected_vectors).min().item()
    return temp_cost + 0.05 * rank_cost - diversity_rerank_weight * min_distance


def write_candidates(rows: List[Dict[str, float]], output_csv: str = "", output_json: str = "") -> None:
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    if output_csv:
        import csv

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank",
            "threshold_ok",
            "pred_cpu_temp",
            "temp_threshold",
            "base_width",
            "base_depth",
            "total_height",
            "base_height",
            "fin_height",
            "fin_thickness",
            "fin_clear_spacing",
            "fin_break_thickness",
            "fin_break_width",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})


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
