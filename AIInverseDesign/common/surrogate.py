"""Forward temperature surrogate training, evaluation, and checkpoint helpers."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import boxcox
from torch.utils.data import DataLoader

from AIInverseDesign.common.data_adapter import (
    CONDITION_KEYS,
    RECOMMEND_KEYS,
    ForwardDataset,
    StandardScaler,
    build_forward_inputs,
    extract_heatsink_ids,
    tensorize_target,
)
from AIInverseDesign.common.models import ConditionBaselineMLP, ForwardMLP

LOGGER = logging.getLogger(__name__)


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
    pairwise_loss_weight: float = 0.0
    pairwise_condition_quantile: float = 0.10
    pairwise_max_pairs: int = 4096
    pairwise_geometry_epsilon: float = 1e-6


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


def split_grouped_validation(samples: list, val_fraction: float, seed: int) -> tuple[list, list, List[str]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--surrogate-val-fraction must be between 0 and 1.")
    heatsinks = sorted(set(extract_heatsink_ids(samples)))
    rng = random.Random(seed + 20_000)
    rng.shuffle(heatsinks)
    val_count = max(1, round(len(heatsinks) * val_fraction))
    val_count = min(val_count, max(1, len(heatsinks) - 1))
    val_ids = set(heatsinks[:val_count])
    train_core = [sample for sample in samples if str(sample["heatsink"]) not in val_ids]
    val_samples = [sample for sample in samples if str(sample["heatsink"]) in val_ids]
    return train_core, val_samples, sorted(val_ids)


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
        return split_grouped_validation(samples, val_fraction, seed)
    raise ValueError("--surrogate-val-mode must be one of: sample-random, grouped-random.")


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
    if config.pairwise_loss_weight < 0.0:
        raise ValueError("--pairwise-loss-weight must be non-negative.")
    if not 0.0 < config.pairwise_condition_quantile <= 1.0:
        raise ValueError("--pairwise-condition-quantile must be in (0, 1].")
    if config.pairwise_max_pairs <= 0:
        raise ValueError("--pairwise-max-pairs must be positive.")
    if config.pairwise_geometry_epsilon < 0.0:
        raise ValueError("--pairwise-geometry-epsilon must be non-negative.")

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
    final_train_pairwise_loss = None
    final_val_scaled_loss = None
    final_val_real_metrics = None
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        running_temp = 0.0
        running_residual = 0.0
        running_pairwise = 0.0
        pairwise_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(config.device)
            y_batch = y_batch.to(config.device)
            pred = model(x_batch)
            temp_loss = surrogate_scaled_loss(pred, y_batch, config.loss_name, config.huber_delta)
            loss = temp_loss
            residual_loss = None
            pairwise_loss = None
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
            if config.pairwise_loss_weight > 0.0:
                pairwise_loss = pairwise_temperature_difference_loss(
                    pred=pred,
                    target=y_batch,
                    x_batch=x_batch,
                    loss_name=config.loss_name,
                    huber_delta=config.huber_delta,
                    condition_quantile=config.pairwise_condition_quantile,
                    max_pairs=config.pairwise_max_pairs,
                    geometry_epsilon=config.pairwise_geometry_epsilon,
                )
                if pairwise_loss is not None:
                    loss = loss + config.pairwise_loss_weight * pairwise_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
            running_temp += temp_loss.item()
            if residual_loss is not None:
                running_residual += residual_loss.item()
            if pairwise_loss is not None:
                running_pairwise += pairwise_loss.item()
                pairwise_batches += 1
        final_train_loss = running / len(loader)
        final_train_temp_loss = running_temp / len(loader)
        final_train_residual_loss = running_residual / len(loader) if teacher is not None else None
        final_train_pairwise_loss = running_pairwise / pairwise_batches if pairwise_batches else None
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
        "pairwise_loss_weight": config.pairwise_loss_weight,
        "pairwise_condition_quantile": config.pairwise_condition_quantile if config.pairwise_loss_weight > 0 else None,
        "pairwise_max_pairs": config.pairwise_max_pairs if config.pairwise_loss_weight > 0 else None,
        "pairwise_geometry_epsilon": config.pairwise_geometry_epsilon if config.pairwise_loss_weight > 0 else None,
        "final_train_pairwise_scaled_loss": final_train_pairwise_loss,
    }
    return model, x_scaler, y_scaler, train_info


def pairwise_temperature_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    x_batch: torch.Tensor,
    loss_name: str,
    huber_delta: float,
    condition_quantile: float,
    max_pairs: int,
    geometry_epsilon: float,
) -> torch.Tensor | None:
    if x_batch.shape[0] < 2:
        return None

    condition = x_batch[:, : len(CONDITION_KEYS)]
    geometry = x_batch[:, 8:]
    condition_dist = torch.cdist(condition, condition, p=2)
    geometry_dist = torch.cdist(geometry, geometry, p=2)
    upper_mask = torch.triu(torch.ones_like(condition_dist, dtype=torch.bool), diagonal=1)
    candidate_mask = upper_mask & (geometry_dist > geometry_epsilon)
    if not torch.any(candidate_mask):
        return None

    candidate_dist = condition_dist[candidate_mask]
    threshold = torch.quantile(candidate_dist, condition_quantile)
    pair_mask = candidate_mask & (condition_dist <= threshold)
    pair_indices = torch.nonzero(pair_mask, as_tuple=False)
    if pair_indices.numel() == 0:
        return None

    if pair_indices.shape[0] > max_pairs:
        selected = torch.randperm(pair_indices.shape[0], device=pair_indices.device)[:max_pairs]
        pair_indices = pair_indices[selected]

    left = pair_indices[:, 0]
    right = pair_indices[:, 1]
    pred_delta = pred[left] - pred[right]
    target_delta = target[left] - target[right]
    return surrogate_scaled_loss(pred_delta, target_delta, loss_name, huber_delta)


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


def infer_forward_model_config(payload: Dict) -> Dict:
    """Return ForwardMLP config, falling back to state_dict shape/key inference."""

    config = dict(payload.get("forward_model_config") or {})
    inferred = infer_forward_model_config_from_state(payload.get("forward_model_state") or {})
    if not config:
        return inferred

    if config.get("architecture") != inferred.get("architecture"):
        LOGGER.warning(
            "ForwardMLP checkpoint config architecture=%s does not match state_dict architecture=%s; "
            "using inferred architecture.",
            config.get("architecture"),
            inferred.get("architecture"),
        )
        return {**config, **inferred}

    return {
        "in_dim": int(config.get("in_dim", inferred.get("in_dim", 13))),
        "hidden_dim": int(config.get("hidden_dim", inferred.get("hidden_dim", 256))),
        "architecture": config.get("architecture", inferred.get("architecture", "flat")),
    }


def infer_forward_model_config_from_state(state: Dict) -> Dict:
    if "context_gate.0.weight" in state:
        architecture = "residual"
    elif "baseline_head.0.weight" in state and "residual_head.0.weight" in state:
        architecture = "two_branch_residual_concat"
    elif "context_encoder.0.weight" in state and "geom_encoder.0.weight" in state:
        architecture = "two_branch_concat"
    else:
        architecture = "flat"

    if architecture == "flat":
        first_weight = state.get("net.0.weight")
        hidden_dim = int(first_weight.shape[0]) if first_weight is not None else 256
        in_dim = int(first_weight.shape[1]) if first_weight is not None else 13
    else:
        first_weight = state.get("context_encoder.0.weight")
        hidden_dim = int(first_weight.shape[0]) if first_weight is not None else 256
        in_dim = 13

    return {"in_dim": in_dim, "hidden_dim": hidden_dim, "architecture": architecture}


def load_surrogate_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[ForwardMLP, ForwardInputScaler, StandardScaler, Dict]:
    payload = torch.load(Path(path), map_location=device)
    forward_model = ForwardMLP(**infer_forward_model_config(payload)).to(device)
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
            pairwise_loss_weight=args.pairwise_loss_weight,
            pairwise_condition_quantile=args.pairwise_condition_quantile,
            pairwise_max_pairs=args.pairwise_max_pairs,
            pairwise_geometry_epsilon=args.pairwise_geometry_epsilon,
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
