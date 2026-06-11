"""Checkpoint helpers for inverse-design generator models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from AIInverseDesign.common.data_adapter import StandardScaler
from AIInverseDesign.common.inverse_split import make_heatsink_split
from AIInverseDesign.common.models import CVAE, DiffusionDenoiser, ForwardMLP
from AIInverseDesign.common.surrogate import ForwardInputScaler, infer_forward_model_config, load_surrogate_checkpoint


@dataclass(frozen=True)
class CheckpointPayloadConfig:
    """Common fields stored in inverse generator checkpoints."""

    method: str
    forward_model: ForwardMLP
    forward_input_scaler: StandardScaler
    target_scaler: StandardScaler
    cond_scaler: StandardScaler
    recommend_scaler: StandardScaler
    train_samples: list
    summary: Dict
    split: Dict | None = None


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
    forward_model = ForwardMLP(**infer_forward_model_config(payload)).to(device)
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
