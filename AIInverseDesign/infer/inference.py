"""Inference utilities for the heatsink inverse-design pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from AIInverseDesign.common.data_adapter import (
    CONDITION_KEYS,
    RECOMMEND_KEYS,
    StandardScaler,
    build_full_geometry_dict,
)
from AIInverseDesign.common.models import CVAE, ForwardMLP


@dataclass
class TrainingArtifacts:
    """Serialized artifacts needed for inference."""

    forward_model: ForwardMLP
    cvae_model: CVAE
    forward_input_scaler: StandardScaler
    cond_scaler: StandardScaler
    recommend_scaler: StandardScaler
    target_scaler: StandardScaler
    heatsink_split: Dict


def load_artifacts(checkpoint_path: str | Path) -> TrainingArtifacts:
    """Load trained artifacts from a checkpoint."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(Path(checkpoint_path), map_location=device)

    forward_model = ForwardMLP(**payload["forward_model_config"])
    forward_model.load_state_dict(payload["forward_model_state"])
    forward_model.eval()
    forward_model = forward_model.to(device)

    cvae_model = CVAE(**payload["cvae_model_config"])
    cvae_model.load_state_dict(payload["cvae_model_state"])
    cvae_model.eval()
    cvae_model = cvae_model.to(device)

    return TrainingArtifacts(
        forward_model=forward_model,
        cvae_model=cvae_model,
        forward_input_scaler=StandardScaler.from_state_dict(payload["scalers"]["forward_input_scaler"]),
        cond_scaler=StandardScaler.from_state_dict(payload["scalers"]["cond_scaler"]),
        recommend_scaler=StandardScaler.from_state_dict(payload["scalers"]["recommend_scaler"]),
        target_scaler=StandardScaler.from_state_dict(payload["scalers"]["target_scaler"]),
        heatsink_split=payload.get("heatsink_split", {}),
    )


def predict_temperature(
    artifacts: TrainingArtifacts,
    condition: Dict[str, float],
    geometry: Dict[str, float],
) -> float:
    """
    Predict temperature using the forward model.
    Input: [condition(5) + bbox(3) + geometry(5)]
    """

    device = next(artifacts.forward_model.parameters()).device
    x = torch.tensor(
        [[
            *(float(condition[k]) for k in CONDITION_KEYS),
            float(geometry["base_width"]),
            float(geometry["base_depth"]),
            float(geometry["total_height"]),
            float(geometry["fin_height"]),
            float(geometry["fin_thickness"]),
            float(geometry["fin_clear_spacing"]),
            float(geometry["fin_break_thickness"]),
            float(geometry["fin_break_width"]),
        ]],
        dtype=torch.float32,
        device=device,
    )

    x_scaled = artifacts.forward_input_scaler.transform(x)
    with torch.no_grad():
        pred_scaled = artifacts.forward_model(x_scaled)
        pred = artifacts.target_scaler.inverse_transform(pred_scaled)

    return float(pred.cpu().item())


def generate_candidates(
    artifacts: TrainingArtifacts,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_limit: float,
    candidate_pool_size: int = 512,
) -> List[Dict[str, float]]:
    """
    Generate candidate geometries from CVAE.

    Note:
    - CVAE condition is [condition(5) + bbox(3)]
    - temp_limit is only used for downstream filtering/sorting
    """

    device = next(artifacts.cvae_model.parameters()).device
    cond_raw = torch.tensor(
        [[
            *(float(condition[k]) for k in CONDITION_KEYS),
            float(bbox["base_width"]),
            float(bbox["base_depth"]),
            float(bbox["total_height"]),
        ]],
        dtype=torch.float32,
        device=device,
    )

    cond_scaled = artifacts.cond_scaler.transform(cond_raw).repeat(candidate_pool_size, 1)
    z = torch.randn(candidate_pool_size, artifacts.cvae_model.latent_dim, device=device)

    with torch.no_grad():
        pred_scaled = artifacts.cvae_model.decode(cond_scaled, z)
        pred = artifacts.recommend_scaler.inverse_transform(pred_scaled).cpu()

    candidates = []
    for row in pred:
        geom = build_full_geometry_dict(bbox, row.tolist())
        geom["pred_cpu_temp"] = predict_temperature(artifacts, condition, geom)
        candidates.append(geom)

    return candidates


def unique_and_sort_candidates(
    candidates: List[Dict[str, float]],
    temp_limit: float,
    top_k: int = 20,
    round_digits: int = 3,
) -> List[Dict[str, float]]:
    """Filter by temp_limit, deduplicate, and sort candidates."""

    feasible = [c for c in candidates if c["pred_cpu_temp"] <= temp_limit]

    seen = set()
    deduped = []
    for candidate in feasible:
        key = tuple(round(candidate[name], round_digits) for name in RECOMMEND_KEYS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    deduped.sort(
        key=lambda x: (
            x["pred_cpu_temp"],
            x["fin_height"],
            x["fin_thickness"],
        )
    )

    return deduped[:top_k]


def infer_designs(
    artifacts: TrainingArtifacts,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_limit: float,
    candidate_pool_size: int = 1024,
    top_k: int = 20,
) -> List[Dict[str, float]]:
    """Run inverse-design inference and return top candidate geometries."""

    candidates = generate_candidates(
        artifacts=artifacts,
        condition=condition,
        bbox=bbox,
        temp_limit=temp_limit,
        candidate_pool_size=candidate_pool_size,
    )
    return unique_and_sort_candidates(candidates, temp_limit=temp_limit, top_k=top_k)
