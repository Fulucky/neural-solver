"""Local source-code inference helpers for heatsink MCP tools."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_INVERSE_DESIGN_ROOT = PROJECT_ROOT / "AIInverseDesign"
INFER_DIR = AI_INVERSE_DESIGN_ROOT / "infer"
DEFAULT_CHECKPOINT = AI_INVERSE_DESIGN_ROOT / "outputs_guided_cvae" / "heatsink" / "best_model.pt"
CHECKPOINT_ENV = "HEATSINK_THRESHOLD_CVAE_CHECKPOINT"
DEVICE_ENV = "HEATSINK_API_DEVICE"
DEFAULT_DEVICE = "cpu"

for path in (PROJECT_ROOT, AI_INVERSE_DESIGN_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

os.environ.setdefault(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT))
os.environ.setdefault(DEVICE_ENV, DEFAULT_DEVICE)


def checkpoint_path(path: str | None = None) -> str:
    return str(Path(path or os.getenv(CHECKPOINT_ENV) or DEFAULT_CHECKPOINT).expanduser())


def infer_device(device: str | None = None) -> str:
    return device or os.getenv(DEVICE_ENV) or DEFAULT_DEVICE


def temp_threshold(request: dict[str, Any]) -> float:
    value = request.get("temp_threshold", request.get("temp_limit"))
    if value is None:
        raise ValueError("temp_threshold or temp_limit is required")
    return float(value)


def enrich_temperature_margin(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    threshold = float(result["temp_threshold"])
    pred = float(result["pred_cpu_temp"])
    result["temp_margin"] = threshold - pred
    result["is_feasible"] = bool(result.get("threshold_ok", pred <= threshold))
    return result


def make_generate_args(
    request: dict[str, Any],
    checkpoint_path_value: str | None,
    device: str | None,
    num_samples: int | None,
    top_k: int | None,
    latent_opt_steps: int,
    latent_lr: float,
    temperature_weight: float,
    threshold_weight: float,
) -> argparse.Namespace:
    condition = request["condition"]
    bbox = request["bbox"]
    return argparse.Namespace(
        checkpoint_path=checkpoint_path(checkpoint_path_value),
        output_csv="",
        output_json="",
        surrogate_checkpoint="",
        num_samples=int(num_samples or request.get("candidate_pool_size") or request.get("num_samples") or 1024),
        top_k=int(top_k or request.get("top_k") or 10),
        temp_threshold=temp_threshold(request),
        chip_length=float(condition["chip_length"]),
        rjc=float(condition["Rjc"]),
        rjb=float(condition["Rjb"]),
        power=float(condition["power"]),
        wind_speed=float(condition["wind_speed"]),
        base_width=float(bbox["base_width"]),
        base_depth=float(bbox["base_depth"]),
        total_height=float(bbox["total_height"]),
        device=infer_device(device),
        latent_opt_steps=int(latent_opt_steps),
        latent_lr=float(latent_lr),
        temperature_weight=float(temperature_weight),
        threshold_weight=float(threshold_weight),
        diversity_rerank_weight=float(request.get("diversity_rerank_weight", 0.15)),
        diversity_temp_tolerance=float(request.get("diversity_temp_tolerance", 2.0)),
    )


@lru_cache(maxsize=4)
def load_model_payload(path: str, device: str) -> dict[str, Any]:
    import torch
    from AIInverseDesign.common.heatsink_inverse_common import load_checkpoint

    return load_checkpoint(path, torch.device(device))


def geometry_values(geometry: dict[str, Any]) -> list[float]:
    clear_spacing = geometry.get("fin_clear_spacing", geometry.get("fin_spacing"))
    if clear_spacing is None:
        raise ValueError("geometry must contain fin_clear_spacing")
    return [
        float(geometry["fin_height"]),
        float(geometry["fin_thickness"]),
        float(clear_spacing),
        float(geometry["fin_break_thickness"]),
        float(geometry["fin_break_width"]),
    ]


def generate_local_candidates(
    request: dict[str, Any],
    checkpoint_path_value: str | None,
    device: str | None,
    num_samples: int | None,
    top_k: int | None,
    latent_opt_steps: int,
    latent_lr: float,
    temperature_weight: float,
    threshold_weight: float,
) -> dict[str, Any]:
    from AIInverseDesign.infer.cvae_inferencer import generate_rows

    args = make_generate_args(
        request,
        checkpoint_path_value,
        device,
        num_samples,
        top_k,
        latent_opt_steps,
        latent_lr,
        temperature_weight,
        threshold_weight,
    )
    rows = generate_rows(args, guided=True)
    return {
        "method": "threshold-cvae",
        "checkpoint_path": args.checkpoint_path,
        "device": args.device,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "temp_threshold": args.temp_threshold,
        "candidates": [enrich_temperature_margin(row) for row in rows],
    }


def score_local_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    checkpoint_path_value: str | None,
    device: str | None,
    top_k: int | None = None,
) -> dict[str, Any]:
    from AIInverseDesign.common.heatsink_inverse_common import score_candidates

    condition = request["condition"]
    bbox = request["bbox"]
    infer_device_value = infer_device(device)
    payload = load_model_payload(checkpoint_path(checkpoint_path_value), infer_device_value)
    rows = score_candidates(
        payload,
        {
            "chip_length": float(condition["chip_length"]),
            "Rjc": float(condition["Rjc"]),
            "Rjb": float(condition["Rjb"]),
            "power": float(condition["power"]),
            "wind_speed": float(condition["wind_speed"]),
        },
        {
            "base_width": float(bbox["base_width"]),
            "base_depth": float(bbox["base_depth"]),
            "total_height": float(bbox["total_height"]),
        },
        [geometry_values(candidate) for candidate in candidates],
        temp_threshold(request),
        top_k or len(candidates),
    )
    return {
        "method": "forward-surrogate-ranking",
        "temp_threshold": temp_threshold(request),
        "candidates": [enrich_temperature_margin(row) for row in rows],
    }


def predict_local_temperature(
    request: dict[str, Any],
    geometry: dict[str, Any],
    checkpoint_path_value: str | None,
    device: str | None,
) -> dict[str, Any]:
    result = score_local_candidates(
        request=request,
        candidates=[geometry],
        checkpoint_path_value=checkpoint_path_value,
        device=device,
        top_k=1,
    )
    candidates = result["candidates"]
    if not candidates:
        raise RuntimeError("temperature prediction returned no rows")
    return candidates[0]


def refine_local_candidate(
    request: dict[str, Any],
    candidate: dict[str, Any],
    updates: dict[str, float] | None,
    instruction: str,
    checkpoint_path_value: str | None,
    device: str | None,
) -> dict[str, Any]:
    from AIInverseDesign.common.data_adapter import GEOMETRY_BOUNDS, clip_fin_clear_spacing_for_pitch, clip_value

    bbox = request["bbox"]
    refined = dict(candidate)
    if "fin_spacing" in refined and "fin_clear_spacing" not in refined:
        refined["fin_clear_spacing"] = refined["fin_spacing"]
    changed: dict[str, dict[str, float]] = {}

    def apply_value(key: str, value: float) -> None:
        old = float(refined[key])
        new = clip_value(key, float(value))
        refined[key] = new
        changed[key] = {"from": old, "to": new}

    for key, value in (updates or {}).items():
        normalized_key = "fin_clear_spacing" if key == "fin_spacing" else key
        if normalized_key in GEOMETRY_BOUNDS and normalized_key in refined:
            apply_value(normalized_key, value)

    text = instruction.lower()
    if any(token in text for token in ["spacing", "间距"]):
        if any(token in text for token in ["larger", "increase", "放大", "调大", "增大"]):
            apply_value("fin_clear_spacing", float(refined["fin_clear_spacing"]) + 0.25)
        if any(token in text for token in ["smaller", "decrease", "缩小", "调小", "减小"]):
            apply_value("fin_clear_spacing", float(refined["fin_clear_spacing"]) - 0.25)
    if any(token in text for token in ["thin", "thinner", "薄", "更薄"]):
        apply_value("fin_thickness", float(refined["fin_thickness"]) - 0.1)
    if any(token in text for token in ["thick", "thicker", "厚", "更厚"]):
        apply_value("fin_thickness", float(refined["fin_thickness"]) + 0.1)
    if any(token in text for token in ["fin height", "鳍片高度", "fin_h"]):
        if any(token in text for token in ["increase", "更高", "增高", "调高"]):
            apply_value("fin_height", float(refined["fin_height"]) + 0.5)
        if any(token in text for token in ["decrease", "更低", "降低", "调低"]):
            apply_value("fin_height", float(refined["fin_height"]) - 0.5)

    if "base_height" in changed:
        refined["fin_height"] = float(bbox["total_height"]) - float(refined["base_height"])
    if "fin_height" in refined:
        base_height = float(bbox["total_height"]) - float(refined["fin_height"])
        refined["base_height"] = clip_value("base_height", base_height)
        refined["fin_height"] = float(bbox["total_height"]) - float(refined["base_height"])
    refined["fin_clear_spacing"] = clip_fin_clear_spacing_for_pitch(
        float(refined["fin_thickness"]),
        float(refined["fin_clear_spacing"]),
    )

    predicted = predict_local_temperature(
        request=request,
        geometry=refined,
        checkpoint_path_value=checkpoint_path_value,
        device=device,
    )
    return {"changes": changed, "candidate": predicted}


def export_local_candidates(candidates: list[dict[str, Any]], export_format: str) -> dict[str, Any]:
    if export_format == "json":
        return {
            "format": "json",
            "filename": "heatsink_candidates.json",
            "content": json.dumps(candidates, ensure_ascii=False, indent=2),
        }

    if export_format == "csv":
        if not candidates:
            return {"format": "csv", "filename": "heatsink_candidates.csv", "content": ""}
        fieldnames = sorted({key for candidate in candidates for key in candidate.keys()})
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(candidate)
        return {
            "format": "csv",
            "filename": "heatsink_candidates.csv",
            "content": output.getvalue(),
        }

    if export_format == "simulation_input":
        return {
            "format": "simulation_input",
            "filename": "heatsink_simulation_input.json",
            "content": json.dumps({"candidates": candidates}, ensure_ascii=False, indent=2),
        }

    raise ValueError(f"Unsupported export_format: {export_format}")
