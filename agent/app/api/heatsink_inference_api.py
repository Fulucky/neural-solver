"""FastAPI service for heatsink threshold-CVAE inference.

This API is the backend AI inference layer. The MCP server should call these
HTTP endpoints instead of loading model checkpoints directly.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AI_INVERSE_DESIGN_ROOT = PROJECT_ROOT / "AIInverseDesign"

for path in (PROJECT_ROOT, AI_INVERSE_DESIGN_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


class Condition(BaseModel):
    chip_length: float
    Rjc: float
    Rjb: float
    power: float
    wind_speed: float


class BoundingBox(BaseModel):
    base_width: float
    base_depth: float
    total_height: float


class InferenceRequest(BaseModel):
    condition: Condition
    bbox: BoundingBox
    temp_threshold: float | None = None
    temp_limit: float | None = None
    top_k: int | None = None
    candidate_pool_size: int | None = Field(default=None, alias="num_samples")
    optimization_priority: list[str] | None = None
    diversity_rerank_weight: float = 0.15
    diversity_temp_tolerance: float = 2.0

    model_config = {"populate_by_name": True}


class Geometry(BaseModel):
    base_width: float | None = None
    base_depth: float | None = None
    total_height: float | None = None
    base_height: float | None = None
    fin_height: float
    fin_thickness: float
    fin_clear_spacing: float
    fin_break_thickness: float
    fin_break_width: float


class GenerateRequest(BaseModel):
    request: InferenceRequest
    method: Literal["cvae", "threshold-cvae", "diffusion"] | None = None
    checkpoint_path: str | None = None
    surrogate_checkpoint: str | None = None
    device: str | None = None
    num_samples: int | None = None
    top_k: int | None = None
    latent_opt_steps: int | None = None
    latent_lr: float | None = None
    temperature_weight: float | None = None
    threshold_weight: float | None = None
    guidance_scale: float | None = None


class PredictRequest(BaseModel):
    request: InferenceRequest
    geometry: Geometry
    method: Literal["cvae", "threshold-cvae", "diffusion"] | None = None
    checkpoint_path: str | None = None
    surrogate_checkpoint: str | None = None
    device: str | None = None


class ScoreRequest(BaseModel):
    request: InferenceRequest
    candidates: list[Geometry]
    method: Literal["cvae", "threshold-cvae", "diffusion"] | None = None
    checkpoint_path: str | None = None
    surrogate_checkpoint: str | None = None
    device: str | None = None
    top_k: int | None = None


class RefineRequest(BaseModel):
    request: InferenceRequest
    candidate: Geometry
    updates: dict[str, float] | None = None
    instruction: str = ""
    method: Literal["cvae", "threshold-cvae", "diffusion"] | None = None
    checkpoint_path: str | None = None
    surrogate_checkpoint: str | None = None
    device: str | None = None


class ExportRequest(BaseModel):
    candidates: list[dict[str, Any]]
    export_format: Literal["json", "csv"] = "json"


app = FastAPI(
    title="Heatsink Threshold-CVAE Inference API",
    version="0.1.0",
    description="Backend AI inference API for heatsink inverse design.",
)


def _active_config():
    from AIInverseDesign.common.inference_config import load_inference_config

    return load_inference_config()


def _method(method: str | None = None) -> str:
    return method or _active_config().method


def _checkpoint_path(path: str | None = None, method: str | None = None) -> str:
    if path:
        return str(Path(path).expanduser())
    if method:
        from AIInverseDesign.common.inference_config import default_checkpoint_for_method

        return default_checkpoint_for_method(method)
    return _active_config().checkpoint_path


def _surrogate_checkpoint(path: str | None = None) -> str:
    return str(Path(path).expanduser()) if path else _active_config().surrogate_checkpoint


def _device(device: str | None = None) -> str:
    return device or _active_config().device


def _temp_threshold(request: InferenceRequest) -> float:
    if request.temp_threshold is not None:
        return float(request.temp_threshold)
    if request.temp_limit is not None:
        return float(request.temp_limit)
    raise ValueError("temp_threshold or temp_limit is required")


def _condition_dict(request: InferenceRequest) -> dict[str, float]:
    return request.condition.model_dump()


def _bbox_dict(request: InferenceRequest) -> dict[str, float]:
    return request.bbox.model_dump()


def _geometry_values(geometry: Geometry) -> list[float]:
    return [
        float(geometry.fin_height),
        float(geometry.fin_thickness),
        float(geometry.fin_clear_spacing),
        float(geometry.fin_break_thickness),
        float(geometry.fin_break_width),
    ]


def _row_with_margin(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    threshold = float(enriched["temp_threshold"])
    pred = float(enriched["pred_cpu_temp"])
    enriched["temp_margin"] = threshold - pred
    enriched["is_feasible"] = bool(enriched.get("threshold_ok", pred <= threshold))
    return enriched


def _make_args(payload: GenerateRequest) -> argparse.Namespace:
    request = payload.request
    condition = request.condition
    bbox = request.bbox
    config = _active_config()
    return argparse.Namespace(
        checkpoint_path=_checkpoint_path(payload.checkpoint_path, payload.method),
        output_csv="",
        output_json="",
        surrogate_checkpoint=_surrogate_checkpoint(payload.surrogate_checkpoint),
        num_samples=int(payload.num_samples or request.candidate_pool_size or config.num_samples),
        top_k=int(payload.top_k or request.top_k or config.top_k),
        temp_threshold=_temp_threshold(request),
        chip_length=condition.chip_length,
        rjc=condition.Rjc,
        rjb=condition.Rjb,
        power=condition.power,
        wind_speed=condition.wind_speed,
        base_width=bbox.base_width,
        base_depth=bbox.base_depth,
        total_height=bbox.total_height,
        device=_device(payload.device),
        latent_opt_steps=int(payload.latent_opt_steps if payload.latent_opt_steps is not None else config.latent_opt_steps),
        latent_lr=float(payload.latent_lr if payload.latent_lr is not None else config.latent_lr),
        temperature_weight=float(payload.temperature_weight if payload.temperature_weight is not None else config.temperature_weight),
        threshold_weight=float(payload.threshold_weight if payload.threshold_weight is not None else config.threshold_weight),
        guidance_scale=float(payload.guidance_scale if payload.guidance_scale is not None else config.guidance_scale),
        diversity_rerank_weight=float(request.diversity_rerank_weight),
        diversity_temp_tolerance=float(request.diversity_temp_tolerance),
    )


@lru_cache(maxsize=4)
def _load_payload(checkpoint_path: str, device: str, surrogate_checkpoint: str) -> dict[str, Any]:
    import torch
    from AIInverseDesign.common.heatsink_inverse_common import load_checkpoint

    return load_checkpoint(checkpoint_path, torch.device(device), surrogate_checkpoint)


def _score_rows(
    request: InferenceRequest,
    geometry_rows: list[list[float]],
    method: str | None,
    checkpoint_path: str | None,
    surrogate_checkpoint: str | None,
    device: str | None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    from AIInverseDesign.common.heatsink_inverse_common import score_candidates as score_with_surrogate

    payload = _load_payload(_checkpoint_path(checkpoint_path, method), _device(device), _surrogate_checkpoint(surrogate_checkpoint))
    rows = score_with_surrogate(
        payload,
        _condition_dict(request),
        _bbox_dict(request),
        geometry_rows,
        _temp_threshold(request),
        top_k or len(geometry_rows),
    )
    return [_row_with_margin(row) for row in rows]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "heatsink-threshold-cvae-inference"}


@app.post("/api/candidates/generate")
def generate_candidates(payload: GenerateRequest) -> dict[str, Any]:
    args = _make_args(payload)
    method = _method(payload.method)
    if method == "diffusion":
        from AIInverseDesign.infer.diffusion_inferencer import generate_rows

        rows = generate_rows(args)
    else:
        from AIInverseDesign.infer.cvae_inferencer import generate_rows

        rows = generate_rows(args, guided=(method == "threshold-cvae"))
    return {
        "method": method,
        "checkpoint_path": args.checkpoint_path,
        "surrogate_checkpoint": args.surrogate_checkpoint,
        "device": args.device,
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "temp_threshold": args.temp_threshold,
        "candidates": [_row_with_margin(row) for row in rows],
    }


@app.post("/api/temperature/predict")
def predict_temperature(payload: PredictRequest) -> dict[str, Any]:
    rows = _score_rows(
        payload.request,
        [_geometry_values(payload.geometry)],
        payload.method,
        payload.checkpoint_path,
        payload.surrogate_checkpoint,
        payload.device,
        top_k=1,
    )
    if not rows:
        raise RuntimeError("temperature prediction returned no rows")
    return rows[0]


@app.post("/api/candidates/score")
def score_candidates(payload: ScoreRequest) -> dict[str, Any]:
    rows = _score_rows(
        payload.request,
        [_geometry_values(candidate) for candidate in payload.candidates],
        payload.method,
        payload.checkpoint_path,
        payload.surrogate_checkpoint,
        payload.device,
        top_k=payload.top_k or len(payload.candidates),
    )
    return {
        "method": "forward-surrogate-ranking",
        "checkpoint_path": _checkpoint_path(payload.checkpoint_path, payload.method),
        "temp_threshold": _temp_threshold(payload.request),
        "candidates": rows,
    }


@app.post("/api/candidates/refine")
def refine_candidate(payload: RefineRequest) -> dict[str, Any]:
    from AIInverseDesign.common.data_adapter import GEOMETRY_BOUNDS, clip_fin_clear_spacing_for_pitch, clip_value

    bbox = _bbox_dict(payload.request)
    refined = payload.candidate.model_dump(exclude_none=True)
    changed: dict[str, dict[str, float]] = {}

    def apply_value(key: str, value: float) -> None:
        old = float(refined[key])
        new = clip_value(key, float(value))
        refined[key] = new
        changed[key] = {"from": old, "to": new}

    for key, value in (payload.updates or {}).items():
        if key in GEOMETRY_BOUNDS and key in refined:
            apply_value(key, value)

    text = payload.instruction.lower()
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

    predicted = predict_temperature(
        PredictRequest(
            request=payload.request,
            geometry=Geometry(**refined),
            method=payload.method,
            checkpoint_path=payload.checkpoint_path,
            surrogate_checkpoint=payload.surrogate_checkpoint,
            device=payload.device,
        )
    )
    return {"changes": changed, "candidate": predicted}


@app.post("/api/candidates/export")
def export_candidates(payload: ExportRequest) -> dict[str, Any]:
    if payload.export_format == "json":
        return {
            "format": "json",
            "filename": "heatsink_candidates.json",
            "content": json.dumps(payload.candidates, ensure_ascii=False, indent=2),
        }

    if not payload.candidates:
        return {"format": "csv", "filename": "heatsink_candidates.csv", "content": ""}
    fieldnames = sorted({key for candidate in payload.candidates for key in candidate.keys()})
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for candidate in payload.candidates:
        writer.writerow(candidate)
    return {
        "format": "csv",
        "filename": "heatsink_candidates.csv",
        "content": output.getvalue(),
    }
