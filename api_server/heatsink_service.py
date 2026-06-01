from __future__ import annotations

from typing import Any

from .config import configure_environment, configure_import_paths


configure_import_paths()
configure_environment()

from agent.app.api.heatsink_inference_api import (  # noqa: E402
    GenerateRequest,
    PredictRequest,
    generate_candidates,
    predict_temperature,
)


def _request_payload(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("request") or data


def recommend_size(data: dict[str, Any]) -> dict[str, Any]:
    """Generate heatsink size candidates from the public API payload shape."""

    payload = GenerateRequest(
        request=_request_payload(data),
        checkpoint_path=data.get("checkpoint_path"),
        device=data.get("device"),
        num_samples=data.get("num_samples"),
        top_k=data.get("top_k"),
        latent_opt_steps=data.get("latent_opt_steps", 40),
        latent_lr=data.get("latent_lr", 5e-2),
        temperature_weight=data.get("temperature_weight", 1.0),
        threshold_weight=data.get("threshold_weight", 2.0),
    )
    return generate_candidates(payload)


def predict_candidate_temperature(data: dict[str, Any]) -> dict[str, Any]:
    """Predict CPU temperature for one candidate geometry."""

    if "geometry" not in data:
        raise ValueError("missing required field: geometry")

    payload = PredictRequest(
        request=_request_payload(data),
        geometry=data["geometry"],
        checkpoint_path=data.get("checkpoint_path"),
        device=data.get("device"),
    )
    return predict_temperature(payload)
