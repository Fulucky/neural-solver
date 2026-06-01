"""Tool: generate heatsink recommendations."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import generate_local_candidates


def generate_candidates(
    request: dict[str, Any],
    method: str | None = None,
    checkpoint_path: str | None = None,
    surrogate_checkpoint: str | None = None,
    device: str | None = None,
    num_samples: int | None = None,
    top_k: int | None = None,
    latent_opt_steps: int | None = None,
    latent_lr: float | None = None,
    temperature_weight: float | None = None,
    threshold_weight: float | None = None,
    guidance_scale: float | None = None,
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """生成推荐：按 route 选择调用 FastAPI API 或本地源码。"""

    if route == "local":
        return generate_local_candidates(
            request=request,
            checkpoint_path_value=checkpoint_path,
            method=method,
            surrogate_checkpoint_value=surrogate_checkpoint,
            device=device,
            num_samples=num_samples,
            top_k=top_k,
            latent_opt_steps=latent_opt_steps,
            latent_lr=latent_lr,
            temperature_weight=temperature_weight,
            threshold_weight=threshold_weight,
            guidance_scale=guidance_scale,
        )
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/candidates/generate",
        {
            "request": request,
            "method": method,
            "checkpoint_path": checkpoint_path,
            "surrogate_checkpoint": surrogate_checkpoint,
            "device": device,
            "num_samples": num_samples,
            "top_k": top_k,
            "latent_opt_steps": latent_opt_steps,
            "latent_lr": latent_lr,
            "temperature_weight": temperature_weight,
            "threshold_weight": threshold_weight,
            "guidance_scale": guidance_scale,
        },
        api_base_url=api_base_url,
        timeout_seconds=120.0,
    )
