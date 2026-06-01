"""Tool: generate heatsink recommendations."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import generate_local_candidates


def generate_candidates(
    request: dict[str, Any],
    checkpoint_path: str | None = None,
    device: str | None = None,
    num_samples: int | None = None,
    top_k: int | None = None,
    latent_opt_steps: int = 40,
    latent_lr: float = 5e-2,
    temperature_weight: float = 1.0,
    threshold_weight: float = 2.0,
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """生成推荐：按 route 选择调用 FastAPI API 或本地源码。"""

    if route == "local":
        return generate_local_candidates(
            request=request,
            checkpoint_path_value=checkpoint_path,
            device=device,
            num_samples=num_samples,
            top_k=top_k,
            latent_opt_steps=latent_opt_steps,
            latent_lr=latent_lr,
            temperature_weight=temperature_weight,
            threshold_weight=threshold_weight,
        )
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/candidates/generate",
        {
            "request": request,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "num_samples": num_samples,
            "top_k": top_k,
            "latent_opt_steps": latent_opt_steps,
            "latent_lr": latent_lr,
            "temperature_weight": temperature_weight,
            "threshold_weight": threshold_weight,
        },
        api_base_url=api_base_url,
        timeout_seconds=120.0,
    )
