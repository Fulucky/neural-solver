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
    candidate_pool_size: int | None = None,
    top_k: int | None = None,
    latent_opt_steps: int | None = None,
    latent_lr: float | None = None,
    temperature_weight: float | None = None,
    threshold_weight: float | None = None,
    guidance_scale: float | None = None,
    engineering_variant_mode: str | None = None,
    engineering_variant_count_per_candidate: int | None = None,
    engineering_variant_max_trials: int | None = None,
    engineering_variant_scale: float | None = None,
    engineering_variant_required_temp_margin: float | None = None,
    engineering_variant_min_unique_ratio: float | None = None,
    engineering_variant_min_norm_mean_dist: float | None = None,
    engineering_variant_min_norm_min_dist: float | None = None,
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
            candidate_pool_size=candidate_pool_size,
            top_k=top_k,
            latent_opt_steps=latent_opt_steps,
            latent_lr=latent_lr,
            temperature_weight=temperature_weight,
            threshold_weight=threshold_weight,
            guidance_scale=guidance_scale,
            engineering_variant_mode=engineering_variant_mode,
            engineering_variant_count_per_candidate=engineering_variant_count_per_candidate,
            engineering_variant_max_trials=engineering_variant_max_trials,
            engineering_variant_scale=engineering_variant_scale,
            engineering_variant_required_temp_margin=engineering_variant_required_temp_margin,
            engineering_variant_min_unique_ratio=engineering_variant_min_unique_ratio,
            engineering_variant_min_norm_mean_dist=engineering_variant_min_norm_mean_dist,
            engineering_variant_min_norm_min_dist=engineering_variant_min_norm_min_dist,
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
            "candidate_pool_size": candidate_pool_size,
            "top_k": top_k,
            "latent_opt_steps": latent_opt_steps,
            "latent_lr": latent_lr,
            "temperature_weight": temperature_weight,
            "threshold_weight": threshold_weight,
            "guidance_scale": guidance_scale,
            "engineering_variant_mode": engineering_variant_mode,
            "engineering_variant_count_per_candidate": engineering_variant_count_per_candidate,
            "engineering_variant_max_trials": engineering_variant_max_trials,
            "engineering_variant_scale": engineering_variant_scale,
            "engineering_variant_required_temp_margin": engineering_variant_required_temp_margin,
            "engineering_variant_min_unique_ratio": engineering_variant_min_unique_ratio,
            "engineering_variant_min_norm_mean_dist": engineering_variant_min_norm_mean_dist,
            "engineering_variant_min_norm_min_dist": engineering_variant_min_norm_min_dist,
        },
        api_base_url=api_base_url,
        timeout_seconds=120.0,
    )
