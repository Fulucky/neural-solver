from __future__ import annotations

from typing import Any

from .config import configure_environment, configure_import_paths


configure_import_paths()
configure_environment()

# 这里复用 agent/app/api 中已有的 Pydantic 入参模型和纯函数式推理入口。
# api_server 只负责把公开 HTTP 请求转换成内部模型需要的结构，不直接实现算法。
from agent.app.api.heatsink_inference_api import (  # noqa: E402
    GenerateRequest,
    PredictRequest,
    generate_candidates,
    predict_temperature,
)


def _request_payload(data: dict[str, Any]) -> dict[str, Any]:
    """兼容两种请求格式：直接传 request，或把 request 字段包在外层。"""

    return data.get("request") or data


def recommend_size(data: dict[str, Any]) -> dict[str, Any]:
    """根据公开 API 入参生成散热器候选尺寸。"""

    payload = GenerateRequest(
        request=_request_payload(data),
        method=data.get("method"),
        checkpoint_path=data.get("checkpoint_path"),
        surrogate_checkpoint=data.get("surrogate_checkpoint"),
        device=data.get("device"),
        candidate_pool_size=data.get("candidate_pool_size"),
        top_k=data.get("top_k"),
        latent_opt_steps=data.get("latent_opt_steps"),
        latent_lr=data.get("latent_lr"),
        temperature_weight=data.get("temperature_weight"),
        threshold_weight=data.get("threshold_weight"),
        guidance_scale=data.get("guidance_scale"),
        engineering_variant_mode=data.get("engineering_variant_mode"),
        engineering_variant_count_per_candidate=data.get("engineering_variant_count_per_candidate"),
        engineering_variant_max_trials=data.get("engineering_variant_max_trials"),
        engineering_variant_scale=data.get("engineering_variant_scale"),
        engineering_variant_required_temp_margin=data.get("engineering_variant_required_temp_margin"),
        engineering_variant_min_unique_ratio=data.get("engineering_variant_min_unique_ratio"),
        engineering_variant_min_norm_mean_dist=data.get("engineering_variant_min_norm_mean_dist"),
        engineering_variant_min_norm_min_dist=data.get("engineering_variant_min_norm_min_dist"),
    )
    return generate_candidates(payload)


def predict_candidate_temperature(data: dict[str, Any]) -> dict[str, Any]:
    """预测单个候选几何尺寸对应的 CPU 温度。"""

    if "geometry" not in data:
        raise ValueError("missing required field: geometry")

    payload = PredictRequest(
        request=_request_payload(data),
        geometry=data["geometry"],
        method=data.get("method"),
        checkpoint_path=data.get("checkpoint_path"),
        surrogate_checkpoint=data.get("surrogate_checkpoint"),
        device=data.get("device"),
    )
    return predict_temperature(payload)
