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
    """预测单个候选几何尺寸对应的 CPU 温度。"""

    if "geometry" not in data:
        raise ValueError("missing required field: geometry")

    payload = PredictRequest(
        request=_request_payload(data),
        geometry=data["geometry"],
        checkpoint_path=data.get("checkpoint_path"),
        device=data.get("device"),
    )
    return predict_temperature(payload)
