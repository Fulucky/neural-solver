"""Tool: predict candidate temperature."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import predict_local_temperature


def predict_temperature(
    request: dict[str, Any],
    geometry: dict[str, Any],
    checkpoint_path: str | None = None,
    device: str | None = None,
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """尺寸调参与温度预测：按 route 选择调用 FastAPI API 或本地源码。"""

    if route == "local":
        return predict_local_temperature(
            request=request,
            geometry=geometry,
            checkpoint_path_value=checkpoint_path,
            device=device,
        )
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/temperature/predict",
        {
            "request": request,
            "geometry": geometry,
            "checkpoint_path": checkpoint_path,
            "device": device,
        },
        api_base_url=api_base_url,
    )
