"""Tool: score and rank candidates."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import score_local_candidates


def score_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    method: str | None = None,
    checkpoint_path: str | None = None,
    surrogate_checkpoint: str | None = None,
    device: str | None = None,
    top_k: int | None = None,
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """模块评分条和综合排序：按 route 选择调用 FastAPI API 或本地源码。"""

    if route == "local":
        return score_local_candidates(
            request=request,
            candidates=candidates,
            checkpoint_path_value=checkpoint_path,
            device=device,
            top_k=top_k,
            method=method,
            surrogate_checkpoint_value=surrogate_checkpoint,
        )
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/candidates/score",
        {
            "request": request,
            "candidates": candidates,
            "method": method,
            "checkpoint_path": checkpoint_path,
            "surrogate_checkpoint": surrogate_checkpoint,
            "device": device,
            "top_k": top_k,
        },
        api_base_url=api_base_url,
    )
