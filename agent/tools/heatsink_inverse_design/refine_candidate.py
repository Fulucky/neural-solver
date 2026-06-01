"""Tool: refine a candidate from explicit updates or user intent."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import refine_local_candidate


def refine_candidate(
    request: dict[str, Any],
    candidate: dict[str, Any],
    updates: dict[str, float] | None = None,
    instruction: str = "",
    checkpoint_path: str | None = None,
    device: str | None = None,
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """用户修改意图：按 route 选择调用 FastAPI API 或本地源码。"""

    if route == "local":
        return refine_local_candidate(
            request=request,
            candidate=candidate,
            updates=updates,
            instruction=instruction,
            checkpoint_path_value=checkpoint_path,
            device=device,
        )
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/candidates/refine",
        {
            "request": request,
            "candidate": candidate,
            "updates": updates,
            "instruction": instruction,
            "checkpoint_path": checkpoint_path,
            "device": device,
        },
        api_base_url=api_base_url,
    )
