"""Tool: export candidates."""

from __future__ import annotations

from typing import Any

from .http_client import post_json
from .local_inference import export_local_candidates


def export_candidates(
    candidates: list[dict[str, Any]],
    export_format: str = "json",
    api_base_url: str | None = None,
    route: str = "api",
) -> dict[str, Any]:
    """导出 JSON / CSV / 验证集：按 route 选择调用 FastAPI API 或本地生成。"""

    if route == "local":
        return export_local_candidates(candidates=candidates, export_format=export_format)
    if route != "api":
        raise ValueError("route must be 'api' or 'local'")

    return post_json(
        "/api/candidates/export",
        {
            "candidates": candidates,
            "export_format": export_format,
        },
        api_base_url=api_base_url,
    )
