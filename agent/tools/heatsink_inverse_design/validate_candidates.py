"""Tool: prepare or submit candidates for simulation validation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def validate_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    simulation_api_url: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """提交仿真求解：未配置仿真 API 时返回待提交 payload。"""

    payload = {
        "request": request,
        "candidates": candidates,
        "source": "heatsink_inverse_design_mcp",
    }
    if not simulation_api_url:
        return {
            "status": "not_submitted",
            "message": "simulation_api_url not provided; returning request payload only.",
            "payload": payload,
        }

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        simulation_api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return {
                "status": "submitted",
                "http_status": response.status,
                "response": json.loads(response_body) if response_body else {},
            }
    except urllib.error.URLError as exc:
        return {
            "status": "submit_failed",
            "error": str(exc),
            "payload": payload,
        }
