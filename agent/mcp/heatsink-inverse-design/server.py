"""FastMCP server for heatsink inverse design.

This file registers the six business tools. Tool implementations live in
`agent/tools/heatsink_inverse_design/`.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP


AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from tools.heatsink_inverse_design import (  # noqa: E402
    export_candidates as export_candidates_tool,
    generate_candidates as generate_candidates_tool,
    predict_temperature as predict_temperature_tool,
    refine_candidate as refine_candidate_tool,
    score_candidates as score_candidates_tool,
    validate_candidates as validate_candidates_tool,
)


mcp = FastMCP("heatsink_inverse_design")
DEFAULT_ROUTE_ENV = "HEATSINK_MCP_DEFAULT_ROUTE"


def selected_route(route: str | None) -> str:
    return route or os.getenv(DEFAULT_ROUTE_ENV, "api")


@mcp.tool()
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
    route: str | None = None,
) -> dict[str, Any]:
    """生成推荐：生成 threshold-CVAE 散热器候选方案。route 可选 api 或 local。"""

    return generate_candidates_tool(
        request=request,
        method=method,
        checkpoint_path=checkpoint_path,
        surrogate_checkpoint=surrogate_checkpoint,
        device=device,
        num_samples=num_samples,
        top_k=top_k,
        latent_opt_steps=latent_opt_steps,
        latent_lr=latent_lr,
        temperature_weight=temperature_weight,
        threshold_weight=threshold_weight,
        guidance_scale=guidance_scale,
        api_base_url=api_base_url,
        route=selected_route(route),
    )


@mcp.tool()
def predict_temperature(
    request: dict[str, Any],
    geometry: dict[str, Any],
    method: str | None = None,
    checkpoint_path: str | None = None,
    surrogate_checkpoint: str | None = None,
    device: str | None = None,
    api_base_url: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """尺寸调参与温度预测：预测单个候选方案的 CPU 温度。route 可选 api 或 local。"""

    return predict_temperature_tool(
        request=request,
        geometry=geometry,
        method=method,
        checkpoint_path=checkpoint_path,
        surrogate_checkpoint=surrogate_checkpoint,
        device=device,
        api_base_url=api_base_url,
        route=selected_route(route),
    )


@mcp.tool()
def score_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    method: str | None = None,
    checkpoint_path: str | None = None,
    surrogate_checkpoint: str | None = None,
    device: str | None = None,
    top_k: int | None = None,
    api_base_url: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """模块评分条和综合排序：评分并排序候选方案。route 可选 api 或 local。"""

    return score_candidates_tool(
        request=request,
        candidates=candidates,
        method=method,
        checkpoint_path=checkpoint_path,
        surrogate_checkpoint=surrogate_checkpoint,
        device=device,
        top_k=top_k,
        api_base_url=api_base_url,
        route=selected_route(route),
    )


@mcp.tool()
def refine_candidate(
    request: dict[str, Any],
    candidate: dict[str, Any],
    updates: dict[str, float] | None = None,
    instruction: str = "",
    method: str | None = None,
    checkpoint_path: str | None = None,
    surrogate_checkpoint: str | None = None,
    device: str | None = None,
    api_base_url: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """用户修改意图：根据自然语言意图或显式参数改型。route 可选 api 或 local。"""

    return refine_candidate_tool(
        request=request,
        candidate=candidate,
        updates=updates,
        instruction=instruction,
        method=method,
        checkpoint_path=checkpoint_path,
        surrogate_checkpoint=surrogate_checkpoint,
        device=device,
        api_base_url=api_base_url,
        route=selected_route(route),
    )


@mcp.tool()
def validate_candidates(
    request: dict[str, Any],
    candidates: list[dict[str, Any]],
    simulation_api_url: str | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """提交仿真求解：准备或提交候选方案到仿真 API。"""

    return validate_candidates_tool(
        request=request,
        candidates=candidates,
        simulation_api_url=simulation_api_url,
        timeout_seconds=timeout_seconds,
    )


@mcp.tool()
def export_candidates(
    candidates: list[dict[str, Any]],
    export_format: str = "json",
    api_base_url: str | None = None,
    route: str | None = None,
) -> dict[str, Any]:
    """导出 JSON / CSV / 验证集：导出候选方案。route 可选 api 或 local。"""

    return export_candidates_tool(
        candidates=candidates,
        export_format=export_format,
        api_base_url=api_base_url,
        route=selected_route(route),
    )


if __name__ == "__main__":
    mcp.run()
