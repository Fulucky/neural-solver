from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_SELECTION_ROOT = REPO_ROOT / "AISelection"
AI_INVERSE_DESIGN_ROOT = REPO_ROOT / "AIInverseDesign"
AGENT_ROOT = REPO_ROOT / "agent"

MCP_DEFAULT_ROUTE_ENV = "HEATSINK_MCP_DEFAULT_ROUTE"
DEFAULT_MCP_ROUTE = "local"

API_HOST = os.getenv("AI_SELECTION_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("AI_SELECTION_API_PORT", "8080"))


def configure_import_paths() -> None:
    """配置本地包导入路径。

    这里显式把 AISelection 和 AIInverseDesign 都作为仓库根目录下的平级模块加入
    Python 搜索路径。这样可以避免代码层面表现成 "AIInverseDesign 隶属于
    AISelection" 的关系；API 层只是同时编排这两个能力。
    """

    paths = [
        REPO_ROOT,
        AI_SELECTION_ROOT,
        AI_INVERSE_DESIGN_ROOT,
        AGENT_ROOT,
    ]
    for path in paths:
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def configure_environment() -> None:
    """设置 API/MCP 运行时默认环境变量。"""

    os.environ.setdefault(MCP_DEFAULT_ROUTE_ENV, DEFAULT_MCP_ROUTE)
