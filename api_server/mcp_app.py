from __future__ import annotations

import importlib.util
from typing import Any

from .config import AGENT_ROOT, configure_environment, configure_import_paths


configure_import_paths()
configure_environment()


def load_local_mcp() -> Any:
    """从 agent/mcp 加载本地 MCP 服务。

    MCP 目录名包含连字符，不能直接用普通 import，因此这里通过文件路径加载。
    """

    server_path = AGENT_ROOT / "mcp" / "heatsink-inverse-design" / "server.py"
    spec = importlib.util.spec_from_file_location("heatsink_inverse_design_mcp_server", server_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load MCP server from {server_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mcp


mcp = load_local_mcp()
