from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import configure_environment, configure_import_paths
from .mcp_app import mcp
from .middleware import HostRewriteMiddleware
from .routes import router


configure_import_paths()
configure_environment()

log = logging.getLogger("NeuralSolverAPI")
log.setLevel(logging.INFO)

_mcp_started = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 FastAPI 时同步启动本地 MCP 会话管理器。

    MCP 用于给 Agent 暴露逆向设计工具；普通 HTTP API 和 MCP 服务共用同一个
    FastAPI 进程，方便本地联调。
    """

    global _mcp_started
    if not _mcp_started:
        async with mcp._session_manager.run():
            _mcp_started = True
            yield
    else:
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Neural Solver API",
        description="Unified API for AISelection and AIInverseDesign.",
        version="1.0.0",
        lifespan=lifespan,
    )
    # 兼容部分本地/代理环境下 Host 头不一致导致的 MCP streamable HTTP 访问问题。
    app.add_middleware(HostRewriteMiddleware)

    # 普通 HTTP 路由：健康检查、AISelection 推理、AIInverseDesign 推荐/温度预测。
    app.include_router(router)

    # MCP 路由：供 Agent 以工具协议调用散热器逆向设计能力。
    app.mount("/ai_heat_sink_gener/", mcp.streamable_http_app())
    return app


app = create_app()
