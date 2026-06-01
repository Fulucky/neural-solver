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

log = logging.getLogger("AiSelection")
log.setLevel(logging.INFO)

_mcp_started = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_started
    if not _mcp_started:
        async with mcp._session_manager.run():
            _mcp_started = True
            yield
    else:
        yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AiSelection API",
        description="AiSelection web service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(HostRewriteMiddleware)
    app.include_router(router)
    app.mount("/ai_heat_sink_gener/", mcp.streamable_http_app())
    return app


app = create_app()

