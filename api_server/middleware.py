from __future__ import annotations

from collections.abc import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


class HostRewriteMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        # 本地 MCP streamable HTTP 对 Host 较敏感；统一改成本地 API 地址。
        scope = request.scope
        headers = list(scope.get("headers", []))
        for index, (key, _value) in enumerate(headers):
            if key == b"host":
                headers[index] = (b"host", b"127.0.0.1:8080")
                break
        scope["headers"] = headers
        return await call_next(request)
