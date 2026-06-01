from __future__ import annotations

from api_server.app import app
from api_server.config import API_HOST, API_PORT


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
        proxy_headers=False,
        forwarded_allow_ips=["*"],
    )
