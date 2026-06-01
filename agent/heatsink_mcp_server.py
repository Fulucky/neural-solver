"""Compatibility entrypoint for the heatsink inverse-design MCP server.

The canonical FastMCP server lives at:
`agent/mcp/heatsink-inverse-design/server.py`.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    server_path = Path(__file__).resolve().parent / "mcp" / "heatsink-inverse-design" / "server.py"
    runpy.run_path(str(server_path), run_name="__main__")
