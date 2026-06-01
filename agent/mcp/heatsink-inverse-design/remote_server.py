"""Run the heatsink inverse-design MCP server over Streamable HTTP."""

from __future__ import annotations

import argparse

from mcp.server.transport_security import TransportSecuritySettings

from server import mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run remote heatsink MCP server over Streamable HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--path", default="/")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--disable-host-check",
        action="store_true",
        help="Disable MCP DNS rebinding/Host header protection for LAN or tunnel testing.",
    )
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.settings.streamable_http_path = args.path
    mcp.settings.log_level = args.log_level.upper()
    if args.disable_host_check:
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
