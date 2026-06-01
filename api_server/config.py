from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AI_INVERSE_DESIGN_ROOT = REPO_ROOT / "AIInverseDesign"
AGENT_ROOT = REPO_ROOT / "agent"

MCP_DEFAULT_ROUTE_ENV = "HEATSINK_MCP_DEFAULT_ROUTE"
DEFAULT_MCP_ROUTE = "local"

API_HOST = os.getenv("AI_SELECTION_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("AI_SELECTION_API_PORT", "8080"))


def configure_import_paths() -> None:
    """Make migrated local packages importable from direct script execution."""

    paths = [
        REPO_ROOT,
        REPO_ROOT / "AISelection",
        AI_INVERSE_DESIGN_ROOT,
        AGENT_ROOT,
        AI_INVERSE_DESIGN_ROOT / "common",
        AI_INVERSE_DESIGN_ROOT / "infer",
    ]
    for path in paths:
        path_text = str(path)
        if path.exists() and path_text not in sys.path:
            sys.path.insert(0, path_text)


def configure_environment() -> None:
    os.environ.setdefault(MCP_DEFAULT_ROUTE_ENV, DEFAULT_MCP_ROUTE)
