# Tools

This folder contains the concrete implementations behind the Agent-facing MCP tools.

`heatsink_inverse_design/` contains the six business tools:

- `generate_candidates.py`
- `predict_temperature.py`
- `score_candidates.py`
- `refine_candidate.py`
- `validate_candidates.py`
- `export_candidates.py`

The FastMCP server in `../mcp/heatsink-inverse-design/server.py` registers these functions as MCP tools.
