# Heatsink Inverse Design Prompt

Use the `heatsink-inverse-design` skill when the user asks for heatsink candidate generation, threshold-CVAE inference, temperature prediction, candidate ranking, geometry refinement, simulation validation, or export.

Prefer the `heatsink_inverse_design` MCP tools under `agent`. In remote `main.py` deployment, MCP tools should use `route: "local"` and call the packaged inference code under `AIHeatsinkInverseDesign/infer` and `AIHeatsinkInverseDesign/common`.
