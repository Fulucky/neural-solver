# Heatsink Agent Test Cases

Use these prompts to test the complete Agent workflow: natural language -> skill -> MCP tools -> FastAPI -> threshold-CVAE.

## Case 1: Candidate Generation

```text
使用散热器逆向设计能力：在 40 x 40 x 20 mm 空间约束下，芯片长度 35 mm，Rjc=0.6 degC/W，Rjb=1.1 degC/W，功率 85 W，风速 4 m/s，温度阈值 80 degC，生成 5 个散热器候选方案。
```

Expected behavior:

- Agent extracts `condition`, `bbox`, `temp_threshold`, `top_k`.
- Agent calls `generate_candidates`.
- Agent returns a compact table with temperature, margin, feasibility, and geometry.

## Case 2: Temperature Prediction

```text
基于刚才第 1 个候选方案，重新预测一次温度，并说明它距离 80 degC 阈值还有多少裕量。
```

Expected behavior:

- Agent reuses the previous request context.
- Agent passes the selected candidate geometry to `predict_temperature`.
- Agent explains `pred_cpu_temp`, `temp_margin`, and feasibility.

## Case 3: Natural-Language Refinement

```text
把第 1 个候选方案的鳍片间距调大一点，同时保持总高度 20 mm 不变，再预测温度变化。
```

Expected behavior:

- Agent calls `refine_candidate`.
- Agent uses `instruction` rather than inventing new hidden parameters.
- Agent reports changed fields and updated temperature.

## Case 4: Scoring And Ranking

```text
把当前这些候选方案按温度优先重新排序，并说明前三个方案的主要差异。
```

Expected behavior:

- Agent calls `score_candidates`.
- Agent returns ranked candidates and a short engineering comparison.

## Case 5: Simulation Validation Payload

```text
把前 3 个候选方案提交仿真验证。如果还没有仿真 API，就先给我准备提交 payload。
```

Expected behavior:

- Agent calls `validate_candidates`.
- If no `simulation_api_url` is configured, Agent says no real CFD job was submitted.
- Agent returns or summarizes the structured validation payload.

## Case 6: Export

```text
把当前候选方案导出成 CSV。
```

Expected behavior:

- Agent calls `export_candidates` with `export_format="csv"`.
- Agent returns filename/content or explains where the generated content is available.

