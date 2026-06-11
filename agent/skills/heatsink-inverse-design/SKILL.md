---
name: heatsink-inverse-design
description: 当用户进行散热器逆向设计、生成候选尺寸、切换 cvae/threshold-cvae/diffusion 推理路径、指定生成模型或温度代理模型、温度预测、候选评分排序、工程扰动改型、仿真验证 payload 准备或候选方案导出时，使用本 skill，并通过 heatsink_inverse_design MCP tools 执行。
---

# 散热器逆向设计

## 概览

使用 `heatsink_inverse_design` MCP Server 作为 Agent 工具层。本 skill 指导 Agent 将用户的自然语言设计需求转成结构化 MCP tool 调用，并返回简洁的工程解释。

当前逆向设计默认推理配置由 `AIInverseDesign/config/inference_config.json` 管理。默认技术路径、生成模型 checkpoint、温度代理模型 checkpoint、设备、rerank 和工程扰动参数都从该配置读取。用户也可以在单次 tool 调用中传入 `method`、`checkpoint_path`、`surrogate_checkpoint` 或其它参数临时覆盖配置。

MCP Server `agent/mcp/heatsink-inverse-design/server.py` 暴露 6 个业务 tools。生成、预测、评分、改型和导出支持 `route` 参数：

- `route="api"`：默认方式，通过 FastAPI 服务调用推理能力。
- `route="local"`：直接调用本地源码执行推理。

除非用户明确要求直接调用本地源码或调试本地模型，否则优先使用默认 `route="api"`。

## 技术路径与模型

支持的逆向生成技术路径：

- `threshold-cvae`：温度阈值条件 CVAE，当前默认路径。
- `cvae`：阈值无关 CVAE，温度阈值通过 latent optimization 和温度代理排序生效。
- `diffusion`：条件扩散生成器，支持 surrogate guidance。

模型路径规则：

- `checkpoint_path`：生成模型 checkpoint，可选。不传时使用 `AIInverseDesign/config/inference_config.json`。
- `surrogate_checkpoint`：温度代理模型 checkpoint，可选。不传时使用生成模型 checkpoint 内嵌的 ForwardMLP 代理模型。
- 两个路径都支持相对路径；相对路径按项目根目录解析。

不要把 `checkpoint_path` 理解为唯一模型。它是生成模型；温度预测和排序使用内嵌代理模型或 `surrogate_checkpoint` 指定的外部代理模型。

## 必要输入

调用 `generate_candidates` 前，从用户请求中提取：

- `condition.chip_length`
- `condition.Rjc`
- `condition.Rjb`
- `condition.power`
- `condition.wind_speed`
- `bbox.base_width`
- `bbox.base_depth`
- `bbox.total_height`
- `temp_threshold` 或 `temp_limit`

常用可选字段：

- `method`：`threshold-cvae`、`cvae` 或 `diffusion`。
- `checkpoint_path`：生成模型 checkpoint。
- `surrogate_checkpoint`：温度代理模型 checkpoint。
- `device`：例如 `cpu` 或 `cuda`。
- `top_k`：未指定时使用配置文件默认值。
- `candidate_pool_size`：未指定时使用配置文件默认值。
- `route`：可选，默认 `api`；用户明确要求本地源码推理时使用 `local`。

排序与工程扰动可选字段：

- `diversity_rerank_weight`：多样性重排序权重；`0` 表示关闭 rerank。
- `diversity_temp_tolerance`：rerank 时允许的温度窗口。
- `engineering_variant_mode`：`off`、`auto` 或 `on`；默认来自配置文件，当前为 `auto`。
- `engineering_variant_count_per_candidate`
- `engineering_variant_max_trials`
- `engineering_variant_scale`
- `engineering_variant_required_temp_margin`

如果缺少必要字段，且不能安全推断，只追问一个最关键的问题。不要自行编造热边界条件。

## 调用流程

1. 新的设计生成请求：调用 `generate_candidates`。
2. 用户询问某个方案温度或要求预测：调用 `predict_temperature`。
3. 用户要求比较、排序或重排已有候选：调用 `score_candidates`。
4. 用户用自然语言要求改型，例如“鳍片更薄一点”“间距调大”：调用 `refine_candidate`。
5. 用户要求仿真、验证、CFD、求解或准备仿真输入：调用 `validate_candidates`。
6. 用户要求导出 JSON、CSV 或仿真输入：调用 `export_candidates`。

优先直接调用 MCP tools，不要绕过 MCP 去 shell 执行推理脚本。需要切换执行方式时，在同一个 MCP tool 请求中设置 `route`。

## 请求结构

生成和预测类工具使用以下 `request` 结构：

```json
{
  "condition": {
    "chip_length": 35.0,
    "Rjc": 0.6,
    "Rjb": 1.1,
    "power": 85.0,
    "wind_speed": 4.0
  },
  "bbox": {
    "base_width": 40.0,
    "base_depth": 40.0,
    "total_height": 20.0
  },
  "temp_threshold": 80.0,
  "top_k": 10,
  "candidate_pool_size": 1024
}
```

单次调用覆盖默认模型示例：

```json
{
  "method": "diffusion",
  "checkpoint_path": "AIInverseDesign/outputs_conditional_diffusion/heatsink/best_model.pt",
  "surrogate_checkpoint": "AIInverseDesign/outputs_surrogate/heatsink/surrogate.pt",
  "device": "cpu",
  "request": {
    "condition": {
      "chip_length": 35.0,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85.0,
      "wind_speed": 4.0
    },
    "bbox": {
      "base_width": 40.0,
      "base_depth": 40.0,
      "total_height": 20.0
    },
    "temp_threshold": 80.0,
    "top_k": 10,
    "candidate_pool_size": 1024
  }
}
```

候选方案几何字段：

```json
{
  "base_width": 40.0,
  "base_depth": 40.0,
  "total_height": 20.0,
  "base_height": 2.5,
  "fin_height": 17.5,
  "fin_thickness": 1.2,
  "fin_clear_spacing": 3.0,
  "fin_break_thickness": 1.5,
  "fin_break_width": 2.0
}
```

## 默认排序与工程扰动

生成候选时默认启用多样性 rerank：

- `diversity_rerank_weight = 0.15`
- `diversity_temp_tolerance = 2.0`

工程扰动默认由配置控制，当前为：

- `engineering_variant_mode = "auto"`

`auto` 只在 Top-K 多样性不足且候选有温度余量时尝试工程扰动；`on` 会强制尝试；`off` 关闭。工程扰动生成的候选可能包含：

- `engineering_variant`
- `variant_parent_pred_cpu_temp`

## 回复方式

候选方案优先用紧凑表格返回，字段包括：

- rank
- `threshold_ok` 或可行性
- `pred_cpu_temp`
- `temp_margin`
- `base_height`
- `fin_height`
- `fin_thickness`
- `fin_clear_spacing`
- `fin_break_thickness`
- `fin_break_width`
- `engineering_variant`，如有

表格后补充两到三条工程观察：最佳温度方案、温度裕量、几何多样性、工程扰动是否参与、用户关注的权衡点。

如果没有候选方案满足阈值，直接说明，并建议增加 `candidate_pool_size`、放宽 `temp_threshold`、增加 `total_height`、提高风速，或进入仿真验证。

## 仿真验证

当用户说“提交验证”“仿真”“求解”“跑 CFD”“准备求解输入”等表达时，使用 `validate_candidates`。

如果没有配置或传入 `simulation_api_url`，说明 MCP tool 当前只返回未来仿真 API 所需的结构化 payload，并不会真的提交 CFD 任务。

对于长耗时求解器集成，使用异步任务表述：

- 创建仿真任务
- 返回 `job_id`
- 查询任务状态
- 获取结果文件

## 约束

- 把 MCP 视为 Agent 的工具层，不要把 MCP 说成模型本体。
- 把后端 AI 模型和仿真求解器视为能力层；默认经 FastAPI 访问，明确指定 `route="local"` 时可用 MCP tool 直接调用本地推理源码。
- 除非 `validate_candidates` 返回真实求解结果，否则不要声称已经完成 CFD 验证。
- 不要把前端 Demo 当成执行路径；前端只用于说明业务功能。
- 回复中保留单位：mm、W、m/s、degC、degC/W。
