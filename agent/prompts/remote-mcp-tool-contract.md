# 远端散热器逆向设计 MCP Tool 调用说明

本文档面向远端调用侧 Agent，用于说明如何通过远端 MCP URL 正确调用散热器逆向设计 tools。

远端 MCP 连接地址示例：

```text
https://<your-domain>/ai_heat_sink_gener/mcp
```

本服务通过 `main.py` 挂载 MCP，底层 MCP Server 来自：

```text
agent/mcp/heatsink-inverse-design/server.py
```

## 必须使用 local 路由

远端部署场景下，MCP tool 不应再调用单独的 FastAPI 推理服务，而应直接调用镜像内本地推理代码。因此调用侧 Agent 必须遵守：

1. 连接 MCP 时使用 Streamable HTTP。
2. MCP URL 使用 `/ai_heat_sink_gener/mcp`。
3. 除 `validate_candidates` 外，所有支持路由的 tools 都显式传入：

```json
{
  "route": "local"
}
```

如果误传：

```json
{
  "route": "api"
}
```

tool 会尝试访问默认推理 API，例如：

```text
http://127.0.0.1:8001/api/candidates/generate
```

在远端镜像没有单独启动该 API 服务时，会出现 connection refused。

虽然 `main.py` 已经默认设置 `HEATSINK_MCP_DEFAULT_ROUTE=local`，但调用侧 Agent 仍建议显式传 `route: "local"`，避免被其它环境变量、客户端缓存或历史请求覆盖。

## MCP Tools 总览

当前只暴露 6 个业务 tools：

| Tool | 用途 | 是否需要 `route: "local"` |
| --- | --- | --- |
| `generate_candidates` | 生成推荐尺寸候选 | 是 |
| `predict_temperature` | 预测单个候选的 CPU 温度 | 是 |
| `score_candidates` | 对多个候选评分和排序 | 是 |
| `refine_candidate` | 根据显式修改或自然语言意图调整候选 | 是 |
| `validate_candidates` | 准备或提交仿真验证 payload | 否，不支持 route |
| `export_candidates` | 导出 JSON / CSV / 仿真输入 | 是 |

推荐调用顺序：

```text
generate_candidates
  -> predict_temperature 或 score_candidates
  -> refine_candidate
  -> validate_candidates
  -> export_candidates
```

并不是每次都必须调用全部 tools。Agent 应根据用户意图选择。

## 通用输入结构

多数 tools 都需要 `request` 字段。`request` 描述热设计工况、外包络和温度阈值。

### request.condition

| 字段 | 类型 | 必填 | 单位 | 含义 |
| --- | --- | --- | --- | --- |
| `chip_length` | number | 是 | mm | 芯片边长或特征长度 |
| `Rjc` | number | 是 | degC/W | 结到壳热阻 |
| `Rjb` | number | 是 | degC/W | 结到板热阻 |
| `power` | number | 是 | W | 芯片功耗 |
| `wind_speed` | number | 是 | m/s | 来流风速 |

### request.bbox

| 字段 | 类型 | 必填 | 单位 | 含义 |
| --- | --- | --- | --- | --- |
| `base_width` | number | 是 | mm | 散热器底座宽度 |
| `base_depth` | number | 是 | mm | 散热器底座深度 |
| `total_height` | number | 是 | mm | 散热器总高度 |

### request 阈值与推荐参数

| 字段 | 类型 | 必填 | 单位 | 含义 |
| --- | --- | --- | --- | --- |
| `temp_threshold` | number | 是 | degC | CPU 温度阈值 |
| `temp_limit` | number | 否 | degC | `temp_threshold` 的别名，二选一 |
| `candidate_pool_size` | integer | 否 | - | 生成候选池大小 |
| `num_samples` | integer | 否 | - | 生成候选池大小，优先级低于 tool 顶层 `num_samples` |
| `top_k` | integer | 否 | - | 返回候选数量，优先级低于 tool 顶层 `top_k` |

标准 `request` 示例：

```json
{
  "condition": {
    "chip_length": 35,
    "Rjc": 0.6,
    "Rjb": 1.1,
    "power": 85,
    "wind_speed": 4
  },
  "bbox": {
    "base_width": 40,
    "base_depth": 40,
    "total_height": 20
  },
  "temp_threshold": 80,
  "candidate_pool_size": 64,
  "top_k": 5
}
```

## 通用候选几何结构

候选几何字段用于 `predict_temperature`、`score_candidates`、`refine_candidate`、`validate_candidates` 和 `export_candidates`。

| 字段 | 类型 | 必填 | 单位 | 含义 |
| --- | --- | --- | --- | --- |
| `base_width` | number | 否 | mm | 底座宽度，通常来自 bbox 或生成结果 |
| `base_depth` | number | 否 | mm | 底座深度，通常来自 bbox 或生成结果 |
| `total_height` | number | 否 | mm | 总高度，通常来自 bbox 或生成结果 |
| `base_height` | number | 否 | mm | 底座高度 |
| `fin_height` | number | 是 | mm | 鳍片高度 |
| `fin_thickness` | number | 是 | mm | 鳍片厚度 |
| `fin_clear_spacing` | number | 是 | mm | 鳍片净间距 |
| `fin_spacing` | number | 否 | mm | `fin_clear_spacing` 的兼容别名 |
| `fin_break_thickness` | number | 是 | mm | 断槽厚度 |
| `fin_break_width` | number | 是 | mm | 断槽宽度 |

候选几何示例：

```json
{
  "base_width": 40,
  "base_depth": 40,
  "total_height": 20,
  "base_height": 2.5,
  "fin_height": 17.5,
  "fin_thickness": 1.2,
  "fin_clear_spacing": 3.0,
  "fin_break_thickness": 1.5,
  "fin_break_width": 2.0
}
```

模型预测或排序后，候选结果通常会额外包含：

| 字段 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `rank` | integer | - | 排名 |
| `pred_cpu_temp` | number | degC | 预测 CPU 温度 |
| `temp_threshold` | number | degC | 温度阈值 |
| `threshold_ok` | boolean | - | 是否满足温度阈值 |
| `temp_margin` | number | degC | 温度裕量，等于 `temp_threshold - pred_cpu_temp` |
| `is_feasible` | boolean | - | 是否可行，通常与 `threshold_ok` 一致 |

## Tool 1: generate_candidates

用途：根据工况、外包络和温度阈值，使用 threshold-CVAE 生成推荐候选尺寸。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request` | object | 是 | 通用输入结构 |
| `route` | string | 是 | 远端部署必须传 `"local"` |
| `checkpoint_path` | string or null | 否 | checkpoint 路径；通常不传，使用服务默认值 |
| `device` | string or null | 否 | 推理设备，例如 `"cpu"` 或 `"cuda"`；通常不传 |
| `num_samples` | integer or null | 否 | 生成候选池大小；建议远端测试先用 64 |
| `top_k` | integer or null | 否 | 返回候选数量 |
| `latent_opt_steps` | integer | 否 | 潜变量优化步数，默认 40 |
| `latent_lr` | number | 否 | 潜变量优化学习率，默认 0.05 |
| `temperature_weight` | number | 否 | 温度目标权重，默认 1.0 |
| `threshold_weight` | number | 否 | 阈值约束权重，默认 2.0 |
| `api_base_url` | string or null | 否 | local 路由下不要传 |

### 调用示例

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80,
    "candidate_pool_size": 64,
    "top_k": 5
  },
  "num_samples": 64,
  "top_k": 5
}
```

### 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `method` | string | 通常为 `"threshold-cvae"` |
| `checkpoint_path` | string | 实际使用的 checkpoint |
| `device` | string | 实际推理设备 |
| `num_samples` | integer | 实际采样数量 |
| `top_k` | integer | 返回候选数量 |
| `temp_threshold` | number | 温度阈值 |
| `candidates` | array | 候选列表，每项为候选几何加预测指标 |

## Tool 2: predict_temperature

用途：对单个候选几何进行温度预测。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request` | object | 是 | 通用输入结构 |
| `geometry` | object | 是 | 单个候选几何 |
| `route` | string | 是 | 远端部署必须传 `"local"` |
| `checkpoint_path` | string or null | 否 | 通常不传 |
| `device` | string or null | 否 | 通常不传 |
| `api_base_url` | string or null | 否 | local 路由下不要传 |

### 调用示例

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "geometry": {
    "fin_height": 17.5,
    "fin_thickness": 1.2,
    "fin_clear_spacing": 3.0,
    "fin_break_thickness": 1.5,
    "fin_break_width": 2.0
  }
}
```

### 输出字段

返回单个候选结果 object，常见字段包括：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `rank` | integer | 排名，通常为 1 |
| `base_width` | number | 底座宽度 |
| `base_depth` | number | 底座深度 |
| `total_height` | number | 总高度 |
| `base_height` | number | 底座高度 |
| `fin_height` | number | 鳍片高度 |
| `fin_thickness` | number | 鳍片厚度 |
| `fin_clear_spacing` | number | 鳍片净间距 |
| `fin_break_thickness` | number | 断槽厚度 |
| `fin_break_width` | number | 断槽宽度 |
| `pred_cpu_temp` | number | 预测 CPU 温度 |
| `temp_threshold` | number | 温度阈值 |
| `threshold_ok` | boolean | 是否满足阈值 |
| `temp_margin` | number | 温度裕量 |
| `is_feasible` | boolean | 是否可行 |

## Tool 3: score_candidates

用途：对多个候选进行温度预测、可行性判断和综合排序。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request` | object | 是 | 通用输入结构 |
| `candidates` | array | 是 | 候选几何列表 |
| `route` | string | 是 | 远端部署必须传 `"local"` |
| `checkpoint_path` | string or null | 否 | 通常不传 |
| `device` | string or null | 否 | 通常不传 |
| `top_k` | integer or null | 否 | 返回前 N 个；不传则返回全部输入候选的排序结果 |
| `api_base_url` | string or null | 否 | local 路由下不要传 |

### 调用示例

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "top_k": 2,
  "candidates": [
    {
      "fin_height": 17.5,
      "fin_thickness": 1.2,
      "fin_clear_spacing": 3.0,
      "fin_break_thickness": 1.5,
      "fin_break_width": 2.0
    },
    {
      "fin_height": 16.8,
      "fin_thickness": 1.0,
      "fin_clear_spacing": 2.6,
      "fin_break_thickness": 1.4,
      "fin_break_width": 2.2
    }
  ]
}
```

### 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `method` | string | 通常为 `"forward-surrogate-ranking"` |
| `temp_threshold` | number | 温度阈值 |
| `candidates` | array | 排序后的候选列表 |

`candidates` 中每一项包含候选几何和预测指标，字段同通用候选结果。

## Tool 4: refine_candidate

用途：根据用户明确修改值或自然语言意图，对单个候选几何进行调整，并重新预测温度。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request` | object | 是 | 通用输入结构 |
| `candidate` | object | 是 | 待修改的候选几何 |
| `route` | string | 是 | 远端部署必须传 `"local"` |
| `updates` | object or null | 否 | 显式字段修改，例如 `{ "fin_thickness": 1.1 }` |
| `instruction` | string | 否 | 自然语言改型意图，例如“鳍片更薄一点，间距调大” |
| `checkpoint_path` | string or null | 否 | 通常不传 |
| `device` | string or null | 否 | 通常不传 |
| `api_base_url` | string or null | 否 | local 路由下不要传 |

支持显式修改的典型字段：

```text
fin_height
fin_thickness
fin_clear_spacing
fin_spacing
fin_break_thickness
fin_break_width
base_height
```

### 调用示例

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "candidate": {
    "fin_height": 17.5,
    "fin_thickness": 1.2,
    "fin_clear_spacing": 3.0,
    "fin_break_thickness": 1.5,
    "fin_break_width": 2.0
  },
  "updates": {
    "fin_thickness": 1.1
  },
  "instruction": "鳍片稍微薄一点"
}
```

### 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `changes` | object | 实际修改字段，包含修改前后值 |
| `candidate` | object | 修改并重新预测后的候选结果 |

`changes` 示例：

```json
{
  "fin_thickness": {
    "from": 1.2,
    "to": 1.1
  }
}
```

`candidate` 字段同通用候选结果。

## Tool 5: validate_candidates

用途：准备或提交仿真验证 payload。

注意：这个 tool 不走 AI 推理，不支持 `route` 字段。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `request` | object | 是 | 通用输入结构 |
| `candidates` | array | 是 | 要验证的候选列表 |
| `simulation_api_url` | string or null | 否 | 仿真服务地址；不传则只返回待提交 payload |
| `timeout_seconds` | number | 否 | 提交仿真 API 的超时时间，默认 10 |

### 调用示例：只生成仿真 payload

```json
{
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "candidates": [
    {
      "rank": 1,
      "fin_height": 17.5,
      "fin_thickness": 1.2,
      "fin_clear_spacing": 3.0,
      "fin_break_thickness": 1.5,
      "fin_break_width": 2.0,
      "pred_cpu_temp": 76.3,
      "temp_threshold": 80,
      "threshold_ok": true
    }
  ]
}
```

### 输出字段：未提交仿真 API

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `"not_submitted"` |
| `message` | string | 未提交原因 |
| `payload` | object | 可提交给仿真服务的结构化 payload |

### 输出字段：提交成功

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `"submitted"` |
| `http_status` | integer | 仿真 API HTTP 状态码 |
| `response` | object | 仿真 API 返回体 |

### 输出字段：提交失败

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `status` | string | `"submit_failed"` |
| `error` | string | 错误信息 |
| `payload` | object | 原始提交 payload |

## Tool 6: export_candidates

用途：导出候选为 JSON、CSV 或仿真输入 JSON。

### 输入字段

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `candidates` | array | 是 | 候选列表 |
| `route` | string | 是 | 远端部署必须传 `"local"` |
| `export_format` | string | 否 | `json`、`csv`、`simulation_input`，默认 `json` |
| `api_base_url` | string or null | 否 | local 路由下不要传 |

### 调用示例：导出 JSON

```json
{
  "route": "local",
  "export_format": "json",
  "candidates": [
    {
      "rank": 1,
      "fin_height": 17.5,
      "fin_thickness": 1.2,
      "fin_clear_spacing": 3.0,
      "fin_break_thickness": 1.5,
      "fin_break_width": 2.0,
      "pred_cpu_temp": 76.3,
      "temp_threshold": 80,
      "threshold_ok": true
    }
  ]
}
```

### 输出字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `format` | string | 导出格式 |
| `filename` | string | 建议文件名 |
| `content` | string | 文件内容字符串 |

不同格式的输出：

| `export_format` | `filename` | `content` |
| --- | --- | --- |
| `json` | `heatsink_candidates.json` | JSON 字符串 |
| `csv` | `heatsink_candidates.csv` | CSV 字符串 |
| `simulation_input` | `heatsink_simulation_input.json` | 包含 `candidates` 的 JSON 字符串 |

## 调用侧 Agent 行为规范

调用侧 Agent 应遵守以下规则：

1. 用户要求“推荐尺寸”“生成方案”“给几个候选”时，调用 `generate_candidates`。
2. 用户给定一个几何方案并询问温度时，调用 `predict_temperature`。
3. 用户给出多个候选并要求比较、评分、排序时，调用 `score_candidates`。
4. 用户要求“鳍片更薄”“间距调大”“高度降低”等改型时，调用 `refine_candidate`。
5. 用户要求“仿真”“验证”“提交求解”“生成 CFD 输入”时，调用 `validate_candidates`。
6. 用户要求“导出”“保存 JSON”“生成 CSV”“给验证集”时，调用 `export_candidates`。
7. 不要把 MCP 说成模型本体；MCP 是 Agent 工具层。
8. 不要声称 `validate_candidates` 已经完成真实 CFD，除非返回了真实仿真 API 的结果。
9. 远端部署下，不要传 `api_base_url`。
10. 远端部署下，除 `validate_candidates` 外，所有 tool 请求必须显式传 `route: "local"`。

## 最小端到端示例

第一步：生成候选。

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80,
    "candidate_pool_size": 64,
    "top_k": 5
  },
  "num_samples": 64,
  "top_k": 5
}
```

第二步：从返回的 `candidates` 里选一个，预测或解释温度。

```json
{
  "route": "local",
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "geometry": {
    "fin_height": 17.5,
    "fin_thickness": 1.2,
    "fin_clear_spacing": 3.0,
    "fin_break_thickness": 1.5,
    "fin_break_width": 2.0
  }
}
```

第三步：用户确认后，准备仿真 payload。

```json
{
  "request": {
    "condition": {
      "chip_length": 35,
      "Rjc": 0.6,
      "Rjb": 1.1,
      "power": 85,
      "wind_speed": 4
    },
    "bbox": {
      "base_width": 40,
      "base_depth": 40,
      "total_height": 20
    },
    "temp_threshold": 80
  },
  "candidates": [
    {
      "rank": 1,
      "fin_height": 17.5,
      "fin_thickness": 1.2,
      "fin_clear_spacing": 3.0,
      "fin_break_thickness": 1.5,
      "fin_break_width": 2.0,
      "pred_cpu_temp": 76.3,
      "temp_threshold": 80,
      "threshold_ok": true
    }
  ]
}
```

## 常见错误

### 错误 1：访问 127.0.0.1:8001 失败

错误示例：

```text
Cannot reach heatsink inference API at http://127.0.0.1:8001/api/candidates/generate
```

原因：tool 走了 `route="api"`。

处理方式：请求里显式传：

```json
{
  "route": "local"
}
```

并确认连接的是：

```text
/ai_heat_sink_gener/mcp
```

### 错误 2：缺少 request 字段

原因：调用侧 Agent 只传了几何或自然语言，没有组织成 MCP tool schema。

处理方式：按本文档的 `request.condition`、`request.bbox`、`temp_threshold` 组织输入。

### 错误 3：缺少 fin_clear_spacing

原因：候选几何缺少鳍片净间距。

处理方式：传入 `fin_clear_spacing`。兼容情况下可传 `fin_spacing`，但推荐统一使用 `fin_clear_spacing`。

### 错误 4：以为 validate_candidates 已完成 CFD

原因：未传 `simulation_api_url` 时，tool 只返回仿真 payload。

处理方式：只有返回 `status: "submitted"` 且仿真 API 返回真实结果时，才能说明已提交仿真。
