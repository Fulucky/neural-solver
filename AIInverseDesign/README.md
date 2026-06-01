# AIInverseDesign 推理脚本说明

`AIInverseDesign` 是从上层工程中抽出的散热器逆向设计推理运行子工程，用来承载推荐尺寸、温度预测和 Agent/MCP 调用所需的最小运行代码。

当前目录只关注推理和 Agent 调用，不包含训练入口脚本、数据处理流程、实验分析脚本或前端页面。`common/` 中保留少量与 checkpoint 兼容相关的公共 helper，运行时入口只使用推理、加载、评分和导出能力。

## 目录内容

```text
AIInverseDesign/
  common/
    data_adapter.py
    experiment_config.py
    heatsink_inverse_common.py
    models.py
  infer/
    infer.py
    cvae_inferencer.py
    guided_cvae_inferencer.py
    diffusion_inferencer.py
  agent/
    mcp/
    tools/
    skills/
    prompts/
    requirements.txt
  README_THREE_PATHS.md
```

主要文件说明：

- `infer/infer.py`：统一推理入口，根据 `--method` 分发到三种技术路径。
- `infer/cvae_inferencer.py`：阈值无关 CVAE 推理。
- `infer/guided_cvae_inferencer.py`：温度阈值条件 CVAE 推理。
- `infer/diffusion_inferencer.py`：条件扩散模型推理。
- `common/heatsink_inverse_common.py`：checkpoint 加载、候选评分、温度预测、结果导出等公共逻辑。
- `common/data_adapter.py`：工况、外包络、几何尺寸和模型张量之间的数据适配。
- `common/models.py`：ForwardMLP、CVAE、DiffusionDenoiser 等模型定义。
- `agent/`：MCP Server、tools、skills 和 prompts，供 Agent 对话调用。

## 支持的三种技术路径

```text
cvae
threshold-cvae
diffusion
```

三者含义：

- `cvae`：阈值无关 CVAE。生成器条件为 `condition + bbox`，温度阈值在推理阶段通过 latent optimization 和 surrogate 排序过滤生效。
- `threshold-cvae`：温度阈值条件 CVAE。生成器条件为 `condition + bbox + temp_threshold`，推荐作为当前默认路径。
- `diffusion`：条件扩散生成器。通过 reverse sampling、surrogate guidance 和最终排序过滤满足温度阈值。

## Checkpoint 默认位置

```text
cvae:           AIInverseDesign/outputs_thresholdfree_cvae/heatsink/best_model.pt
threshold-cvae: AIInverseDesign/outputs_guided_cvae/heatsink/best_model.pt
diffusion:      AIInverseDesign/outputs_conditional_diffusion/heatsink/best_model.pt
```

如果 checkpoint 不在默认路径，请在命令行中通过 `--checkpoint-path` 指定。

## 命令行推理

在项目根目录执行：

```cmd
cd /d D:\ZhouWJ\InverseDesign
python -m AIInverseDesign.infer.infer --method threshold-cvae -- ^
  --checkpoint-path AIInverseDesign\outputs_guided_cvae\heatsink\best_model.pt ^
  --output-csv threshold_heatsink_candidates.csv ^
  --num-samples 1024 ^
  --top-k 20 ^
  --temp-threshold 80 ^
  --chip-length 35 ^
  --rjc 0.6 ^
  --rjb 1.1 ^
  --power 85 ^
  --wind-speed 4 ^
  --base-width 40 ^
  --base-depth 40 ^
  --total-height 20
```

切换技术路径时，只需要替换 `--method` 和对应 checkpoint。

## 输入字段

推荐尺寸和温度预测都需要工况与外包络：

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
    "temp_threshold": 80,
    "top_k": 5,
    "candidate_pool_size": 64
  }
}
```

温度预测还需要候选几何：

```json
{
  "geometry": {
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
}
```

注意字段名使用 `fin_clear_spacing`，不是 `fin_spacing`。

## 输出字段

候选方案会包含：

```text
rank
pred_cpu_temp
temp_threshold
threshold_ok
temp_margin
is_feasible
base_width
base_depth
total_height
base_height
fin_height
fin_thickness
fin_clear_spacing
fin_break_thickness
fin_break_width
```

排序规则：

```text
1. threshold_ok 为 true 的方案优先
2. pred_cpu_temp 更低优先
3. fin_height 更低优先
```
