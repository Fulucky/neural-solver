# Neural Solver

`Neural Solver` 当前由两个平级业务模块和一个统一 API 层组成：

- `AISelection/`：正向选择/模型评估流程，入口是 `AISelection.run_controller`。
- `AIInverseDesign/`：散热器逆向设计，负责尺寸推荐、候选评分和温度预测。
- `api_server/`：FastAPI 适配层，同时暴露 AISelection 和 AIInverseDesign 的 HTTP 接口。
- `agent/`：Agent、MCP Server、工具脚本和技能描述。

注意：`AIInverseDesign` 不是 `AISelection` 的子模块。代码里通过 `api_server/config.py` 把二者作为仓库根目录下的平级包加入导入路径，API 层只是统一编排它们。

## 本地启动 API

建议在项目根目录执行：

```powershell
cd D:\ZhouWJ\hdp-neural-solver
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

默认监听地址：

```text
http://127.0.0.1:8080
```

如果需要改端口：

```powershell
$env:AI_SELECTION_API_PORT = "8090"
python main.py
```

## 不加载模型的连通性测试

打开另一个 PowerShell：

```powershell
curl.exe http://127.0.0.1:8080/health
```

预期返回：

```text
Success!
```

再测试 POST 链路：

```powershell
curl.exe -X POST http://127.0.0.1:8080/test
```

预期返回：

```json
{"status":"success","message":"AI selection test"}
```

这两个接口只验证 API 服务是否跑通，不会加载 checkpoint。

## 逆向设计推荐接口

接口地址：

```text
POST /recommendSize
POST /heatsink/recommend-size
```

示例：

```powershell
curl.exe -X POST http://127.0.0.1:8080/heatsink/recommend-size `
  -H "Content-Type: application/json" `
  -d "{\"request\":{\"condition\":{\"chip_length\":35,\"Rjc\":0.6,\"Rjb\":1.1,\"power\":85,\"wind_speed\":4},\"bbox\":{\"base_width\":40,\"base_depth\":40,\"total_height\":20},\"temp_threshold\":80,\"top_k\":3,\"candidate_pool_size\":16},\"checkpoint_path\":\"D:\\path\\to\\best_model.pt\",\"device\":\"cpu\"}"
```

说明：

- `condition` 是芯片和环境工况。
- `bbox` 是散热器外包络尺寸。
- `temp_threshold` 或 `temp_limit` 是目标温度上限。
- `checkpoint_path` 指向本地训练好的 `best_model.pt`；如果不传，会使用 `config/inverse_design_inference.json` 里的默认模型。

## 更换逆向设计技术路径和模型

默认配置文件：

```text
AIInverseDesign/config/inference_config.json
```

支持的技术路径：

```text
cvae
threshold-cvae
diffusion
```

查看当前配置：

```powershell
python scripts\configure_inverse_design.py --show
```

切换到 threshold-CVAE：

```powershell
python scripts\configure_inverse_design.py `
  --method threshold-cvae `
  --checkpoint AIInverseDesign/outputs_guided_cvae/heatsink/best_model.pt `
  --device cpu
```

切换到 diffusion：

```powershell
python scripts\configure_inverse_design.py `
  --method diffusion `
  --checkpoint AIInverseDesign/outputs_conditional_diffusion/heatsink/best_model.pt `
  --guidance-scale 0.08 `
  --device cpu
```

切换到 threshold-free CVAE：

```powershell
python scripts\configure_inverse_design.py `
  --method cvae `
  --checkpoint AIInverseDesign/outputs_thresholdfree_cvae/heatsink/best_model.pt `
  --device cpu
```

配置修改后，新的 API 请求会读取新配置。为了避免已经缓存的旧 checkpoint 继续占用内存，建议重启 `python main.py` 进程。

默认配置中：

- `diversity_rerank_weight` 为 `0.15`，表示默认启用多样性 rerank；设为 `0` 可关闭。
- `engineering_variant_mode` 为 `auto`，表示默认启用自动工程扰动兜底；只有 Top-K 多样性不足且候选有温度余量时才会生成扰动变体。
- `engineering_variant_mode` 可选 `off`、`auto`、`on`。`off` 关闭，`on` 强制尝试工程扰动。

修改工程扰动配置示例：

```powershell
python scripts\configure_inverse_design.py `
  --engineering-variant-mode auto `
  --engineering-variant-count-per-candidate 2 `
  --engineering-variant-scale 0.08
```

单次请求也可以临时覆盖配置，不会改 JSON 文件：

```json
{
  "method": "diffusion",
  "checkpoint_path": "D:\\path\\to\\diffusion\\best_model.pt",
  "device": "cpu",
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
    "top_k": 3,
    "candidate_pool_size": 16
  }
}
```

## 温度预测接口

接口地址：

```text
POST /predictTemperature
POST /heatsink/predict-temperature
```

示例：

```powershell
curl.exe -X POST http://127.0.0.1:8080/heatsink/predict-temperature `
  -H "Content-Type: application/json" `
  -d "{\"request\":{\"condition\":{\"chip_length\":35,\"Rjc\":0.6,\"Rjb\":1.1,\"power\":85,\"wind_speed\":4},\"bbox\":{\"base_width\":40,\"base_depth\":40,\"total_height\":20},\"temp_threshold\":80},\"geometry\":{\"base_width\":40,\"base_depth\":40,\"total_height\":20,\"base_height\":2.5,\"fin_height\":17.5,\"fin_thickness\":1.2,\"fin_clear_spacing\":3.0,\"fin_break_thickness\":1.5,\"fin_break_width\":2.0},\"checkpoint_path\":\"D:\\path\\to\\best_model.pt\",\"device\":\"cpu\"}"
```

## AISelection 推理接口

接口地址：

```text
POST /aiSelectionInfer
```

示例：

```powershell
curl.exe -X POST http://127.0.0.1:8080/aiSelectionInfer `
  -H "Content-Type: application/json" `
  -d "{\"models_path\":\"D:\\path\\to\\AISelection\\models\",\"results_path\":\"D:\\path\\to\\results\",\"input_argv\":[\"1225\",\"85\",\"0.6\",\"1.1\",\"4\"]}"
```

`input_argv` 顺序是：

```text
芯片面积、功率、Rjc、Rjb、风速
```

## 常见问题

- 如果 `/health` 通，但推荐/预测接口失败，优先检查 `checkpoint_path` 是否存在。
- 如果提示缺少 `torch`、`fastapi`、`uvicorn`，重新执行 `pip install -r requirements.txt`。
- 如果端口被占用，设置 `$env:AI_SELECTION_API_PORT` 后重启。
- 如果只想验证 API 服务是否启动，不需要准备模型文件，测试 `/health` 和 `/test` 即可。
