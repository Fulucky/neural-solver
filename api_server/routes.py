from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .heatsink_service import predict_candidate_temperature, recommend_size


router = APIRouter()
log = logging.getLogger("NeuralSolverAPI")


@router.get("/health")
async def health_check() -> str:
    """最轻量的连通性检查，不加载模型。"""

    log.info("health check")
    return "Success!"


@router.post("/test")
async def test_endpoint() -> dict[str, str]:
    """用于确认 POST 请求链路正常，不触发模型推理。"""

    return {"status": "success", "message": "AI selection test"}


@router.post("/aiSelectionInfer")
async def ai_selection_infer(request: Request):
    try:
        from AIHeatsinkSelection.run_controller import start_ai_selection_infer

        data = await request.json()
        # AIHeatsinkSelection 是正向选择流程，模型目录和结果目录由调用方传入。
        models_path = data.get("models_path") or "/home/ma-user/work/AIHeatsinkSelection/models"
        results_path = data.get("results_path") or "/home/ma-user/work/AIHeatsinkSelection"
        input_argv = data.get("input_argv")

        await asyncio.to_thread(start_ai_selection_infer, models_path, results_path, input_argv)
        return {"status": "success", "message": "AI selection inference completed"}
    except Exception as exc:
        log.exception("AI selection inference failed")
        return JSONResponse(status_code=400, content={"error": f"AI selection inference failed: {exc}"})


@router.post("/recommendSize")
@router.post("/heatsink/recommend-size")
async def recommend_size_endpoint(request: Request):
    try:
        # AIHeatsinkInverseDesign 逆向推荐：根据工况、外包络和温度阈值生成候选几何尺寸。
        data = await request.json()
        return await asyncio.to_thread(recommend_size, data)
    except Exception as exc:
        log.exception("heatsink size recommendation failed")
        return JSONResponse(status_code=400, content={"error": f"heatsink size recommendation failed: {exc}"})


@router.post("/predictTemperature")
@router.post("/heatsink/predict-temperature")
async def predict_temperature_endpoint(request: Request):
    try:
        # AIHeatsinkInverseDesign 正向温度预测：给定候选几何尺寸，返回预测 CPU 温度。
        data = await request.json()
        return await asyncio.to_thread(predict_candidate_temperature, data)
    except Exception as exc:
        log.exception("temperature prediction failed")
        return JSONResponse(status_code=400, content={"error": f"temperature prediction failed: {exc}"})
