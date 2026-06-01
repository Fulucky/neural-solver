"""逆向设计推理配置读写工具。

这个模块只管理“默认使用哪条技术路径、哪个 checkpoint、哪个设备”等运行配置。
API、MCP 本地工具和命令行脚本都读取同一份 JSON，避免默认模型散落在多处。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH_ENV = "HEATSINK_INFERENCE_CONFIG"
METHOD_ENV = "HEATSINK_INFERENCE_METHOD"
LEGACY_CHECKPOINT_ENV = "HEATSINK_THRESHOLD_CVAE_CHECKPOINT"
CHECKPOINT_ENV = "HEATSINK_INFERENCE_CHECKPOINT"
DEVICE_ENV = "HEATSINK_API_DEVICE"

DEFAULT_CONFIG_PATH = REPO_ROOT / "AIInverseDesign" / "config" / "inference_config.json"
SUPPORTED_METHODS = ("cvae", "threshold-cvae", "diffusion")

DEFAULT_CHECKPOINTS = {
    "cvae": "AIInverseDesign/outputs_thresholdfree_cvae/heatsink/best_model.pt",
    "threshold-cvae": "AIInverseDesign/outputs_guided_cvae/heatsink/best_model.pt",
    "diffusion": "AIInverseDesign/outputs_conditional_diffusion/heatsink/best_model.pt",
}


@dataclass(frozen=True)
class InferenceConfig:
    method: str
    checkpoint_path: str
    surrogate_checkpoint: str
    device: str
    num_samples: int
    top_k: int
    latent_opt_steps: int
    latent_lr: float
    temperature_weight: float
    threshold_weight: float
    guidance_scale: float
    diversity_rerank_weight: float
    diversity_temp_tolerance: float
    engineering_variant_mode: str
    engineering_variant_count_per_candidate: int
    engineering_variant_max_trials: int
    engineering_variant_scale: float
    engineering_variant_required_temp_margin: float
    engineering_variant_min_unique_ratio: float
    engineering_variant_min_norm_mean_dist: float
    engineering_variant_min_norm_min_dist: float


def config_path() -> Path:
    """返回当前启用的配置文件路径。"""

    return Path(os.getenv(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH).expanduser()


def _default_data() -> dict[str, Any]:
    return {
        "method": "threshold-cvae",
        "checkpoint_path": DEFAULT_CHECKPOINTS["threshold-cvae"],
        "surrogate_checkpoint": "",
        "device": "cpu",
        "num_samples": 1024,
        "top_k": 10,
        "latent_opt_steps": 40,
        "latent_lr": 5e-2,
        "temperature_weight": 1.0,
        "threshold_weight": 2.0,
        "guidance_scale": 0.08,
        "diversity_rerank_weight": 0.15,
        "diversity_temp_tolerance": 2.0,
        "engineering_variant_mode": "auto",
        "engineering_variant_count_per_candidate": 2,
        "engineering_variant_max_trials": 20,
        "engineering_variant_scale": 0.08,
        "engineering_variant_required_temp_margin": 1.0,
        "engineering_variant_min_unique_ratio": 0.8,
        "engineering_variant_min_norm_mean_dist": 1.0,
        "engineering_variant_min_norm_min_dist": 0.3,
    }


def _resolve_path(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _validate_method(method: str) -> str:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported inverse design method: {method}")
    return method


def default_checkpoint_for_method(method: str) -> str:
    """返回某条技术路径的默认 checkpoint 绝对路径。"""

    return _resolve_path(DEFAULT_CHECKPOINTS[_validate_method(method)])


def read_config_data(path: str | Path | None = None) -> dict[str, Any]:
    """读取配置文件；不存在时返回内置默认值。"""

    target = Path(path).expanduser() if path else config_path()
    data = _default_data()
    if target.exists():
        with target.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        data.update({key: value for key, value in loaded.items() if value is not None})
    return data


def load_inference_config(path: str | Path | None = None) -> InferenceConfig:
    """读取配置并应用环境变量覆盖。

    环境变量优先级最高，便于临时调试；普通情况下推荐改 JSON 文件。
    """

    data = read_config_data(path)
    method = _validate_method(os.getenv(METHOD_ENV) or str(data["method"]))
    checkpoint = (
        os.getenv(CHECKPOINT_ENV)
        or os.getenv(LEGACY_CHECKPOINT_ENV)
        or str(data.get("checkpoint_path") or DEFAULT_CHECKPOINTS[method])
    )
    if not checkpoint:
        checkpoint = DEFAULT_CHECKPOINTS[method]

    return InferenceConfig(
        method=method,
        checkpoint_path=_resolve_path(checkpoint),
        surrogate_checkpoint=_resolve_path(str(data.get("surrogate_checkpoint") or "")),
        device=os.getenv(DEVICE_ENV) or str(data.get("device") or "cpu"),
        num_samples=int(data.get("num_samples") or 1024),
        top_k=int(data.get("top_k") or 10),
        latent_opt_steps=int(data.get("latent_opt_steps") or 40),
        latent_lr=float(data.get("latent_lr") or 5e-2),
        temperature_weight=float(data.get("temperature_weight") or 1.0),
        threshold_weight=float(data.get("threshold_weight") or 2.0),
        guidance_scale=float(data.get("guidance_scale") or 0.08),
        diversity_rerank_weight=float(data.get("diversity_rerank_weight") or 0.15),
        diversity_temp_tolerance=float(data.get("diversity_temp_tolerance") or 2.0),
        engineering_variant_mode=str(data.get("engineering_variant_mode") or "auto"),
        engineering_variant_count_per_candidate=int(data.get("engineering_variant_count_per_candidate") or 2),
        engineering_variant_max_trials=int(data.get("engineering_variant_max_trials") or 20),
        engineering_variant_scale=float(data.get("engineering_variant_scale") or 0.08),
        engineering_variant_required_temp_margin=float(data.get("engineering_variant_required_temp_margin") or 1.0),
        engineering_variant_min_unique_ratio=float(data.get("engineering_variant_min_unique_ratio") or 0.8),
        engineering_variant_min_norm_mean_dist=float(data.get("engineering_variant_min_norm_mean_dist") or 1.0),
        engineering_variant_min_norm_min_dist=float(data.get("engineering_variant_min_norm_min_dist") or 0.3),
    )


def write_config_data(data: dict[str, Any], path: str | Path | None = None) -> Path:
    """写入配置文件，保持 UTF-8 和缩进，方便人工查看。"""

    target = Path(path).expanduser() if path else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return target


def update_config(updates: dict[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    """更新配置文件并返回更新后的配置字典。"""

    data = read_config_data(path)
    method = updates.get("method")
    if method is not None:
        updates["method"] = _validate_method(str(method))
        if not updates.get("checkpoint_path"):
            updates["checkpoint_path"] = DEFAULT_CHECKPOINTS[str(method)]

    for key, value in updates.items():
        if value is not None:
            data[key] = value

    write_config_data(data, path)
    return data
