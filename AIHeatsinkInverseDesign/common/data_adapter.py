"""
数据适配器：将现有散热器数据格式适配到 CVAE_V2 框架

现有数据字段保持不变，通过这个适配器映射到模型需要的格式
"""

import torch
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import random
import math

LOGGER = logging.getLogger(__name__)

# 现有数据的字段名称
CONDITION_KEYS = [
    "chip_length",
    "Rjc",
    "Rjb",
    "power",
    "wind_speed",
]

# 现有数据的几何字段
GEOMETRY_KEYS = [
    "base_width",
    "base_depth",
    "base_height",
    "fin_height",
    "fin_thickness",
    "fin_clear_spacing",
    "fin_break_thickness",
    "fin_break_width",
]

# 用户给定的约束（推理时）
BOX_KEYS = [
    "base_width",
    "base_depth",
    "total_height",
]

# 模型需要生成的几何参数（保持现有字段名，但不包括 base_height）
# base_height 将由 total_height - fin_height 计算
RECOMMEND_KEYS = [
    "fin_height",
    "fin_thickness",
    "fin_clear_spacing",
    "fin_break_thickness",
    "fin_break_width",
]

TARGET_KEY = "cpu_temp"

# 几何参数边界
GEOMETRY_BOUNDS = {
    "base_width": (20.0, 60.0),
    "base_depth": (20.0, 60.0),
    "base_height": (1.5, 3.0),
    "fin_height": (3.0, 26.0),
    "fin_thickness": (0.8, 1.6),
    "fin_clear_spacing": (0.4, 4.2),
    "fin_break_thickness": (1.0, 3.0),
    "fin_break_width": (1.0, 3.0),
    "total_height": (4.5, 29.0),
}

FIN_SPACING_PITCH_BOUNDS = (2.0, 5.0)


class ForwardDataset(torch.utils.data.Dataset):
    """前向模型数据集"""
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], self.y[idx]


class InverseDataset(torch.utils.data.Dataset):
    """逆向模型数据集"""
    def __init__(self, cond: torch.Tensor, target_geom: torch.Tensor) -> None:
        self.cond = cond
        self.target_geom = target_geom

    def __len__(self) -> int:
        return len(self.cond)

    def __getitem__(self, idx: int) -> tuple:
        return self.cond[idx], self.target_geom[idx]


class StandardScaler:
    def __init__(self) -> None:
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(self, values: torch.Tensor) -> "StandardScaler":
        self.mean = values.mean(dim=0)
        self.std = values.std(dim=0).clamp_min(1e-6)
        return self

    def _require_fitted(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before use.")
        return self.mean, self.std

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        mean_raw, std_raw = self._require_fitted()
        mean = mean_raw.to(values.device)
        std = std_raw.to(values.device)
        return (values - mean) / std

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        mean_raw, std_raw = self._require_fitted()
        mean = mean_raw.to(values.device)
        std = std_raw.to(values.device)
        return values * std + mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        mean, std = self._require_fitted()
        return {"mean": mean, "std": std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "StandardScaler":
        scaler = cls()
        scaler.mean = state["mean"]
        scaler.std = state["std"]
        return scaler


def load_json_samples(path: str | Path) -> List[dict]:
    """加载现有格式的 JSON 数据"""
    path = Path(path)
    if path.is_dir():
        samples: List[dict] = []
        for file in sorted(path.glob("*.json")):
            with file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                samples.extend(payload)
            else:
                samples.append(payload)
        return normalize_geometry_samples(samples)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload if isinstance(payload, list) else [payload]
    return normalize_geometry_samples(samples)


def normalize_geometry_samples(samples: List[dict]) -> List[dict]:
    """Normalize legacy pitch spacing into decoupled clear spacing."""
    for sample in samples:
        geometry = sample.get("geometry", {})
        if "fin_clear_spacing" not in geometry and "fin_spacing" in geometry:
            geometry["fin_clear_spacing"] = float(geometry["fin_spacing"]) - float(geometry["fin_thickness"])
    return samples


def tensorize(samples: List[dict], keys: List[str], root_key: str) -> torch.Tensor:
    """从样本中提取指定字段并转换为张量"""
    rows = []
    for sample in samples:
        row = []
        for key in keys:
            value = sample[root_key][key]
            # 处理 None 值
            if value is None:
                value = 0.0
            row.append(float(value))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32)


def tensorize_target(samples: List[dict]) -> torch.Tensor:
    """提取性能目标（cpu_temp）"""
    rows = [[float(sample["performance"][TARGET_KEY])] for sample in samples]
    return torch.tensor(rows, dtype=torch.float32)


def extract_heatsink_ids(samples: List[dict]) -> List[str]:
    """提取散热器 ID"""
    return [str(sample["heatsink"]) for sample in samples]


def make_group_split(
    samples: List[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """按散热器分组划分数据"""
    rng = random.Random(seed)
    heatsink_ids = sorted(set(extract_heatsink_ids(samples)))
    rng.shuffle(heatsink_ids)

    n_test = max(1, math.ceil(len(heatsink_ids) * test_ratio))
    test_ids = set(heatsink_ids[:n_test])

    train_samples = [s for s in samples if s["heatsink"] not in test_ids]
    test_samples = [s for s in samples if s["heatsink"] in test_ids]
    return train_samples, test_samples


def build_forward_inputs(samples: List[dict]) -> torch.Tensor:
    """
    构建前向模型输入
    输入: [condition(5) + bbox(3) + geometry(5)]
    输出: cpu_temp
    注意: base_height 由 total_height - fin_height 计算，不作为显式输入
    """
    # 工况条件 (5维)
    cond = tensorize(samples, CONDITION_KEYS, "condition")

    # BBox (3维) - 用户给定的约束
    bbox = torch.tensor([
        [
            float(sample["geometry"]["base_width"]),
            float(sample["geometry"]["base_depth"]),
            float(sample["derived"]["total_height"]),
        ]
        for sample in samples
    ], dtype=torch.float32)

    # 几何参数 (5维) - 不包括 base_height
    geom = torch.tensor([
        [
            float(sample["geometry"]["fin_height"]),
            float(sample["geometry"]["fin_thickness"]),
            float(sample["geometry"]["fin_clear_spacing"]),
            float(sample["geometry"]["fin_break_thickness"]),
            float(sample["geometry"]["fin_break_width"]),
        ]
        for sample in samples
    ], dtype=torch.float32)

    return torch.cat([cond, bbox, geom], dim=1)


def build_inverse_condition_inputs(samples: List[dict]) -> torch.Tensor:
    """
    构建逆向模型条件输入
    输入: [condition(5) + bbox(3) + temp_limit(1)]
    输出: geometry(6)
    """
    # 工况条件 (5维)
    cond = tensorize(samples, CONDITION_KEYS, "condition")

    # BBox (3维)
    bbox = torch.tensor([
        [
            float(sample["geometry"]["base_width"]),
            float(sample["geometry"]["base_depth"]),
            float(sample["derived"]["total_height"]),
        ]
        for sample in samples
    ], dtype=torch.float32)

    # 温度上限 (1维) - 训练时用实际温度
    return torch.cat([cond, bbox], dim=1)


def build_recommend_targets(samples: List[dict]) -> torch.Tensor:
    """构建逆向模型目标输出"""
    return tensorize(samples, RECOMMEND_KEYS, "geometry")


def build_threshold_condition_inputs(samples: List[dict]) -> torch.Tensor:
    """Build guided-CVAE inputs: condition(5) + bbox(3) + observed cpu_temp(1)."""

    base_cond = build_inverse_condition_inputs(samples)
    temps = tensorize_target(samples)
    return torch.cat([base_cond, temps], dim=1)


def build_threshold_augmented_training_tensors(
    train_samples: List[dict],
    n_threshold_samples: int = 3,
    upper_strategy: str = "global_max",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Expand feasible geometry rows across looser threshold conditions.

    Each observed sample is feasible at its observed temperature and any looser
    threshold up to either the global or per-heatsink maximum training
    temperature.
    """

    if n_threshold_samples <= 0:
        raise ValueError("n_threshold_samples must be positive.")
    if upper_strategy not in {"global_max", "heatsink_max"}:
        raise ValueError("upper_strategy must be one of: global_max, heatsink_max.")

    base_cond = build_inverse_condition_inputs(train_samples)
    geom = build_recommend_targets(train_samples)
    temps = tensorize_target(train_samples).reshape(-1)
    rng = random.Random(seed)
    global_max = float(torch.max(temps).item())
    heatsink_max: Dict[str, float] = {}
    for idx, sample in enumerate(train_samples):
        heatsink = str(sample["heatsink"])
        heatsink_max[heatsink] = max(heatsink_max.get(heatsink, -float("inf")), float(temps[idx].item()))

    cond_rows = []
    geom_rows = []
    observed_rows = []
    for idx, sample in enumerate(train_samples):
        temp = float(temps[idx].item())
        upper = global_max if upper_strategy == "global_max" else heatsink_max[str(sample["heatsink"])]
        thresholds = [temp]
        for _ in range(n_threshold_samples - 1):
            thresholds.append(temp if upper <= temp + 1e-8 else float(rng.uniform(temp, upper)))
        for threshold in thresholds:
            cond_rows.append(torch.cat([base_cond[idx], torch.tensor([threshold], dtype=torch.float32)]))
            geom_rows.append(geom[idx])
            observed_rows.append(torch.tensor([temp], dtype=torch.float32))

    stats = {
        "threshold_rows": float(len(cond_rows)),
        "threshold_samples_per_layout": float(n_threshold_samples),
        "threshold_min": float(torch.min(temps).item()),
        "threshold_max": global_max,
    }
    return torch.stack(cond_rows), torch.stack(geom_rows), torch.stack(observed_rows), stats


def build_inference_condition_tensor(
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_limit: float | None = None,
) -> torch.Tensor:
    """Build one raw condition tensor for inference."""

    values = [
        *(float(condition[k]) for k in CONDITION_KEYS),
        *(float(bbox[k]) for k in BOX_KEYS),
    ]
    if temp_limit is not None:
        values.append(float(temp_limit))
    return torch.tensor([values], dtype=torch.float32)


def clip_value(name: str, value: float) -> float:
    """将值裁剪到合法范围内"""
    low, high = GEOMETRY_BOUNDS[name]
    return max(low, min(high, value))

def clip_fin_clear_spacing_for_pitch(fin_thickness: float, fin_clear_spacing: float) -> float:
    """Keep clear spacing valid while preserving legacy fin pitch bounds."""
    clear_low, clear_high = GEOMETRY_BOUNDS["fin_clear_spacing"]
    pitch_low, pitch_high = FIN_SPACING_PITCH_BOUNDS
    low = max(clear_low, pitch_low - fin_thickness)
    high = min(clear_high, pitch_high - fin_thickness)
    if low > high:
        return max(clear_low, min(clear_high, fin_clear_spacing))
    return max(low, min(high, fin_clear_spacing))


def build_full_geometry_dict(
    bbox: Dict[str, float],
    recommend_values: List[float],
) -> Dict[str, float]:
    """
    构建完整的几何字典
    recommend_values: [fin_height, fin_thickness, fin_clear_spacing, fin_break_thickness, fin_break_width]
    base_height 将由 total_height - fin_height 计算
    """
    total_height = float(bbox["total_height"])
    fin_height = float(recommend_values[0])

    # 计算 base_height
    base_height = total_height - fin_height

    # 确保 base_height 在合理范围内
    base_height = clip_value("base_height", base_height)

    # 重新计算 fin_height 以满足约束
    fin_height = total_height - base_height

    geom = {
        "base_width": float(bbox["base_width"]),
        "base_depth": float(bbox["base_depth"]),
        "total_height": total_height,
        "base_height": base_height,
        "fin_height": fin_height,
    }

    # 添加其他几何参数
    geom.update({
        "fin_thickness": float(recommend_values[1]),
        "fin_clear_spacing": float(recommend_values[2]),
        "fin_break_thickness": float(recommend_values[3]),
        "fin_break_width": float(recommend_values[4]),
    })

    # 裁剪到合法范围
    geom["fin_thickness"] = clip_value("fin_thickness", geom["fin_thickness"])
    geom["fin_clear_spacing"] = clip_value("fin_clear_spacing", geom["fin_clear_spacing"])
    geom["fin_clear_spacing"] = clip_fin_clear_spacing_for_pitch(
        geom["fin_thickness"],
        geom["fin_clear_spacing"],
    )
    geom["fin_break_thickness"] = clip_value("fin_break_thickness", geom["fin_break_thickness"])
    geom["fin_break_width"] = clip_value("fin_break_width", geom["fin_break_width"])

    return geom


def print_data_summary(samples: List[dict]):
    """打印数据摘要信息"""
    LOGGER.info("数据集摘要:")
    LOGGER.info("  总样本数: %d", len(samples))
    LOGGER.info("  散热器数量: %d", len(set(extract_heatsink_ids(samples))))
    LOGGER.info("  工况参数: %s", CONDITION_KEYS)
    LOGGER.info("  几何参数: %s", GEOMETRY_KEYS)
    LOGGER.info("  输出目标: %s", TARGET_KEY)

    # 打印散热器分布
    heatsink_counts = {}
    for sample in samples:
        heatsink = sample["heatsink"]
        heatsink_counts[heatsink] = heatsink_counts.get(heatsink, 0) + 1

    LOGGER.info("各散热器样本数分布:")
    for heatsink, count in sorted(heatsink_counts.items()):
        LOGGER.info("  %s: %d 样本", heatsink, count)
