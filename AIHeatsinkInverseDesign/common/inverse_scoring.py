"""Candidate scoring, diversity reranking, and engineering variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch

from AIHeatsinkInverseDesign.common.data_adapter import (
    CONDITION_KEYS,
    GEOMETRY_BOUNDS,
    RECOMMEND_KEYS,
    build_full_geometry_dict,
    clip_fin_clear_spacing_for_pitch,
    clip_value,
)

GEOMETRY_DECIMALS = 2
GEOMETRY_OUTPUT_KEYS = (
    "base_width",
    "base_depth",
    "total_height",
    "base_height",
    *RECOMMEND_KEYS,
)


def _predict_temperature_tensor(model, x_scaler, y_scaler, x: torch.Tensor) -> torch.Tensor:
    pred_scaled = model(x_scaler.transform(x))
    return y_scaler.inverse_transform(pred_scaled)


def quantize_geometry_dict(geom: Dict[str, float], decimals: int = GEOMETRY_DECIMALS) -> Dict[str, float]:
    """Round geometry to the precision visible to users before scoring/ranking."""

    bbox = {
        "base_width": round(float(geom["base_width"]), decimals),
        "base_depth": round(float(geom["base_depth"]), decimals),
        "total_height": round(float(geom["total_height"]), decimals),
    }
    recommend = [round(float(geom[name]), decimals) for name in RECOMMEND_KEYS]
    quantized = build_full_geometry_dict(bbox, recommend)
    quantized = {
        key: round(float(value), decimals) if key in GEOMETRY_OUTPUT_KEYS else value
        for key, value in quantized.items()
    }
    for key, value in geom.items():
        if key not in quantized:
            quantized[key] = value
    return quantized


def recommendation_key(row: Dict[str, float], decimals: int = GEOMETRY_DECIMALS) -> tuple[float, ...]:
    return tuple(round(float(row[name]), decimals) for name in RECOMMEND_KEYS)


def score_candidates(
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    geom_rows: Iterable[Iterable[float]],
    temp_threshold: float,
    top_k: int,
    diversity_rerank_weight: float = 0.15,
    diversity_temp_tolerance: float = 2.0,
    engineering_variant_mode: str = "off",
    engineering_variant_count_per_candidate: int = 2,
    engineering_variant_max_trials: int = 20,
    engineering_variant_scale: float = 0.08,
    engineering_variant_required_temp_margin: float = 1.0,
    engineering_variant_min_unique_ratio: float = 0.8,
    engineering_variant_min_norm_mean_dist: float = 1.0,
    engineering_variant_min_norm_min_dist: float = 0.3,
) -> List[Dict[str, float]]:
    rows = score_candidate_pool(payload, condition, bbox, geom_rows, temp_threshold)
    return select_candidates_from_pool(
        rows=rows,
        payload=payload,
        condition=condition,
        bbox=bbox,
        temp_threshold=temp_threshold,
        top_k=top_k,
        diversity_rerank_weight=diversity_rerank_weight,
        diversity_temp_tolerance=diversity_temp_tolerance,
        engineering_variant_mode=engineering_variant_mode,
        engineering_variant_count_per_candidate=engineering_variant_count_per_candidate,
        engineering_variant_max_trials=engineering_variant_max_trials,
        engineering_variant_scale=engineering_variant_scale,
        engineering_variant_required_temp_margin=engineering_variant_required_temp_margin,
        engineering_variant_min_unique_ratio=engineering_variant_min_unique_ratio,
        engineering_variant_min_norm_mean_dist=engineering_variant_min_norm_mean_dist,
        engineering_variant_min_norm_min_dist=engineering_variant_min_norm_min_dist,
    )


def select_candidates_from_pool(
    rows: List[Dict[str, float]],
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_threshold: float,
    top_k: int,
    diversity_rerank_weight: float = 0.15,
    diversity_temp_tolerance: float = 2.0,
    engineering_variant_mode: str = "off",
    engineering_variant_count_per_candidate: int = 2,
    engineering_variant_max_trials: int = 20,
    engineering_variant_scale: float = 0.08,
    engineering_variant_required_temp_margin: float = 1.0,
    engineering_variant_min_unique_ratio: float = 0.8,
    engineering_variant_min_norm_mean_dist: float = 1.0,
    engineering_variant_min_norm_min_dist: float = 0.3,
) -> List[Dict[str, float]]:
    rows.sort(key=lambda item: (not item["threshold_ok"], item["pred_cpu_temp"], item["fin_height"]))
    selected = diversity_rerank_candidates(
        rows,
        payload,
        top_k,
        diversity_rerank_weight=diversity_rerank_weight,
        diversity_temp_tolerance=diversity_temp_tolerance,
    )
    selected = apply_engineering_variants(
        selected=selected,
        pool_rows=rows,
        payload=payload,
        condition=condition,
        bbox=bbox,
        temp_threshold=temp_threshold,
        top_k=top_k,
        diversity_rerank_weight=diversity_rerank_weight,
        diversity_temp_tolerance=diversity_temp_tolerance,
        mode=engineering_variant_mode,
        count_per_candidate=engineering_variant_count_per_candidate,
        max_trials=engineering_variant_max_trials,
        scale=engineering_variant_scale,
        required_temp_margin=engineering_variant_required_temp_margin,
        min_unique_ratio=engineering_variant_min_unique_ratio,
        min_norm_mean_dist=engineering_variant_min_norm_mean_dist,
        min_norm_min_dist=engineering_variant_min_norm_min_dist,
    )
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def score_candidate_pool(
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    geom_rows: Iterable[Iterable[float]],
    temp_threshold: float,
) -> List[Dict[str, float]]:
    rows = []
    seen = set()
    for raw in geom_rows:
        geom = quantize_geometry_dict(build_full_geometry_dict(bbox, list(raw)))
        key = recommendation_key(geom)
        if key in seen:
            continue
        seen.add(key)
        rows.append(score_geometry(payload, condition, geom, temp_threshold))
    return rows


def score_geometry(
    payload: Dict,
    condition: Dict[str, float],
    geom: Dict[str, float],
    temp_threshold: float,
) -> Dict[str, float]:
    device = next(payload["forward_model"].parameters()).device
    row = quantize_geometry_dict(geom)
    x = torch.tensor(
        [[
            *(float(condition[k]) for k in CONDITION_KEYS),
            float(row["base_width"]),
            float(row["base_depth"]),
            float(row["total_height"]),
            float(row["fin_height"]),
            float(row["fin_thickness"]),
            float(row["fin_clear_spacing"]),
            float(row["fin_break_thickness"]),
            float(row["fin_break_width"]),
        ]],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        pred = _predict_temperature_tensor(
            payload["forward_model"],
            payload["forward_input_scaler"],
            payload["target_scaler"],
            x,
        )
    row["pred_cpu_temp"] = float(pred.cpu().item())
    row["temp_threshold"] = float(temp_threshold)
    row["threshold_ok"] = bool(row["pred_cpu_temp"] <= temp_threshold)
    return row


def apply_engineering_variants(
    selected: List[Dict[str, float]],
    pool_rows: List[Dict[str, float]],
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_threshold: float,
    top_k: int,
    diversity_rerank_weight: float,
    diversity_temp_tolerance: float,
    mode: str,
    count_per_candidate: int,
    max_trials: int,
    scale: float,
    required_temp_margin: float,
    min_unique_ratio: float,
    min_norm_mean_dist: float,
    min_norm_min_dist: float,
) -> List[Dict[str, float]]:
    if mode == "off" or top_k <= 0 or not selected:
        return selected
    if mode not in {"auto", "on"}:
        raise ValueError("--engineering-variant-mode must be one of: off, auto, on.")
    if mode == "auto" and not needs_engineering_variants(
        selected,
        payload,
        top_k,
        min_unique_ratio,
        min_norm_mean_dist,
        min_norm_min_dist,
    ):
        return selected
    if mode == "auto" and not has_variant_temperature_margin(selected, required_temp_margin):
        return selected

    variants = generate_engineering_variants(
        selected=selected,
        payload=payload,
        condition=condition,
        bbox=bbox,
        temp_threshold=temp_threshold,
        count_per_candidate=count_per_candidate,
        max_trials=max_trials,
        scale=scale,
        required_temp_margin=required_temp_margin,
    )
    if not variants:
        return selected

    merged = []
    seen = set()
    for row in [*pool_rows, *variants]:
        key = recommendation_key(row)
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    merged.sort(key=lambda item: (not item["threshold_ok"], item["pred_cpu_temp"], item["fin_height"]))
    selected_with_variants = diversity_rerank_candidates(
        merged,
        payload,
        top_k,
        diversity_rerank_weight=diversity_rerank_weight,
        diversity_temp_tolerance=diversity_temp_tolerance,
    )
    for row in selected_with_variants:
        row.setdefault("engineering_variant", False)
    return selected_with_variants


def needs_engineering_variants(
    rows: List[Dict[str, float]],
    payload: Dict,
    top_k: int,
    min_unique_ratio: float,
    min_norm_mean_dist: float,
    min_norm_min_dist: float,
) -> bool:
    unique_ratio = geometry_unique_count(rows) / max(top_k, 1)
    metrics = normalized_geometry_metrics(rows, payload)
    return (
        unique_ratio < min_unique_ratio
        or metrics["normalized_mean_pairwise_distance"] < min_norm_mean_dist
        or metrics["normalized_min_pairwise_distance"] < min_norm_min_dist
    )


def has_variant_temperature_margin(rows: List[Dict[str, float]], required_temp_margin: float) -> bool:
    if required_temp_margin <= 0:
        return any(bool(row["threshold_ok"]) for row in rows)
    return any(float(row["temp_threshold"]) - float(row["pred_cpu_temp"]) >= required_temp_margin for row in rows)


def generate_engineering_variants(
    selected: List[Dict[str, float]],
    payload: Dict,
    condition: Dict[str, float],
    bbox: Dict[str, float],
    temp_threshold: float,
    count_per_candidate: int,
    max_trials: int,
    scale: float,
    required_temp_margin: float,
) -> List[Dict[str, float]]:
    variants: List[Dict[str, float]] = []
    if count_per_candidate <= 0 or max_trials <= 0 or scale <= 0:
        return variants
    for base_idx, base in enumerate(selected):
        if not bool(base["threshold_ok"]):
            continue
        if float(base["temp_threshold"]) - float(base["pred_cpu_temp"]) < required_temp_margin:
            continue
        accepted = 0
        for trial in range(max_trials):
            if accepted >= count_per_candidate:
                break
            candidate = perturb_geometry(base, bbox, scale, base_idx, trial)
            scored = score_geometry(payload, condition, candidate, temp_threshold)
            if not bool(scored["threshold_ok"]):
                continue
            scored["engineering_variant"] = True
            scored["variant_parent_pred_cpu_temp"] = float(base["pred_cpu_temp"])
            variants.append(scored)
            accepted += 1
    return variants


def perturb_geometry(
    base: Dict[str, float],
    bbox: Dict[str, float],
    scale: float,
    base_idx: int,
    trial: int,
) -> Dict[str, float]:
    values = {name: float(base[name]) for name in RECOMMEND_KEYS}
    fields = [
        "fin_height",
        "fin_clear_spacing",
        "fin_thickness",
        "fin_break_thickness",
        "fin_break_width",
    ]
    field = fields[(base_idx + trial) % len(fields)]
    direction = -1.0 if ((base_idx + trial) % 2) else 1.0
    low, high = GEOMETRY_BOUNDS[field]
    step = max((high - low) * scale, 1e-6)
    values[field] = values[field] + direction * step
    values["fin_height"] = clip_value("fin_height", values["fin_height"])
    values["fin_thickness"] = clip_value("fin_thickness", values["fin_thickness"])
    values["fin_clear_spacing"] = clip_fin_clear_spacing_for_pitch(
        values["fin_thickness"],
        values["fin_clear_spacing"],
    )
    values["fin_break_thickness"] = clip_value("fin_break_thickness", values["fin_break_thickness"])
    values["fin_break_width"] = clip_value("fin_break_width", values["fin_break_width"])
    return quantize_geometry_dict(build_full_geometry_dict(bbox, [values[name] for name in RECOMMEND_KEYS]))


def diversity_rerank_candidates(
    rows: List[Dict[str, float]],
    payload: Dict,
    top_k: int,
    diversity_rerank_weight: float = 0.15,
    diversity_temp_tolerance: float = 2.0,
) -> List[Dict[str, float]]:
    if top_k <= 0 or not rows:
        return []
    if diversity_rerank_weight <= 0.0 or top_k == 1:
        return rows[:top_k]

    ok_rows = [row for row in rows if row["threshold_ok"]]
    fallback_rows = [row for row in rows if not row["threshold_ok"]]
    selected = _select_diverse_group(
        ok_rows,
        payload,
        min(top_k, len(ok_rows)),
        diversity_rerank_weight,
        diversity_temp_tolerance,
    )
    if len(selected) < top_k:
        selected.extend(
            _select_diverse_group(
                fallback_rows,
                payload,
                top_k - len(selected),
                diversity_rerank_weight,
                diversity_temp_tolerance,
            )
        )
    return selected[:top_k]


def _select_diverse_group(
    rows: List[Dict[str, float]],
    payload: Dict,
    top_k: int,
    diversity_rerank_weight: float,
    diversity_temp_tolerance: float,
) -> List[Dict[str, float]]:
    if top_k <= 0 or not rows:
        return []
    if len(rows) <= top_k:
        return rows[:]

    vectors = _normalized_geometry_vectors(rows, payload)
    selected_indices = [0]
    remaining = set(range(1, len(rows)))
    best_temp = float(rows[0]["pred_cpu_temp"])
    temp_span = max(
        max(float(row["pred_cpu_temp"]) for row in rows) - best_temp,
        abs(float(diversity_temp_tolerance)),
        1e-6,
    )

    while len(selected_indices) < top_k and remaining:
        preferred = [
            idx
            for idx in remaining
            if diversity_temp_tolerance <= 0.0
            or float(rows[idx]["pred_cpu_temp"]) <= best_temp + diversity_temp_tolerance
        ]
        pool = preferred or list(remaining)
        best_idx = min(
            pool,
            key=lambda idx: _diverse_candidate_cost(
                rows,
                vectors,
                idx,
                selected_indices,
                best_temp,
                temp_span,
                diversity_rerank_weight,
            ),
        )
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [rows[idx] for idx in selected_indices]


def _normalized_geometry_vectors(rows: List[Dict[str, float]], payload: Dict) -> torch.Tensor:
    device = next(payload["forward_model"].parameters()).device
    raw = torch.tensor(
        [[float(row[name]) for name in RECOMMEND_KEYS] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    return payload["recommend_scaler"].transform(raw)


def geometry_unique_count(rows: List[Dict[str, float]]) -> int:
    return len({recommendation_key(row) for row in rows})


def normalized_geometry_metrics(rows: List[Dict[str, float]], payload: Dict) -> Dict[str, float]:
    if len(rows) < 2:
        return {
            "normalized_mean_pairwise_distance": 0.0,
            "normalized_min_pairwise_distance": 0.0,
        }
    vectors = _normalized_geometry_vectors(rows, payload)
    dists = torch.pdist(vectors, p=2)
    if dists.numel() == 0:
        return {
            "normalized_mean_pairwise_distance": 0.0,
            "normalized_min_pairwise_distance": 0.0,
        }
    return {
        "normalized_mean_pairwise_distance": float(dists.mean().item()),
        "normalized_min_pairwise_distance": float(dists.min().item()),
    }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    q = max(0.0, min(1.0, q))
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return float(ordered[lower])
    upper_weight = pos - lower
    lower_weight = 1.0 - upper_weight
    return float(ordered[lower] * lower_weight + ordered[upper] * upper_weight)


def candidate_pool_summary(rows: List[Dict[str, float]], payload: Dict) -> Dict[str, float]:
    if not rows:
        return {
            "pool_candidate_count": 0,
            "pool_threshold_ok_count": 0,
            "pool_threshold_ok_rate": 0.0,
            "pool_best_pred_cpu_temp": 0.0,
            "pool_p10_pred_cpu_temp": 0.0,
            "pool_median_pred_cpu_temp": 0.0,
            "pool_p90_pred_cpu_temp": 0.0,
            "pool_threshold_margin_mean": 0.0,
            "pool_threshold_margin_min": 0.0,
            "pool_unique_count": 0,
            "pool_mean_pairwise_geometry_distance": 0.0,
            "pool_min_pairwise_geometry_distance": 0.0,
            "pool_normalized_mean_pairwise_distance": 0.0,
            "pool_normalized_min_pairwise_distance": 0.0,
        }
    ok_count = sum(1 for row in rows if bool(row["threshold_ok"]))
    pred_values = [float(row["pred_cpu_temp"]) for row in rows]
    threshold_margins = [float(row["temp_threshold"]) - float(row["pred_cpu_temp"]) for row in rows]
    raw_metrics = raw_geometry_metrics(rows)
    metrics = normalized_geometry_metrics(rows, payload)
    return {
        "pool_candidate_count": len(rows),
        "pool_threshold_ok_count": ok_count,
        "pool_threshold_ok_rate": ok_count / len(rows),
        "pool_best_pred_cpu_temp": min(pred_values),
        "pool_p10_pred_cpu_temp": _percentile(pred_values, 0.10),
        "pool_median_pred_cpu_temp": _percentile(pred_values, 0.50),
        "pool_p90_pred_cpu_temp": _percentile(pred_values, 0.90),
        "pool_threshold_margin_mean": sum(threshold_margins) / len(threshold_margins),
        "pool_threshold_margin_min": min(threshold_margins),
        "pool_unique_count": geometry_unique_count(rows),
        "pool_mean_pairwise_geometry_distance": raw_metrics["mean_pairwise_geometry_distance"],
        "pool_min_pairwise_geometry_distance": raw_metrics["min_pairwise_geometry_distance"],
        "pool_normalized_mean_pairwise_distance": metrics["normalized_mean_pairwise_distance"],
        "pool_normalized_min_pairwise_distance": metrics["normalized_min_pairwise_distance"],
    }


def raw_geometry_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if len(rows) < 2:
        return {
            "mean_pairwise_geometry_distance": 0.0,
            "min_pairwise_geometry_distance": 0.0,
        }
    vectors = torch.tensor(
        [[float(row[name]) for name in RECOMMEND_KEYS] for row in rows],
        dtype=torch.float32,
    )
    dists = torch.pdist(vectors, p=2)
    if dists.numel() == 0:
        return {
            "mean_pairwise_geometry_distance": 0.0,
            "min_pairwise_geometry_distance": 0.0,
        }
    return {
        "mean_pairwise_geometry_distance": float(dists.mean().item()),
        "min_pairwise_geometry_distance": float(dists.min().item()),
    }


def write_pool_summary(rows: List[Dict[str, float]], payload: Dict, output_json: str = "") -> None:
    if not output_json:
        return
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(candidate_pool_summary(rows, payload), f, ensure_ascii=False, indent=2)


def _diverse_candidate_cost(
    rows: List[Dict[str, float]],
    vectors: torch.Tensor,
    idx: int,
    selected_indices: List[int],
    best_temp: float,
    temp_span: float,
    diversity_rerank_weight: float,
) -> float:
    temp_cost = (float(rows[idx]["pred_cpu_temp"]) - best_temp) / temp_span
    rank_cost = idx / max(len(rows) - 1, 1)
    selected_vectors = vectors[selected_indices]
    min_distance = torch.cdist(vectors[idx : idx + 1], selected_vectors).min().item()
    return temp_cost + 0.05 * rank_cost - diversity_rerank_weight * min_distance


def write_candidates(rows: List[Dict[str, float]], output_csv: str = "", output_json: str = "") -> None:
    rows = [quantize_geometry_dict(row) for row in rows]
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank",
            "threshold_ok",
            "pred_cpu_temp",
            "temp_threshold",
            "base_width",
            "base_depth",
            "total_height",
            "base_height",
            "fin_height",
            "fin_thickness",
            "fin_clear_spacing",
            "fin_break_thickness",
            "fin_break_width",
            "engineering_variant",
            "variant_parent_pred_cpu_temp",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                csv_row = {name: row.get(name, "") for name in fieldnames}
                for name in GEOMETRY_OUTPUT_KEYS:
                    if name in csv_row and csv_row[name] != "":
                        csv_row[name] = f"{float(csv_row[name]):.{GEOMETRY_DECIMALS}f}"
                writer.writerow(csv_row)
