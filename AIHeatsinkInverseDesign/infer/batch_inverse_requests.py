"""Batch inverse-design inference for a JSON request set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from AIHeatsinkInverseDesign.common.checkpoints import (
    load_checkpoint,
    load_cvae_from_payload,
    load_diffusion_from_payload,
)
from AIHeatsinkInverseDesign.common.heatsink_inverse_common import make_inference_cond
from AIHeatsinkInverseDesign.common.inverse_scoring import score_candidate_pool, select_candidates_from_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inverse design on many requests from one JSON file.")
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--checkpoint-path", "--checkpoint", dest="checkpoint_path", required=True)
    parser.add_argument("--method", choices=["cvae", "threshold-cvae", "diffusion"], default="threshold-cvae")
    parser.add_argument("--surrogate-checkpoint", default="")
    parser.add_argument("--output-csv", default="reports/inverse_batch_candidates.csv")
    parser.add_argument("--summary-json", default="reports/inverse_batch_summary.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--candidate-pool-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--latent-opt-steps", type=int, default=40)
    parser.add_argument("--latent-lr", type=float, default=5e-2)
    parser.add_argument("--guidance-scale", type=float, default=0.08)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--threshold-weight", type=float, default=2.0)
    parser.add_argument("--diversity-rerank-weight", type=float, default=0.15)
    parser.add_argument("--diversity-temp-tolerance", type=float, default=2.0)
    parser.add_argument("--engineering-variant-mode", choices=["off", "auto", "on"], default="auto")
    parser.add_argument("--engineering-variant-count-per-candidate", type=int, default=2)
    parser.add_argument("--engineering-variant-max-trials", type=int, default=20)
    parser.add_argument("--engineering-variant-scale", type=float, default=0.08)
    parser.add_argument("--engineering-variant-required-temp-margin", type=float, default=1.0)
    parser.add_argument("--engineering-variant-min-unique-ratio", type=float, default=0.8)
    parser.add_argument("--engineering-variant-min-norm-mean-dist", type=float, default=1.0)
    parser.add_argument("--engineering-variant-min-norm-min-dist", type=float, default=0.3)
    return parser


def load_requests(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("requests"), list):
        return payload["requests"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported request JSON format: {path}")


def predict_from_parts(payload: Dict, cond_raw: torch.Tensor, geom_raw: torch.Tensor) -> torch.Tensor:
    forward_x = torch.cat([cond_raw[:, :8], geom_raw], dim=1)
    pred_scaled = payload["forward_model"](payload["forward_input_scaler"].transform(forward_x))
    return payload["target_scaler"].inverse_transform(pred_scaled)


def generate_cvae_pool(payload: Dict, request: Dict, args: argparse.Namespace, guided: bool) -> List[Dict]:
    device = torch.device(args.device)
    cvae = load_cvae_from_payload(payload, device)
    condition = request["condition"]
    bbox = request["bbox"]
    temp_threshold = float(request.get("temp_threshold", request.get("temp_limit")))
    pool_size = args.candidate_pool_size
    cond_scaled = make_inference_cond(payload, condition, bbox, temp_threshold, guided, pool_size, device)
    cond_raw = payload["cond_scaler"].inverse_transform(cond_scaled).to(device)
    z = torch.randn(pool_size, cvae.latent_dim, device=device)

    if args.latent_opt_steps > 0:
        z = z.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=args.latent_lr)
        for _ in range(args.latent_opt_steps):
            geom_scaled = cvae.decode(cond_scaled, z)
            geom_raw = payload["recommend_scaler"].inverse_transform(geom_scaled)
            pred_temp = predict_from_parts(payload, cond_raw, geom_raw)
            threshold = torch.full_like(pred_temp, temp_threshold)
            temp_loss = pred_temp.mean() / payload["target_scaler"].std.to(device).mean()
            threshold_loss = torch.relu(pred_temp - threshold).pow(2).mean() / payload["target_scaler"].std.to(device).pow(2).mean()
            loss = args.temperature_weight * temp_loss + args.threshold_weight * threshold_loss + 1e-3 * z.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        geom_scaled = cvae.decode(cond_scaled, z.detach())
        geom_rows = payload["recommend_scaler"].inverse_transform(geom_scaled).detach().cpu().tolist()
    return score_candidate_pool(payload, condition, bbox, geom_rows, temp_threshold)


def generate_diffusion_pool(payload: Dict, request: Dict, args: argparse.Namespace) -> List[Dict]:
    device = torch.device(args.device)
    model = load_diffusion_from_payload(payload, device)
    condition = request["condition"]
    bbox = request["bbox"]
    temp_threshold = float(request.get("temp_threshold", request.get("temp_limit")))
    pool_size = args.candidate_pool_size
    cond_scaled = make_inference_cond(payload, condition, bbox, temp_threshold, guided=False, n=pool_size, device=device)
    cond_raw = payload["cond_scaler"].inverse_transform(cond_scaled).to(device)

    cfg = payload["diffusion_config"]
    timesteps = int(cfg["timesteps"])
    betas = torch.linspace(float(cfg["beta_start"]), float(cfg["beta_end"]), timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    x = torch.randn(pool_size, model.target_dim, device=device)

    for step in reversed(range(timesteps)):
        t = torch.full((pool_size,), step, dtype=torch.long, device=device)
        with torch.no_grad():
            pred_noise = model(x, cond_scaled, t)
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bars[step]
            beta_t = betas[step]
            mean_x = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
            x = mean_x + torch.sqrt(beta_t) * torch.randn_like(x) if step > 0 else mean_x

        if args.guidance_scale > 0.0:
            x = x.detach().requires_grad_(True)
            geom_raw = payload["recommend_scaler"].inverse_transform(x)
            pred_temp = predict_from_parts(payload, cond_raw, geom_raw)
            threshold = torch.full_like(pred_temp, temp_threshold)
            temp_loss = pred_temp.mean() / payload["target_scaler"].std.to(device).mean()
            threshold_loss = torch.relu(pred_temp - threshold).pow(2).mean() / payload["target_scaler"].std.to(device).pow(2).mean()
            loss = args.temperature_weight * temp_loss + args.threshold_weight * threshold_loss
            grad = torch.autograd.grad(loss, x)[0]
            x = (x - args.guidance_scale * grad).detach()

    geom_rows = payload["recommend_scaler"].inverse_transform(x).detach().cpu().tolist()
    return score_candidate_pool(payload, condition, bbox, geom_rows, temp_threshold)


def select_rows(payload: Dict, request: Dict, args: argparse.Namespace) -> List[Dict]:
    if args.method == "diffusion":
        pool = generate_diffusion_pool(payload, request, args)
    else:
        pool = generate_cvae_pool(payload, request, args, guided=args.method == "threshold-cvae")
    temp_threshold = float(request.get("temp_threshold", request.get("temp_limit")))
    rows = select_candidates_from_pool(
        rows=pool,
        payload=payload,
        condition=request["condition"],
        bbox=request["bbox"],
        temp_threshold=temp_threshold,
        top_k=args.top_k,
        diversity_rerank_weight=args.diversity_rerank_weight,
        diversity_temp_tolerance=args.diversity_temp_tolerance,
        engineering_variant_mode=args.engineering_variant_mode,
        engineering_variant_count_per_candidate=args.engineering_variant_count_per_candidate,
        engineering_variant_max_trials=args.engineering_variant_max_trials,
        engineering_variant_scale=args.engineering_variant_scale,
        engineering_variant_required_temp_margin=args.engineering_variant_required_temp_margin,
        engineering_variant_min_unique_ratio=args.engineering_variant_min_unique_ratio,
        engineering_variant_min_norm_mean_dist=args.engineering_variant_min_norm_mean_dist,
        engineering_variant_min_norm_min_dist=args.engineering_variant_min_norm_min_dist,
    )
    for row in rows:
        row["pool_candidate_count"] = len(pool)
    return rows


def flatten_rows(request: Dict, rows: List[Dict]) -> List[Dict]:
    flat_rows = []
    temp_threshold = float(request.get("temp_threshold", request.get("temp_limit")))
    for row in rows:
        flat_rows.append(
            {
                "request": request.get("name", ""),
                "chip_length": request["condition"]["chip_length"],
                "Rjc": request["condition"]["Rjc"],
                "Rjb": request["condition"]["Rjb"],
                "power": request["condition"]["power"],
                "wind_speed": request["condition"]["wind_speed"],
                "bbox_base_width": request["bbox"]["base_width"],
                "bbox_base_depth": request["bbox"]["base_depth"],
                "bbox_total_height": request["bbox"]["total_height"],
                "temp_threshold": temp_threshold,
                **row,
            }
        )
    return flat_rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(requests: List[Dict], rows: List[Dict]) -> Dict:
    best_rows = [row for row in rows if int(row.get("rank", 0)) == 1]
    ok_best = [row for row in best_rows if bool(row.get("threshold_ok"))]
    pred_values = [float(row["pred_cpu_temp"]) for row in best_rows]
    margins = [float(row["temp_threshold"]) - float(row["pred_cpu_temp"]) for row in best_rows]
    return {
        "request_count": len(requests),
        "candidate_row_count": len(rows),
        "best_row_count": len(best_rows),
        "best_threshold_ok_count": len(ok_best),
        "best_threshold_ok_rate": len(ok_best) / len(best_rows) if best_rows else 0.0,
        "best_pred_cpu_temp_mean": sum(pred_values) / len(pred_values) if pred_values else 0.0,
        "best_margin_mean": sum(margins) / len(margins) if margins else 0.0,
        "best_margin_min": min(margins) if margins else 0.0,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    payload = load_checkpoint(args.checkpoint_path, device, args.surrogate_checkpoint)
    requests = load_requests(Path(args.request_json))

    all_rows = []
    for idx, request in enumerate(requests, start=1):
        torch.manual_seed(args.seed + idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + idx)
        rows = select_rows(payload, request, args)
        all_rows.extend(flatten_rows(request, rows))
        if idx % 50 == 0 or idx == len(requests):
            print(f"Processed {idx}/{len(requests)} requests")

    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    write_csv(output_csv, all_rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summarize(requests, all_rows), f, ensure_ascii=False, indent=2)
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_json}")


if __name__ == "__main__":
    main()
