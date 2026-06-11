"""Infer heatsink designs with threshold-free CVAE."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.heatsink_inverse_common import (
    add_common_infer_args,
    build_forward_input_from_parts,
    load_checkpoint,
    load_cvae_from_payload,
    make_inference_cond,
    predict_temperature_tensor,
    request_from_args,
    score_candidate_pool,
    select_candidates_from_pool,
    write_pool_summary,
    write_candidates,
)


def build_parser(description: str = "Generate heatsink candidates with threshold-free CVAE.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_infer_args(parser)
    parser.add_argument("--latent-opt-steps", type=int, default=40)
    parser.add_argument("--latent-lr", type=float, default=5e-2)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--threshold-weight", type=float, default=2.0)
    return parser


def generate_rows(args: argparse.Namespace, guided: bool = False):
    device = torch.device(args.device)
    payload = load_checkpoint(args.checkpoint_path, device, args.surrogate_checkpoint)
    cvae = load_cvae_from_payload(payload, device)
    condition, bbox, temp_threshold = request_from_args(args)
    cond_scaled = make_inference_cond(
        payload, condition, bbox, temp_threshold, guided=guided, n=args.num_samples, device=device
    )
    cond_raw = cond_scaled.new_tensor(
        payload["cond_scaler"].inverse_transform(cond_scaled).detach().cpu().numpy()
    ).to(device)
    z = torch.randn(args.num_samples, cvae.latent_dim, device=device)

    if args.latent_opt_steps > 0:
        z = z.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=args.latent_lr)
        for _ in range(args.latent_opt_steps):
            geom_scaled = cvae.decode(cond_scaled, z)
            geom_raw = payload["recommend_scaler"].inverse_transform(geom_scaled)
            pred_temp = predict_temperature_tensor(
                payload["forward_model"],
                payload["forward_input_scaler"],
                payload["target_scaler"],
                build_forward_input_from_parts(cond_raw, geom_raw),
            )
            threshold = torch.full_like(pred_temp, temp_threshold)
            temp_loss = pred_temp.mean() / payload["target_scaler"].std.to(device).mean()
            threshold_loss = torch.relu(pred_temp - threshold).pow(2).mean() / payload["target_scaler"].std.to(device).pow(2).mean()
            prior_loss = z.pow(2).mean() * 1e-3
            loss = args.temperature_weight * temp_loss + args.threshold_weight * threshold_loss + prior_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        geom_scaled = cvae.decode(cond_scaled, z.detach())
        geom_raw = payload["recommend_scaler"].inverse_transform(geom_scaled).cpu().tolist()

    pool_rows = score_candidate_pool(payload, condition, bbox, geom_raw, temp_threshold)
    write_pool_summary(pool_rows, payload, getattr(args, "pool_summary_json", ""))
    return select_candidates_from_pool(
        rows=pool_rows,
        payload=payload,
        condition=condition,
        bbox=bbox,
        temp_threshold=temp_threshold,
        top_k=args.top_k,
        diversity_rerank_weight=getattr(args, "diversity_rerank_weight", 0.15),
        diversity_temp_tolerance=getattr(args, "diversity_temp_tolerance", 2.0),
        engineering_variant_mode=getattr(args, "engineering_variant_mode", "off"),
        engineering_variant_count_per_candidate=getattr(args, "engineering_variant_count_per_candidate", 2),
        engineering_variant_max_trials=getattr(args, "engineering_variant_max_trials", 20),
        engineering_variant_scale=getattr(args, "engineering_variant_scale", 0.08),
        engineering_variant_required_temp_margin=getattr(args, "engineering_variant_required_temp_margin", 1.0),
        engineering_variant_min_unique_ratio=getattr(args, "engineering_variant_min_unique_ratio", 0.8),
        engineering_variant_min_norm_mean_dist=getattr(args, "engineering_variant_min_norm_mean_dist", 1.0),
        engineering_variant_min_norm_min_dist=getattr(args, "engineering_variant_min_norm_min_dist", 0.3),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = build_parser().parse_args()
    rows = generate_rows(args, guided=False)
    write_candidates(rows, args.output_csv, args.output_json)
    LOGGER.info("Generated %d candidates.", len(rows))
    for row in rows[: min(10, len(rows))]:
        LOGGER.info(
            "#%02d ok=%s pred=%.3fC fin_h=%.3f",
            row["rank"],
            row["threshold_ok"],
            row["pred_cpu_temp"],
            row["fin_height"],
        )


if __name__ == "__main__":
    main()
