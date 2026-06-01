"""Infer heatsink designs with conditional diffusion and surrogate guidance."""

from __future__ import annotations

import argparse
import logging

import torch

from AIInverseDesign.common.heatsink_inverse_common import (
    add_common_infer_args,
    build_forward_input_from_parts,
    configure_logging,
    load_checkpoint,
    load_diffusion_from_payload,
    make_inference_cond,
    predict_temperature_tensor,
    request_from_args,
    score_candidates,
    write_candidates,
)


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate heatsink candidates with conditional diffusion.")
    add_common_infer_args(parser)
    parser.add_argument("--guidance-scale", type=float, default=0.08)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--threshold-weight", type=float, default=2.0)
    return parser


def generate_rows(args: argparse.Namespace):
    device = torch.device(args.device)
    payload = load_checkpoint(args.checkpoint_path, device, args.surrogate_checkpoint)
    model = load_diffusion_from_payload(payload, device)
    condition, bbox, temp_threshold = request_from_args(args)
    cond_scaled = make_inference_cond(
        payload, condition, bbox, temp_threshold, guided=False, n=args.num_samples, device=device
    )
    cond_raw = payload["cond_scaler"].inverse_transform(cond_scaled)

    cfg = payload["diffusion_config"]
    timesteps = int(cfg["timesteps"])
    betas = torch.linspace(float(cfg["beta_start"]), float(cfg["beta_end"]), timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x = torch.randn(args.num_samples, model.target_dim, device=device)
    for step in reversed(range(timesteps)):
        t = torch.full((args.num_samples,), step, dtype=torch.long, device=device)
        with torch.no_grad():
            pred_noise = model(x, cond_scaled, t)
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bars[step]
            beta_t = betas[step]
            mean = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_t)
            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

        if args.guidance_scale > 0.0:
            x = x.detach().requires_grad_(True)
            geom_raw = payload["recommend_scaler"].inverse_transform(x)
            pred_temp = predict_temperature_tensor(
                payload["forward_model"],
                payload["forward_input_scaler"],
                payload["target_scaler"],
                build_forward_input_from_parts(cond_raw, geom_raw),
            )
            threshold = torch.full_like(pred_temp, temp_threshold)
            temp_loss = pred_temp.mean() / payload["target_scaler"].std.to(device).mean()
            threshold_loss = torch.relu(pred_temp - threshold).pow(2).mean() / payload["target_scaler"].std.to(device).pow(2).mean()
            loss = args.temperature_weight * temp_loss + args.threshold_weight * threshold_loss
            grad = torch.autograd.grad(loss, x)[0]
            x = (x - args.guidance_scale * grad).detach()

    geom_raw = payload["recommend_scaler"].inverse_transform(x).detach().cpu().tolist()
    return score_candidates(
        payload,
        condition,
        bbox,
        geom_raw,
        temp_threshold,
        args.top_k,
        diversity_rerank_weight=args.diversity_rerank_weight,
        diversity_temp_tolerance=args.diversity_temp_tolerance,
    )


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    rows = generate_rows(args)
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
