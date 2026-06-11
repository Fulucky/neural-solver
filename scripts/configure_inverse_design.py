"""修改逆向设计默认推理配置。

示例：
python scripts/configure_inverse_design.py --method diffusion --checkpoint AIInverseDesign/outputs_conditional_diffusion/heatsink/best_model.pt
python scripts/configure_inverse_design.py --show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AIInverseDesign.common.inference_config import (  # noqa: E402
    SUPPORTED_METHODS,
    config_path,
    load_inference_config,
    read_config_data,
    update_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Configure inverse-design inference defaults.")
    parser.add_argument("--show", action="store_true", help="Only print the current config.")
    parser.add_argument("--method", choices=SUPPORTED_METHODS, help="Inference method.")
    parser.add_argument("--checkpoint", dest="checkpoint_path", help="Generator checkpoint path.")
    parser.add_argument("--surrogate-checkpoint", help="Optional ForwardMLP surrogate checkpoint.")
    parser.add_argument("--device", help="Inference device, for example cpu or cuda.")
    parser.add_argument("--candidate-pool-size", type=int, help="Default generated candidate pool size.")
    parser.add_argument("--top-k", type=int, help="Default number of returned candidates.")
    parser.add_argument("--latent-opt-steps", type=int, help="CVAE latent optimization steps.")
    parser.add_argument("--latent-lr", type=float, help="CVAE latent optimization learning rate.")
    parser.add_argument("--temperature-weight", type=float, help="Temperature loss weight.")
    parser.add_argument("--threshold-weight", type=float, help="Threshold violation loss weight.")
    parser.add_argument("--guidance-scale", type=float, help="Diffusion guidance scale.")
    parser.add_argument("--diversity-rerank-weight", type=float, help="Diversity-aware Top-K reranking weight.")
    parser.add_argument("--diversity-temp-tolerance", type=float, help="Preferred temperature window for reranking.")
    parser.add_argument("--engineering-variant-mode", choices=["off", "auto", "on"], help="Engineering perturbation mode.")
    parser.add_argument("--engineering-variant-count-per-candidate", type=int, help="Accepted variants per selected candidate.")
    parser.add_argument("--engineering-variant-max-trials", type=int, help="Maximum perturbation trials per selected candidate.")
    parser.add_argument("--engineering-variant-scale", type=float, help="Perturbation step as a fraction of each field range.")
    parser.add_argument("--engineering-variant-required-temp-margin", type=float, help="Required temperature margin for variants.")
    parser.add_argument("--engineering-variant-min-unique-ratio", type=float, help="Auto-mode unique ratio threshold.")
    parser.add_argument("--engineering-variant-min-norm-mean-dist", type=float, help="Auto-mode normalized mean-distance threshold.")
    parser.add_argument("--engineering-variant-min-norm-min-dist", type=float, help="Auto-mode normalized min-distance threshold.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.show:
        data = read_config_data()
        resolved = load_inference_config()
        print(json.dumps({"config_path": str(config_path()), "raw": data, "resolved": resolved.__dict__}, ensure_ascii=False, indent=2))
        return

    updates = {
        "method": args.method,
        "checkpoint_path": args.checkpoint_path,
        "surrogate_checkpoint": args.surrogate_checkpoint,
        "device": args.device,
        "candidate_pool_size": args.candidate_pool_size,
        "top_k": args.top_k,
        "latent_opt_steps": args.latent_opt_steps,
        "latent_lr": args.latent_lr,
        "temperature_weight": args.temperature_weight,
        "threshold_weight": args.threshold_weight,
        "guidance_scale": args.guidance_scale,
        "diversity_rerank_weight": args.diversity_rerank_weight,
        "diversity_temp_tolerance": args.diversity_temp_tolerance,
        "engineering_variant_mode": args.engineering_variant_mode,
        "engineering_variant_count_per_candidate": args.engineering_variant_count_per_candidate,
        "engineering_variant_max_trials": args.engineering_variant_max_trials,
        "engineering_variant_scale": args.engineering_variant_scale,
        "engineering_variant_required_temp_margin": args.engineering_variant_required_temp_margin,
        "engineering_variant_min_unique_ratio": args.engineering_variant_min_unique_ratio,
        "engineering_variant_min_norm_mean_dist": args.engineering_variant_min_norm_mean_dist,
        "engineering_variant_min_norm_min_dist": args.engineering_variant_min_norm_min_dist,
    }
    if not any(value is not None for value in updates.values()):
        raise SystemExit("No updates provided. Use --show to inspect current config.")

    data = update_config(updates)
    print(json.dumps({"config_path": str(config_path()), "updated": data}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
