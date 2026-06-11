"""Infer heatsink designs with temp-threshold-conditioned CVAE."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cvae_inferencer import build_parser, generate_rows
from common.heatsink_inverse_common import write_candidates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    args = build_parser("Generate heatsink candidates with temp-threshold-conditioned CVAE.").parse_args()
    rows = generate_rows(args, guided=True)
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
