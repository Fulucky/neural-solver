"""Infer heatsink designs with temp-threshold-conditioned CVAE."""

from __future__ import annotations

import logging

from AIHeatsinkInverseDesign.infer.cvae_inferencer import build_parser, generate_rows
from AIHeatsinkInverseDesign.common.heatsink_inverse_common import configure_logging, write_candidates


LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
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
