# Heatsink Inverse Design: Three Paths

This directory mirrors the Heatpipe inverse-design architecture with three
training and inference paths:

- `cvae`: threshold-free CVAE. The generator is conditioned on `condition + bbox`;
  `temp_threshold` is applied during inference through latent optimization and
  surrogate ranking/filtering.
- `threshold-cvae`: threshold-conditioned CVAE. The generator is
  conditioned on `condition + bbox + temp_threshold`, and training includes a
  forward-surrogate threshold penalty.
- `diffusion`: conditional diffusion generator. The denoiser is conditioned on
  `condition + bbox`; `temp_threshold` is applied during reverse sampling through
  forward-surrogate guidance and final ranking/filtering.

## Training

```powershell
python ..\train\train.py --method cvae -- --data "D:\path\to\training_data_filtered.json"
python ..\train\train.py --method threshold-cvae -- --data "D:\path\to\training_data_filtered.json"
python ..\train\train.py --method diffusion -- --data "D:\path\to\training_data_filtered.json"
```

For `threshold-cvae`, each observed sample is expanded into
multiple threshold-conditioned rows. The first row uses:

```text
temp_threshold = observed cpu_temp
```

The extra rows sample thresholds from:

```text
observed cpu_temp <= temp_threshold <= upper_temperature
```

This mirrors the heatpipe guided-CVAE setup: one geometry that reaches a given
temperature is also feasible for any looser threshold. Control this with:

```powershell
--threshold-samples-per-layout 3
--threshold-upper-strategy global_max
```

`global_max` samples up to the max training temperature; `heatsink_max` samples
up to the max temperature observed within the same heatsink.

## Three-Path Comparison

Use `compare_three_paths.py` to run a fair comparison. It trains one shared
ForwardMLP surrogate, reuses that surrogate for all three generation paths, runs
the same inference request set, and writes a summary table.

Use `--request-json` for the fixed representative validation set. The request
file can contain one request object, a top-level request list, or
`{"requests": [...]}`. Each request object must contain `condition`, `bbox`,
and `temp_limit` or `temp_threshold`.

```powershell
python ..\train\compare_three_paths.py `
  --data "D:\path\to\training_data_filtered.json" `
  --output-dir .\reports\three_path_comparison `
  --request-json .\validation_requests\request.json `
  --device cuda `
  --condition-transform boxcox `
  --surrogate-scheduler onecycle `
  --num-samples 1024 `
  --top-k 20
```

For a quick wiring check, add `--quick`; this drops the training epochs and
sampling count so the run is only a smoke test.

The main outputs are:

```text
reports/three_path_comparison/comparison_summary.csv
reports/three_path_comparison/comparison_summary.json
reports/three_path_comparison/candidates/{method}/{request}.csv
reports/three_path_comparison/logs/*.log
```

The summary compares:

- `threshold_ok_rate`: Top-K candidates satisfying the requested temperature threshold.
- `best_pred_cpu_temp`: lowest surrogate-predicted temperature in Top-K.
- `mean_pred_cpu_temp`: average surrogate-predicted temperature in Top-K.
- `first_ok_rank`: first threshold-satisfying rank, or `0` if none.
- `unique_count`: number of unique Top-K geometries after rounding generated geometry fields.
- `mean_pairwise_geometry_distance`: diversity proxy over the 5 generated geometry fields.
- `min_pairwise_geometry_distance`: nearest-neighbor geometry distance in raw units.
- `normalized_mean_pairwise_distance`: average pairwise distance after per-request z-score normalization.
- `normalized_min_pairwise_distance`: nearest-neighbor distance after per-request z-score normalization.
- `pool_*`: diversity diagnostics for the raw generated candidate pool before Top-K selection.
- `engineering_variant_count`: number of final Top-K rows produced by optional local engineering perturbation.

Inference uses diversity-aware Top-K reranking by default. Candidates are still
split by threshold feasibility first and seeded from the best predicted
temperature, but later picks trade a small amount of predicted-temperature
optimality for normalized geometry distance from already selected candidates.
Use the following flags for ablation:

```powershell
--diversity-rerank-weight 0      # pure temperature ranking
--diversity-rerank-weight 0.15   # default light diversity reranking
--diversity-temp-tolerance 2.0   # preferred degC window for diverse picks
```

Default output directories:

```text
outputs_thresholdfree_cvae/heatsink/best_model.pt
outputs_guided_cvae/heatsink/best_model.pt
outputs_conditional_diffusion/heatsink/best_model.pt
```

## Forward Surrogate And Feature Importance

Train only the shared temperature surrogate:

```powershell
python ..\train\train.py --method surrogate -- `
  --data "D:\path\to\training_data_filtered.json" `
  --output-dir .\reports\surrogate `
  --device cuda `
  --test-mode grouped-random `
  --test-fraction 0.10 `
  --surrogate-val-mode grouped-random `
  --surrogate-val-fraction 0.10 `
  --surrogate-loss huber `
  --surrogate-best-metric rmse `
  --huber-delta 0.2
```

The surrogate uses a context-gated residual structure instead of a flat
concatenation MLP:

```text
context = Encoder(condition(5) + bbox(3))
geom    = Encoder(geometry(5))

T_base  = Head(context)
delta_T = Head(context, Gate(context) * geom)
T_pred  = T_base + delta_T
```

This matches the observed feature hierarchy: user inputs set the baseline
thermal state, while recommended geometry variables model design-specific
temperature corrections and interactions with that context.

`--surrogate-loss huber` trains in scaled-temperature space. `--huber-delta 0.2`
is roughly a 0.2-standard-deviation error threshold; keep `--surrogate-loss mse`
when reproducing older MSE-only runs.

The best checkpoint is selected by real-temperature validation metrics after
inverse-transforming predictions back to degC. The default is
`--surrogate-best-metric rmse`; use `mae` when average absolute error is the
primary target.

To let every heatsink take one turn as the test set and identify problematic
geometries, run leave-one-heatsink-out evaluation:

```powershell
python ..\train\leave_one_heatsink_out_surrogate.py `
  --data "D:\path\to\training_data_filtered.json" `
  --output-dir .\reports\leave_one_heatsink_out_surrogate `
  --device cuda `
  --surrogate-val-mode grouped-random `
  --surrogate-val-fraction 0.10 `
  --surrogate-loss mse `
  --surrogate-best-metric mae `
  --surrogate-epochs 80
```

The summary is sorted by held-out MAE:

```text
reports/leave_one_heatsink_out_surrogate/leave_one_heatsink_out_metrics.csv
reports/leave_one_heatsink_out_surrogate/leave_one_heatsink_out_metrics.json
```

When trusted training data excludes Iter0, validate Iter0 as a completely
external source instead of mixing it back into leave-one-heatsink-out training:

```powershell
conda run -n heatpipe python ..\train\evaluate_surrogate_external.py `
  --train-data ".\dataset\training_data_iter12.json" `
  --external-data ".\dataset\training_data_iter0.json" `
  --external-label iter0 `
  --output-dir ".\reports\surrogate_external_iter0" `
  --surrogate-epochs 100 `
  --condition-transform boxcox `
  --surrogate-scheduler onecycle `
  --surrogate-best-metric mae `
  --write-residuals `
  --device cuda
```

If a trusted surrogate checkpoint already exists, reuse it and only evaluate
Iter0:

```powershell
conda run -n heatpipe python ..\train\evaluate_surrogate_external.py `
  --surrogate-checkpoint ".\reports\surrogate_iter12\surrogate.pt" `
  --external-data ".\dataset\training_data_iter0.json" `
  --external-label iter0 `
  --output-dir ".\reports\surrogate_external_iter0" `
  --write-residuals `
  --device cuda
```

External validation outputs:

```text
reports/surrogate_external_iter0/iter0_external_summary.json
reports/surrogate_external_iter0/iter0_metrics_by_heatsink.csv
reports/surrogate_external_iter0/iter0_residuals.csv
reports/surrogate_external_iter0/surrogate.pt
```

Analyze all 13 ForwardMLP inputs:

```powershell
python ..\train\analyze_surrogate_feature_importance.py `
  --data "D:\path\to\training_data_filtered.json" `
  --surrogate-checkpoint .\reports\surrogate\surrogate.pt `
  --output-dir .\reports\surrogate_feature_importance `
  --device cuda `
  --permutation-repeats 5 `
  --max-eval-samples 50000
```

The 13 inputs are:

```text
condition(5): chip_length, Rjc, Rjb, power, wind_speed
bbox(3):      base_width, base_depth, total_height
geometry(5):  fin_height, fin_thickness, fin_clear_spacing,
              fin_break_thickness, fin_break_width
```

Outputs:

```text
reports/surrogate_feature_importance/surrogate_13_input_feature_importance.csv
reports/surrogate_feature_importance/surrogate_13_input_feature_importance.json
reports/surrogate_feature_importance/surrogate_13_input_feature_importance.png
```

The primary ranking is `scaled_gradient_importance`: mean absolute gradient of
predicted temperature with respect to each standardized model input. This keeps
all 13 inputs on the same model-input scale and is the preferred metric when the
goal is feature importance rather than data-distribution contribution.

The CSV also keeps two auxiliary diagnostics:

- `delta_rmse` and `r2_drop`: permutation importance, useful for seeing how much
  the current test distribution depends on each input.
- `raw_sensitivity`: local perturbation sensitivity in degC per raw input unit.

## Inference

```powershell
python -m AIInverseDesign.infer.infer --method threshold-cvae -- `
  --checkpoint-path .\outputs_guided_cvae\heatsink\best_model.pt `
  --output-csv .\threshold_heatsink_candidates.csv `
  --num-samples 1024 `
  --top-k 20 `
  --temp-threshold 80 `
  --chip-length 35 `
  --rjc 0.6 `
  --rjb 1.1 `
  --power 85 `
  --wind-speed 4 `
  --base-width 40 `
  --base-depth 40 `
  --total-height 20
```

All inference paths return candidates ranked by:

```text
1. threshold_ok first
2. lower pred_cpu_temp first
3. lower fin_height first
```
