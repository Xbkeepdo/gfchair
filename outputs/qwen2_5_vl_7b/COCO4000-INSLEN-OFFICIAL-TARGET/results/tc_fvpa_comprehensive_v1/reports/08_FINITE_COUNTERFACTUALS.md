# Finite counterfactuals

Status: **NOT RUN**

Model: `qwen2_5_vl_7b`

## Evidence

No finite counterfactual rows.

## Cohort denominator

```json
{
  "fp32_intervention_rows": 0,
  "frozen_write_rows": 0,
  "images": 0,
  "labels": {},
  "layers": [],
  "model": "qwen2_5_vl_7b",
  "path_convergence_rows": 0,
  "shapley_rows": 0,
  "spatial_metric_rows": 0,
  "target_layer_cases": 0
}
```

## Interpretation boundary

`a_m` is a source-wise value-path write conditional on the observed clean attention pattern. It is not the complete causal effect of removing image patch `m`. The current-block zero-write baseline removes only current-block visual writes and is not a no-image state. Missing experiments are not inferred from existing measurements.
