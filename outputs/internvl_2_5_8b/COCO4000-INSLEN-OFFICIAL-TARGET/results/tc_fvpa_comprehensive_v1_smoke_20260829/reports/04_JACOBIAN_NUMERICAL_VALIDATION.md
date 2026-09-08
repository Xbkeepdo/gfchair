# Jacobian numerical validation

Status: **PARTIAL**

Model: `internvl_2_5_8b`

## Evidence

Synthetic FP64 tests are separate; real-case reconstruction is available but no formal all-model audit.

## Cohort denominator

```json
{
  "fp32_intervention_rows": 336,
  "frozen_write_rows": 33,
  "images": 1,
  "labels": {
    "HALL": 1,
    "REAL": 0
  },
  "layers": [
    32
  ],
  "model": "internvl_2_5_8b",
  "path_convergence_rows": 12,
  "shapley_rows": 0,
  "spatial_metric_rows": 0,
  "target_layer_cases": 1
}
```

## Interpretation boundary

`a_m` is a source-wise value-path write conditional on the observed clean attention pattern. It is not the complete causal effect of removing image patch `m`. The current-block zero-write baseline removes only current-block visual writes and is not a no-image state. Missing experiments are not inferred from existing measurements.
