# Jacobian numerical validation

Status: **PARTIAL**

Model: `qwen2_5_vl_7b`

## Evidence

Synthetic FP64 tests are separate; real-case reconstruction is available but no formal all-model audit.

## Cohort denominator

```json
{
  "fp32_intervention_rows": 216384,
  "frozen_write_rows": 43428,
  "images": 500,
  "labels": {
    "HALL": 1148,
    "REAL": 2348
  },
  "layers": [
    7,
    14,
    21,
    28
  ],
  "model": "qwen2_5_vl_7b",
  "path_convergence_rows": 39480,
  "shapley_rows": 300,
  "spatial_metric_rows": 53200,
  "target_layer_cases": 3496
}
```

## Interpretation boundary

`a_m` is a source-wise value-path write conditional on the observed clean attention pattern. It is not the complete causal effect of removing image patch `m`. The current-block zero-write baseline removes only current-block visual writes and is not a no-image state. Missing experiments are not inferred from existing measurements.
