# Efficiency and scaling

Status: **PARTIAL**

Model: `llava_1_5_7b`

## Evidence

Per-case runtime exists for 3656 cases; formal throughput and peak-memory scaling are incomplete.

## Cohort denominator

```json
{
  "fp32_intervention_rows": 60144,
  "frozen_write_rows": 46815,
  "images": 500,
  "labels": {
    "HALL": 1128,
    "REAL": 2528
  },
  "layers": [
    8,
    16,
    24,
    32
  ],
  "local_images": 500,
  "model": "llava_1_5_7b",
  "path_convergence_rows": 42600,
  "path_images": 200,
  "path_target_layer_cases": 1420,
  "shapley_rows": 300,
  "spatial_metric_rows": 57464,
  "target_layer_cases": 3656
}
```

## Interpretation boundary

`a_m` is a source-wise value-path write conditional on the observed clean attention pattern. It is not the complete causal effect of removing image patch `m`. The current-block zero-write baseline removes only current-block visual writes and is not a no-image state. Missing experiments are not inferred from existing measurements.
