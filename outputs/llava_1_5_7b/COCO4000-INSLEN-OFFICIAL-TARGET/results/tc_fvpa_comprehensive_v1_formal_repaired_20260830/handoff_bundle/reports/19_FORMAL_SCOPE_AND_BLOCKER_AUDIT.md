# Formal scope and blocker audit

This audit is generated from the persisted artifacts in this formal output root. It does not import smoke or cross-model results.

Model: `llava_1_5_7b`

## Frozen-protocol execution

| Family | Persisted formal evidence | Status |
|---|---|---|
| Local Riesz | 500 images; 3656 unique target-layer cases; layers [8, 16, 24, 32]; three target scalars | **PASS** |
| Path attribution | 200 images; 1420 unique target-layer cases; 42600 convergence rows (3 scalars x K=1/4/8/16/32 x trapezoid/Gauss-Legendre) | **PASS** |
| True FP32 causal | 60144 measured rows; final layer 32 included; 6 persisted lower-layer blocker entries; 0 failures | **PARTIAL / BLOCKED** |
| Shapley | 300 final-layer true-FP32 rows; 128 permutations; non-final layers [8, 16, 24] are outside implemented scope | **PARTIAL** |
| Frozen-write counterfactual | 46830 rows; family status `{"activation_patching": "NOT_RUN", "fixed_qk": "NOT_RUN", "frozen_write": "PASS", "pixel_counterfactual": "NOT_RUN"}` | **PARTIAL / BLOCKED** |
| Analyze | authoritative PT/CSV.GZ tables and metrics generated | **PASS** |
| Reports | dynamic evidence reports and handoff bundle generated | **PASS** |

## Explicitly blocked or not run

- Lower-layer true-FP32 downstream execution (layers [8, 16, 24]): **BLOCKED** when listed in the persisted FP32 summary. No lower-layer result is approximated or relabeled measured.
- Fixed-QK, activation patching, and pixel counterfactuals: **NOT_RUN**.
- Real-neuron interventions, train-only geometry calibration, hallucination detector fitting, and paired 10,000-replicate spatial/detection bootstrap: **NOT_RUN**.
- Cross-model generalization and external QA/VQA benchmarks: **NOT_RUN**. They were outside this single-model execution.
- Parquet exports: **BLOCKED** because optional pandas/Parquet dependencies are unavailable. PT and CSV.GZ outputs are authoritative.

## Persisted stage gate

```json
{
  "analysis:llava_1_5_7b": "PASS",
  "counterfactuals:llava_1_5_7b": "BLOCKED",
  "fp32_causal:llava_1_5_7b:shard0": "BLOCKED",
  "fp32_causal:llava_1_5_7b:shard1": "BLOCKED",
  "local_riesz:llava_1_5_7b:shard0": "PASS",
  "local_riesz:llava_1_5_7b:shard1": "PASS",
  "path_attribution:llava_1_5_7b:shard0": "PASS",
  "path_attribution:llava_1_5_7b:shard1": "PASS",
  "reports:llava_1_5_7b": "PASS",
  "shapley:llava_1_5_7b:shard0": "PASS",
  "shapley:llava_1_5_7b:shard1": "PASS"
}
```

The report status `PASS (scope audit)` means this audit faithfully records measured, blocked, and not-run items; it does not promote partial scientific families to PASS.
