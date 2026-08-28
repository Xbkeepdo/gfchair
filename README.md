# gfchair: InsLen official-target control

This isolated workspace keeps the existing generation, DGST/spatial-cost,
ADS/CGC, MLP and split settings, but replaces the COCO object-target chain with
the released Instruction-Lens first-subtoken/first-occurrence protocol.

The formal pipeline does not compute exact response offsets or shadow spans.
Objects that the official resolver cannot find are skipped; resolved objects
use the first occurrence of the selected vocabulary ID.  The decoder state is
the causal row immediately before that response token, and Logit Lens indexes
the same selected ID.  See `utils/inslen_targeting.py` and
`configs/model_configs_inslen_official_target.yaml`.

## Core Feature

For each labeled object token, DGST-T runs one prefix forward pass and captures
decoder-layer internals:

- `o_ffn` is the only source update used for `source_dist`.
- `target_dist` is attention over support tokens multiplied by semantic
  probability of the target token.
- exact Wasserstein/OT transport risk is computed on top-k union support.
- prompt cosine features compare the prediction state with prompt last/mean
  states.
- context confidence is prompt logit-lens confidence multiplied by top visual
  target-alignment cosine.

The main algorithm lives in:

- `models/dgst_capture.py`
- `features/dgst_t.py`
- `features/extractor.py`

## JFFN experiments

The JFFN path decomposes each visual token's attention residual write and uses
forward-mode JVP to measure its local response through `FFN(Norm(z))`. The
implementation and the complete numerical/scientific audit are available in:

- `features/visual_ffn_jacobian.py`
- `features/jffn_experiment.py`
- `jacobian_visual_ffn_validation_report.md`
- `jffn_second_round_incremental_validation_report.md`
- `docs/JFFN_EXPERIMENT_SUMMARY_FOR_DISCUSSION.md`

Large extracted features, checkpoints, predictions, and generated figures are
runtime artifacts under `outputs/` and are intentionally excluded from Git.

## Main Pipeline

Run from this directory:

```bash
python -m nltk.downloader -d "$HOME/nltk_data" \
  punkt averaged_perceptron_tagger wordnet omw-1.4

MODEL=qwen3_vl_8b \
OUTPUT=outputs/qwen3_vl_8b/COCO4000-INSLEN-OFFICIAL-TARGET \
bash run.sh
```

Shell entrypoints use the `python` from the currently activated environment.
Activate the intended Conda/venv before running them; an explicit interpreter
can still be selected with `PYTHON_BIN=/path/to/python bash run.sh` (or the
corresponding QA launcher).

To reuse captions from the original workspace without copying its outputs into
gfchair, pass the source experiment directory:

```bash
MODEL=qwen2_5_vl_7b \
REUSE_GENERATIONS_FROM=/home/apulis-dev/userdata/CODEX/test-cocochair/token-detector/outputs/qwen2_5_vl_7b/COCO4000-COST-spatial \
bash run.sh
```

A complete copied `generations.json` is accepted automatically after validating
the selected image cohort and every row's `generated_text` and actual
`response_token_ids`. The new output receives its own generation manifest;
manual registration is not required. Partial legacy generation shards still
require explicit adoption because their content is incomplete.

The gfchair labeling file keeps two views, both derived from CHAIR + InsLen:

- `all_object_token_spans` retains resolved/skipped InsLen decisions for CHAIR
  accounting; it contains no exact-token shadow location.
- `object_token_spans` contains only the official resolver's successful,
  per-label unique detected words and is the sole feature/training cohort.
- `official_svar_samples` is disabled and remains empty.

For a formal target at response index `i`, wrappers use the prediction row for
`response_ids[i]` (the last prompt row when `i=0`, otherwise the row belonging
to `response_ids[i-1]`).


## POPE, CLEVR, and AMBER VQA comparisons

The unified QA path compares the current six-branch DGST method, ADS+CGC, and
MetaToken/SVAR-controlled/DHCP/ProjectAway/HalLoc on one shared, image-level
8:2 split with no validation set. Training monitors train loss, restores the
minimum-train-loss checkpoint, and reports both a fixed 0.5 threshold and a
Real-F1 threshold selected on train only. COCO and QA share
`configs/model_configs_unified.yaml`;
`configs/model_configs_server_fj01.yaml` contains the same settings with fj01
paths. The `qa_benchmarks` section stores only QA dataset/protocol settings,
while models, feature extraction and probe hyperparameters are shared.

Run one complete benchmark with the same compact launcher style as the original
POPE script:

```bash
MODEL=qwen3_vl_8b bash run_pope.sh
MODEL=qwen3_vl_8b bash run_clevr.sh
MODEL=qwen3_vl_8b bash run_amber.sh
```

`run_qa.sh` is the common implementation. It prepares the fixed split, resumes
generation/labeling/extraction, trains seeds 42/43/44, trains the configured
baselines, and writes comparison tables. QA extraction is selected in the same
YAML with `qa_benchmarks.extraction_mode`: `all`, `method_only`,
`ads_cgc_only`, or `baseline_only`. In `all` mode the prompt-last wrapper is
called once per question and the same `ModelOutput` is consumed by DGST,
ADS+CGC, and every enabled baseline. Baseline payloads remain isolated under
`baseline/<label_protocol>/`; `baseline_only` never creates or overwrites the
root `features.pkl`.

QA probe combinations are read verbatim from `training.feature_sets.method`
and `training.feature_sets.ads_cgc`; the trainer only appends the configured QA
position suffix. There is no separate hard-coded QA feature matrix.
QA schema `qa-prompt-last-token-v6` serializes every enabled support scope:
VV keeps its historical unprefixed method names, while VP uses explicit
`vp_...` method names. VP-only runs therefore do not require dummy VV tensors.
The compact payload also retains configured state-update risk curves and
Prompt CAFE so every YAML-selected QA feature can be read directly by probes.

QA baseline training uses the same `training.baseline.trainers` list and report
protocol as COCO. The default runs both `native_paper` and
`shared_torch_mlp` on the same strict 8:2 split and seeds, then writes their
side-by-side comparison. The shared MLP receives MetaToken's canonical `10+H`
vector, SVAR's selected `layer×head` vector, the flattened DHCP spatial tensor,
and ProjectAway's global plus per-layer internal-confidence vector. Both heads
report fixed-0.5 and train-F1 thresholds with real-positive and
hallucination-positive metrics. `qa_benchmarks.baseline_trainer` only selects
which trained baseline family appears in the cross-family QA headline table.

By default QA baselines use the POPE-style
`object_hallucination_yes_only` protocol. To additionally run the all-answer
correctness track:

Generation and the joint feature extraction default to two question-sharded
workers on `cuda:0 cuda:1`. Each worker writes isolated
resume shards and the parent process validates and atomically consolidates the
artifacts. Probe and baseline training remain on `DEVICE` (default `cuda:0`).
For a deliberate single-GPU run, set both device lists explicitly:

```bash
GENERATION_DEVICES="cuda:0" FEATURE_DEVICES="cuda:0" \
MODEL=qwen3_vl_8b bash run_pope.sh
```

```bash
BASELINE_LABEL_PROTOCOLS="object_hallucination_yes_only answer_correctness_all" \
MODEL=qwen3_vl_8b bash run_pope.sh
```

The active VQA experiment compares `prompt_last_token` with
`question_object_pre_token`. For `prompt_last_token`, extraction fixes the
response target index to `0`, so the strict prefix contains no generated
response tokens and the selected causal row is exactly the final token of the
complete prompt. That row predicts `response_token_ids[0]` and does not move if
a model later emits a preamble before its semantic yes/no answer. Generation
still saves and validates the actual semantic yes/no token for labeling, but
that semantic location does not choose the feature row.

For `question_object_pre_token`, extraction locates the queried object surface
in the complete contextualized question. If its first actual sub-token is `j`,
the strict prefix ends before `j` and the final causal row predicts that
contextual token. The target token ID is the actual first contextual sub-token,
not a separately tokenized approximation. POPE provides exact object spans,
while CLEVR uses the entity head consumed by the terminal `exist` program and
reports coverage explicitly. Each method feature set is trained and summarized
separately at both available positions.
The two reporting protocols remain separate: `object_hallucination_yes_only`
measures false-positive object hallucination, while `answer_correctness_all`
measures general yes/no answer errors. Both use `0=hallucination/error, 1=real`,
report real as the headline positive class, and also report hallucination
metrics.

AMBER uses all 14,216 official discriminative Yes/No questions over 1004
images (existence, attribute, and relation). The deterministic seed-42 outer
split is defined over physical images (803 train / 201 test); question counts
need not be exactly 80/20 because AMBER has a variable number of questions per
image. AMBER does not expose one canonical queried-object span across all of its
existence, attribute, and relation questions, so its rows retain
`prompt_last_token` and report the object position as unavailable.

Some official AMBER source images are as large as 54 MP. The unified YAML
therefore applies `max_pixels: 200704` only to `amber_discriminative`, using
the same bounded visual grid for generation and feature extraction. COCO,
POPE, and CLEVR preprocessing is unchanged.
