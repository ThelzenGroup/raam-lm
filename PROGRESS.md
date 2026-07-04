# RAAM-LM Progress

## 2026-07-03 - MLOps MCP Integration

- Installed `mlops-mcp-server` globally outside the repository at `/home/lumalgo/.codex/tools/mlops-mcp-server/.venv`.
- Added a global wrapper at `/home/lumalgo/.codex/bin/mlops-mcp-server` and a Codex MCP config entry.
- Added dependency-free `.mlops/experiments` support in `src/raam_lm/mlops.py`.
- Added `scripts/backfill_mlops_runs.py` to import historical pulled run evidence from output folders.
- Updated `scripts/train.py` with opt-in live tracker logging via `--mlops-project-path` or `RAAM_MLOPS_PROJECT_PATH`.
- Ignored `.mlops/` in Git so local run state and artifact references are not published accidentally.
- Added `docs/MLOPS.md` with backfill and live-training commands.

## 2026-07-02

- Read the pasted goal text and treated it as source of truth.
- Inspected the workspace: it was empty, and no `AGENTS.md` project instructions were present.
- Created an installable package under `src/raam_lm`.
- Added debug and scale configs.
- Added core modules, baselines, RAAM model, losses, data, probes, training utilities, profiling, and CLI scripts.
- Added tests and documentation.

## Environment Notes

- Initial system Python was `/usr/bin/python` 3.13.14.
- Initial system Python did not have `torch` installed.
- Bundled Codex Python 3.12.13 also did not have `torch`, `pytest`, or `yaml` installed.

## Verification Log

All commands below were run with the local project virtual environment first on `PATH`:

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest -q
```

Result: passed, `12 passed, 1 warning in 2.06s`.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/smoke_train.py --config configs/debug/raam_tiny.yaml --steps 30 --device auto
```

Result: passed on CPU, no NaNs, final loss `7.428411`, log path `runs/debug_raam_tiny/train_log.jsonl`.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/run_debug_ablation_matrix.py --steps 30 --device auto
```

Result: passed on CPU for `transformer_tiny`, `pure_mamba_like_tiny`, `raam_tiny`, `raam_no_compression`, `raam_no_anchors`, `raam_no_attention_islands`, and `raam_no_mtp`. All rows reported `nan_status: false` and `causal_test_status: true`. Summary path: `runs/debug_ablation_matrix/summary.json`.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/profile_step.py --config configs/debug/raam_tiny.yaml --device auto
```

Result: passed on CPU and wrote `runs/profile_manifest.json`. Manifest includes parameter counts, estimated FLOPs/token, measured tokens/sec, mean and p95 step time, peak allocated/reserved memory, device, dtype, config hash, and git SHA field.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/eval_probes.py --config configs/debug/raam_tiny.yaml --device auto
```

Result: passed on CPU and wrote `runs/probe_results.json` for all registry models and all four probes.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/estimate_flops.py --config configs/debug/raam_tiny.yaml
```

Result: passed, estimated FLOPs/token `444531`.

## Limitations

- `mamba-ssm` is not installed, so debug runs used the clearly named `fallback_gated_conv` mixer. This is logged as `mixer_backend`.
- CUDA was not available in this environment, so GPU behavior was not measured.
- PyTorch emitted a warning because NumPy is not installed in the local venv; NumPy seeding remains optional and the commands completed successfully.
- Smoke losses and probe scores are code-path checks only, not scientific architecture evidence.

## Follow-Up Next-Token Comparison

After the initial ablation run, the comparison script was updated to report `final_next_token_loss` separately from total training loss and to support multiple seeds.

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/run_debug_ablation_matrix.py --steps 100 --seeds 17,29,41 --device auto --quiet --output runs/next_token_comparison_3seed_100step/summary.json
```

Result: passed on CPU for 7 configs x 3 seeds. All rows reported `nan_status: false` and `causal_test_status: true`.

| Rank | Config | Next-Token Loss Mean | Std | Est. FLOPs/Token |
| ---: | --- | ---: | ---: | ---: |
| 1 | `raam_no_anchors` | 5.6233 | 0.0133 | 415731 |
| 2 | `raam_no_attention_islands` | 5.6315 | 0.0299 | 509427 |
| 3 | `raam_tiny` | 5.6400 | 0.0156 | 444531 |
| 4 | `raam_no_compression` | 5.6511 | 0.0157 | 522931 |
| 5 | `raam_no_mtp` | 5.6581 | 0.0216 | 431424 |
| 6 | `pure_mamba_like_tiny` | 5.7107 | 0.0162 | 507187 |
| 7 | `transformer_tiny` | 5.7915 | 0.0542 | 570163 |

Interpretation: this short generated-data run is stronger plumbing evidence than the 30-step smoke run, but still not architecture evidence. It suggests anchors and attention islands are not justified by this tiny setup, while the compressed RAAM family is worth testing at longer runs and with probes.

## RAAM-AgentCoder Scratch Pipeline

Implemented the from-scratch chat-first agentic coding training path:

- tokenizer training: `scripts/train_tokenizer.py`
- dataset packing: `scripts/pack_dataset.py`
- checkpointable packed-data training: `scripts/train.py`
- generation: `scripts/generate.py`
- chat eval smoke test: `scripts/eval_chat.py`
- agentic coding eval smoke test: `scripts/eval_agentic_coding.py`
- Vast preflight: `scripts/vast_preflight.sh`
- scratch configs: `configs/scratch/raam_agentcoder_debug.yaml`, `raam_agentcoder_50m.yaml`, `raam_agentcoder_100m.yaml`, `raam_agentcoder_300m.yaml`, `transformer_agentcoder_100m.yaml`, `pure_mamba_like_agentcoder_100m.yaml`
- docs: `docs/DATA_FORMAT.md`, `docs/SCRATCH_TRAINING_PLAN.md`, `docs/AGENTIC_CODING_EVALS.md`, `docs/VAST_TRAINING.md`

Verification:

```bash
PATH="$PWD/.venv/bin:$PATH" python -m pytest -q
```

Result: passed, latest run `17 passed, 1 warning in 25.56s`.

Tiny end-to-end run:

```bash
rm -rf runs/agentcoder_e2e
PATH="$PWD/.venv/bin:$PATH" python scripts/train_tokenizer.py examples/tiny_agentic.jsonl --output runs/agentcoder_e2e/tokenizer.json --vocab-size 512
PATH="$PWD/.venv/bin:$PATH" python scripts/pack_dataset.py examples/tiny_agentic.jsonl --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/packed --seq-len 64 --val-fraction 0.34
PATH="$PWD/.venv/bin:$PATH" python scripts/train.py --config configs/scratch/raam_agentcoder_debug.yaml --train-bin runs/agentcoder_e2e/packed/train.bin --val-bin runs/agentcoder_e2e/packed/val.bin --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/train --steps 20 --device auto
PATH="$PWD/.venv/bin:$PATH" python scripts/train.py --config configs/scratch/raam_agentcoder_debug.yaml --train-bin runs/agentcoder_e2e/packed/train.bin --val-bin runs/agentcoder_e2e/packed/val.bin --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/train --steps 25 --resume runs/agentcoder_e2e/train/checkpoints/last.pt --device auto
PATH="$PWD/.venv/bin:$PATH" python scripts/generate.py --config configs/scratch/raam_agentcoder_debug.yaml --tokenizer runs/agentcoder_e2e/tokenizer.json --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt --prompt $'<|user|>\nFix add().\n<|assistant|>\n' --device auto --max-new-tokens 16 > runs/agentcoder_e2e/generation.txt
PATH="$PWD/.venv/bin:$PATH" python scripts/eval_agentic_coding.py --config configs/scratch/raam_agentcoder_debug.yaml --tokenizer runs/agentcoder_e2e/tokenizer.json --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt --device auto --output runs/agentcoder_e2e/agentic_eval.json
```

Result: passed on CPU. Tokenizer vocab size was `411`. Packed manifest recorded `249` train tokens and `210` validation tokens. Training logged validation loss through step `24`; final resumed `val_next_token_loss` was `5.203440189361572`. Checkpoint path: `runs/agentcoder_e2e/train/checkpoints/last.pt`.

Additional checks:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/eval_chat.py --config configs/scratch/raam_agentcoder_debug.yaml --tokenizer runs/agentcoder_e2e/tokenizer.json --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt --device auto --output runs/agentcoder_e2e/chat_eval.json
bash -n scripts/vast_preflight.sh
```

Result: chat eval ran and wrote `runs/agentcoder_e2e/chat_eval.json`; Vast preflight shell syntax passed. The full Vast preflight was not run here because CUDA/RTX 5090 is not available in this local environment.

Agentic eval caveat: the tiny model is intentionally undertrained. The smoke eval ran across 8 task prompts and recorded `next_token_validation_loss: 5.203440189361572`, but `mean_patch_apply_rate` and `json_tool_call_validity` were both `0.0`, which is expected for a 25-step toy run and is not evidence of useful coding ability.

## Vast Dataset Research and Setup Plan

Researched the current Vast.ai RTX 5-series requirements and the first practical RAAM-AgentCoder dataset mix.

Added:

- `docs/TRAINING_DATA_AND_VAST_RESEARCH.md`
- `scripts/prepare_agentcoder_research_data.py`
- `tests/test_prepare_agentcoder_research_data.py`

The recommended first Vast dataset is a staged corpus: selected StarCoder2 extra subsets for code-adjacent base text, OpenAssistant/oasst1 and WildChat for chat behavior, NVIDIA Open-SWE-Traces and NVIDIA SWE-Zero OpenHands trajectories for agentic coding, plus local permissive repo/patch/test data. SWE-bench remains evaluation-only.

The first paid training target remains `configs/scratch/raam_agentcoder_50m.yaml` for a 1000-step rehearsal, followed by resume/eval checks before moving to `configs/scratch/raam_agentcoder_100m.yaml`.

Updated the Vast docs again after the publishing decision changed. Public GitHub is now the documented default path, with `.gitignore` added to keep datasets, packed token files, checkpoints, logs, caches, and local secrets out of the repository.

## Vast 100M RAAM Mechanism Gate

Added matched 100M RAAM mechanism configs and runbook commands:

- `configs/scratch/raam_agentcoder_100m_full.yaml`
- `configs/scratch/raam_agentcoder_100m_no_anchors.yaml`
- `configs/scratch/raam_agentcoder_100m_no_attention_islands.yaml`
- updated `docs/STAGED_TRAINING_RUNBOOK.md`
- updated `docs/VAST_TRAINING.md`

Local validation before pushing:

```bash
python3 - <<'PY'
from pathlib import Path
import yaml
paths = [
    Path('configs/scratch/raam_agentcoder_100m_full.yaml'),
    Path('configs/scratch/raam_agentcoder_100m_no_anchors.yaml'),
    Path('configs/scratch/raam_agentcoder_100m_no_attention_islands.yaml'),
]
for path in paths:
    data = yaml.safe_load(path.read_text())
    print(path, data.get('attention_island_layers'), data.get('compression', {}).get('anchors_per_block'))
PY
bash -n scripts/vast_stage3_baselines.sh scripts/vast_pull_artifacts.sh scripts/vast_train_50m.sh
git diff --check
```

Result: passed. Pushed commit `825fefb` to `main`.

Vast run:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_raam_mechanisms_20260702T211130Z \
CONFIGS='configs/scratch/raam_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m_no_attention_islands.yaml configs/scratch/raam_agentcoder_100m_no_anchors.yaml configs/scratch/raam_agentcoder_100m_full.yaml' \
STEPS=5 RESUME_STEPS=6 SAVE_EVERY=5 EVAL_EVERY=5 EXPORT_CHECKPOINT=0 \
bash scripts/vast_stage3_baselines.sh
```

Environment: Vast RTX 5090, Torch `2.12.0+cu130`, CUDA available, `mixer_backend: fallback_gated_conv`.

| Variant | Config | Last Val Loss | Tokens/sec | Peak VRAM MB | FLOPs/token | Compression Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| compression-only RAAM | `raam_agentcoder_100m.yaml` | 10.3226 | 27746.2 | 12636.3 | 157843558 | 0.0625 |
| anchors-only RAAM | `raam_agentcoder_100m_no_attention_islands.yaml` | 10.3277 | 27084.0 | 13372.4 | 157844582 | 0.1875 |
| attention-islands-only RAAM | `raam_agentcoder_100m_no_anchors.yaml` | 10.3135 | 27533.3 | 12603.0 | 142137446 | 0.0625 |
| full RAAM | `raam_agentcoder_100m_full.yaml` | 10.3221 | 25998.4 | 13320.7 | 144497766 | 0.1875 |

Artifact pull: completed via SSH tar with `*.pt` excluded. Pulled artifacts contained logs, manifests, configs, tokenizer copies, train logs, generation smoke text, `agentic_eval.json`, `summary.json`, and `summary.md`, with zero checkpoint files.

Interpretation: all four 100M RAAM mechanism variants fit and resume on the RTX 5090. In this very short 5-to-6-step gate, attention-islands-only RAAM had the lowest validation loss and lowest estimated FLOPs/token among the mechanism variants. Full RAAM fit but did not beat attention-islands-only on this tiny gate. Agentic generation scores remained `0.0` for JSON tool-call validity and patch apply rate, which is expected at this tiny token count and is not evidence of useful coding ability.

Next highest-value experiment: run a longer matched 100M ablation, probably 100 to 110 steps first, for compression-only, attention-islands-only, full RAAM, and pure Mamba-like. Do not move to a full paid training run until that longer gate confirms which variant is worth scaling.

## Vast 100M Long Matched Gate

Ran the longer matched 100M gate on Vast RTX 5090 using the same packed AgentCoder data and tokenizer:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_long_gate_20260702T212134Z \
CONFIGS='configs/scratch/pure_mamba_like_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m_no_anchors.yaml configs/scratch/raam_agentcoder_100m_full.yaml' \
STEPS=100 RESUME_STEPS=110 SAVE_EVERY=50 EVAL_EVERY=10 EXPORT_CHECKPOINT=0 \
bash scripts/vast_stage3_baselines.sh
```

The first run filled the instance disk while writing checkpoint archives after the third config. The generated `.pt` files were removed from `/root/raam-lm/runs`, the missing eval and full-RAAM segment were completed with checkpoint exports disabled, and the final artifact pull excluded checkpoint weights. Local artifact pull: `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage4_100m_long_gate_20260702T212134Z`. No `.pt` files were pulled.

Environment: Vast RTX 5090, Torch `2.12.0+cu130`, CUDA available, `mixer_backend: fallback_gated_conv`.

| Variant | Config | Last Step | Tokens Seen | Last Val Loss | Val Loss Delta | Tokens/sec | Peak VRAM MB | Non-Emb Params | FLOPs/token | Compression Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure Mamba-like | `pure_mamba_like_agentcoder_100m.yaml` | 109 | 7208960 | 6.01303 | -4.45784 | 152322 | 17637.6 | 66947584 | 157777510 | 1.0 |
| compression-only RAAM | `raam_agentcoder_100m.yaml` | 109 | 7208960 | 6.28747 | -4.10276 | 227282 | 12636.3 | 67867138 | 157843558 | 0.0625 |
| attention-islands-only RAAM | `raam_agentcoder_100m_no_anchors.yaml` | 109 | 7208960 | 6.30943 | -4.08739 | 209365 | 12603.0 | 66806274 | 142137446 | 0.0625 |
| full RAAM | `raam_agentcoder_100m_full.yaml` | 109 | 7208960 | 6.32306 | -4.06966 | 227984 | 13327.9 | 66806274 | 144497766 | 0.1875 |

Agentic eval remained at `0.0` JSON tool-call validity and `0.0` patch apply rate for all variants. That is expected at this tiny training budget and should be treated as a smoke signal only, not a useful-code capability result.

Interpretation: pure Mamba-like is currently the best-quality model in this matched 100M gate. The best RAAM variant is compression-only RAAM: it has the lowest RAAM validation loss and is much faster with materially lower peak VRAM than pure Mamba-like. The earlier 5-to-6-step attention-islands lead did not hold up at 100-to-110 steps. Full RAAM fit, resumed, and ran quickly, but it did not beat the cheaper compression-only RAAM variant, so it should not be the next scaling default.

Next highest-value experiment: run a longer two-way 100M quality/efficiency gate between `pure_mamba_like_agentcoder_100m.yaml` and `raam_agentcoder_100m.yaml`, such as 500 to 550 or 1000 to 1100 steps, before spending on a larger full training run. The decision question is whether compression-only RAAM can close enough validation-loss gap to justify its speed and VRAM advantage for the from-scratch chat and agentic coding target.

## Vast 100M Two-Way 1000-Step Gate

Added disk-safe checkpoint cleanup to the Vast runners before this run:

- `scripts/vast_stage3_baselines.sh`
- `scripts/vast_train_50m.sh`
- `docs/STAGED_TRAINING_RUNBOOK.md`
- `docs/VAST_TRAINING.md`

Validation before pushing:

```bash
bash -n scripts/vast_stage3_baselines.sh scripts/vast_train_50m.sh
git diff --check
```

Result: passed. Pushed commit `4d49274` to `main`.

Ran the longer two-way 100M gate on Vast RTX 5090:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_two_way_1000step_20260702T213737Z \
CONFIGS='configs/scratch/pure_mamba_like_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m.yaml' \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=50 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_stage3_baselines.sh
```

Artifact pull: `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage4_100m_two_way_1000step_20260702T213737Z`. No `.pt` files were pulled. Both Vast instances were confirmed `stopped/exited` after the artifact pull.

Environment: Vast RTX 5090, Torch `2.12.0+cu130`, CUDA available, `mixer_backend: fallback_gated_conv`.

| Variant | Config | Last Step | Tokens Seen | Last Val Loss | Val Loss Delta | Tokens/sec | Peak VRAM MB | Non-Emb Params | FLOPs/token | Compression Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure Mamba-like | `pure_mamba_like_agentcoder_100m.yaml` | 1099 | 72089600 | 3.84068 | -6.63019 | 128065 | 20741.7 | 66947584 | 157777510 | 1.0 |
| compression-only RAAM | `raam_agentcoder_100m.yaml` | 1099 | 72089600 | 3.17354 | -7.21669 | 189302 | 15744.2 | 67867138 | 157843558 | 0.0625 |

Agentic eval still reported `0.0` JSON tool-call validity and `0.0` mean patch apply rate for both models. The generated text improved compared with very short runs but is not yet useful chat or coding behavior.

Interpretation: compression-only RAAM is now the best 100M candidate under this matched gate. It beat pure Mamba-like on validation loss while also running about 48% faster at the final logged step and using about 24% less peak VRAM. This reverses the 100-to-110-step gate and makes compression-only RAAM the next scaling default, with the caveat that the current dataset is still small and this is base-LM evidence, not final chat/agentic coding evidence.

Next highest-value experiment: expand the real AgentCoder corpus substantially, repack with the same tokenizer policy, and run compression-only RAAM as the main 100M candidate long enough for chat and agentic coding evals to become meaningful. Keep pure Mamba-like as the fallback baseline, but do not spend on full RAAM until compression-only RAAM has a stronger data-scale result.

## Stage 5 Candidate Entrypoint

Added `scripts/vast_train_100m_candidate.sh` as a dedicated next-step wrapper. It initially defaulted to the winning compression-only `configs/scratch/raam_agentcoder_100m.yaml`, a separate `/root/data/agentcoder_stage5` data root, moderate expanded dataset source limits, `5000 -> 5500` steps, `SAVE_EVERY=0`, a compact model-only export, and one optimizer-resumable `last.pt` for continuation. After the expanded Stage 5 candidate destabilized, the wrapper default was changed to `configs/scratch/raam_agentcoder_100m_stage5_stable.yaml`.

Updated:

- `docs/STAGED_TRAINING_RUNBOOK.md`
- `docs/VAST_TRAINING.md`
- `docs/TRAINING_DATA_AND_VAST_RESEARCH.md`

Validation:

```bash
bash -n scripts/vast_train_100m_candidate.sh scripts/vast_train_50m.sh scripts/vast_stage3_baselines.sh
git diff --check
```

Result: passed.

## Stage 5 Candidate Smoke on Vast

Updated `scripts/vast_train_100m_candidate.sh` to explicitly forward runtime overrides such as batch size, training sequence length, tokenizer vocab size, pack sequence length, gradient accumulation, and eval batch count. Validation:

```bash
bash -n scripts/vast_train_100m_candidate.sh scripts/vast_train_50m.sh scripts/vast_stage3_baselines.sh
git diff --check
```

Result: passed. Pushed commit `7a63b7f` to `main`.

Ran a tiny Stage 5 wrapper smoke on Vast RTX 5090:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5_smoke \
RAW_DIR=/root/data/agentcoder_stage5_smoke/raw \
PACKED_DIR=/root/data/agentcoder_stage5_smoke/packed_128 \
TOKENIZER=/root/data/agentcoder_stage5_smoke/tokenizer.json \
RUN_DIR=/root/raam-lm/runs/stage5_100m_candidate_smoke_20260702T220239Z/train \
MAX_OPEN_SWE=2 MAX_SWE_ZERO=2 MAX_WILDCHAT=4 MAX_OASST=4 \
STARCODER2_EXTRAS='documentation=4 issues=4' \
VOCAB_SIZE=1024 SEQ_LEN=128 \
BATCH_SIZE=1 TRAIN_SEQ_LEN=128 GRAD_ACCUMULATION_STEPS=1 EVAL_BATCHES=1 \
STEPS=2 RESUME_STEPS=3 SAVE_EVERY=0 EVAL_EVERY=1 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_train_100m_candidate.sh
```

Artifact pull: `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_100m_candidate_smoke_20260702T220239Z`. No `.pt` files were pulled.

Smoke evidence:

- raw dataset manifest wrote records from Open-SWE-Traces, SWE-Zero OpenHands, WildChat, OASST1, StarCoder2 documentation, and StarCoder2 issues
- tokenizer trained with vocab size `1024`
- packed manifest wrote `298522` train tokens and `1453` validation tokens at sequence length `128`
- train/resume completed through global step `2`
- final smoke validation loss was `6.875584602355957`
- generation and agentic eval completed
- JSON tool-call validity and mean patch apply rate remained `0.0`, as expected for a 3-step smoke
- `KEEP_TRAINING_CHECKPOINTS=0` removed all `.pt` files after eval
- Vast disk remained at `1%` used

## Stage 5 Expanded Candidate Run

Pushed `6a0ce85` after adding a guard that lets `scripts/vast_train_50m.sh`
continue when dataset preparation exits non-zero after writing a valid
`manifest.json`. This handled a Python finalization crash after the expanded raw
manifest had already been written.

Ran the expanded Stage 5 candidate on Vast RTX 5090:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RUN_DIR=/root/raam-lm/runs/stage5_raam_agentcoder_100m_candidate_retry_20260702T224633Z/train \
STEPS=5000 RESUME_STEPS=5500 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=1 KEEP_TRAINING_CHECKPOINTS=1 \
bash scripts/vast_train_100m_candidate.sh
```

Raw Stage 5 manifest records:

| Source | Records |
| --- | ---: |
| `OpenAssistant/oasst1` | 5000 |
| `allenai/WildChat` | 10000 |
| `bigcode/starcoder2data-extras/documentation` | 10000 |
| `bigcode/starcoder2data-extras/issues` | 10000 |
| `bigcode/starcoder2data-extras/kaggle` | 5000 |
| `bigcode/starcoder2data-extras/stackoverflow` | 10000 |
| `nvidia/Open-SWE-Traces` | 10000 |
| `nvidia/SWE-Zero-openhands-trajectories` | 10000 |

Packed corpus:

- tokenizer vocab size: `32768`
- train docs: `68600`
- train tokens: `1887569077`
- validation docs: `1400`
- validation tokens: `40123219`
- sequence length: `2048`

Training result:

| Metric | Value |
| --- | ---: |
| Last logged step | 5499 |
| Tokens seen | 360448000 |
| Best validation loss | 3.0947578072547914 at step 500 |
| First validation loss | 10.370828104019164 |
| Final validation loss | 13.07088327407837 |
| Final train loss | 81.24600219726562 |
| Final next-token loss | 15.282869338989258 |
| Final reconstruction loss | 169.55172729492188 |
| Final MTP h2 loss | 261.0045166015625 |
| Final MTP h3 loss | 730.0146484375 |
| Final tokens/sec | 153872.55807689478 |
| Peak allocated VRAM MB | 18847.81689453125 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_candidate_retry_20260702T224633Z`.
The pull included `model_only_fp16.pt` and excluded optimizer `last.pt`.

Interpretation: the expanded Stage 5 data path works, but the older
`raam_agentcoder_100m.yaml` auxiliary-loss schedule is not stable at this scale.
The model learned sharply at first, then validation loss worsened, generation
collapsed into repetitive text, and agentic scores stayed at zero. Do not promote
this run. The next candidate is
`configs/scratch/raam_agentcoder_100m_stage5_stable.yaml`, which keeps
compression-only RAAM but disables early reconstruction loss and curriculum MTP.

Local validation note while preparing the stable config:

```text
$ python3 scripts/estimate_flops.py --config configs/scratch/raam_agentcoder_100m_stage5_stable.yaml
Traceback (most recent call last):
  File "/home/lumalgo/Documents/exp2/scripts/estimate_flops.py", line 11, in <module>
    from raam_lm.config import config_hash, load_config
  File "/home/lumalgo/Documents/exp2/src/raam_lm/__init__.py", line 4, in <module>
    from .registry import build_model, available_models
  File "/home/lumalgo/Documents/exp2/src/raam_lm/registry.py", line 5, in <module>
    from .baselines import DenseTransformerForCausalLM, PureMambaLikeForCausalLM
  File "/home/lumalgo/Documents/exp2/src/raam_lm/baselines/__init__.py", line 1, in <module>
    from .transformer import DenseTransformerForCausalLM
  File "/home/lumalgo/Documents/exp2/src/raam_lm/baselines/transformer.py", line 5, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Fallback validation passed with system Python: YAML loaded successfully,
`use_curriculum_mtp` was false, `compression.recon_loss_weight` was `0.0`, and
`mtp.enabled` was false. Torch-side validation must run on Vast or in a local
environment with PyTorch installed.

## Stage 5 Stable Schedule Gate

Pushed `306ad3c` with the stable Stage 5 config and updated wrapper default:

- config: `configs/scratch/raam_agentcoder_100m_stage5_stable.yaml`
- default runner: `scripts/vast_train_100m_candidate.sh`
- early reconstruction loss disabled with `compression.recon_loss_weight: 0.0`
- curriculum MTP disabled with `use_curriculum_mtp: false` and `mtp.enabled: false`

Ran a bounded Vast RTX 5090 gate on the existing expanded Stage 5 packed corpus:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
RUN_DIR=/root/raam-lm/runs/stage5_raam_agentcoder_100m_stable_gate_20260703T003821Z/train \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_train_100m_candidate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_stable_gate_20260703T003821Z`.
The pull is about 2 MB and contains no `.pt` files. Both Vast RTX 5090 instances
were stopped/exited after the pull.

Stable gate metrics:

| Metric | Value |
| --- | ---: |
| Last logged step | 1099 |
| Tokens seen | 72089600 |
| First validation loss | 10.387241506576538 |
| Best validation loss | 3.130998957157135 at step 500 |
| Final validation loss | 4.5944470286369326 |
| Final train loss | 3.9415853023529053 |
| Final tokens/sec | 243616.58096179334 |
| Peak allocated VRAM MB | 12630.3056640625 |
| Non-embedding params | 67080706 |
| Estimated FLOPs/token | 151132672 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.387241506576538 |
| 100 | 6.659002757072448 |
| 200 | 5.01783173084259 |
| 300 | 3.725554144382477 |
| 400 | 3.1932442784309387 |
| 500 | 3.130998957157135 |
| 600 | 3.23995920419693 |
| 700 | 3.318199861049652 |
| 800 | 3.3735808610916136 |
| 900 | 3.475276291370392 |
| 999 | 4.031751370429992 |
| 1000 | 4.032040143013001 |
| 1099 | 4.5944470286369326 |

Interpretation: disabling early reconstruction/MTP fixes the catastrophic collapse
from the previous expanded Stage 5 run, but it does not make this setup ready for
full training. The validation curve still peaks around step 500 and then degrades
while the LR is still warming up. The next highest-value experiment is a stable
Stage 5 learning-rate gate: same data and loss setup, but cap LR around the step
500 value or use a shorter/lower warmup before spending on a longer run.

Added next candidate config:
`configs/scratch/raam_agentcoder_100m_stage5_lr1e4.yaml`. It keeps the same
compression-only Stage 5 setup with reconstruction and MTP disabled, but changes
the optimizer schedule to `lr: 0.0001` and `warmup_steps: 500` so the LR is capped
near the previous run's best-validation region instead of continuing to rise past
step 500.

## Stage 5 Capped-LR Gate

Pushed `1c6d0db` with
`configs/scratch/raam_agentcoder_100m_stage5_lr1e4.yaml`, then ran a bounded Vast
RTX 5090 gate on the existing expanded Stage 5 packed corpus:

```bash
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
RUN_DIR=/root/raam-lm/runs/stage5_raam_agentcoder_100m_lr1e4_gate_20260703T005205Z/train \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr1e4.yaml \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_train_100m_candidate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_lr1e4_gate_20260703T005205Z`.
The pull is about 1.9 MB and contains no `.pt` files. Both Vast RTX 5090
instances were stopped/exited after the pull.

Capped-LR gate metrics:

| Metric | Value |
| --- | ---: |
| Last logged step | 1099 |
| Tokens seen | 72089600 |
| First validation loss | 10.387824440002442 |
| Best validation loss | 3.0503190875053408 at step 500 |
| Final validation loss | 3.337503743171692 |
| Final train loss | 2.9091925621032715 |
| Final tokens/sec | 227739.13384071097 |
| Peak allocated VRAM MB | 12630.3056640625 |
| Non-embedding params | 67080706 |
| Estimated FLOPs/token | 151132672 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.387824440002442 |
| 100 | 6.776480412483215 |
| 200 | 5.214842581748963 |
| 300 | 3.9693182826042177 |
| 400 | 3.3113513946533204 |
| 500 | 3.0503190875053408 |
| 600 | 3.143168342113495 |
| 700 | 3.2386062860488893 |
| 800 | 3.2452985048294067 |
| 900 | 3.2523025393486025 |
| 999 | 3.344883382320404 |
| 1000 | 3.318834912776947 |
| 1099 | 3.337503743171692 |

Interpretation: capping LR at `1e-4` materially improves the final Stage 5 gate
versus the previous stable config (`3.3375` final validation instead of `4.5944`)
and improves the best point (`3.0503` instead of `3.1310`). It still peaks at
step 500 and drifts upward afterward, so the model is still not ready for a full
training spend. The next decision is either to export/check around the current
best region or run one more lower-LR gate (`5e-5` or `7.5e-5`) to test whether the
validation curve can keep improving beyond 500 steps.

Added next candidate config:
`configs/scratch/raam_agentcoder_100m_stage5_lr75e6.yaml`. It keeps the same
compression-only Stage 5 setup with reconstruction and MTP disabled, but caps LR
at `0.000075` with `warmup_steps: 500`. This is the next gate to test whether the
post-500 validation drift keeps shrinking without slowing learning as much as a
`5e-5` cap.

## Stage 5 75e-6 LR Gate

Pushed `9155b35` with
`configs/scratch/raam_agentcoder_100m_stage5_lr75e6.yaml` and
`scripts/vast_launch_stage5_gate.sh`, then ran a bounded Vast RTX 5090 gate on
the existing expanded Stage 5 packed corpus:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_lr75e6_gate_20260703T010459Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr75e6.yaml \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_lr75e6_gate_20260703T010459Z`.
The pull is about 1.9 MB and contains no `.pt` files. Both Vast RTX 5090
instances were stopped/exited after the pull.

75e-6 LR gate metrics:

| Metric | Value |
| --- | ---: |
| Last logged step | 1099 |
| Tokens seen | 72089600 |
| First validation loss | 10.388726663589477 |
| Best validation loss | 3.0213409900665282 at step 600 |
| Final validation loss | 3.237572705745697 |
| Final train loss | 2.863391637802124 |
| Final tokens/sec | 244333.9745951525 |
| Peak allocated VRAM MB | 12630.3056640625 |
| Non-embedding params | 67080706 |
| Estimated FLOPs/token | 151132672 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.388726663589477 |
| 100 | 6.9750683307647705 |
| 200 | 5.547672200202942 |
| 300 | 4.390269994735718 |
| 400 | 3.6601709485054017 |
| 500 | 3.136573326587677 |
| 600 | 3.0213409900665282 |
| 700 | 3.145030105113983 |
| 800 | 3.1879626750946044 |
| 900 | 3.167621982097626 |
| 999 | 3.290269196033478 |
| 1000 | 3.2663076400756834 |
| 1099 | 3.237572705745697 |

Interpretation: `lr75e6` is now the best Stage 5 schedule tested. It beats
`lr1e4` on best validation (`3.0213` vs `3.0503`) and final validation (`3.2376`
vs `3.3375`) while moving the best point from step 500 to step 600. Agentic
scores remain zero, so this is base-LM stability evidence only. Full training is
still not cleared. The next useful decision is either a `5e-5` gate to see whether
the best point moves later again, or a short checkpoint-export run around the
current step-600 best region for qualitative inspection.

Added next candidate config:
`configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml`. It keeps the same
compression-only Stage 5 setup with reconstruction and MTP disabled, but caps LR
at `0.00005` with `warmup_steps: 500`. This tests whether the best-validation
point continues moving later and whether post-best drift shrinks further.

## Stage 5 50e-6 LR Gate

Pushed `82848e3` with
`configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml`, then ran a bounded Vast
RTX 5090 gate on the existing expanded Stage 5 packed corpus:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_lr5e5_gate_20260703T011705Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_lr5e5_gate_20260703T011705Z`.
The pull is about 1.9 MB and contains no `.pt` files. Both Vast RTX 5090
instances were stopped/exited after the pull.

50e-6 LR gate metrics:

| Metric | Value |
| --- | ---: |
| Last logged step | 1099 |
| Tokens seen | 72089600 |
| First validation loss | 10.389572143554688 |
| Best validation loss | 3.0210490942001345 at step 800 |
| Final validation loss | 3.1759371876716616 |
| Final train loss | 2.8481225967407227 |
| Final tokens/sec | 237922.05490398325 |
| Peak allocated VRAM MB | 12630.3056640625 |
| Non-embedding params | 67080706 |
| Estimated FLOPs/token | 151132672 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.389572143554688 |
| 100 | 7.264174509048462 |
| 200 | 5.953322315216065 |
| 300 | 4.944909071922302 |
| 400 | 4.30405638217926 |
| 500 | 3.5682665586471556 |
| 600 | 3.1166778445243835 |
| 700 | 3.0475279688835144 |
| 800 | 3.0210490942001345 |
| 900 | 3.036092829704285 |
| 999 | 3.179477167129517 |
| 1000 | 3.1620751857757567 |
| 1099 | 3.1759371876716616 |

Interpretation: `lr5e5` is now the safest Stage 5 schedule tested. It is slower
early than `lr75e6`, but it moves the best point later, nearly ties the best loss
(`3.0210` vs `3.0213`), and has the best final validation loss so far (`3.1759`
vs `3.2376`). Agentic scores remain zero, so this is still base-LM schedule
evidence only. The next highest-value experiment is a longer `5e-5` continuation
gate or a checkpoint-export pass around the current step-800 best region for
qualitative generation inspection.

## Stage 5 50e-6 Step-800 Export

Ran a bounded `5e-5` export pass to capture the current best measured Stage 5
base-LM region as a compact model-only checkpoint:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
STEPS=801 RESUME_STEPS=801 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=1 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z`.
The pull is about 194 MB and includes exactly one `.pt` file:
`current/train/checkpoints/model_only_fp16.pt` with size `201313187` bytes. No
optimizer `last.pt` or `step_*.pt` checkpoint was pulled. Both Vast RTX 5090
instances were stopped/exited after the pull.

Export metrics:

| Metric | Value |
| --- | ---: |
| Last logged step | 800 |
| Tokens seen | 52494336 |
| First validation loss | 10.389572143554688 |
| Final/best validation loss | 3.0210490942001345 at step 800 |
| Final train loss | 2.831112861633301 |
| Final tokens/sec | 239013.06591040603 |
| Peak allocated VRAM MB | 12309.60498046875 |
| Non-embedding params | 67080706 |
| Estimated FLOPs/token | 151132672 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Interpretation: this is the current best measured base-LM checkpoint artifact,
not a useful chat/coding model yet. It is suitable for qualitative generation
inspection, model-only storage, or as a baseline point before a longer `5e-5`
continuation gate.

Added model-only resume support:

- `scripts/train.py` now loads checkpoints without `optimizer_state`, starts from
  `checkpoint.step + 1`, uses a fresh optimizer, and records `resume_mode:
  model_only` plus `resume_optimizer_loaded: false` in the manifest.
- `scripts/vast_train_50m.sh`, `scripts/vast_train_100m_candidate.sh`, and
  `scripts/vast_launch_stage5_gate.sh` now accept `START_CHECKPOINT` for the first
  training call.

This makes the exported step-800 `model_only_fp16.pt` usable for continuation
smoke tests and longer fresh-optimizer gates.

## Stage 5 Model-Only Resume Smoke

Pushed `c83bdac` with model-only resume support, then ran a short Vast RTX 5090
continuation smoke from the exported step-800 model-only checkpoint:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_model_only_resume_smoke_20260703T020833Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
START_CHECKPOINT=/root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z/train/checkpoints/model_only_fp16.pt \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
STEPS=805 RESUME_STEPS=805 SAVE_EVERY=0 EVAL_EVERY=2 EVAL_BATCHES=1 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_model_only_resume_smoke_20260703T020833Z`.
The pull is about 768 KB and contains no `.pt` files. Both Vast RTX 5090
instances were stopped/exited after the pull.

Manifest evidence:

- `resume_mode: model_only`
- `resume_optimizer_loaded: false`
- `resume_start_step: 801`
- `resume_from: /root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z/train/checkpoints/model_only_fp16.pt`

Training evidence:

| Metric | Value |
| --- | ---: |
| Logged steps | 4 |
| First logged step | 801 |
| Last logged step | 804 |
| Final train loss | 3.034564256668091 |
| Final validation loss | 3.590768814086914 |
| Final tokens/sec | 239544.6837340605 |
| Peak allocated VRAM MB | 12309.60498046875 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Interpretation: this smoke validates that the exported `model_only_fp16.pt`
checkpoint can seed further training with a fresh optimizer. The validation
numbers are not a quality result because the smoke used `EVAL_BATCHES=1` and only
four training steps. The next quality gate should use the same model-only start
with normal eval batches and a longer continuation window.

## Stage 5 Model-Only Continuation Gate

Ran a normal-eval continuation gate from the exported step-800 model-only
checkpoint:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_model_only_continue_20260703T021343Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
START_CHECKPOINT=/root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z/train/checkpoints/model_only_fp16.pt \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
STEPS=1201 RESUME_STEPS=1201 SAVE_EVERY=0 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_model_only_continue_20260703T021343Z`.
The pull is about 1.2 MB and contains no `.pt` files. Both Vast RTX 5090
instances were stopped/exited after the pull.

Manifest evidence:

- `resume_mode: model_only`
- `resume_optimizer_loaded: false`
- `resume_start_step: 801`

Continuation metrics:

| Metric | Value |
| --- | ---: |
| Logged steps | 400 |
| First logged step | 801 |
| Last logged step | 1200 |
| Best resumed validation loss | 3.111972713470459 at step 900 |
| Final validation loss | 3.159172797203064 |
| Final train loss | 2.3228018283843994 |
| Final tokens/sec | 244426.19807304617 |
| Peak allocated VRAM MB | 12309.60498046875 |
| JSON tool-call validity | 0.0 |
| Mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 900 | 3.111972713470459 |
| 1000 | 3.1630563139915466 |
| 1100 | 3.1767639040946962 |
| 1200 | 3.159172797203064 |

Interpretation: model-only continuation with a fresh optimizer is viable, but it
does not improve on the exported step-800 checkpoint. The best artifact remains
`stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z` at validation
`3.0210490942001345`. A future continuation should either keep optimizer state
from the best region or test an even lower LR; otherwise use the step-800 export
for qualitative inspection and as the current base-LM candidate.

## Stage 5 Qualitative Checkpoint Inspection

Added `scripts/qualitative_checkpoint_inspect.py` and ran it on the current best
model-only checkpoint, `stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z`
step `800`.

```bash
cd /root/raam-lm
/venv/main/bin/python scripts/qualitative_checkpoint_inspect.py \
  --config configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
  --tokenizer /root/data/agentcoder_stage5/tokenizer.json \
  --checkpoint /root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z/train/checkpoints/model_only_fp16.pt \
  --device cuda \
  --max-new-tokens 96 \
  --temperature 0.8 \
  --top-k 50 \
  --seeds 17 \
  --output-json /root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_qual_20260703T022622Z/qualitative_samples.json \
  --output-md /root/raam-lm/runs/stage5_raam_agentcoder_100m_lr5e5_qual_20260703T022622Z/qualitative_samples.md
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_stage5_raam_agentcoder_100m_lr5e5_qual_20260703T022622Z`.
The pull includes `qualitative_samples.json` and `qualitative_samples.md`; no
checkpoint files were pulled. Both Vast RTX 5090 instances were verified
`exited/stopped` after the pull.

Qualitative evidence:

| Prompt group | Samples | Useful completions | Valid JSON | Test-command mentions | EOS generated |
| --- | ---: | ---: | ---: | ---: | ---: |
| chat/coding/software-engineering/agentic coding | 8 | 0 | 0 | 0 | 0 |

The samples remain incoherent and mostly look like fragments of paths, symbols,
and code-adjacent tokens. One completion tripped the loose diff-marker flag by
emitting `---`, but it was not an applicable patch. This means the step-800
checkpoint is the best measured base-LM artifact so far, not a usable chat or
agentic coding model.

Decision: do not start expensive full chat/coding training from this checkpoint
as if it were ready. The next highest-value work is a data/tokenizer/objective
sanity pass: verify chat templates and packing boundaries, inspect memorized
training examples, add a small overfit test on curated chat/coding records, and
only then run the next paid Stage 5 continuation or larger scratch run.

## AgentCoder Tiny Overfit Sanity Gate

Added and ran a curated overfit sanity gate before any larger paid chat/coding
run:

- `examples/agentcoder_overfit_sanity.jsonl`
- `configs/scratch/raam_agentcoder_overfit.yaml`
- `scripts/eval_overfit_sanity.py`
- `scripts/run_agentcoder_overfit_sanity.py`
- `scripts/pack_dataset.py --mirror-val`

Local validation before the Vast run:

```bash
python3 -m py_compile scripts/eval_overfit_sanity.py scripts/run_agentcoder_overfit_sanity.py scripts/pack_dataset.py scripts/train_tokenizer.py src/raam_lm/__init__.py src/raam_lm/agent_data.py
python3 scripts/train_tokenizer.py examples/agentcoder_overfit_sanity.jsonl --output work/overfit_tokenizer_smoke/tokenizer.json --vocab-size 1024
git diff --check
```

Remote validation on Vast RTX 5090:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k mirror_validation
/venv/main/bin/python -m py_compile scripts/eval_overfit_sanity.py scripts/run_agentcoder_overfit_sanity.py
/venv/main/bin/python -m pytest -q
```

## AgentCoder Gate Comparison Report

The Vast instance remained unavailable on the next retry:

```text
Required resources are currently unavailable, state change queued.
```

After a 75-second poll, both known Vast RTX 5090 instances still reported
`actual_status: exited`, `cur_state: stopped`, and `next_state: stopped`.

Added `scripts/compare_agentcoder_gates.py` to compare pulled curated gate
artifacts without requiring `torch`. The report reads `summary.json` and
`curated_eval.json`, then emits:

- exact pass rate
- behavior accuracy and confusion matrix
- failed cases with expected/predicted behavior
- missing required substrings and JSON correctness
- train/validation token counts and final loss/speed metrics

Validation:

```bash
python3 -m py_compile scripts/compare_agentcoder_gates.py scripts/eval_overfit_sanity.py
python3 -m pytest -q tests/test_compare_agentcoder_gates.py
python3 scripts/compare_agentcoder_gates.py \
  /home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_20260703T030727Z \
  /home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_cmdfix_20260703T031643Z \
  --output-json /home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_latest.json \
  --output-md /home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_latest.md
git diff --check
```

Resulting comparison:

| Run | Exact pass | Behavior accuracy | Final validation loss |
| --- | ---: | ---: | ---: |
| Curated v1 | 8 / 10 | 9 / 10 | 1.5991248786449432 |
| Command-disambiguation v2 | 7 / 10 | 8 / 10 | 0.7426003813743591 |

The report makes the next balanced-gate target concrete: the new balanced
curriculum should beat the current best exact gate (`8 / 10`) while preserving
or improving behavior accuracy (`9 / 10`). Lower validation loss alone is not a
good enough signal, because v2 lowered validation loss while regressing exact
held-out behavior.

## AgentCoder Balanced-Curriculum Gate Run

Started Vast instance `43634442` successfully and ran the balanced curated gate:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_balanced_20260703T034213Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote validation:

- targeted generator test: `1 passed, 7 deselected in 1.50s`
- full suite: `26 passed in 32.63s`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_balanced_20260703T034213Z`.

Generated comparison report:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_balanced_20260703T034213Z.md`.

| Metric | Curated v1 | Command-disambiguation v2 | Balanced v3 |
| --- | ---: | ---: | ---: |
| Held-out exact pass rate | 8 / 10 | 7 / 10 | 8 / 10 |
| Behavior accuracy | 9 / 10 | 8 / 10 | 9 / 10 |
| Train records | 60 | 73 | 96 |
| Train tokens | 7679 | 8695 | 10567 |
| Validation tokens | 1543 | 2049 | 2804 |
| Final validation loss | 1.5991248786449432 | 0.7426003813743591 | 0.9329699128866196 |
| Final tokens/sec | 68901.96115779184 | 83803.19876940553 | 66871.47363583921 |

Balanced v3 matched the best exact and behavior result while using equalized
behavior-family counts, but it did not beat the `8 / 10` exact-pass target.
Failures:

- `curated_debugging`: answered with an `is_nonempty` function completion
  instead of the debugging process.
- `curated_repo_lookup`: used the repo-context behavior but named `config.py`
  instead of the expected `calc.py`.

Interpretation: balancing removed the command/risky/flag regressions introduced
by v2, but the tiny model still confuses nearby template slots when the same
symbol names appear across behaviors. The next useful step is a collision-aware
curated generator/eval split: avoid overlapping function names and filenames
between train and held-out cases unless they are intentionally testing lookup,
then add explicit negative examples for "debugging is not code completion" and
"repo-context answers must cite the file containing the implementation."

## AgentCoder Collision-Aware Gate Run

Added collision-aware changes to the curated generator:

- debugging supervision now explicitly says to explain the debugging process
  without writing code
- repo-context lookup supervision now says to cite the file that defines the
  function, not the file that imports it
- the held-out repo lookup case now uses `normalize_title` in `titles.py`, avoiding
  the previous overlap with training examples around `add`, `calc.py`, and
  `config.py`
- the curated manifest format is now `agentcoder-curated-sft-v3`

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/eval_overfit_sanity.py scripts/compare_agentcoder_gates.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_collision_v3/train.jsonl --cases-output work/curated_sft_collision_v3/cases.json --manifest-output work/curated_sft_collision_v3/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_collision_v3/train.jsonl --output work/curated_sft_collision_v3/tokenizer.json --vocab-size 1536
git diff --check
```

Remote validation and gate:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_collision_20260703T035241Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote test results:

- targeted generator test: `1 passed, 7 deselected in 1.46s`
- full suite: `27 passed in 32.18s`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_collision_20260703T035241Z`.

Generated comparison report:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_collision_20260703T035241Z.md`.

| Metric | Curated v1 | Command-disambiguation v2 | Balanced v3 | Collision-aware v3 |
| --- | ---: | ---: | ---: | ---: |
| Held-out exact pass rate | 8 / 10 | 7 / 10 | 8 / 10 | 8 / 10 |
| Behavior accuracy | 9 / 10 | 8 / 10 | 9 / 10 | 10 / 10 |
| Train records | 60 | 73 | 96 | 96 |
| Train tokens | 7679 | 8695 | 10567 | 11069 |
| Validation tokens | 1543 | 2049 | 2804 | 2856 |
| Final validation loss | 1.5991248786449432 | 0.7426003813743591 | 0.9329699128866196 | 0.9868301898241043 |

Collision-aware v3 improved the diagnostic signal: the model now chose the
correct behavior family for every held-out case. Exact pass rate did not improve
because the remaining misses are slot-copying/detail errors:

- `curated_repo_lookup`: used the right repo-lookup behavior, but answered
  `slugify` / `names.py` instead of copying `normalize_title` / `titles.py` from
  the provided context.
- `curated_flag_patch`: used the right boolean-flag patch behavior, but patched
  the remembered `cache_enabled(... == "on")` example instead of the held-out
  `is_enabled(... == "true")` file/value pair.

Interpretation: the next gate should target exact slot binding, not broader
behavior classification. Add examples that force copying the requested symbol,
file path, and target literal from the prompt/context, plus eval checks that
distinguish "right behavior, wrong slot" from "wrong behavior."

Results: targeted mirror-val test passed, script syntax passed, and the full
test suite passed with `22 passed in 32.73s`.

Overfit run:

```bash
/venv/main/bin/python scripts/run_agentcoder_overfit_sanity.py \
  --config configs/scratch/raam_agentcoder_overfit.yaml \
  --data examples/agentcoder_overfit_sanity.jsonl \
  --output-dir /root/raam-lm/runs/agentcoder_overfit_sanity_20260703T024025Z \
  --device cuda \
  --clean
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_overfit_sanity_20260703T024025Z`.
The pull includes the summary, exact eval JSON, qualitative samples, tokenizer,
packed manifests, train log, and the small final checkpoint. Both Vast RTX 5090
instances were verified stopped after the pull.

| Metric | Value |
| --- | ---: |
| Overfit pass rate | 8 / 8 |
| EOS generated | 8 / 8 |
| Valid JSON command case | 1 / 1 |
| Final train loss | 0.023516744375228882 |
| Final mirrored validation loss | 0.01699875178746879 |
| Last checkpoint step | 1199 |
| Tokens seen | 1843200 |
| Final tokens/sec | 50214.59541508371 |
| Peak allocated VRAM MB | 129.876953125 |
| Non-embedding params | 1244802 |
| Estimated FLOPs/token | 2238592 |

Interpretation: the data renderer, tokenizer coverage, prompt boundaries,
generation path, EOS handling, JSON/tool formatting, and patch/test-command
targets can be learned by a tiny RAAM model on a deliberately mirrored toy set.
This does not prove general chat or coding ability, but it does clear the most
basic pipeline sanity gate that failed qualitatively at Stage 5 scale. The next
step is to run a small real-data slice gate that is not mirrored: keep the same
exact eval prompts, train on a slightly larger curated subset, and require
held-out validation plus qualitative behavior to improve before launching a
larger paid continuation.

## AgentCoder Non-Mirrored Slice Gate

Added the next tiny non-mirrored gate:

- `examples/agentcoder_slice_train.jsonl`
- `examples/agentcoder_slice_eval_cases.json`
- `configs/scratch/raam_agentcoder_slice_gate.yaml`
- `scripts/run_agentcoder_slice_gate.py`
- `scripts/eval_overfit_sanity.py --cases-json`

Local validation before the Vast run:

```bash
python3 -m py_compile scripts/eval_overfit_sanity.py scripts/run_agentcoder_slice_gate.py scripts/run_agentcoder_overfit_sanity.py
python3 scripts/train_tokenizer.py examples/agentcoder_slice_train.jsonl --output work/slice_tokenizer_smoke/tokenizer.json --vocab-size 1024
git diff --check
```

Overfit-style but non-mirrored run on Vast RTX 5090:

```bash
/venv/main/bin/python scripts/run_agentcoder_slice_gate.py \
  --config configs/scratch/raam_agentcoder_slice_gate.yaml \
  --data examples/agentcoder_slice_train.jsonl \
  --cases-json examples/agentcoder_slice_eval_cases.json \
  --output-dir /root/raam-lm/runs/agentcoder_slice_gate_20260703T025341Z \
  --device cuda \
  --clean \
  --no-fail
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_slice_gate_20260703T025341Z`.
The pull includes `summary.json`, `slice_eval.json`, qualitative samples,
manifests, tokenizer, train log, runner log, and the small final checkpoint.
Both Vast RTX 5090 instances were verified stopped after the pull.

| Metric | Value |
| --- | ---: |
| Held-out slice pass rate | 3 / 6 |
| Train docs | 10 |
| Validation docs | 2 |
| Train tokens | 1295 |
| Validation tokens | 200 |
| Final train loss | 0.024610433727502823 |
| Final validation loss | 6.021189093589783 |
| Last checkpoint step | 1599 |
| Tokens seen | 2457600 |
| Final tokens/sec | 51556.69678444392 |
| Peak allocated VRAM MB | 130.5009765625 |

Held-out checks:

| Case | Result |
| --- | --- |
| add patch + pytest | pass |
| strict JSON Python-file command | pass |
| risky-edit clarifying question | fail |
| plain debugging process | fail |
| held-out `is_even` completion | fail |
| default Python test command | pass |

The failures are informative: the model copied the nearest memorized pattern
instead of generalizing. The risky-edit prompt produced the JSON command, the
debugging prompt produced the boolean-flag patch, and `is_even` became the
trained `is_odd` function. This means the pipeline is healthy enough to learn,
but the tiny non-mirrored gate is still memorization-heavy. The next training
step should be a broader curated supervised set with multiple paraphrases per
behavior and a separate held-out eval set, not an immediate large paid run.

## AgentCoder Curated Paraphrase SFT Gate

Added a deterministic broader SFT gate with multiple paraphrases per behavior:

- `scripts/make_agentcoder_curated_sft.py`
- `scripts/run_agentcoder_curated_gate.py`
- `configs/scratch/raam_agentcoder_curated_gate.yaml`

The generated gate contains 60 training records across 11 behavior families and
10 held-out eval cases. Behavior families include patching, strict JSON command
output, risky-edit clarification, debugging, function completion, stack-trace
diagnosis, repo-context lookup, test-command recommendation, code review, and
commit summaries.

Local validation before the Vast run:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/run_agentcoder_curated_gate.py scripts/eval_overfit_sanity.py scripts/run_agentcoder_slice_gate.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_smoke/train.jsonl --cases-output work/curated_sft_smoke/cases.json --manifest-output work/curated_sft_smoke/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_smoke/train.jsonl --output work/curated_sft_smoke/tokenizer.json --vocab-size 1536
git diff --check
```

Remote validation on Vast RTX 5090:

```bash
/venv/main/bin/python -m py_compile scripts/make_agentcoder_curated_sft.py scripts/run_agentcoder_curated_gate.py scripts/eval_overfit_sanity.py
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python -m pytest -q
```

Results: targeted generator test passed and the full test suite passed with
`23 passed in 32.76s`.

Curated gate run:

```bash
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_20260703T030727Z \
  --device cuda \
  --clean \
  --no-fail
```

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_20260703T030727Z`.
The pull includes generated train/eval data, manifests, tokenizer, packed data,
train log, runner log, exact eval JSON, qualitative samples, and the small final
checkpoint. Both Vast RTX 5090 instances were verified stopped after the pull.

| Metric | Value |
| --- | ---: |
| Held-out curated pass rate | 8 / 10 |
| Train records | 60 |
| Eval cases | 10 |
| Train docs | 48 |
| Validation docs | 12 |
| Train tokens | 7679 |
| Validation tokens | 1543 |
| Final train loss | 0.027493080124258995 |
| Final validation loss | 1.5991248786449432 |
| Last checkpoint step | 2599 |
| Tokens seen | 5324800 |
| Final tokens/sec | 68901.96115779184 |
| Peak allocated VRAM MB | 151.90576171875 |
| Non-embedding params | 1244802 |
| Estimated FLOPs/token | 2289792 |

Held-out checks:

| Case | Result |
| --- | --- |
| add patch + pytest | pass |
| strict JSON Python-file command | fail |
| risky-edit clarifying question | pass |
| plain debugging process | pass |
| held-out `is_even` completion | pass |
| stack-trace diagnosis | pass |
| repo-context lookup | pass |
| default Python test command | fail |
| `parse_port` review | pass |
| boolean flag patch | pass |

This is a meaningful improvement over the previous non-mirrored slice gate:
held-out pass rate improved from `3 / 6` to `8 / 10`, and final validation loss
improved from `6.0212` to `1.5991`. The remaining failures are command-intent
confusions: the Python-files JSON prompt produced the pytest JSON command, and
the default test-command prompt produced a repo-context answer. The next useful
step is to add a focused command-disambiguation mini-curriculum, then rerun this
curated gate before considering a larger supervised run or 100M continuation.

## AgentCoder Command-Disambiguation Rerun

Expanded the curated generator with a focused command-disambiguation
mini-curriculum:

- added 9 `command_disambiguation` examples
- increased direct `test_command` examples from 4 to 8
- updated the generator coverage test and eval docs

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/run_agentcoder_curated_gate.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_smoke_v2/train.jsonl --cases-output work/curated_sft_smoke_v2/cases.json --manifest-output work/curated_sft_smoke_v2/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_smoke_v2/train.jsonl --output work/curated_sft_smoke_v2/tokenizer.json --vocab-size 1536
git diff --check
```

Remote validation and rerun:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_cmdfix_20260703T031643Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Results: targeted generator test passed and the full test suite passed with
`23 passed in 32.38s`.

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_cmdfix_20260703T031643Z`.
The pull includes generated train/eval data, manifests, tokenizer, packed data,
train log, runner log, exact eval JSON, qualitative samples, and the small final
checkpoint. Both Vast RTX 5090 instances were verified stopped after the pull.

| Metric | Curated v1 | Command-disambiguation v2 |
| --- | ---: | ---: |
| Held-out pass rate | 8 / 10 | 7 / 10 |
| Train records | 60 | 73 |
| Train tokens | 7679 | 8695 |
| Validation tokens | 1543 | 2049 |
| Final train loss | 0.027493080124258995 | 0.029969634488224983 |
| Final validation loss | 1.5991248786449432 | 0.7426003813743591 |
| Final tokens/sec | 68901.96115779184 | 83803.19876940553 |
| Peak allocated VRAM MB | 151.90576171875 | 151.93505859375 |

The command-focused examples fixed the two original command failures:

- strict JSON Python-file command: fail -> pass
- default Python test command: fail -> pass

But the update introduced three exact-behavior regressions:

- risky-edit clarifying question produced the Python-file JSON command
- held-out `is_even` completion regressed to `is_odd`
- boolean flag patch regressed to an add patch

Interpretation: the lower validation loss shows the broader synthetic
curriculum is easier for the model to model, but exact held-out behavior is
still unstable and sensitive to small distribution shifts. The best exact gate
result remains curated v1 at `8 / 10`; command-disambiguation v2 is useful
diagnostic evidence, not a better candidate. The next step should be a balanced
curriculum with equalized behavior-family counts and a confusion-matrix style
eval summary, rather than simply adding more examples for whichever behavior
failed last.

## AgentCoder Balanced-Curriculum Gate Prep

Implemented the next diagnostic gate step:

- balanced the deterministic curated SFT generator to 8 examples for each of 12
  behavior families, for 96 train records total
- added `expected_behavior` labels to all 10 held-out eval cases
- added behavior inference and a confusion matrix to `scripts/eval_overfit_sanity.py`
- surfaced behavior accuracy/confusion in the curated gate summary
- updated the generator coverage test and eval docs

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/eval_overfit_sanity.py scripts/run_agentcoder_curated_gate.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_balanced_smoke/train.jsonl --cases-output work/curated_sft_balanced_smoke/cases.json --manifest-output work/curated_sft_balanced_smoke/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_balanced_smoke/train.jsonl --output work/curated_sft_balanced_smoke/tokenizer.json --vocab-size 1536
git diff --check
```

Generated manifest:

```json
{
  "balanced_behavior_target": 8,
  "train_records": 96,
  "eval_cases": 10,
  "format": "agentcoder-curated-sft-v2"
}
```

The local focused pytest collection could not run because this host's Python
does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Attempted to start Vast instance `43634442` for the remote RTX 5090 gate, but
Vast returned:

```text
Required resources are currently unavailable, state change queued.
```

After a longer poll, both Vast RTX 5090 instances still reported
`actual_status: exited` and `cur_state: stopped`, so no paid gate was run. The
next resume step is to retry starting `43634442` and run:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_balanced_<UTC_TIMESTAMP> \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

## AgentCoder Slot-Diagnostic Gate Run

Implemented and ran the next AgentCoder gate increment to make exact
slot-binding failures measurable instead of only visible by manual inspection.

Code changes:

- added optional `forbidden_substrings` to curated eval cases
- made `scripts/eval_overfit_sanity.py` fail cases when forbidden text appears
- added `present_forbidden_substrings` and `slot_error` to eval artifacts
- extended `scripts/compare_agentcoder_gates.py` to report forbidden hits and
  slot errors in JSON and Markdown
- updated the curated generator to format boolean-flag examples with stronger
  file/helper/literal copy instructions
- documented slot-copy diagnostics in `docs/AGENTIC_CODING_EVALS.md`

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/eval_overfit_sanity.py scripts/compare_agentcoder_gates.py
python3 -m pytest -q tests/test_compare_agentcoder_gates.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_slot_v4/train.jsonl --cases-output work/curated_sft_slot_v4/cases.json --manifest-output work/curated_sft_slot_v4/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_slot_v4/train.jsonl --output work/curated_sft_slot_v4/tokenizer.json --vocab-size 1536
git diff --check
```

Results: syntax checks passed, focused comparison tests passed with
`5 passed`, the v4 generator emitted `96` train records and `10` eval cases,
and tokenizer training produced `vocab_size=849`.

The local focused generator pytest collection still cannot run on this host
because local Python does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Remote RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_slot_20260703T040601Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote results:

- focused generator test: `1 passed, 7 deselected`
- full remote suite: `28 passed in 32.49s`
- run id: `agentcoder_curated_gate_slot_20260703T040601Z`
- pass rate: `8 / 10`
- behavior accuracy: `9 / 10`
- final validation loss: `0.9754665791988373`
- final train loss: `0.037981100380420685`
- final tokens/sec: `68409.00513840611`
- train records: `96`
- train tokens: `11459`
- validation tokens: `3042`
- non-embedding params: `1244802`
- estimated FLOPs/token: `2327936`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_slot_20260703T040601Z`.

Comparison report:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_slot_20260703T040601Z.md`.

| Metric | Curated v1 | Command-disambiguation v2 | Balanced v3 | Collision-aware v3 | Slot-diagnostic v4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Held-out pass rate | 8 / 10 | 7 / 10 | 8 / 10 | 8 / 10 | 8 / 10 |
| Behavior accuracy | 9 / 10 | 8 / 10 | 9 / 10 | 10 / 10 | 9 / 10 |
| Train records | 60 | 73 | 96 | 96 | 96 |
| Train tokens | 7679 | 8695 | 10567 | 11069 | 11459 |
| Validation tokens | 1543 | 2049 | 2804 | 2856 | 3042 |
| Final validation loss | 1.5991248786449432 | 0.7426003813743591 | 0.9329699128866196 | 0.9868301898241043 | 0.9754665791988373 |
| Final tokens/sec | 68901.96115779184 | 83803.19876940553 | 66871.47363583921 | 65994.03447703178 | 68409.00513840611 |

Slot-diagnostic v4 did not improve exact pass rate over collision-aware v3.
It did improve artifact quality by preserving forbidden-substring and
`slot_error` fields for future comparisons, but the training change regressed
one behavior-family decision:

- `curated_repo_lookup`: predicted repo lookup behavior, but copied
  `render_invoice` / `invoices.py` instead of `normalize_title` / `titles.py`;
  `slot_error=true`
- `curated_flag_patch`: produced the addition-patch template for `calc.py`
  instead of the flag patch for `flags.py`; predicted behavior was
  `patch_addition`, so this is a behavior regression rather than a pure slot
  error

Interpretation: the best diagnostic run remains collision-aware v3 because it
had `10 / 10` behavior accuracy even though exact pass stayed `8 / 10`.
Slot-diagnostic v4 is useful as scoring infrastructure, but not as a better
candidate model. The next model-quality step should separate patch families
more strongly before training: make boolean-flag prompts structurally distinct
from arithmetic patch prompts, add held-out file/helper/value copy drills, and
keep forbidden-substring scoring enabled so exact-slot regressions are visible.

After artifact pull, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## AgentCoder Patch-Family Separation Gate Run

Implemented and ran the next curated AgentCoder gate step, focused on separating
boolean-flag repair from arithmetic/addition patch templates.

Code changes:

- rewrote boolean-flag training prompts as explicit "Boolean flag task, not
  arithmetic" records
- removed focused pytest-command language from boolean-flag assistant traces
- made the held-out flag case require exact `flags.py`, `is_enabled`, and
  enabled-literal slots
- added forbidden substrings for stale addition-patch completions such as
  `calc.py`, `def add`, and `return a + b`
- strengthened repo-lookup prompts to ask for the exact requested symbol and
  defining file
- documented patch-family collision handling in
  `docs/AGENTIC_CODING_EVALS.md`

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/eval_overfit_sanity.py scripts/compare_agentcoder_gates.py
python3 -m pytest -q tests/test_compare_agentcoder_gates.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_patchsplit_v5/train.jsonl --cases-output work/curated_sft_patchsplit_v5/cases.json --manifest-output work/curated_sft_patchsplit_v5/manifest.json
python3 scripts/train_tokenizer.py work/curated_sft_patchsplit_v5/train.jsonl --output work/curated_sft_patchsplit_v5/tokenizer.json --vocab-size 1536
git diff --check
```

Results: syntax checks passed, focused comparison tests passed with
`5 passed`, the v5 generator emitted `96` train records and `10` eval cases,
and tokenizer training produced `vocab_size=855`.

The local focused generator pytest collection still cannot run on this host
because local Python does not have `torch` installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Remote RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_curated_gate_patchsplit_20260703T042042Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote results:

- focused generator test: `1 passed, 7 deselected`
- full remote suite: `28 passed in 33.04s`
- run id: `agentcoder_curated_gate_patchsplit_20260703T042042Z`
- pass rate: `9 / 10`
- behavior accuracy: `10 / 10`
- final validation loss: `1.028306633234024`
- final train loss: `0.01835857890546322`
- final tokens/sec: `68380.61845104837`
- train records: `96`
- train tokens: `11757`
- validation tokens: `3120`
- non-embedding params: `1244802`
- estimated FLOPs/token: `2329472`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_patchsplit_20260703T042042Z`.

Comparison report:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_patchsplit_20260703T042042Z.md`.

| Metric | Curated v1 | Command-disambiguation v2 | Balanced v3 | Collision-aware v3 | Slot-diagnostic v4 | Patch-split v5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Held-out pass rate | 8 / 10 | 7 / 10 | 8 / 10 | 8 / 10 | 8 / 10 | 9 / 10 |
| Behavior accuracy | 9 / 10 | 8 / 10 | 9 / 10 | 10 / 10 | 9 / 10 | 10 / 10 |
| Train records | 60 | 73 | 96 | 96 | 96 | 96 |
| Train tokens | 7679 | 8695 | 10567 | 11069 | 11459 | 11757 |
| Validation tokens | 1543 | 2049 | 2804 | 2856 | 3042 | 3120 |
| Final validation loss | 1.5991248786449432 | 0.7426003813743591 | 0.9329699128866196 | 0.9868301898241043 | 0.9754665791988373 | 1.028306633234024 |
| Final tokens/sec | 68901.96115779184 | 83803.19876940553 | 66871.47363583921 | 65994.03447703178 | 68409.00513840611 | 68380.61845104837 |

Patch-split v5 is the best curated gate result so far. It fixed the prior
boolean-flag regression and kept behavior accuracy at `10 / 10`. The only
remaining held-out failure is a repo-lookup slot-copy error:

- `curated_repo_lookup`: predicted repo lookup behavior, but copied
  `add` / `calc.py` instead of `normalize_title` / `titles.py`;
  `present_forbidden_substrings=["add is implemented", "calc.py"]`;
  `slot_error=true`

Interpretation: the next useful model-quality step is targeted symbol/file
binding, not broader behavior-family balancing. Add an exact repo-context lookup
copy drill that varies the requested symbol and import/definition file while
keeping the answer format fixed, then rerun the curated gate. If that reaches
`10 / 10`, graduate this tiny gate from debugging to a reusable preflight before
larger chat/coding training.

After artifact pull, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## AgentCoder Repo-Lookup and Slot-Copy Follow-Up

Ran three follow-up curated gates after patch-split v5 to test whether the tiny
RAAM-AgentCoder setup can bind exact file/symbol slots from repo context instead
of replaying a nearby training example.

Code changes pushed:

- `3ed2b1f Add repo lookup distractor curriculum`
- `f01f533 Tighten AgentCoder slot diagnostics`
- `04de904 Fix curated repo lookup prompt contract`
- `17975e3 Add explicit slot-copy AgentCoder drills`

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_curated_sft.py scripts/eval_overfit_sanity.py scripts/compare_agentcoder_gates.py
python3 scripts/make_agentcoder_curated_sft.py --train-output work/curated_sft_slotcopy_v8/train.jsonl --cases-output work/curated_sft_slotcopy_v8/cases.json --manifest-output work/curated_sft_slotcopy_v8/manifest.json
git diff --check
```

Result: syntax and generator checks passed. The v8 generator emitted `96`
records, `10` eval cases, balanced `8` records per behavior, and format
`agentcoder-curated-sft-v8`. Local focused pytest still cannot collect on this
host because local Python lacks `torch`; the same focused test passed on Vast.

Remote RTX 5090 validation command shape for each run:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k curated_sft_generator
/venv/main/bin/python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/<RUN_ID> \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

All three follow-up runs completed on the RTX 5090 and full remote pytest passed
with `28 passed` each time.

| Run | Format focus | Exact pass | Behavior accuracy | Final val loss | Final tokens/sec | Failed cases |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `agentcoder_curated_gate_lookup_20260703T043243Z` | repo distractors | `8 / 10` | `9 / 10` | `1.0250842720270157` | `66937.11065241446` | add patch became test-command; repo lookup copied `build_parser` / `title_tools.py` |
| `agentcoder_curated_gate_v7_20260703T044053Z` | safer title/normalize drills | `8 / 10` | `10 / 10` | `0.9177761673927307` | `68737.39410439291` | add patch copied `arithmetic.py`; repo lookup copied `normalize_user` / `validators.py` |
| `agentcoder_curated_gate_v8_20260703T044746Z` | explicit slot-copy prompts | `8 / 10` | `10 / 10` | `0.8717429041862488` | `68172.28443860989` | add patch copied `mathlib.py`; repo lookup copied `save_user` / `slugs.py` |

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_lookup_20260703T043243Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_v7_20260703T044053Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_curated_gate_v8_20260703T044746Z`

Comparison report:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/agentcoder_gate_comparison_slotcopy_20260703T044746Z.md`.

Interpretation: patch-split v5 remains the best exact-pass curated gate at
`9 / 10`. v7 and v8 improved behavior accuracy to `10 / 10` and lowered
validation loss, but exact pass rate stayed at `8 / 10` because the model still
replays stale file/symbol slots. This is no longer a behavior-family problem; it
is a context slot-binding problem. Do not launch a larger paid chat/coding run
as if this is solved. The next useful step is a bigger programmatic slot-copy
curriculum and eval set with many repo-context file/function/literal
combinations, plus a stronger held-out report, before returning to Stage 5
continuation.

After artifact pulls, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## 2026-07-03 - Request-Key Value Copy Route

Goal: improve the one-value AgentCoder key-value copy gate for the chat/coding
target without retraining on external data. The route is deliberately a
falsifiable scratch mechanism, not evidence that RAAM works scientifically.

Commits:

- `2bfeb86` - add request-key value copy route
- `56484bb` - make the route eval-only and softer
- `3c81771` - constrain the query to the requested-key span
- `d36a500` - ignore query separators and source only from repo_context
- `d7c8180` - prevent continuation from crossing value-line stop tokens

Local validation:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/copy_head.py tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- syntax checks passed
- focused non-Torch local tests passed: `43 passed in 0.25s`
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_pipeline.py -k "copy_head or request_value or keyvalue or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards"
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_request_value_evalonly_onefield_compare_20260703T121915Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --max-new-tokens 48 --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_request_value_evalonly_onefield_compare_20260703T121915Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --max-new-tokens 48 --clean --no-fail
```

Remote focused tests after the final route fix passed:

- `36 passed, 8 deselected, 1 warning`

Request-route eval results on the same trained checkpoints:

| Route / Model | Pass | Seen | Covered Value | Heldout | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Initial strong request route, RAAM | 0 / 96 | 0 / 32 | 0 / 32 | 0 / 32 | Overpowered and copied key/file-like tokens instead of values. |
| Eval-only query-filtered, RAAM | 58 / 96 | 32 / 32 | 26 / 32 | 0 / 32 | Fixed first-token value lookup but missed `.py` continuations. |
| Eval-only query-filtered, Transformer | 58 / 96 | 32 / 32 | 26 / 32 | 0 / 32 | Same result as RAAM. |
| Continuation 24 without segment stop, RAAM | 17 / 96 | 9 / 32 | 8 / 32 | 0 / 32 | Over-copied into later repo-context lines. |
| Continuation 24 with same-line stop, RAAM | 64 / 96 | 32 / 32 | 32 / 32 | 0 / 32 | Best RAAM result; all seen and tokenizer-covered cases pass. |
| Continuation 24 with same-line stop, Transformer | 63 / 96 | 32 / 32 | 31 / 32 | 0 / 32 | One covered duplication remains. |

Representative corrected completions:

- `copy_service_000` -> `copy_service_000\n<eos>`
- `disabled_file_001` -> `disabled_file_001\n<eos>`
- `copy_adapter_002.py` -> `copy_adapter_002.py\n<eos>`

Remaining failure:

- heldout byte-fallback values still fail, e.g. `case_adapter_096` generated
  `cadadadadadadadadadadadadadadadadadadadadadadada`.
- This means the route can retrieve single learned value tokens and
  tokenizer-covered multi-token filename values, but not unseen byte-level value
  continuations yet.

Artifacts pulled without checkpoint weights:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_request_value_evalonly_onefield_compare_20260703T121915Z`

Remote cleanup:

- deleted `last.pt` for both RAAM and Transformer under the request-route run
- verified no `.pt` or `.safetensors` remained under that run directory
- verified both Vast RTX 5090 instances were `exited` after stop:
  `43627905 exited`, `43634442 exited`

Interpretation: the request-key copy route is now a useful diagnostic scaffold
for exact repo-context value retrieval. It improves the one-value gate versus
the previous unwrapped one-value results (`RAAM 32/96`, Transformer `35/96`),
but the improvement is not RAAM-specific because the Transformer baseline gets
nearly the same lift. The next highest-value step is to make byte-fallback
continuation copy robust without hardcoding token ids, or to switch this probe
to tokenizer-stable delimiters before returning to larger chat/coding training.

## Pair-Consistency Copy-Head Gate

Added an opt-in copy-head consistency bias for the fragile atomic binding gate.
The bias compares the current token plus recent context against the causal
source-side neighborhood before each candidate copied token. The goal is to
prefer a file token whose source row also contained the just-emitted symbol,
instead of copying a valid file token from a different row.

Added:

- `CopyHeadConfig.consistency_strength`
- `CopyHeadConfig.consistency_recent_tokens`
- `CopyHeadConfig.consistency_source_window`
- `configs/scratch/raam_agentcoder_atomic_hybrid1_pair_copy_head_gate.yaml`
- a focused same-row preference test in `tests/test_copy_head.py`
- config coverage in `tests/test_agentcoder_atomic_cardinality_sweep.py`

Local validation:

```bash
python3 -m py_compile src/raam_lm/copy_head.py src/raam_lm/config.py src/raam_lm/flops.py tests/test_copy_head.py tests/test_agentcoder_atomic_cardinality_sweep.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Results:

- syntax checks passed
- focused non-Torch local tests passed: `33 passed in 0.12s`
- `git diff --check` passed
- local `python3 -m pytest -q tests/test_copy_head.py` is blocked in this
  workspace by `ModuleNotFoundError: No module named 'torch'`; the same test was
  run on the Vast PyTorch environment below

Remote validation first exposed a real test failure:

- run root: `agentcoder_atomic_hybrid1_pair_copy_head_seed029_steps2400_20260703T092545Z`
- failure: `test_causal_copy_head_consistency_bias_prefers_same_source_row`
- cause: the current symbol and an older recent token could tie the two source
  rows
- fix: weight the current token more strongly than older recent tokens in the
  consistency bias

Vast RTX 5090 validation after the fix:

```bash
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_pair_copy_head_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_pair_copy_head_seed029_steps2400_20260703T092706Z/raam_seed029_pair_copy_head \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --clean \
  --no-fail
```

Remote results:

- copy-head and masking tests: `10 passed, 8 deselected`
- atomic config/generator tests: `33 passed in 0.15s`
- train records: `64`
- eval cases: `64`
- assistant-only loss: `true`
- behavior accuracy: `64 / 64`
- exact pass: `63 / 64`
- final validation next-token loss: `0.266976`
- tokens/sec: `14011.4`
- estimated FLOPs/token: `1964000`

Remaining failure:

- `atomic_mirror_009`
- expected `symbol=copy_symbol_009` and `file=copy_file_009.py`
- completed `symbol=copy_symbol_009` and `file=copy_file_010.py`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_pair_copy_head_seed029_steps2400_20260703T092706Z`.
Checkpoint weights were not pulled.

Interpretation: the pair-consistency bias is valid enough to keep as an
opt-in diagnostic, but it did not improve the prior best `63 / 64` seed-29
copy-head result. It changed the remaining error from the earlier far-row
`009`/`058` swap to an adjacent-row `009`/`010` file slip. The next useful model
step is a stronger binding-carry diagnostic: after emitting the copied symbol,
track or reselect the source row position and constrain the following copied
slot value to come from that same row, then re-run the same `64`-binding gate.

## Binding-Carry and Key-Follow Copy-Head Gates

Added two opt-in diagnostics after the pair-consistency bias failed to clear the
seed-29 gate:

- `binding_carry_*`: a rare-token row-tail carry route that boosts source tokens
  following a recently emitted rare anchor
- `key_follow_*`: a slot-key route that, after an answer prefix such as
  `file=`, copies the value token at a fixed offset after the same key in a
  bounded source region

Added:

- `configs/scratch/raam_agentcoder_atomic_hybrid1_binding_carry_copy_head_gate.yaml`
- `configs/scratch/raam_agentcoder_atomic_hybrid1_key_follow_copy_head_gate.yaml`
- focused copy-head tests for binding-carry, key-follow, and causal
  future-token perturbation
- config coverage in `tests/test_agentcoder_atomic_cardinality_sweep.py`
- FLOP accounting for the extra diagnostic routes

Local validation:

```bash
python3 -m py_compile src/raam_lm/copy_head.py src/raam_lm/config.py src/raam_lm/flops.py tests/test_copy_head.py tests/test_agentcoder_atomic_cardinality_sweep.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 scripts/estimate_flops.py --config configs/scratch/raam_agentcoder_atomic_hybrid1_binding_carry_copy_head_gate.yaml
python3 scripts/estimate_flops.py --config configs/scratch/raam_agentcoder_atomic_hybrid1_key_follow_copy_head_gate.yaml
git diff --check
```

Results:

- syntax checks passed
- focused non-Torch local tests passed after binding-carry: `34 passed in 0.13s`
- focused non-Torch local tests passed after key-follow: `35 passed in 0.13s`
- binding-carry config FLOPs/token estimate: `2259168`
- key-follow config FLOPs/token estimate: `2107296`
- `git diff --check` passed
- local `tests/test_copy_head.py` remains blocked by missing local `torch`; the
  same tests were run on Vast below

Vast RTX 5090 binding-carry gate:

```bash
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_binding_carry_copy_head_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_binding_carry_copy_head_seed029_steps2400_20260703T094155Z/raam_seed029_binding_carry_copy_head \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --clean \
  --no-fail
```

Binding-carry results:

- copy-head and masking tests: `12 passed, 8 deselected`
- atomic config/generator tests: `34 passed in 0.16s`
- exact pass: `52 / 64`
- behavior accuracy: `64 / 64`
- final validation next-token loss: `0.214981`
- tokens/sec: `15176.7`
- estimated FLOPs/token: `2111456`

Representative binding-carry failures:

- `atomic_mirror_002` completed `copy_symbol_021` / `copy_file_021.py`
- `atomic_mirror_009` completed `copy_symbol_063` / `copy_file_063.py`
- `atomic_mirror_061` completed `copy_symbol_012` / `copy_file_012.py`

Interpretation: direct rare-token binding carry is a negative result. It lowers
validation loss but worsens exact binding by encouraging wrong but internally
consistent pairs. Keep it as an opt-in diagnostic only; do not use it for the
current AgentCoder training path.

Copy-logit-scale diagnostic on the prior best copy-head checkpoint:

```bash
/venv/main/bin/python scripts/eval_overfit_sanity.py \
  --config <copy-head config with logit_scale swept over 4,5,6,8,10,12> \
  --tokenizer /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/tokenizer.json \
  --checkpoint /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/train/checkpoints/last.pt \
  --device cuda \
  --cases-json /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/generated/atomic_eval_cases.json \
  --max-new-tokens 24 \
  --no-fail
```

Results:

- scales `4`, `5`, `6`, `8`, and `10`: still `63 / 64`, same
  `atomic_mirror_009` file failure, completing `copy_file_058.py`
- scale `12`: still `63 / 64`, but over-boosted into
  `file=copy_symbol_009.py`

Interpretation: the remaining copy-head failure is source selection, not just
copy-logit strength.

Vast RTX 5090 key-follow eval-only diagnostic:

```bash
/venv/main/bin/python scripts/eval_overfit_sanity.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_key_follow_copy_head_gate.yaml \
  --tokenizer /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/tokenizer.json \
  --checkpoint /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/train/checkpoints/last.pt \
  --device cuda \
  --cases-json /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head/generated/atomic_eval_cases.json \
  --max-new-tokens 24 \
  --no-fail
```

Result: `64 / 64` exact pass, behavior accuracy `64 / 64`.

Vast RTX 5090 key-follow full train/eval gate:

```bash
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_key_follow_copy_head_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_key_follow_copy_head_seed029_steps2400_20260703T095120Z/raam_seed029_key_follow_copy_head \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --clean \
  --no-fail
```

Key-follow results:

- copy-head and masking tests: `14 passed, 8 deselected`
- atomic config/generator tests: `35 passed in 0.15s`
- exact pass: `64 / 64`
- behavior accuracy: `64 / 64`
- final validation next-token loss: `0.227288`
- tokens/sec: `21475.8`
- estimated FLOPs/token: `1959584`

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_binding_carry_copy_head_seed029_steps2400_20260703T094155Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_copy_head_logit_scale_eval_20260703T094621Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_key_follow_eval_only_20260703T095041Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_key_follow_copy_head_seed029_steps2400_20260703T095120Z`

Checkpoint weights were not pulled.

Interpretation: key-follow is the first efficient RAAM route to clear the
fragile seed-29 `64`-binding mirror gate without preserving every token as an
anchor. It is still a narrow structured-copy diagnostic, not evidence that the
model is ready for broad chat/coding training. The next useful step is to
generalize the slot-copy gate beyond the synthetic `symbol`/`file` pair:
multiple key names, multiple value offsets/formats, repo-context distractors,
held-out keys/values, and then compare RAAM key-follow against the Transformer
copy-head baseline under the same gate.

## Key-Value Copy Ladder

Added a broader key-value copy gate to stress the key-follow route beyond the
atomic `symbol`/`file` pair. The new gate asks the model to copy three requested
`key=value` lines from a repo-context table containing seven `key: value` lines:
three targets and four distractors.

Added:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml`
- `configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml`
- `tests/test_agentcoder_keyvalue_copy_generator.py`

The generator covers:

- keys: `symbol`, `file`, `helper`, `module`, `route`, `fixture`, `setting`,
  `adapter`, `service`, `endpoint`, `task`, `parser`
- value formats: identifier-like strings, `.py` filenames, flag-like strings,
  and case-id strings
- seen and held-out eval tiers
- forbidden `key=value` strings for distractor fields

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py scripts/eval_overfit_sanity.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
rm -rf work/keyvalue_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/keyvalue_smoke/train.jsonl --cases-output work/keyvalue_smoke/cases.json --manifest-output work/keyvalue_smoke/manifest.json --eval-mode ladder --train-records 12 --eval-cases 8
python3 scripts/estimate_flops.py --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml
python3 scripts/estimate_flops.py --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml
git diff --check
```

Results:

- syntax checks passed
- focused local tests passed: `36 passed in 0.24s`
- smoke manifest emitted `12` train records and `16` eval cases
- key-value RAAM config FLOPs/token estimate after aligned key-follow:
  `2788608`
- key-value Transformer config FLOPs/token estimate after aligned key-follow:
  `3381376`
- `git diff --check` passed

Implementation fixes prompted by the gate:

- Fixed seen-tier generation so seen eval cases reuse the exact trained slot
  tuples rather than only the same record index.
- Extended `infer_behavior` to classify all new key names as
  `copy_slot_values`.
- Added aligned key-follow mode so a generated `key=<partial value>` can copy
  the corresponding later token from `key: <value>` in source context.
- Added an assistant-output guard so key-follow does not trigger from
  instructional text such as `key=value lines` in the user prompt.

Vast RTX 5090 comparison:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_key_follow_compare_20260703T100438Z/raam \
  --device cuda --seed 29 --train-records 96 --eval-cases 32 \
  --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode ladder --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_key_follow_compare_20260703T100438Z/transformer \
  --device cuda --seed 29 --train-records 96 --eval-cases 32 \
  --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode ladder --clean --no-fail
```

Remote results before aligned/OOV fixes:

- focused remote tests: `20 passed, 8 deselected`

| Model | Exact Pass | Seen Pass | Held-Out Pass | Behavior Accuracy | Val Loss | Tokens/sec | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM key-value key-follow | 29 / 64 | 29 / 32 | 0 / 32 | 64 / 64 | 0.122228 | 26101.1 | 2141440 |
| Transformer key-value key-follow | 32 / 64 | 32 / 32 | 0 / 32 | 64 / 64 | 0.060538 | 28913.9 | 2734208 |

Held-out failures were not RAAM-specific. Both models preserved the requested
behavior but failed all held-out exact values. Representative held-out
completions copied only value prefixes such as `c`, `h`, or fragments from seen
training rows.

Aligned key-follow eval-only diagnostic on the same checkpoints:

- Remote tests for aligned key-follow: `9 passed, 9 deselected`
- RAAM aligned eval-only: `4 / 64`
- Transformer aligned eval-only: `4 / 64`

This was worse because the aligned route fired from prompt text before the
assistant answer began.

Assistant-guarded aligned key-follow eval-only diagnostic on the same
checkpoints:

- Remote tests for guarded key-follow: `10 passed, 9 deselected`
- RAAM guarded aligned eval-only: `29 / 64`
- Transformer guarded aligned eval-only: `32 / 64`

The guard restored the original comparison result but did not solve held-out
exact copying. The remaining failures are byte/OOV value copying failures caused
by training the tiny tokenizer only on training JSONL: held-out values such as
`copy_file_096.py` are split into byte-fallback or partial tokens, and the
current model/copy route does not yet reliably generate the full byte-token
sequence.

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_keyvalue_key_follow_compare_20260703T100438Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_keyvalue_aligned_eval_only_20260703T101030Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_keyvalue_aligned_guard_eval_only_20260703T101319Z`

Checkpoint weights were not pulled.

Prefix-matched span-copy follow-up:

Added `key_follow_match_value_prefix` so continuation tokens must follow the
same already-generated value prefix in the source span. Direct route inspection
on `keyvalue_heldout_000` showed the deterministic route itself points to the
right bytes:

- `adapter=` -> `c`
- `adapter=c` -> `a`
- `adapter=ca` -> `s`
- `adapter=case_` -> `a`

However, combined-logit inspection on the RAAM checkpoint showed the learned
model still strongly prefers newline after the first byte:

| Partial | Correct Route Token | Correct Logit at Strength 10 | Strong Competing Token | Competing Logit |
| --- | --- | ---: | --- | ---: |
| `adapter=` | `c` | 10.0 | seen whole-token adapter | 5.433 |
| `adapter=c` | `a` | 10.014 | newline | 25.571 |
| `adapter=ca` | `s` | 10.0 | `module` | 15.599 |

Eval-only prefix-match results on the same checkpoints:

- remote focused tests: `11 passed, 9 deselected`
- RAAM prefix-match eval-only: `29 / 64`
- Transformer prefix-match eval-only: `32 / 64`

RAAM key-follow strength sweep on the same checkpoint:

| Key-follow Strength | Exact Pass |
| ---: | ---: |
| 10 | 29 / 64 |
| 14 | 28 / 64 |
| 18 | 3 / 64 |
| 24 | 0 / 64 |
| 32 | 0 / 64 |
| 48 | 0 / 64 |

Higher strength over-copied source fragments and collapsed generation rather
than solving held-out byte-copying.

Split first-token vs continuation-copy diagnostic:

- kept first-token key-follow strength at `10`
- tried continuation strength `30`
- blocked newline source tokens from the deterministic copy route

Remote eval-only results:

- focused tests: `11 passed, 9 deselected`
- RAAM split-continuation eval-only: `2 / 64`
- Transformer split-continuation eval-only: `0 / 64`

This was also a negative result: the high continuation route copied source
context fragments such as `module: ...` and repeated source rows. The key-value
configs were restored to conservative continuation strength `0.0`, which falls
back to the base key-follow strength.

Additional local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_keyvalue_prefix_match_eval_only_20260703T102150Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_keyvalue_prefix_strength_sweep_20260703T102303Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_keyvalue_split_continuation_eval_only_20260703T103147Z`

Checkpoint weights were not pulled.

Interpretation: key-follow cleared the atomic in-vocabulary `symbol`/`file`
mirror gate, but the broader key-value ladder falsifies the idea that it is
already enough for agentic coding. The next useful step is an OOV-aware copy
diagnostic: either train/evaluate with a tokenizer coverage split that separates
slot-binding from byte-generation, or add a true sequential span-copy route that
can copy byte-fallback values from the current context without relying on the
value being a learned whole token.

## AgentCoder Key-Value Coverage Ladder

Added a `coverage_ladder` eval mode to
`scripts/make_agentcoder_keyvalue_copy_sft.py` and
`scripts/run_agentcoder_keyvalue_copy_gate.py`. It keeps the original
`seen_slot` and `heldout_slot` tiers and adds `covered_value_slot`, where the
prompt/target selection changes but every target and distractor value is drawn
from key-value rows that appeared in the training JSONL. This separates
tokenizer/value coverage from true key-lookup generalization.

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
rm -rf work/keyvalue_coverage_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/keyvalue_coverage_smoke/train.jsonl --cases-output work/keyvalue_coverage_smoke/cases.json --manifest-output work/keyvalue_coverage_smoke/manifest.json --eval-mode coverage_ladder --train-records 12 --eval-cases 4
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
bash -n scripts/vast_pull_artifacts.sh
```

Results:

- generator test file: `7 passed`
- related copy-generator bundle: `37 passed`
- smoke manifest: 12 train records and 12 eval cases split evenly across
  `seen_slot`, `covered_value_slot`, and `heldout_slot`
- `git diff --check` passed
- `vast_pull_artifacts.sh` syntax check passed

Also tightened `scripts/vast_pull_artifacts.sh` so tar-mode pulls exclude nested
checkpoint weights under paths such as `raam/train/checkpoints/*.pt` and
`transformer/train/checkpoints/*.pt`.

Pushed `d420acc` and ran the coverage ladder on Vast RTX 5090 instance
`43634442`:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_coverage_ladder_compare_20260703T104034Z/raam \
  --device cuda --seed 29 --train-records 96 --eval-cases 32 \
  --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode coverage_ladder --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_coverage_ladder_compare_20260703T104034Z/transformer \
  --device cuda --seed 29 --train-records 96 --eval-cases 32 \
  --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode coverage_ladder --clean --no-fail
```

Remote focused tests: `24 passed, 8 deselected`.

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out Pass | Behavior Accuracy | Val Loss | Tokens/sec | FLOPs/token | Non-embedding Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM key-value key-follow | 31 / 96 | 31 / 32 | 0 / 32 | 0 / 32 | 96 / 96 | 0.094837 | 26782.8 | 2524160 | 1208962 |
| Transformer key-value key-follow | 16 / 96 | 16 / 32 | 0 / 32 | 0 / 32 | 96 / 96 | 0.342092 | 22171.1 | 3116928 | 1082496 |

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_keyvalue_coverage_ladder_compare_20260703T104034Z`

Checkpoint weights were removed from the local artifact copy after verifying the
tar-mode pull exclusion gap.

Interpretation: the previous OOV-only explanation was incomplete. The new
covered-value tier uses values present in training text, yet both RAAM and the
Transformer fail `0 / 32`. The tiny gates are mostly memorizing exact prompt
layouts rather than learning a reusable repo-context key lookup algorithm. RAAM
is currently better than the Transformer on exact seen prompts for this run, but
neither model has the slot-binding robustness needed for agentic coding.

Next useful step: change the training generator, not the architecture knob. For
each value row, train multiple randomized target/distractor/order variants so
the model cannot pass by memorizing one exact context layout, then rerun
`coverage_ladder`. A good gate should first reach high `covered_value_slot`
accuracy before spending more time on OOV byte-copy mechanics.

## AgentCoder Key-Value Variant Training Gate

Added `--train-variants-per-row` to the key-value copy generator and gate runner.
Variant `0` preserves the old exact training record for each base row; additional
variants reshuffle target keys, distractors, and repo-context order for the same
underlying values. The goal is to remove the one-prompt-per-row memorization
path exposed by the previous coverage ladder.

Changed files:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `tests/test_agentcoder_keyvalue_copy_generator.py`
- `scripts/eval_overfit_sanity.py`
- `scripts/run_agentcoder_slotcopy_gate.py`

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
rm -rf work/keyvalue_variant_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/keyvalue_variant_smoke/train.jsonl --cases-output work/keyvalue_variant_smoke/cases.json --manifest-output work/keyvalue_variant_smoke/manifest.json --eval-mode coverage_ladder --train-records 12 --train-variants-per-row 3 --eval-cases 4
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
python3 -m py_compile scripts/eval_overfit_sanity.py scripts/run_agentcoder_slotcopy_gate.py scripts/run_agentcoder_keyvalue_copy_gate.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- key-value generator file: `8 passed`
- related copy-generator bundle: `38 passed`
- evaluator/summary focused tests: `14 passed`
- smoke manifest: 12 base train rows, 3 variants per row, 36 train records, 12
  eval cases split evenly across `seen_slot`, `covered_value_slot`, and
  `heldout_slot`
- `git diff --check` passed

Pushed:

- `af75d4c Add key-value training variants`
- `969c16f Track key sequence accuracy in copy eval`

Remote Vast RTX 5090 setup:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_variant4_coverage_compare_20260703T105408Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyvalue_variant4_coverage_compare_20260703T105408Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --clean --no-fail
```

Remote focused tests: `25 passed, 8 deselected`.

Variant-4 coverage ladder results:

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out Pass | Behavior Accuracy | Val Loss | Tokens/sec | Train Records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM key-value key-follow | 1 / 96 | 1 / 32 | 0 / 32 | 0 / 32 | 96 / 96 | 0.600969 | 26344.7 | 384 |
| Transformer key-value key-follow | 1 / 96 | 1 / 32 | 0 / 32 | 0 / 32 | 82 / 96 | 0.712969 | 26923.4 | 384 |

The four-variant data removed the exact-layout memorization path but did not
produce reusable key lookup under the 1600-step tiny schedule. Common failures
switched from clean exact-prompt memorization to wrong requested key triplets.
For example, a seen-row case requesting `service`, `file`, and `fixture` could
produce `service`, `module`, and `setting`: valid-looking values from the same
source row, but not the user-requested keys.

Added key-sequence scoring to `scripts/eval_overfit_sanity.py` and re-scored
the saved checkpoints without retraining:

```bash
/venv/main/bin/python scripts/eval_overfit_sanity.py ... --output /root/raam-lm/runs/agentcoder_keyvalue_variant4_coverage_compare_20260703T105408Z/raam/keyvalue_eval_keyseq.json --no-fail
/venv/main/bin/python scripts/eval_overfit_sanity.py ... --output /root/raam-lm/runs/agentcoder_keyvalue_variant4_coverage_compare_20260703T105408Z/transformer/keyvalue_eval_keyseq.json --no-fail
```

Key-sequence results:

| Model | Overall Key Sequence | Seen | Covered-Value | Held-Out/OOV |
| --- | ---: | ---: | ---: | ---: |
| RAAM key-value key-follow | 33 / 96 | 1 / 32 | 0 / 32 | 32 / 32 |
| Transformer key-value key-follow | 1 / 96 | 1 / 32 | 0 / 32 | 0 / 32 |

Interpretation:

- RAAM can preserve the requested key order on the held-out/OOV tier, but still
  fails value continuation there (`adapter=c`, `file=c`, etc.).
- RAAM and Transformer both fail key selection on the covered-value tier, where
  values are tokenizer-covered but the requested key combination/layout differs.
- The next gate should split the problem explicitly: first train/evaluate
  requested-key emission without values, then reintroduce value copying. Until
  key-sequence accuracy is high on `covered_value_slot`, exact slot-copy pass
  rates are a muddled signal.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_keyvalue_variant4_coverage_compare_20260703T105408Z`

Checkpoint weights were not pulled.

## AgentCoder Key-Only Variant Gate

Added a key-only mode to the existing key-value generator, runner, and
evaluator. The mode keeps the same repo-context rows, target-key sampling,
distractor sampling, and coverage tiers, but changes the assistant target from
`key=value` lines to one requested key name per line. This directly tests
whether the model can follow the user-requested key order before value-copying
is reintroduced.

Changed files:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `scripts/eval_overfit_sanity.py`
- `tests/test_agentcoder_keyvalue_copy_generator.py`

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py scripts/eval_overfit_sanity.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
rm -rf work/keyonly_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/keyonly_smoke/train.jsonl --cases-output work/keyonly_smoke/cases.json --manifest-output work/keyonly_smoke/manifest.json --completion-mode key_only --eval-mode coverage_ladder --train-records 12 --train-variants-per-row 3 --eval-cases 4
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- focused key-value/slotcopy tests: `15 passed`
- related generator bundle: `39 passed`
- key-only smoke manifest: 12 base train rows, 3 variants per row, 36 train
  records, `completion_mode: key_only`, and 12 eval cases split evenly across
  `seen_slot`, `covered_value_slot`, and `heldout_slot`
- `git diff --check` passed

Pushed `b4aed1c Add key-only repo key gate`.

Remote Vast RTX 5090 setup:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyonly_variant4_compare_20260703T111138Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode key_only --max-new-tokens 16 \
  --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_keyonly_variant4_compare_20260703T111138Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode key_only --max-new-tokens 16 \
  --clean --no-fail
```

Remote focused tests: `26 passed, 8 deselected`.

Key-only coverage ladder results:

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out/OOV Pass | Key Sequence Accuracy | Behavior Accuracy | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM key-only | 96 / 96 | 32 / 32 | 32 / 32 | 32 / 32 | 96 / 96 | 96 / 96 | 0.299722 | 26515.3 |
| Transformer key-only | 0 / 96 | 0 / 32 | 0 / 32 | 0 / 32 | 0 / 96 | 53 / 96 | 0.866070 | 26558.5 |

Sample completions:

- RAAM seen: expected `service`, `file`, `fixture`; generated
  `service\nfile\nfixture\n<eos>`.
- RAAM covered-value: expected `setting`, `parser`, `service`; generated
  `setting\nparser\nservice\n<eos>`.
- RAAM held-out/OOV: expected `adapter`, `file`, `service`; generated
  `adapter\nfile\nservice\n<eos>`.
- Transformer seen: expected `service`, `file`, `fixture`; generated
  `endpoint\nroute\nsymbol\n<eos>`.
- Transformer covered/held-out samples generated `endpoint\nroute\nendpoint\n<eos>`.

Interpretation: this is the first clean positive separation for RAAM on the
current AgentCoder synthetic ladder. RAAM can learn the user-requested key-order
selection rule under the 4-variant regime, while the matched Transformer does
not under the same tiny schedule. This still does not prove the full model is
useful for coding, but it narrows the next bottleneck: key selection can work;
value copying/continuation is the failing component.

Next useful step: add a key-conditioned value-only gate. It should provide the
repo context plus the requested key sequence and ask for only the corresponding
values in order. That separates value retrieval/continuation from key-selection
and tells us whether to invest in a value span-copy route or in more data/schedule
first.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_keyonly_variant4_compare_20260703T111138Z`

Checkpoint weights were not pulled.

## AgentCoder Value-Only Variant Gate

Added a key-conditioned value-only mode to the key-value repo-context gate. This
mode gives the model the same `repo_context` rows plus the requested key order,
but asks it to emit only the corresponding values, one per line. The purpose is
to separate value retrieval/continuation from key-order selection after the
key-only gate showed RAAM could solve requested-key emission.

Changed files:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `scripts/eval_overfit_sanity.py`
- `tests/test_agentcoder_keyvalue_copy_generator.py`

Commits:

- `aebeb70 Add value-only repo value gate`
- `7409b50 Allow value-only keyvalue gate runner`
- `c66648d Only score enforced sequence metrics`

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py scripts/eval_overfit_sanity.py scripts/run_agentcoder_slotcopy_gate.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
rm -rf work/valueonly_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/valueonly_smoke/train.jsonl --cases-output work/valueonly_smoke/cases.json --manifest-output work/valueonly_smoke/manifest.json --completion-mode value_only --eval-mode coverage_ladder --train-records 12 --train-variants-per-row 3 --eval-cases 4
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
python3 -m py_compile scripts/run_agentcoder_keyvalue_copy_gate.py scripts/make_agentcoder_keyvalue_copy_sft.py
python3 scripts/run_agentcoder_keyvalue_copy_gate.py --help | grep 'value_only'
python3 -m py_compile scripts/eval_overfit_sanity.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- focused key-value/slotcopy tests: `16 passed`
- related generator bundle: `40 passed`
- value-only smoke manifest: 12 base train rows, 3 variants per row, 36 train
  records, `completion_mode: value_only`, and 12 eval cases split evenly across
  `seen_slot`, `covered_value_slot`, and `heldout_slot`
- runner help now accepts `value_only`
- `git diff --check` passed

Remote Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --max-new-tokens 48 \
  --clean --no-fail
/venv/main/bin/python scripts/eval_overfit_sanity.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --tokenizer /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/raam/tokenizer.json \
  --checkpoint /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/raam/train/checkpoints/last.pt \
  --device cuda \
  --cases-json /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/raam/generated/keyvalue_eval_cases.json \
  --output /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/raam/keyvalue_eval_cleanseq.json \
  --max-new-tokens 48 --min-pass-rate 1.0 --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueonly_variant4_compare_20260703T112451Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --max-new-tokens 48 \
  --clean --no-fail
```

Remote focused tests: `27 passed, 8 deselected`.

One failed command was hit and fixed: the first eval-only RAAM re-score assigned
`RUN_ID=...` in the same shell command where `$RUN_ID` appeared in arguments, so
the shell expanded it empty and the tokenizer path became
`/root/raam-lm/runs/raam/tokenizer.json`. The rerun above used the explicit path
and succeeded.

Value-only coverage ladder results:

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out/OOV Pass | Value Sequence Accuracy | Behavior Accuracy | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM value-only | 9 / 96 | 9 / 32 | 0 / 32 | 0 / 32 | 9 / 96 | 89 / 96 | 1.167214 | 25464.9 |
| Transformer value-only | 2 / 96 | 2 / 32 | 0 / 32 | 0 / 32 | 2 / 96 | 96 / 96 | 1.406007 | 27498.7 |

The RAAM result above uses the cleaned eval file
`raam/keyvalue_eval_cleanseq.json`; the original RAAM `summary.json` was produced
before `c66648d` and still includes a stale key-sequence labeled count for this
value-only run. The cleaned eval reports `key_sequence_labeled_cases: 0` and
`value_sequence_labeled_cases: 96`.

Representative completions:

- RAAM passing seen sample: expected `case_module_004`, `allowed_setting_004`,
  `copy_fixture_004.py`; generated the exact same three values.
- RAAM covered-value sample: expected `enabled_setting_000`, `case_parser_000`,
  `copy_service_000`; generated the seen-row values `copy_service_000`,
  `copy_file_000.py`, `copy_fixture_000.py`.
- RAAM held-out/OOV sample: expected `case_adapter_096`, `copy_file_096.py`,
  `copy_service_096`; generated `e.py`, `e.py`.
- Transformer passing seen sample: expected `disabled_file_001`,
  `case_setting_001`, `copy_symbol_001.py`; generated the exact same three
  values.
- Transformer held-out/OOV sample: expected `case_adapter_096`,
  `copy_file_096.py`, `copy_service_096`; generated `e`, `e`, `e`.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_valueonly_variant4_compare_20260703T112451Z`

Checkpoint weights were not pulled.

After artifact pull, the remote RAAM and Transformer `last.pt` files for this
run were deleted, and both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

Interpretation: RAAM has a small advantage over the matched Transformer on exact
seen value retrieval under this tiny schedule, but neither model has learned
covered-value recombination or held-out value continuation. Together with the
key-only result, the current bottleneck is no longer requested-key selection; it
is value span retrieval/copying. The next highest-value experiment is a
span-copy/objective change for values, such as explicit pointer-style value-copy
targets, value-boundary sentinel tokens, or a curriculum that first copies one
requested value before scaling back to three-value ordered outputs.

## AgentCoder One-Value Value-Only Gate

Added a configurable target-field count to the existing key-value generator and
runner so the value-only gate can train/evaluate one requested value before
scaling back to three ordered values. The default remains three targets, so prior
gates are unchanged unless `--target-fields` is passed.

Changed files:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `tests/test_agentcoder_keyvalue_copy_generator.py`

Commit:

- `b6c9bec Add one-value keyvalue gate curriculum knob`

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
rm -rf work/valueonly_one_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/valueonly_one_smoke/train.jsonl --cases-output work/valueonly_one_smoke/cases.json --manifest-output work/valueonly_one_smoke/manifest.json --completion-mode value_only --eval-mode coverage_ladder --train-records 12 --train-variants-per-row 3 --eval-cases 4 --target-fields 1
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
python3 scripts/run_agentcoder_keyvalue_copy_gate.py --help | grep -E 'target-fields|distractor-fields|value_only'
git diff --check
```

Results:

- focused key-value generator tests: `11 passed`
- related generator bundle: `41 passed`
- one-value smoke manifest: 12 base train rows, 3 variants per row, 36 train
  records, `target_fields: 1`, `completion_mode: value_only`, and 12 eval cases
  split evenly across `seen_slot`, `covered_value_slot`, and `heldout_slot`
- runner help exposes `--target-fields` and `--distractor-fields`
- `git diff --check` passed

Remote Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueonly_onefield_compare_20260703T114312Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --max-new-tokens 24 --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueonly_onefield_compare_20260703T114312Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --max-new-tokens 24 --clean --no-fail
```

Remote focused tests: `28 passed, 8 deselected`.

One failed command was hit before training: the first RAAM launch tried to write
`work/latest_onefield_run_id.txt` before the remote `work/` directory existed.
Creating `work/` and rerunning the same gate fixed it.

One-value coverage ladder results:

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out/OOV Pass | Value Sequence Accuracy | Behavior Accuracy | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM one-value | 32 / 96 | 27 / 32 | 5 / 32 | 0 / 32 | 32 / 96 | 96 / 96 | 0.685678 | 26499.6 |
| Transformer one-value | 35 / 96 | 30 / 32 | 5 / 32 | 0 / 32 | 35 / 96 | 96 / 96 | 0.699391 | 28818.5 |

Representative completions:

- RAAM seen pass: expected `copy_service_000`; generated
  `copy_service_000`.
- RAAM seen fail: expected `case_module_004`; generated `case_adapter_004`.
- RAAM covered-value pass: expected `copy_adapter_002.py`; generated
  `copy_adapter_002.py`.
- RAAM covered-value fail: expected `copy_route_000`; generated
  `copy_service_000`.
- RAAM held-out/OOV fail: expected `case_adapter_096`; generated `a`.
- Transformer seen pass: expected `copy_service_000`; generated
  `copy_service_000`.
- Transformer covered-value pass: expected `copy_adapter_002.py`; generated
  `copy_adapter_002.py`.
- Transformer held-out/OOV fail: expected `case_adapter_096`; generated `,`.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_valueonly_onefield_compare_20260703T114312Z`

Checkpoint weights were not pulled. After artifact pull, the remote RAAM and
Transformer `last.pt` files for this run were deleted, and both Vast RTX 5090
instances were verified stopped:

```text
43627905 exited
43634442 exited
```

Interpretation: reducing the task from three values to one value substantially
improves exact seen retrieval for both models and gives both a small covered-value
signal, but it does not solve held-out/OOV value continuation. RAAM is no longer
ahead on this simplified value-only gate; the matched Transformer is slightly
better overall under the same tiny schedule. The next useful change should focus
on span/value generalization rather than key selection: either add explicit
value-boundary sentinels plus a one-value-to-three-value curriculum, or add a
copy/span pointer head and compare it against the current copy-head-only path.

## AgentCoder Value-Boundary One-Value Gate

Added an optional value-boundary sentinel format for the value-only key-value
gate. With `--value-boundaries`, repo-context values and assistant outputs are
wrapped as `<value>...</value>`, while eval still compares the plain underlying
value sequence. This tested whether explicit span boundaries help value copying
without changing model architecture.

Changed files:

- `scripts/make_agentcoder_keyvalue_copy_sft.py`
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
- `scripts/eval_overfit_sanity.py`
- `tests/test_agentcoder_keyvalue_copy_generator.py`

Commits:

- `aa766c6 Add value boundary sentinel gate option`
- `ae964ba Fix eval script root import path`

Local verification:

```bash
python3 -m py_compile scripts/make_agentcoder_keyvalue_copy_sft.py scripts/run_agentcoder_keyvalue_copy_gate.py scripts/eval_overfit_sanity.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
rm -rf work/valueboundary_smoke && python3 scripts/make_agentcoder_keyvalue_copy_sft.py --train-output work/valueboundary_smoke/train.jsonl --cases-output work/valueboundary_smoke/cases.json --manifest-output work/valueboundary_smoke/manifest.json --completion-mode value_only --value-boundaries --eval-mode coverage_ladder --train-records 12 --train-variants-per-row 3 --eval-cases 4 --target-fields 1
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_slotcopy_generator.py
python3 scripts/run_agentcoder_keyvalue_copy_gate.py --help | grep -E 'value-boundaries|target-fields|value_only'
git diff --check
```

Results:

- focused key-value generator tests: `12 passed`
- related generator bundle: `42 passed`
- boundary smoke manifest: 12 base train rows, 3 variants per row, 36 train
  records, `target_fields: 1`, `value_boundaries: true`, and 12 eval cases split
  evenly across `seen_slot`, `covered_value_slot`, and `heldout_slot`
- runner help exposes `--value-boundaries`
- `git diff --check` passed

Remote Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python - <<'PY'
from scripts.eval_overfit_sanity import generated_value_sequence
assert generated_value_sequence("<value>copy_service_000</value>\n<eos>") == ["copy_service_000"]
PY
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueboundary_onefield_compare_20260703T115728Z/raam \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --value-boundaries --max-new-tokens 32 --clean --no-fail
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_valueboundary_onefield_compare_20260703T115728Z/transformer \
  --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 \
  --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 \
  --value-boundaries --max-new-tokens 32 --clean --no-fail
```

Remote focused tests: `29 passed, 8 deselected`; boundary eval parse check
passed.

One failed command was hit after RAAM training completed: running
`scripts/eval_overfit_sanity.py` as a script could not import
`scripts.make_agentcoder_keyvalue_copy_sft` because the repo root was not on
`sys.path`. `ae964ba` fixed the script root import path. RAAM was then rescored
from the existing checkpoint and its `summary.json` was recovered from the eval,
data, packed, and train manifests.

Value-boundary one-value coverage ladder results:

| Model | Exact Pass | Seen Pass | Covered-Value Pass | Held-Out/OOV Pass | Value Sequence Accuracy | Behavior Accuracy | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM value-boundary one-value | 27 / 96 | 22 / 32 | 5 / 32 | 0 / 32 | 27 / 96 | 96 / 96 | 0.418680 | 25920.3 |
| Transformer value-boundary one-value | 17 / 96 | 14 / 32 | 3 / 32 | 0 / 32 | 17 / 96 | 96 / 96 | 0.417798 | 28305.5 |

Representative completions:

- RAAM seen pass: expected `disabled_file_001`; generated
  `<value>disabled_file_001</value>`.
- RAAM seen fail: expected `copy_service_000`; generated
  `<value>copy_file_000.py</value>`.
- RAAM covered-value pass: expected `copy_adapter_002.py`; generated
  `<value>copy_adapter_002.py</value>`.
- RAAM held-out/OOV fail: expected `case_adapter_096`; generated
  `<value>c</value>`.
- Transformer seen pass: expected `disabled_file_001`; generated
  `<value>disabled_file_001</value>`.
- Transformer seen fail: expected `copy_service_000`; generated
  `<value>copy_fixture_000.py</value>`.
- Transformer held-out/OOV fail: expected `case_adapter_096`; generated
  `<value>py</value>`.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_valueboundary_onefield_compare_20260703T115728Z`

Checkpoint weights were not pulled. After artifact pull, the remote RAAM and
Transformer `last.pt` files for this run were deleted, and both Vast RTX 5090
instances were verified stopped:

```text
43627905 exited
43634442 exited
```

Interpretation: value-boundary sentinels made the wrapper format easy to learn,
but hurt exact value retrieval relative to the unwrapped one-value gate
(`RAAM 27/96` vs `32/96`, Transformer `17/96` vs `35/96`) and still produced
`0/32` held-out/OOV passes. This falsifies the simple boundary-token hypothesis
for the current tiny setup. The next useful direction is model-side copying:
add a value-query/pointer route that can use requested keys before the assistant
boundary, or relax the existing key-follow copy head so value-only prompts can
copy from the requested key instead of relying only on generated keys.

## AgentCoder Assistant-Only Atomic Mask Gate

Added assistant-only SFT loss masking for structured JSONL packing and training:

- `scripts/pack_dataset.py --assistant-loss-only`
- `pack_documents(..., assistant_loss_only=True)` now writes
  `train_loss_mask.bin` and `val_loss_mask.bin`
- `PackedTokenDataset` can return token-aligned loss masks
- `scripts/train.py` auto-detects sibling `*_loss_mask.bin` files, records them
  in the train manifest, and logs `loss_mask_fraction`
- RAAM, Transformer, and pure Mamba-like forward methods accept `loss_mask`
- `scripts/run_agentcoder_atomic_copy_gate.py` defaults to assistant-only loss
  masking, with `--no-assistant-loss-only` available for reproducing older
  all-token gates

Initial local validation:

```bash
python3 -m py_compile src/raam_lm/agent_data.py src/raam_lm/losses.py src/raam_lm/model.py src/raam_lm/baselines/transformer.py src/raam_lm/baselines/mamba_like.py scripts/pack_dataset.py scripts/train.py scripts/run_agentcoder_atomic_copy_gate.py scripts/run_agentcoder_atomic_anchor_seed_sweep.py scripts/run_agentcoder_atomic_cardinality_sweep.py
python3 -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Result: local atomic tests passed, `30 passed in 0.09s`.

The first remote attempt exposed a CLI forwarding bug: the command included
`--assistant-loss-only`, but the packed manifest still reported
`assistant_loss_only: false`. Fixed in commit `739b89f` by forwarding the flag
into `pack_documents`, and added a CLI-level regression test.

Corrected Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_pipeline.py -k 'assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_seed029_masked_steps2400_20260703T085307Z/raam_seed029_masked \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --clean \
  --no-fail
```

Remote test results:

- mask-specific pipeline tests: `3 passed, 8 deselected in 3.37s`
- atomic tests: `30 passed in 0.11s`
- packed manifest: `assistant_loss_only: true`
- supervised loss tokens: `832 / 6272` train tokens and `832 / 6272` validation
  tokens
- train manifest recorded both loss-mask files
- final train log mask fraction: about `0.1328`

Masked seed-29 gate result:

| Objective | Seed | Exact Pass | Behavior Accuracy | Slot Errors | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all-token SFT, previous corrected gate | 29 | 31 / 64 | 64 / 64 | 33 | 0.087371 | 22018.3 |
| assistant-only masked SFT | 29 | 19 / 64 | 64 / 64 | 45 | 0.266936 | 22027.1 |

Representative masked failure:

- requested `symbol=copy_symbol_002` and `file=copy_file_002.py`
- generated `symbol=copy_symbol_001` and `file=copy_file_001.py`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_seed029_masked_steps2400_20260703T085307Z`.
Checkpoint weights were not pulled.

Interpretation: assistant-only masking is correct SFT infrastructure and should
remain available, but it is not the slot-binding fix. On the fragile seed it
made exact binding worse while preserving behavior accuracy, which means the
model still knows the requested output family but does not reliably select the
current prompt's symbol/file pair. The next useful model step is a stronger
current-context binding mechanism or objective, not just more answer-token
weighting. Candidate next gates: add explicit pointer/copy supervision for slot
tokens, add a contrastive wrong-slot penalty, or test a dense-attention
Transformer with the same masked objective to separate an objective issue from
RAAM compression/anchor routing.

## AgentCoder Causal Copy-Head Diagnostic

Added a feature-gated causal pointer-copy path to test whether exact
current-context copying needs a model-side mechanism rather than only a loss
mask:

- `src/raam_lm/copy_head.py`
- `CopyHeadConfig` in `src/raam_lm/config.py`
- RAAM, Transformer, and pure Mamba-like models can mix normal vocabulary logits
  with causal copy logits over previous input tokens
- copy-head FLOPs are included in `estimate_flops_per_token`
- train logs record `copy_head_enabled`
- diagnostic configs:
  - `configs/scratch/raam_agentcoder_atomic_hybrid1_copy_head_gate.yaml`
  - `configs/scratch/transformer_agentcoder_atomic_copy_head_gate.yaml`

Local validation:

```bash
python3 -m py_compile src/raam_lm/copy_head.py src/raam_lm/config.py src/raam_lm/model.py src/raam_lm/baselines/transformer.py src/raam_lm/baselines/mamba_like.py src/raam_lm/flops.py scripts/train.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Result: local focused tests passed, `32 passed in 0.10s`.

The first two remote training attempts exposed copy-head mixed-precision issues:

- `scatter_add_` dtype mismatch under CUDA autocast
- fp16 overflow from the masked score sentinel

Fixed in commits `898c64b` and `a19d33d` by keeping the pointer-copy math in
fp32 with autocast disabled inside the copy head. Added remote-covered
regression tests, including CUDA autocast backward.

Corrected Vast RTX 5090 tests:

```bash
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_pipeline.py -k 'copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards'
```

Result: `9 passed, 8 deselected` with the expected fallback mixer warning.

Seed-29 copy-head gate at `2400` steps:

```bash
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_copy_head_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z/raam_seed029_copy_head \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --clean \
  --no-fail
```

| Gate | Seed | Exact Pass | Behavior Accuracy | Val Loss | Tokens/sec | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all-token SFT, no copy head | 29 | 31 / 64 | 64 / 64 | 0.087371 | 22018.3 | 1868160 |
| assistant-only SFT, no copy head | 29 | 19 / 64 | 64 / 64 | 0.266936 | 22027.1 | 1868160 |
| assistant-only SFT, copy head | 29 | 63 / 64 | 64 / 64 | 0.222196 | 20262.8 | 1950176 |

The single `2400`-step failure copied the correct symbol but the wrong paired
file:

- requested `symbol=copy_symbol_009` and `file=copy_file_009.py`
- generated `symbol=copy_symbol_009` and `file=copy_file_058.py`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_copy_head_seed029_steps2400_20260703T090954Z`.
Checkpoint weights were not pulled.

Longer seed-29 copy-head gate at `3600` steps:

```bash
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_hybrid1_copy_head_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_copy_head_seed029_steps3600_20260703T091247Z/raam_seed029_copy_head \
  --device cuda \
  --seed 29 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 3600 \
  --clean \
  --no-fail
```

Result: still `63 / 64`, behavior accuracy `64 / 64`, final validation loss
`0.238495`. The remaining failure flipped the same pair:

- requested `symbol=copy_symbol_058` and `file=copy_file_058.py`
- generated `symbol=copy_symbol_058` and `file=copy_file_009.py`

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_copy_head_seed029_steps3600_20260703T091247Z`.
Checkpoint weights were not pulled.

Interpretation: the causal copy head is the first change that materially fixes
the fragile seed-29 binding gate, improving exact pass from `31 / 64` to
`63 / 64`. It does not fully clear the gate. The last blocker is pair
consistency: the model can copy each field format and usually the current
symbol, but can still associate the file from a different memorized row. The
next useful step is a pair-consistency objective or architecture tweak, such as
contrastive wrong-pair penalties, explicit source-row supervision for copied
fields, or a copy head that carries a selected source position across adjacent
slot fields. Do not promote the copy-head variant to broad chat/coding training
until the corrected three-seed atomic mirror gate reaches `64 / 64` on every
seed and then passes held-out/decoy slot-copy gates.

## AgentCoder Atomic Hybrid1 Seed-Fixed Repeatability Gate

Discovered that the earlier atomic anchor seed sweep did not actually vary the
training seed. The data generator accepted `--seed`, but `scripts/train.py` did
not have a CLI seed override and `scripts/run_agentcoder_atomic_copy_gate.py`
did not forward the seed into packing or training. This means the older
repeatability rows from that pre-fix run were fixed-seed repeats, not true
initialization and data-order seed evidence.

Fixed in commit `897e201`:

- `scripts/train.py` now accepts `--seed` and applies it before seeding,
  dataset construction, and model initialization.
- `scripts/run_agentcoder_atomic_copy_gate.py` now forwards the same seed to
  `pack_dataset.py` and `train.py`.
- `tests/test_agentcoder_atomic_anchor_seed_sweep.py` now asserts seed
  forwarding into both generated commands.

Local validation before the corrected Vast run:

```bash
python3 -m py_compile scripts/train.py scripts/run_agentcoder_atomic_copy_gate.py scripts/run_agentcoder_atomic_anchor_seed_sweep.py scripts/run_agentcoder_atomic_cardinality_sweep.py
python3 -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Result: focused local tests passed, `30 passed in 0.10s`; syntax and diff checks
passed.

Corrected Vast RTX 5090 run:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_anchor_seed_sweep.py \
  --configs hybrid1=configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml \
  --seeds 17,29,41 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_seed_sweep_steps2400_seedfix_20260703T083416Z \
  --device cuda \
  --clean
```

Remote focused tests passed before the run.

| Config | Seeds | Exact Pass Mean | Min | Max | Total Exact Pass | Mean Val Loss | Mean Tokens/sec | All Passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| one-token hybrid | `17,29,41` | 53 / 64 | 31 / 64 | 64 / 64 | 159 / 192 | 0.072933 | 21910.6 | false |

Per-seed rows:

| Seed | Exact Pass | Behavior Accuracy | Slot Errors | Val Loss | Tokens/sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 64 / 64 | 64 / 64 | 0 | 0.062354 | 22197.1 |
| 29 | 31 / 64 | 64 / 64 | 33 | 0.087371 | 22018.3 |
| 41 | 64 / 64 | 64 / 64 | 0 | 0.069072 | 21516.3 |

Seed `29` failed by copying valid-looking slots from the wrong training rows,
not by choosing the wrong behavior. The first failed case requested
`symbol=copy_symbol_002` and `file=copy_file_002.py`, but generated
`symbol=copy_symbol_033` and `file=copy_file_033.py`. This is a current-context
binding failure, not a format or behavior-classification failure.

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_seed_sweep_steps2400_seedfix_20260703T083416Z`.
Checkpoint weights were not pulled.

Interpretation: the one-token hybrid result at `2400` steps is not
seed-repeatable. The previous `64 / 64` single-seed result remains useful as a
ceiling check, but it does not clear RAAM for broader chat/coding training. The
next useful model step is a seed-sensitive context-binding diagnosis: either
run seed `29` longer/lower-LR to see whether it catches up, or add an auxiliary
slot-alignment/current-context copying objective and rerun the same corrected
three-seed gate. Do not scale to a larger paid chat/coding run until the minimum
per-seed atomic mirror pass rate reaches `64 / 64`, then move to held-out and
decoy slot-copy gates.

## AgentCoder Atomic Anchor Seed Repeatability Gate

Added a focused repeatability harness for RAAM anchor variants:

- `scripts/run_agentcoder_atomic_anchor_seed_sweep.py`
- `tests/test_agentcoder_atomic_anchor_seed_sweep.py`
- `docs/AGENTIC_CODING_EVALS.md` now documents the learned-vs-hybrid1 seed gate

Local validation:

```bash
python3 -m py_compile scripts/run_agentcoder_atomic_anchor_seed_sweep.py scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py
python3 -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Result: focused local tests passed, `29 passed in 0.11s`; syntax and diff checks
passed.

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_anchor_seed_sweep.py tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_anchor_seed_sweep.py \
  --configs learned=configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml hybrid1=configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml \
  --seeds 17,29,41 \
  --train-records 64 \
  --eval-cases 64 \
  --steps 1200 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_anchor_seed_sweep_learned_vs_hybrid1_20260703T081417Z \
  --device cuda \
  --clean
```

Remote focused tests passed, `29 passed in 0.13s`.

| Config | Seeds | Exact Pass Mean | Min | Max | Total Exact Pass | Mean Val Loss | Mean Tokens/sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned 4-anchor | `17,29,41` | 57 / 64 | 57 / 64 | 57 / 64 | 171 / 192 | 0.080535 | 21805.9 |
| one-token hybrid | `17,29,41` | 59 / 64 | 59 / 64 | 59 / 64 | 177 / 192 | 0.077860 | 21511.2 |

Per-seed rows:

| Config | Seed | Exact Pass | Val Loss | Tokens/sec |
| --- | ---: | ---: | ---: | ---: |
| learned 4-anchor | 17 | 57 / 64 | 0.080535 | 21482.6 |
| learned 4-anchor | 29 | 57 / 64 | 0.080535 | 21786.6 |
| learned 4-anchor | 41 | 57 / 64 | 0.080535 | 22148.5 |
| one-token hybrid | 17 | 59 / 64 | 0.077860 | 21577.1 |
| one-token hybrid | 29 | 59 / 64 | 0.077860 | 21505.7 |
| one-token hybrid | 41 | 59 / 64 | 0.077860 | 21450.9 |

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_anchor_seed_sweep_learned_vs_hybrid1_20260703T081417Z`.
Checkpoint weights were not pulled.

Interpretation: the one-token hybrid is repeatably better than the learned
4-anchor route on this mirrored `64`-binding atomic gate, but neither passes the
`64 / 64` repeatability threshold at the default `1200` steps. RAAM is therefore
not cleared for broader chat/coding scale from this evidence alone. The next
useful architecture gate is either a repeatability check at `2400` steps for the
one-token hybrid, or a harder held-out/decoy atomic slot-copy variant that tests
whether the copied binding comes from current context rather than memorized
mirrored training pairs.

## AgentCoder Atomic Cardinality Sweep

Added a reusable cardinality sweep wrapper for the atomic copy gate:

- `scripts/run_agentcoder_atomic_cardinality_sweep.py`
- `tests/test_agentcoder_atomic_cardinality_sweep.py`

Updated `docs/AGENTIC_CODING_EVALS.md` with the sweep command and its
diagnostic purpose. The sweep runs the no-decoy atomic copy gate across binding
counts and writes one aggregate `summary.json`.

Local validation:

```bash
python3 -m py_compile scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py scripts/make_agentcoder_atomic_copy_sft.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
git diff --check
```

Results:

- syntax checks passed
- focused local tests passed: `13 passed in 0.06s`
- `git diff --check` passed
- local end-to-end smoke could not run under system `python3` because that
  interpreter has no `torch` installed:
  `ModuleNotFoundError: No module named 'torch'`

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_20260703T060200Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `13 passed in 0.09s`
- full remote test suite passed: `50 passed in 32.84s`
- sweep values: `1,2,4,8,16,32,64`
- eval policy: mirrored eval with eval cases matched to train-record count
- `mirror_val: true`
- default train budget: `1200` steps per sub-run

| Bindings | RAAM Exact Pass | RAAM Val Loss | Transformer Exact Pass | Transformer Val Loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 / 1 | 0.00000956 | 1 / 1 | 0.0000128 |
| 2 | 2 / 2 | 0.018920 | 2 / 2 | 0.019245 |
| 4 | 2 / 4 | 0.033123 | 4 / 4 | 0.030965 |
| 8 | 8 / 8 | 0.041650 | 8 / 8 | 0.047006 |
| 16 | 13 / 16 | 0.052292 | 16 / 16 | 0.049577 |
| 32 | 2 / 32 | 0.085958 | 32 / 32 | 0.053770 |
| 64 | 10 / 64 | 0.097215 | 0 / 64 | 0.192129 |

First pass-rate failure below `1.0`:

- RAAM: `4` bindings, `2 / 4` exact
- Transformer: `64` bindings, `0 / 64` exact

Representative failures:

- RAAM `n=4`, `atomic_mirror_000`: expected `copy_symbol_000` /
  `copy_file_000.py`, completed `copy_symbol_002` / `copy_file_002.py`.
- RAAM `n=16`, `atomic_mirror_006`: expected `copy_symbol_006` /
  `copy_file_006.py`, completed `copy_symbol_010` / `copy_file_010.py`.
- RAAM `n=32` repeatedly collapsed onto valid but wrong seen slots such as
  `copy_symbol_010` / `copy_file_010.py`.
- Transformer `n=64` collapsed to the mixed pair `copy_symbol_015` /
  `copy_file_003.py` for many prompts.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_20260703T060200Z`

Checkpoint weights were not pulled.

Interpretation: the atomic task now gives us a much clearer floor. Both models
learn one and two exact bindings. The tiny Transformer baseline preserves exact
bindings through `32 / 32`, while RAAM's current compressed path fails at `4`,
recovers at `8`, degrades at `16`, and collapses at `32`. The wrong completions
are usually valid seen symbol/file pairs, not malformed output. That points to
an exact binding preservation problem rather than a chat-format problem. Before
larger chat/coding training, the next useful test is a targeted RAAM ablation
with dynamic hourglass compression disabled on the same sweep.

## AgentCoder RAAM No-Compression Cardinality Ablation

Added:

- `configs/scratch/raam_agentcoder_atomic_no_compression_gate.yaml`

This config keeps the same tiny RAAM backbone scale and training budget as the
atomic copy gate, but disables dynamic hourglass compression:

- `use_dynamic_hourglass_compression: false`
- `compression.enabled: false`

Local validation:

```bash
python3 -m py_compile scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py scripts/make_agentcoder_atomic_copy_sft.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_copy_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_no_compression_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    print(path, data['model_name'], data['use_dynamic_hourglass_compression'], data['compression']['enabled'], data['train']['seq_len'])
PY
git diff --check
```

Results:

- focused local tests passed: `13 passed in 0.05s`
- config parse confirmed full RAAM has compression enabled and the ablation has
  compression disabled
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_no_compression_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_no_compression_20260703T062016Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `13 passed in 0.08s`
- remote config parse confirmed `False False` for
  `use_dynamic_hourglass_compression` and `compression.enabled`
- full remote test suite passed after the run: `50 passed in 33.00s`
- eval policy: mirrored eval with eval cases matched to train-record count
- `mirror_val: true`
- default train budget: `1200` steps per sub-run

| Bindings | RAAM No-Compression Exact Pass | Val Loss | Tokens/sec |
| ---: | ---: | ---: | ---: |
| 4 | 1 / 4 | 0.045245 | 31768 |
| 8 | 1 / 8 | 0.064115 | 31479 |
| 16 | 1 / 16 | 0.073701 | 31784 |
| 32 | 1 / 32 | 0.089397 | 31430 |
| 64 | 1 / 64 | 0.109869 | 31479 |

Representative failures:

- no-compression `n=4` collapsed three failed prompts to
  `copy_symbol_002` / `copy_file_002.py`.
- no-compression `n=16` collapsed failed prompts to `copy_symbol_001` /
  `copy_file_001.py`.
- no-compression `n=64` collapsed failed prompts to `copy_symbol_056` /
  `copy_file_056.py`.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_no_compression_20260703T062016Z`

Checkpoint weights were not pulled.

Interpretation: disabling dynamic hourglass compression did not fix exact
binding; it made this gate worse. Full RAAM's compression is not the sole cause
of the failure and may be helping the tiny model retain some binding signal.
The next useful model-side test is to add an exact token-level path back into
the RAAM atomic config, likely via attention islands or anchor-preserved
local-global routing, and rerun the same cardinality sweep before any larger
chat/coding training.

## AgentCoder RAAM Anchor-Attention Cardinality Ablation

Added:

- `configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml`

Updated:

- `docs/AGENTIC_CODING_EVALS.md` now includes the anchor-attention ablation
  command.
- `tests/test_agentcoder_atomic_cardinality_sweep.py` now asserts the ablation
  keeps compression on while enabling preserved anchors and attention islands.

This config keeps dynamic hourglass compression enabled, but adds a stronger
exact route through the compressed/global stream:

- `use_anchor_preserved_local_global: true`
- `use_attention_islands: true`
- `attention_island_layers: [1, 2]`
- `compression.anchors_per_block: 4`

Local validation:

```bash
python3 -m py_compile scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py scripts/make_agentcoder_atomic_copy_sft.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_copy_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_no_compression_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    comp = data['compression']
    print(path, data['attention_island_layers'], data['use_anchor_preserved_local_global'], data['use_attention_islands'], comp['enabled'], comp['anchors_per_block'], data['train']['seq_len'])
PY
git diff --check
```

Results:

- focused local tests passed: `14 passed in 0.07s`
- config parse confirmed the anchor-attention ablation uses `[1, 2]` attention
  islands and `4` anchors per block
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_anchor_attention_20260703T063228Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `14 passed in 0.11s`
- remote config parse confirmed `[1, 2] True True True 4`
- full remote test suite passed after the run: `51 passed in 33.03s`
- eval policy: mirrored eval with eval cases matched to train-record count
- `mirror_val: true`
- default train budget: `1200` steps per sub-run

| Bindings | Full RAAM Exact Pass | RAAM No-Compression Exact Pass | RAAM Anchor-Attention Exact Pass | Anchor-Attention Val Loss |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 2 / 4 | 1 / 4 | 2 / 4 | 0.030546 |
| 8 | 8 / 8 | 1 / 8 | 8 / 8 | 0.041757 |
| 16 | 13 / 16 | 1 / 16 | 15 / 16 | 0.046363 |
| 32 | 2 / 32 | 1 / 32 | 14 / 32 | 0.068648 |
| 64 | 10 / 64 | 1 / 64 | 57 / 64 | 0.080535 |

Representative failures:

- anchor-attention `n=4` still copied wrong seen pairs for two prompts:
  `copy_symbol_002` / `copy_file_002.py` and `copy_symbol_001` /
  `copy_file_001.py`.
- anchor-attention `n=16` had one mixed-pair failure:
  `symbol=copy_symbol_013` with `file=copy_file_009.py`.
- anchor-attention `n=64` had seven failures; most were valid but wrong seen
  pairs, plus one malformed partial output for `atomic_mirror_060`.

Local artifact pull:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_anchor_attention_20260703T063228Z`

Checkpoint weights were not pulled.

Interpretation: adding preserved anchors plus exact attention islands is the
first architectural change that materially improves the binding floor. It does
not fix low-cardinality instability (`n=4` remains `2 / 4`), but it changes the
large-cardinality result from full RAAM's `10 / 64` to `57 / 64`, while the tiny
Transformer baseline was `0 / 64` at the same `64`-binding gate. The next useful
step is to make the exact route deterministic enough to remove wrong-slot
collapses: either preserve all tokens for this tiny exact-copy control, add a
deterministic special-token/string anchor policy, or add a slot-alignment loss
before broader chat/coding training.

## AgentCoder RAAM All-Anchor Cardinality Ceiling

Added:

- `configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml`

Updated:

- `docs/AGENTIC_CODING_EVALS.md` now includes the all-anchor ceiling command.
- `tests/test_agentcoder_atomic_cardinality_sweep.py` now asserts the
  all-anchor config sets `anchors_per_block == block_size`.

This config keeps dynamic hourglass compression, preserved anchors, and
attention islands enabled, but preserves every token in each compression block:

- `attention_island_layers: [1, 2]`
- `compression.block_size: 8`
- `compression.anchors_per_block: 8`
- `compression.pooled_chunks_per_block: 1`

This is intentionally a diagnostic ceiling, not an efficient target model.

Local validation:

```bash
python3 -m py_compile scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py scripts/make_agentcoder_atomic_copy_sft.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    comp = data['compression']
    print(path, data['attention_island_layers'], data['use_anchor_preserved_local_global'], data['use_attention_islands'], comp['enabled'], comp['block_size'], comp['anchors_per_block'], data['train']['seq_len'])
PY
git diff --check
```

Results:

- focused local tests passed: `15 passed in 0.08s`
- config parse confirmed the all-anchor ablation uses `block_size: 8` and
  `anchors_per_block: 8`
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_all_anchor_attention_20260703T064504Z \
  --device cuda \
  --clean
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_all_anchor_n64_steps2400_20260703T065102Z \
  --device cuda \
  --clean
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml \
  --train-records 64 \
  --eval-cases 64 \
  --steps 4800 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_all_anchor_n64_steps4800_20260703T065403Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `15 passed in 0.11s`
- remote config parse confirmed `[1, 2] 8 8 True`
- full remote test suite passed after the runs: `52 passed in 32.99s`
- eval policy: mirrored eval with eval cases matched to train-record count
- `mirror_val: true`

All-anchor sweep at the default `1200` training steps:

| Bindings | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 4 | 4 / 4 | 0.029271 | 921600 |
| 8 | 8 / 8 | 0.042089 | 921600 |
| 16 | 15 / 16 | 0.046292 | 921600 |
| 32 | 16 / 32 | 0.068097 | 921600 |
| 64 | 2 / 64 | 0.108794 | 921600 |

Focused `64`-binding longer-training probes:

| Steps | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 1200 | 2 / 64 | 0.108794 | 921600 |
| 2400 | 61 / 64 | 0.067554 | 1843200 |
| 4800 | 64 / 64 | 0.065526 | 3686400 |

Representative failures before the 4800-step run:

- default all-anchor `n=16` had one wrong-slot failure:
  expected `copy_symbol_013` / `copy_file_013.py`, completed
  `copy_symbol_012` / `copy_file_012.py`.
- default all-anchor `n=64` collapsed many prompts to `copy_symbol_016` /
  `copy_file_016.py` or `copy_symbol_003` / `copy_file_003.py`.
- at `2400` steps, the remaining `n=64` failures were still valid but wrong
  seen pairs.
- at `4800` steps, all `64` mirrored atomic bindings passed exactly.

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_all_anchor_attention_20260703T064504Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_all_anchor_n64_steps2400_20260703T065102Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_all_anchor_n64_steps4800_20260703T065403Z`

Checkpoint weights were not pulled.

Interpretation: all-anchor attention proves RAAM can close the mirrored
`64`-binding atomic copy gate when every token has an exact route and the tiny
model gets enough optimization. The default `1200` steps underfit the heavier
all-anchor route, but `4800` steps reached `64 / 64`. This separates the
remaining useful-model work into two concrete tracks: make anchor selection
cheap and deterministic enough that the efficient partial-anchor route behaves
like the all-anchor ceiling, and keep broad chat/coding training blocked until
the efficient route passes the same exact-binding gate without needing to
preserve every token.

## AgentCoder RAAM Uniform-Anchor Cardinality Ablation

Added deterministic anchor selection support:

- `CompressionConfig.anchor_selection`, defaulting to `learned_topk`
- `DynamicHourglassCompressor` support for `anchor_selection: uniform`
- compressor metadata now records `anchor_selection`
- RAAM model aux now records `anchor_selection`
- `configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml`

Updated:

- `docs/AGENTIC_CODING_EVALS.md` now includes the uniform-anchor sweep command.
- `tests/test_anchor_selection.py` covers deterministic uniform anchor indices
  and rejects `anchors_per_block > block_size`.
- `tests/test_agentcoder_atomic_cardinality_sweep.py` covers the uniform-anchor
  config.

This config keeps the efficient partial-anchor budget from the previous
anchor-attention ablation, but makes the four anchors deterministic rather than
learned:

- `attention_island_layers: [1, 2]`
- `compression.block_size: 8`
- `compression.anchors_per_block: 4`
- `compression.anchor_selection: uniform`

Local validation:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/compression.py src/raam_lm/model.py scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    comp = data['compression']
    print(path, comp.get('anchor_selection', 'learned_topk'), comp['block_size'], comp['anchors_per_block'], data['attention_island_layers'])
PY
git diff --check
```

Results:

- focused local tests passed: `16 passed in 0.10s`
- config parse confirmed learned `4` anchors, uniform `4` anchors, and all `8`
  anchors are distinct configs
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_anchor_selection.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_uniform_anchor_attention_20260703T070842Z \
  --device cuda \
  --clean
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_uniform_anchor_n64_steps2400_20260703T071433Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `19 passed in 1.60s`
- remote config parse confirmed `uniform 8 4`
- full remote test suite passed after the runs: `55 passed in 33.04s`
- eval policy: mirrored eval with eval cases matched to train-record count
- `mirror_val: true`

Uniform-anchor sweep at the default `1200` training steps:

| Bindings | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 4 | 4 / 4 | 0.029176 | 921600 |
| 8 | 5 / 8 | 0.043210 | 921600 |
| 16 | 11 / 16 | 0.055005 | 921600 |
| 32 | 31 / 32 | 0.069652 | 921600 |
| 64 | 1 / 64 | 0.109745 | 921600 |

Focused `64`-binding longer-training probe:

| Steps | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 1200 | 1 / 64 | 0.109745 | 921600 |
| 2400 | 16 / 64 | 0.090924 | 1843200 |

Representative failures:

- default uniform `n=8` copied wrong seen pairs for three prompts, commonly
  `copy_symbol_001` / `copy_file_001.py`.
- default uniform `n=32` missed one case, copying `copy_symbol_001` /
  `copy_file_001.py` for `atomic_mirror_026`.
- default uniform `n=64` collapsed most prompts to `copy_symbol_003` /
  `copy_file_003.py`.
- after `2400` steps, uniform `n=64` improved but still had `48` wrong-slot
  failures.

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_uniform_anchor_attention_20260703T070842Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_uniform_anchor_n64_steps2400_20260703T071433Z`

Checkpoint weights were not pulled.

Interpretation: deterministic uniform anchors are not the efficient fix. They
do remove the low-cardinality `n=4` failure and almost solve `n=32`, but they
hurt `n=8`, `n=16`, and especially `n=64`. The learned 4-anchor route remains
the best efficient route for high-cardinality binding at this scale (`57 / 64`
at `1200` steps), while the all-anchor ceiling proves the model can reach
`64 / 64` with full token preservation and more optimization. The next useful
step is not naive fixed spacing; it is content-aware deterministic anchoring or
an auxiliary slot-alignment objective that biases partial anchors toward the
identifier/value tokens needed for exact copying.

## AgentCoder RAAM Token-ID Anchor Cardinality Ablation

Added a deterministic content-aware anchor selector:

- `DynamicHourglassCompressor` supports `anchor_selection: token_id_topk`
- `token_id_topk` requires `input_ids` and anchors the highest token IDs inside
  each compression block
- `configs/scratch/raam_agentcoder_atomic_token_anchor_attention_gate.yaml`
- `docs/AGENTIC_CODING_EVALS.md` now includes the token-ID anchor sweep command
- `tests/test_anchor_selection.py` covers token-ID top-k anchor placement and
  the required-`input_ids` error
- `tests/test_agentcoder_atomic_cardinality_sweep.py` covers the new config

The motivation was to test a cheap proxy for rare/code-like tokens. In the
generated atomic copy tokenizer, learned `copy_*` values get higher IDs than
byte fallback tokens, so `token_id_topk` should preserve identifier/value tokens
without learning anchor scores.

Local validation:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/compression.py src/raam_lm/model.py scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_token_anchor_attention_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    comp = data['compression']
    print(path, comp.get('anchor_selection', 'learned_topk'), comp['block_size'], comp['anchors_per_block'], data['attention_island_layers'])
PY
git diff --check
```

Results:

- focused local tests passed: `17 passed in 0.11s`
- config parse confirmed learned `4` anchors, all `8` anchors, uniform `4`
  anchors, and token-ID `4` anchors are distinct configs
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_anchor_selection.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_token_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_token_anchor_attention_20260703T072932Z \
  --device cuda \
  --clean
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_token_anchor_attention_gate.yaml \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_token_anchor_n64_steps2400_20260703T073533Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests passed: `22 passed in 1.69s`
- full remote test suite passed after the runs: `58 passed in 32.95s`
- eval policy: mirrored eval with eval cases matched to train-record count,
  except the focused run used fixed `64` eval cases
- `mirror_val: true`

Token-ID anchor sweep at the default `1200` training steps:

| Bindings | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 4 | 4 / 4 | 0.029163 | 921600 |
| 8 | 8 / 8 | 0.041700 | 921600 |
| 16 | 16 / 16 | 0.045566 | 921600 |
| 32 | 32 / 32 | 0.055519 | 921600 |
| 64 | 21 / 64 | 0.092811 | 921600 |

Focused `64`-binding longer-training probe:

| Steps | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 1200 | 21 / 64 | 0.092811 | 921600 |
| 2400 | 56 / 64 | 0.070240 | 1843200 |

Representative failures:

- default token-ID `n=64` produced behavior-correct `symbol/file` lines but
  copied wrong seen slots, for example `copy_symbol_001` -> `copy_symbol_034`
  and `copy_symbol_009` -> `copy_symbol_063`.
- after `2400` steps, `n=64` still had `8` wrong-slot failures:
  `atomic_mirror_000`, `007`, `009`, `019`, `025`, `026`, `034`, and `040`.

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_token_anchor_attention_20260703T072932Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_token_anchor_n64_steps2400_20260703T073533Z`

Checkpoint weights were not pulled.

Interpretation: token-ID anchors are a useful diagnostic but not the current
efficient route. They are much better than uniform anchors at default steps and
solve `4`, `8`, `16`, and `32` bindings cleanly, but the high-cardinality
`64`-binding result is worse than learned 4-anchor top-k at `1200` steps
(`21 / 64` versus `57 / 64`). Doubling optimization improves token-ID anchors
to `56 / 64`, but that still only roughly matches the learned route while using
twice the steps. The next architecture step should keep learned/content-aware
anchors in play and add a stronger slot-alignment signal or hybrid anchor prior,
not switch wholesale to token-ID anchors before full chat/coding training.

## AgentCoder RAAM Hybrid Anchor Cardinality Ablation

Added a hybrid anchor selector and two diagnostic configs:

- `CompressionConfig.token_id_anchor_count`
- `DynamicHourglassCompressor` support for
  `anchor_selection: hybrid_token_id_learned`
- `configs/scratch/raam_agentcoder_atomic_hybrid_anchor_attention_gate.yaml`
  reserves `2` token-ID anchors and fills `2` learned anchors
- `configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml`
  reserves `1` token-ID anchor and fills `3` learned anchors
- `docs/AGENTIC_CODING_EVALS.md` documents both hybrid sweeps
- `tests/test_anchor_selection.py` covers hybrid reservation, missing
  `input_ids`, and invalid `token_id_anchor_count`
- `tests/test_agentcoder_atomic_cardinality_sweep.py` covers both hybrid configs

The goal was to keep the strong learned/content route while giving at least one
fixed slot per block to high-ID rare/code-like tokens. This directly followed
the token-ID-only result: rare-token preservation helped but replacing all
learned anchors was too costly.

Local validation:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/compression.py src/raam_lm/model.py scripts/run_agentcoder_atomic_cardinality_sweep.py scripts/run_agentcoder_atomic_copy_gate.py
python3 -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py
python3 - <<'PY'
from pathlib import Path
import yaml
for path in [
    Path('configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_all_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_uniform_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_token_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_hybrid_anchor_attention_gate.yaml'),
    Path('configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml'),
]:
    data = yaml.safe_load(path.read_text())
    comp = data['compression']
    print(path, comp.get('anchor_selection', 'learned_topk'), comp.get('token_id_anchor_count', 0), comp['block_size'], comp['anchors_per_block'], data['attention_island_layers'])
PY
git diff --check
```

Results:

- focused local tests passed before the first hybrid commit:
  `18 passed in 0.11s`
- focused local tests passed after adding the one-token hybrid config:
  `19 passed in 0.10s`
- config parse confirmed the hybrid configs use the same `4`-anchor budget with
  distinct `token_id_anchor_count` values
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_anchor_selection.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_hybrid_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_hybrid_anchor_attention_20260703T074802Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_cardinality_sweep.py tests/test_agentcoder_atomic_copy_generator.py tests/test_anchor_selection.py
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_cardinality_sweep_raam_hybrid1_anchor_attention_20260703T075520Z \
  --device cuda \
  --clean
/venv/main/bin/python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml \
  --train-records 64 \
  --eval-cases 64 \
  --steps 2400 \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_hybrid1_n64_steps2400_20260703T080111Z \
  --device cuda \
  --clean
/venv/main/bin/python -m pytest -q
```

Remote validation results:

- focused remote tests after first hybrid config: `26 passed in 1.67s`
- focused remote tests after one-token hybrid config: `27 passed in 1.65s`
- full remote test suite passed after the runs: `63 passed in 33.36s`
- eval policy: mirrored eval with eval cases matched to train-record count,
  except the focused run used fixed `64` eval cases
- `mirror_val: true`

Two-token hybrid sweep at the default `1200` training steps:

| Bindings | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 4 | 4 / 4 | 0.029026 | 921600 |
| 8 | 8 / 8 | 0.041702 | 921600 |
| 16 | 16 / 16 | 0.045974 | 921600 |
| 32 | 31 / 32 | 0.050310 | 921600 |
| 64 | 33 / 64 | 0.090650 | 921600 |

One-token hybrid sweep at the default `1200` training steps:

| Bindings | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 4 | 4 / 4 | 0.029151 | 921600 |
| 8 | 8 / 8 | 0.045266 | 921600 |
| 16 | 15 / 16 | 0.046210 | 921600 |
| 32 | 32 / 32 | 0.057599 | 921600 |
| 64 | 59 / 64 | 0.077860 | 921600 |

Focused one-token hybrid `64`-binding longer-training probe:

| Steps | Exact Pass | Val Loss | Tokens Seen |
| ---: | ---: | ---: | ---: |
| 1200 | 59 / 64 | 0.077860 | 921600 |
| 2400 | 64 / 64 | 0.062354 | 1843200 |

Representative failures:

- two-token hybrid `n=64` only reached `33 / 64`, so reserving two high-ID
  anchors appears to disrupt the learned/content route too much.
- one-token hybrid default `n=16` missed only `atomic_mirror_015`.
- one-token hybrid default `n=64` missed five cases:
  `atomic_mirror_006`, `029`, `032`, `039`, and `057`.
- one-token hybrid `n=64` at `2400` steps had no slot-copy failures.

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_hybrid_anchor_attention_20260703T074802Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_cardinality_sweep_raam_hybrid1_anchor_attention_20260703T075520Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_hybrid1_n64_steps2400_20260703T080111Z`

Checkpoint weights were not pulled.

Interpretation: the one-token hybrid is the best efficient partial-anchor route
so far. It slightly improves the high-cardinality default-step result over the
learned 4-anchor route (`59 / 64` versus `57 / 64`) and closes the `64 / 64`
atomic exact-binding gate at `2400` steps without preserving every token. This
does not yet justify broad chat/coding training: it is one mirrored synthetic
gate, one seed, and one tiny model size. The next useful gate is repeatability:
rerun the one-token hybrid and the learned 4-anchor route across multiple seeds,
then test a harder held-out/decoy slot-copy gate before starting any broad
agentic coding pretraining.

## AgentCoder Programmatic Slot-Copy Gate

Implemented the next diagnostic step after the v6-v8 slot-copy failures: a
larger programmatic slot-copy curriculum and held-out eval set.

Added:

- `scripts/make_agentcoder_slotcopy_sft.py`
- `scripts/run_agentcoder_slotcopy_gate.py`
- `tests/test_agentcoder_slotcopy_generator.py`

Updated:

- `scripts/eval_overfit_sanity.py` now preserves optional `slot_family` and
  `expected_slots` metadata in eval results.
- `docs/AGENTIC_CODING_EVALS.md` documents the new Slot-Copy Diagnostic Gate.

Generator shape:

| Family | Train records | Held-out eval cases | Purpose |
| --- | ---: | ---: | --- |
| `repo_lookup` | 48 | 16 | copy requested symbol and defining file from repo context with unrelated definitions |
| `patch_return` | 48 | 16 | copy exact arithmetic file/helper/return expression/test command |
| `patch_literal` | 48 | 16 | copy exact boolean-flag file/helper/enabled literal/test command |

Total output: `144` train records and `48` held-out eval cases. Train and eval
expected-slot tuples are disjoint. The runner summarizes pass rate, behavior
accuracy, and slot-error count per family.

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_slotcopy_sft.py scripts/run_agentcoder_slotcopy_gate.py scripts/eval_overfit_sanity.py scripts/make_agentcoder_curated_sft.py
python3 scripts/make_agentcoder_slotcopy_sft.py --train-output work/slotcopy_gate_smoke/train.jsonl --cases-output work/slotcopy_gate_smoke/cases.json --manifest-output work/slotcopy_gate_smoke/manifest.json
python3 -m pytest -q tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- syntax checks passed
- generator emitted format `agentcoder-slotcopy-sft-v1`
- train records: `144`
- eval cases: `48`
- train family counts: `48` each for `repo_lookup`, `patch_return`,
  `patch_literal`
- eval family counts: `16` each for `repo_lookup`, `patch_return`,
  `patch_literal`
- focused local tests passed: `4 passed in 0.07s`

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_slotcopy_generator.py
/venv/main/bin/python scripts/run_agentcoder_slotcopy_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_slotcopy_gate_20260703T050322Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote results:

- focused slot-copy generator tests: `4 passed in 0.10s`
- full remote test suite: `32 passed in 32.66s`
- run id: `agentcoder_slotcopy_gate_20260703T050322Z`
- train records: `144`
- eval cases: `48`
- train tokens: `51564`
- validation tokens: `12756`
- non-embedding params: `1244802`
- estimated FLOPs/token: `2336384`
- final train loss: `0.05523088574409485`
- final validation loss: `0.2700197398662567`
- final tokens/sec: `68546.0167564717`
- exact pass rate: `0 / 48`
- behavior accuracy: `48 / 48`

| Family | Cases | Exact Pass | Slot Errors | Behavior Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `repo_lookup` | 16 | 0 | 16 | 1.0 |
| `patch_return` | 16 | 0 | 16 | 1.0 |
| `patch_literal` | 16 | 0 | 16 | 1.0 |

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_slotcopy_gate_20260703T050322Z`.
Checkpoint weights were not pulled.

Representative failures:

- `slot_repo_lookup_00_format_user`: selected repo lookup behavior but copied
  the distractor `extract_title` / `title_extract_00.py` slot instead of
  `format_user` / `user_format_03.py`.
- `slot_patch_return_00_append_capacity`: produced an arithmetic patch in the
  right style but copied unrelated file/helper slots such as
  `amount_blend_08.py`.
- `slot_patch_literal_00_notify_enabled`: produced a boolean flag patch in the
  right style but copied unrelated file/helper slots such as
  `export_open_05.py`.

Interpretation: this is the clearest blocker so far. The tiny RAAM-AgentCoder
model can learn the response family, and the low validation loss is misleading,
but it does not bind held-out file/function/literal slots from context. Do not
launch a larger paid chat/coding run as if this is solved. The next useful step
is a slot-binding ladder that separates seen-slot memorization from held-out
slot generalization before returning to broader Stage 5 continuation.

After artifact pull, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## AgentCoder Slot-Binding Ladder Diagnostic

Added a seen-vs-heldout ladder to the programmatic slot-copy gate so the next
GPU run can distinguish basic memorization/copy failure from true held-out slot
generalization failure.

Updated:

- `scripts/make_agentcoder_slotcopy_sft.py`
  - keeps the previous `--eval-mode heldout` behavior available
  - adds `--eval-mode ladder`
  - ladder eval emits `seen_slot` and `heldout_slot` cases
- `scripts/eval_overfit_sanity.py`
  - preserves `eval_tier` in eval result rows
- `scripts/run_agentcoder_slotcopy_gate.py`
  - defaults to `--eval-mode ladder`
  - adds `slot_ladder_summary` grouped by slot family and eval tier
- `tests/test_agentcoder_slotcopy_generator.py`
  - covers held-out disjointness, seen-slot overlap, and ladder summaries
- `docs/AGENTIC_CODING_EVALS.md`
  - documents the new ladder gate shape

Generator shape:

| Mode | Train Records | Eval Cases | Eval Tiers |
| --- | ---: | ---: | --- |
| `heldout` | 144 | 48 | 48 held-out slots |
| `ladder` | 144 | 96 | 48 seen slots + 48 held-out slots |

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_slotcopy_sft.py scripts/run_agentcoder_slotcopy_gate.py scripts/eval_overfit_sanity.py
rm -rf work/slotcopy_ladder_smoke && python3 scripts/make_agentcoder_slotcopy_sft.py --train-output work/slotcopy_ladder_smoke/heldout_train.jsonl --cases-output work/slotcopy_ladder_smoke/heldout_cases.json --manifest-output work/slotcopy_ladder_smoke/heldout_manifest.json && python3 scripts/make_agentcoder_slotcopy_sft.py --eval-mode ladder --train-output work/slotcopy_ladder_smoke/ladder_train.jsonl --cases-output work/slotcopy_ladder_smoke/ladder_cases.json --manifest-output work/slotcopy_ladder_smoke/ladder_manifest.json
python3 -m pytest -q tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- syntax checks passed
- held-out generator emitted `144` train records and `48` eval cases
- ladder generator emitted `144` train records and `96` eval cases
- ladder eval tier counts: `48` `seen_slot`, `48` `heldout_slot`
- focused local tests passed: `6 passed in 0.08s`
- `git diff --check` passed

Next GPU action:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_slotcopy_generator.py
/venv/main/bin/python scripts/run_agentcoder_slotcopy_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_slotcopy_ladder_<UTC_TIMESTAMP> \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Interpretation: if `seen_slot` also fails badly, the tiny setup is not even
copying/memorizing slot-bound examples reliably and should be fixed before any
larger spend. If `seen_slot` passes but `heldout_slot` fails, the next work is
architecture/objective/curriculum changes for context binding. If both pass,
then the slot-copy blocker is no longer the first reason to delay a broader
chat/coding continuation.

Vast RTX 5090 ladder validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_slotcopy_generator.py
/venv/main/bin/python scripts/run_agentcoder_slotcopy_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_slotcopy_ladder_20260703T051703Z \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote results:

- focused slot-copy generator tests: `6 passed in 0.12s`
- full remote test suite: `34 passed in 32.72s`
- run id: `agentcoder_slotcopy_ladder_20260703T051703Z`
- train records: `144`
- eval cases: `96`
- eval tier counts: `48` `seen_slot`, `48` `heldout_slot`
- train tokens: `51564`
- validation tokens: `12756`
- non-embedding params: `1244802`
- estimated FLOPs/token: `2336384`
- final train loss: `0.05523088574409485`
- final validation loss: `0.2700197398662567`
- final tokens/sec: `68643.99354831324`
- exact pass rate: `19 / 96`
- behavior accuracy: `96 / 96`

| Family | Seen Exact Pass | Held-Out Exact Pass | Seen Slot Errors | Held-Out Slot Errors |
| --- | ---: | ---: | ---: | ---: |
| `repo_lookup` | 12 / 16 | 0 / 16 | 4 | 16 |
| `patch_return` | 2 / 16 | 0 / 16 | 14 | 16 |
| `patch_literal` | 5 / 16 | 0 / 16 | 11 | 16 |

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_slotcopy_ladder_20260703T051703Z`.
Checkpoint weights were not pulled.

Interpretation: this confirms the blocker is earlier than held-out
generalization. The tiny model can classify the requested response family, but
it does not reliably bind even seen file/function/literal slots when context is
regenerated. The next useful model step is not a broader chat/coding run. It is
to make exact slot copying a simpler supervised objective first, likely by
adding a very small copy-only ladder with short outputs, checking whether a
Transformer baseline passes it, and only then reintroducing patch formatting and
repo-context distractors.

After artifact pull, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## AgentCoder Copy-Only Binding Gate

Implemented the simpler diagnostic that the previous slot-binding ladder called
for: short `key=value` copy-only outputs, no diffs, no patch prose, no test
command wording beyond copying a `test=<path>` slot. Also added a matched tiny
Transformer config so the failure can be checked against a dense-attention
baseline before blaming RAAM compression.

Added:

- `scripts/make_agentcoder_copyonly_sft.py`
- `scripts/run_agentcoder_copy_gate.py`
- `configs/scratch/raam_agentcoder_copy_gate.yaml`
- `configs/scratch/transformer_agentcoder_copy_gate.yaml`
- `tests/test_agentcoder_copyonly_generator.py`

Updated:

- `scripts/eval_overfit_sanity.py` now recognizes `copy_slot_values`
  completions containing `symbol=`, `file=`, `helper=`, `return=`, `literal=`,
  or `test=`.
- `docs/AGENTIC_CODING_EVALS.md` documents the copy-only gate and the baseline
  interpretation.

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_copyonly_sft.py scripts/run_agentcoder_copy_gate.py scripts/eval_overfit_sanity.py scripts/make_agentcoder_slotcopy_sft.py scripts/run_agentcoder_slotcopy_gate.py
rm -rf work/copyonly_gate_smoke && python3 scripts/make_agentcoder_copyonly_sft.py --train-output work/copyonly_gate_smoke/train.jsonl --cases-output work/copyonly_gate_smoke/cases.json --manifest-output work/copyonly_gate_smoke/manifest.json --eval-mode ladder
python3 -m pytest -q tests/test_agentcoder_copyonly_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- syntax checks passed
- generator emitted format `agentcoder-copyonly-sft-v1`
- train records: `144`
- eval cases: `96`
- eval tier counts: `48` `seen_slot`, `48` `heldout_slot`
- focused local tests passed: `9 passed in 0.12s`
- `git diff --check` passed

Vast RTX 5090 validation:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_copyonly_generator.py tests/test_agentcoder_slotcopy_generator.py
/venv/main/bin/python scripts/run_agentcoder_copy_gate.py \
  --config configs/scratch/raam_agentcoder_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_copyonly_baseline_20260703T052913Z/raam \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python scripts/run_agentcoder_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_copyonly_baseline_20260703T052913Z/transformer \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote results:

- focused copy/slot generator tests: `9 passed in 0.16s`
- full remote test suite: `37 passed in 32.91s`
- run root: `agentcoder_copyonly_baseline_20260703T052913Z`
- train records: `144`
- eval cases: `96`
- train tokens: `32924`
- validation tokens: `8356`
- behavior accuracy: `96 / 96` for both models

| Model | Exact Pass | Seen Exact Pass | Held-Out Exact Pass | Val Loss | Tokens/sec | Non-Emb Params | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM copy gate | 22 / 96 | 22 / 48 | 0 / 48 | 0.302754 | 50450.0 | 1244802 | 2305152 |
| Transformer copy gate | 18 / 96 | 18 / 48 | 0 / 48 | 0.268684 | 48099.0 | 1049728 | 2684928 |

| Family | RAAM Seen | RAAM Held-Out | Transformer Seen | Transformer Held-Out |
| --- | ---: | ---: | ---: | ---: |
| `repo_lookup_copy` | 14 / 16 | 0 / 16 | 11 / 16 | 0 / 16 |
| `patch_return_copy` | 5 / 16 | 0 / 16 | 3 / 16 | 0 / 16 |
| `patch_literal_copy` | 3 / 16 | 0 / 16 | 4 / 16 | 0 / 16 |

Local artifact pull:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_copyonly_baseline_20260703T052913Z`.
Checkpoint weights were not pulled.

Interpretation: simplifying the output helped only slightly. The held-out
failure is not RAAM-specific because the tiny Transformer baseline also scored
`0 / 48` held-out. RAAM is a little better on seen-slot exact pass, but neither
model can treat the current context as a reliable binding table yet. The next
useful step is to make an even smaller atomic copy task with one family, one
slot pair, no decoys, and mirrored validation first. Only after that passes
should we add decoys, multiple fields, held-out slots, and then patch formatting
back in.

After artifact pull, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## AgentCoder Atomic Copy Gate

Added the smallest copy control after the copy-only ladder still failed: a
single slot family that copies only `symbol=<value>` and `file=<value>` from a
no-decoy context. The runner mirrors packed validation by default.

Added:

- `scripts/make_agentcoder_atomic_copy_sft.py`
- `scripts/run_agentcoder_atomic_copy_gate.py`
- `configs/scratch/raam_agentcoder_atomic_copy_gate.yaml`
- `configs/scratch/transformer_agentcoder_atomic_copy_gate.yaml`
- `tests/test_agentcoder_atomic_copy_generator.py`

Updated:

- `docs/AGENTIC_CODING_EVALS.md` now documents the atomic copy control.
- The atomic generator/runner now support `--train-records` and `--eval-cases`
  so we can distinguish one-record overfit from multi-binding failure.

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_atomic_copy_sft.py scripts/run_agentcoder_atomic_copy_gate.py scripts/eval_overfit_sanity.py scripts/make_agentcoder_copyonly_sft.py scripts/run_agentcoder_copy_gate.py
rm -rf work/atomic_copy_smoke && python3 scripts/make_agentcoder_atomic_copy_sft.py --train-output work/atomic_copy_smoke/train.jsonl --cases-output work/atomic_copy_smoke/cases.json --manifest-output work/atomic_copy_smoke/manifest.json --eval-mode ladder
python3 -m pytest -q tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_copyonly_generator.py tests/test_agentcoder_slotcopy_generator.py
git diff --check
```

Results:

- syntax checks passed
- ladder generator emitted `64` train records and `64` eval cases
- focused local tests passed: `13 passed in 0.16s`
- one-record generator mode emitted `1` train record and `4` ladder eval cases
- one-record focused tests passed: `5 passed in 0.06s`
- `git diff --check` passed

Vast RTX 5090 validation, 64-pair mirror control:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_copy_generator.py tests/test_agentcoder_copyonly_generator.py tests/test_agentcoder_slotcopy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_copy_mirror_20260703T054229Z/raam \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_atomic_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_copy_mirror_20260703T054229Z/transformer \
  --device cuda \
  --clean \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote 64-pair mirror results:

- focused atomic/copy/slot generator tests: `13 passed in 0.24s`
- full remote test suite: `41 passed in 32.80s`
- train records: `64`
- eval cases: `32`
- train tokens: `6272`
- validation tokens: `6272`
- `mirror_val: true`

| Model | Mirror Exact Pass | Val Loss | Train Loss | Tokens Seen |
| --- | ---: | ---: | ---: | ---: |
| RAAM atomic mirror | 6 / 32 | 0.097215 | 0.088383 | 921600 |
| Transformer atomic mirror | 0 / 32 | 0.192129 | 0.193607 | 921600 |

Representative completion check:

- RAAM prompt for `copy_symbol_000` returned `copy_symbol_056`.
- Transformer prompt for multiple cases collapsed to `copy_symbol_015` /
  `copy_file_003.py`.

Vast RTX 5090 validation, one-record mirror control:

```bash
/venv/main/bin/python -m pytest -q tests/test_agentcoder_atomic_copy_generator.py
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_copy_one_20260703T054652Z/raam \
  --device cuda \
  --clean \
  --train-records 1 \
  --eval-cases 1 \
  --steps 400 \
  --no-fail
/venv/main/bin/python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_atomic_copy_gate.yaml \
  --output-dir /root/raam-lm/runs/agentcoder_atomic_copy_one_20260703T054652Z/transformer \
  --device cuda \
  --clean \
  --train-records 1 \
  --eval-cases 1 \
  --steps 400 \
  --no-fail
/venv/main/bin/python -m pytest -q
```

Remote one-record results:

- focused atomic generator tests: `5 passed in 0.08s`
- full remote test suite: `42 passed in 32.88s`
- train records: `1`
- eval cases: `1`
- train tokens: `98`
- validation tokens: `98`
- `mirror_val: true`

| Model | Mirror Exact Pass | Val Loss | Train Loss | Tokens Seen |
| --- | ---: | ---: | ---: | ---: |
| RAAM one-record atomic | 1 / 1 | 0.0000775 | 0.0000784 | 307200 |
| Transformer one-record atomic | 1 / 1 | 0.0002455 | 0.0002557 | 307200 |

Local artifact pulls:

- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_copy_mirror_20260703T054229Z`
- `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_atomic_copy_one_20260703T054652Z`

Checkpoint weights were not pulled.

Interpretation: training/generation/eval plumbing can overfit one exact copy
pair for both models. The failure begins when multiple possible bindings exist,
even with no decoys and mirrored validation. The next useful step is a cardinality
sweep over `--train-records` values such as `1,2,4,8,16,32,64` with mirrored
eval. That will locate the point where exact binding breaks and tell us whether
we need a data/objective change, a decoding/eval change, or a model-capacity
change before returning to chat/coding training.

After artifact pulls, both Vast RTX 5090 instances were verified stopped:

```text
43627905 exited
43634442 exited
```

## 2026-07-03 - Request-Key Stop-EOS Value Copy Gate

The current broad chat checkpoint is still
`stage5_raam_agentcoder_100m_lr5e5_export_20260703T012841Z`, and it is still not
a useful assistant. This step intentionally did not continue broad SFT. It
targeted the smallest reproducible failure left in the exact repo-context
retrieval path: after the request-key copy route found the correct held-out
value prefix, generation often continued into repeated suffix tokens instead of
stopping cleanly.

Code change committed and pushed first:

- commit `a66ab18` - `Emit EOS after request-key value copies`
- added `CopyHeadConfig.request_key_follow_stop_strength`
- added `CopyHeadConfig.request_key_follow_stop_emit_token_id`
- extended the request-key follow route to return an EOS-emission distribution
  after a matched generated value prefix reaches the source value stop token
- enabled the route in the matched RAAM and Transformer request-value configs
- added a focused copy-head regression test for EOS at value stop
- updated FLOP accounting for the request-key follow/stop route

Local validation for the code change:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/copy_head.py tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m py_compile src/raam_lm/flops.py
python3 scripts/estimate_flops.py --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml
python3 scripts/estimate_flops.py --config configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml
git diff --check
```

Results:

- syntax checks passed
- focused key/value generator tests passed: `13 passed in 0.10s`
- RAAM local FLOPs estimate: `2944384` FLOPs/token, config hash
  `608a842b9c759728`
- Transformer local FLOPs estimate: `3537152` FLOPs/token, config hash
  `d490e0c1d3c9c1fe`
- `git diff --check` passed

Rechecked locally after the Vast run:

```bash
python3 -m py_compile src/raam_lm/config.py src/raam_lm/copy_head.py src/raam_lm/flops.py tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py
python3 -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py
```

Result: `13 passed in 0.12s`.

Vast RTX 5090 run:

- instance: `43634442`
- run ID: `agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z`
- remote run root:
  `/root/raam-lm/runs/agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z`
- local artifact path:
  `/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z`

Remote command:

```bash
cd /root/raam-lm
git pull --ff-only
RUN_ID=agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z
RUN_ROOT=/root/raam-lm/runs/$RUN_ID
mkdir -p "$RUN_ROOT/logs" work
/venv/main/bin/python -m pytest -q tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_pipeline.py -k "request_key or keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards" | tee "$RUN_ROOT/logs/tests.log"
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml --output-dir "$RUN_ROOT/raam" --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 --max-new-tokens 48 --clean --no-fail | tee "$RUN_ROOT/logs/raam.log"
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py --config configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml --output-dir "$RUN_ROOT/transformer" --device cuda --seed 29 --train-records 96 --train-variants-per-row 4 --eval-cases 32 --steps 1600 --seq-len 128 --vocab-size 2048 --eval-mode coverage_ladder --completion-mode value_only --target-fields 1 --max-new-tokens 48 --clean --no-fail | tee "$RUN_ROOT/logs/transformer.log"
```

Remote focused tests passed: `38 passed, 8 deselected, 1 warning in 5.11s`.

Comparable previous prefix-route artifact:
`/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_request_value_prefix_onefield_compare_20260703T124955Z`.

| Model | Run | Exact Pass | Seen | Covered Value | Held-Out | Val Loss | Tokens/sec | FLOPs/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM | previous prefix route | 72 / 96 | 32 / 32 | 32 / 32 | 8 / 32 | 0.682547 | 27080.3 | 2639104 |
| Transformer | previous prefix route | 91 / 96 | 31 / 32 | 32 / 32 | 28 / 32 | 0.709046 | 26933.3 | 3231872 |
| RAAM | stop-EOS route | 93 / 96 | 32 / 32 | 32 / 32 | 29 / 32 | 1.870651 | 25023.2 | 2794880 |
| Transformer | stop-EOS route | 93 / 96 | 32 / 32 | 32 / 32 | 29 / 32 | 2.003771 | 21322.4 | 3387648 |

Both stop-EOS runs failed the same three held-out cases:

- `keyvalue_heldout_021`
- `keyvalue_heldout_027`
- `keyvalue_heldout_028`

Artifact pull:

```bash
RUN_ID=agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z
REMOTE=/root/raam-lm/runs/$RUN_ID
LOCAL=/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_$RUN_ID
mkdir -p "$LOCAL"
ssh -o StrictHostKeyChecking=no -p 34442 root@ssh1.vast.ai "cd '$REMOTE' && tar -czf - \
  logs \
  raam/summary.json raam/keyvalue_eval.json raam/generated/keyvalue_eval_cases.json raam/generated/keyvalue_manifest.json raam/packed/manifest.json raam/tokenizer.json raam/train/config.yaml raam/train/manifest.json raam/train/train_log.jsonl \
  transformer/summary.json transformer/keyvalue_eval.json transformer/generated/keyvalue_eval_cases.json transformer/generated/keyvalue_manifest.json transformer/packed/manifest.json transformer/tokenizer.json transformer/train/config.yaml transformer/train/manifest.json transformer/train/train_log.jsonl" | tar -xzf - -C "$LOCAL"
find "$LOCAL" -type f \( -name '*.pt' -o -name '*.safetensors' -o -name '*.bin' -o -name '*train.jsonl' \) -print
```

Result:

- local artifact size: `4.6M`
- no checkpoint files pulled
- no packed token bins pulled
- no generated training JSONL pulled
- pulled logs, summaries, eval cases/results, manifests, tokenizer/config
  copies, and train logs

Remote cleanup:

```bash
ssh -o StrictHostKeyChecking=no -p 34442 root@ssh1.vast.ai 'RUN=/root/raam-lm/runs/agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z; find "$RUN" -type f \( -name "*.pt" -o -name "*.safetensors" \) -print -delete; echo REMAINING; find "$RUN" -type f \( -name "*.pt" -o -name "*.safetensors" \) -print'
```

Deleted:

- `/root/raam-lm/runs/agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z/raam/train/checkpoints/last.pt`
- `/root/raam-lm/runs/agentcoder_request_value_stop_eos_onefield_compare_20260703T143000Z/transformer/train/checkpoints/last.pt`

The final `REMAINING` scan printed no checkpoint/model files.

Vast stop verification:

```bash
vastai stop instance 43634442
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p 34442 root@ssh1.vast.ai 'echo still_up'
vastai show instances-v1 --raw
```

Results:

- stop command returned `stopping instance 43634442.`
- SSH check returned `Connection refused`
- sanitized final status fields:

```text
43627905 actual_status=exited cur_state=stopped intended_status=stopped next_state=stopped
43634442 actual_status=exited cur_state=stopped intended_status=stopped next_state=stopped
```

Interpretation:

- This is a real, evidence-backed improvement over the current RAAM
  request-value copy failure: exact pass improved from `72 / 96` to `93 / 96`,
  and held-out exact retrieval improved from `8 / 32` to `29 / 32`.
- The previous suffix-loop failure was mostly a termination/control problem
  after the correct value prefix was selected.
- The improvement is not RAAM-specific because the matched Transformer also
  reaches `93 / 96` and fails the same held-out cases.
- RAAM is still favorable on this tiny gate's efficiency metrics versus the
  matched Transformer: lower validation loss, higher tokens/sec, fewer
  estimated FLOPs/token, and similar exact pass.
- This does not make the Stage 5 85M chat checkpoint a useful assistant. It
  clears one exact-output repo-context copying blocker. The next useful gate is
  to extend from one requested value to ordered multi-field outputs and then
  back to patch-format validity, while treating exact pass rate rather than
  validation loss as the primary signal.

## 2026-07-03 - Ordered Multi-Field Request-Value Gate

Goal:

- Test the next blocker after one requested value: ordered `target_fields=2`
  value-only repo-context copying.
- Compare RAAM with the matched Transformer before changing the route.
- If it fails, diagnose with generated completions and implement the smallest
  real fix.

Baseline two-field comparison before fixes:

```text
RUN_ID=agentcoder_request_value_stop_eos_multifield2_compare_20260703T141125Z
```

Command shape:

```bash
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml \
  --output-dir "$RUN_ROOT/raam" \
  --device cuda --seed 29 \
  --train-records 128 --train-variants-per-row 4 --eval-cases 48 \
  --steps 2000 --seq-len 128 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only \
  --target-fields 2 --max-new-tokens 96 \
  --clean --no-fail
```

Matched Transformer used the same command with
`configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml`.

Baseline result:

| Model | target fields | exact pass | seen | covered | held-out | behavior accuracy | val loss | tokens/sec | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM | 2 | 0 / 144 | 0 / 48 | 0 / 48 | 0 / 48 | 1.0 | 2.083217 | 26373.7 | 2944384 |
| Transformer | 2 | 0 / 144 | 0 / 48 | 0 / 48 | 0 / 48 | 1.0 | 1.998388 | 27450.0 | 3537152 |

Failure diagnosis:

- Both models emitted exactly one requested value and then `<eos>`.
- Example: expected `["copy_service_000", "copy_file_000.py"]`, generated
  `["copy_file_000.py"]`.
- This was not RAAM-specific. The request-key stop route treated every copied
  source value boundary as answer termination.

Implemented route fixes:

- `src/raam_lm/copy_head.py`
  - Track request-key rank and current assistant value index from generated
    separators.
  - Emit source separator tokens between requested values.
  - Emit `<eos>` only after the final requested key.
  - Reset value-prefix matching after assistant-generated separators.
  - Add optional `request_key_follow_source_separator_token_id` to avoid using
    source anchors that are not actual `key:` positions.
- `src/raam_lm/config.py`
  - Add `request_key_follow_source_separator_token_id`.
- `configs/scratch/*_agentcoder_keyvalue_request_value_gate.yaml`
  - Enable source separator anchoring with token id `271` (`:`).
- Tests:
  - ordered multi-value stop/separator behavior
  - source-separator anchor constraint
  - config assertion for the new gate field

Verification:

```text
python3 -m py_compile src/raam_lm/config.py src/raam_lm/copy_head.py tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py

/venv/main/bin/python -m pytest tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py \
  -k "request_key or keyvalue_request_value_configs" -q
# 11 passed, 26 deselected

/venv/main/bin/python -m pytest tests/test_agentcoder_keyvalue_copy_generator.py tests/test_agentcoder_pipeline.py tests/test_copy_head.py \
  -k "request_key or keyvalue or copy_head or assistant_loss_mask or dataset_packing_writes or pack_dataset_cli_forwards" -q
# 40 passed, 8 deselected, 1 warning
```

Patched RAAM diagnostic rerun:

```text
RUN_ID=agentcoder_request_value_stop_eos_multifield2_orderedfix_compare_20260703T142933Z
```

The full original `max_new_tokens=96` eval was interrupted after it became clear
the patched route could run to long generations on failures. Capped diagnostic
evals used `max_new_tokens=24` against the completed RAAM checkpoint.

| Eval | exact pass | seen | covered | held-out | behavior accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 12-case smoke | 10 / 12 | 10 / 12 | n/a | n/a | 1.0 |
| full capped, ordered stop fix | 79 / 144 | 39 / 48 | 40 / 48 | 0 / 48 | 1.0 |
| full capped, plus source separator anchor | 79 / 144 | 39 / 48 | 40 / 48 | 0 / 48 | 1.0 |

Interpretation:

- The ordered stop/separator fix is real: the gate moved from `0 / 144` exact
  to `79 / 144` on the capped diagnostic eval.
- It is not strong enough to escalate to `target_fields=3`.
- Held-out remains `0 / 48`; the source-separator anchor did not move this,
  so the simple false-anchor hypothesis is not the main blocker.
- Remaining failures are mostly:
  - separator-repeat loops on seen/covered cases, for example first value then
    repeated `<|assistant|>` markers;
  - partial byte-fallback copies on held-out values, for example `ce_164` or
    truncated second values.
- Validation loss is secondary here; exact value sequence is the primary signal.

Artifact pull:

```text
/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_request_value_stop_eos_multifield2_compare_20260703T141125Z
/home/lumalgo/Documents/Codex/2026-07-02/g/outputs/vast_agentcoder_request_value_stop_eos_multifield2_orderedfix_compare_20260703T142933Z
```

Remote cleanup:

- Deleted remote `.pt`, `.bin`, and `.safetensors` files for the patched run.
- Stopped Vast instance `43634442`; SSH no longer accepts a session, and Vast
  reports `cur_state=stopped`, `intended_status=stopped`, `next_state=stopped`.

Next step:

- Do not escalate to `target_fields=3`.
- Add a focused held-out byte-fallback/value-boundary gate:
  - one and two requested values where values are intentionally out-of-vocab
    byte fallback strings;
  - exact first-token, continuation, separator, and final EOS assertions;
  - explicit no-repeat/no-special-token checks after generated separators.
- Fix the request-key route until that synthetic held-out gate passes, then
  rerun the same `target_fields=2` coverage ladder against both RAAM and the
  matched Transformer.

## 2026-07-03 - focused byte-fallback boundary gate and target_fields=2 rerun

Objective:

- Fix the remaining `target_fields=2` request-value copy blocker before any
  escalation to `target_fields=3`.
- Add a focused held-out byte-fallback/value-boundary gate covering one and two
  requested values.
- Rerun the full `target_fields=2` coverage ladder only after the focused gate
  passed.

Implementation:

- `src/raam_lm/copy_head.py`
  - Added deterministic assistant output-token indexing for request-value
    routing, so byte-fallback continuation advances by exact source offset
    rather than ambiguous repeated prefix matches.
  - Kept multi-key request ordering and source-separator anchoring.
- `src/raam_lm/config.py`
  - Added tokenizer-aware `resolve_copy_head_token_ids(config, tokenizer)` for
    newline, colon, period, space, and comma ids that move after tokenizer
    training.
- `scripts/train.py`, `scripts/eval_overfit_sanity.py`, `scripts/generate.py`,
  and `scripts/qualitative_checkpoint_inspect.py`
  - Resolve copy-head route-control token ids after loading the trained
    tokenizer.
- `scripts/run_agentcoder_keyvalue_copy_gate.py`
  - Added a sequence-window preflight. The gate now fails before training if
    the generated training records do not fit `--seq-len` or if eval prompt
    plus expected completion cannot fit `config.max_seq_len`.
- `configs/scratch/*_agentcoder_keyvalue_request_value_gate.yaml`
  - Raised `max_seq_len`, `train.seq_len`, and eval long-context length from
    `160` / `128` to `384`.
  - Raised `request_key_follow_strength` from `12.0` to `20.0` so the
    deterministic first-token route beats learned EOS after nonterminal
    separators.
- Tests:
  - Added held-out byte-fallback/value-boundary cases for `target_fields=1`
    and `target_fields=2`.
  - Added a request-value config/window test for the focused gate shape.
  - Added a copy-head guard that the second value's first token beats strong
    learned EOS and `<|assistant|>` logits after a generated nonterminal
    separator.
  - Added config resolver coverage for trained-tokenizer punctuation/newline
    ids.

Key diagnosis:

- The first focused one-value run failed at `1 / 24` before token-id
  resolution because trained tokenizer punctuation/newline ids did not match
  the hard-coded config ids.
- Re-evaluating that same checkpoint after resolver wiring improved to
  `12 / 24`.
- Prompt/completion length audit then found the larger blocker: focused
  boundary prompts were already `248-288` tokens before generation and up to
  `341` tokens with expected completion, while eval kept only `160` tokens.
  This truncated the source table during generation.
- Re-evaluating the old one-field checkpoint with a `384` context changed it
  from `12 / 24` to `24 / 24`, confirming truncation as a real failure cause.
- The clean two-field focused run initially reached only `7 / 24`. Failures
  copied the first wrapped value, emitted the separator, then chose EOS. A
  logit probe on `keyvalue_heldout_002` showed EOS at `13.65` and the correct
  second-value `<` route at `12.38`; increasing first-token request route
  strength to `20.0` made `<` top at `20.00`.

Verification:

```text
python3 -m py_compile tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py scripts/run_agentcoder_keyvalue_copy_gate.py src/raam_lm/config.py src/raam_lm/copy_head.py

PYTHONPATH=src:. python3 -m pytest tests/test_agentcoder_keyvalue_copy_generator.py -k 'focus_window or request_value_configs or byte_fallback' -q
# 3 passed, 12 deselected

/venv/main/bin/python -m pytest tests/test_config.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py \
  -k "resolve_copy_head or byte_fallback or value_boundary or focus_window or request_value_configs or request_key_follow_copies_byte_fallback_offsets" -q
# 5 passed, 36 deselected
```

Focused gate run:

```text
RUN_ID=agentcoder_request_value_bytefallback_boundary_seq384_20260703T160000Z
```

| Run | exact pass | behavior accuracy | value accuracy | val loss | tokens/sec | FLOPs/token | max eval full tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM value-boundary held-out, target_fields=1 | 24 / 24 | 1.0 | 1.0 | 1.033544 | 81259.7 | 6899328 | 286 |
| RAAM value-boundary held-out, target_fields=2 | 24 / 24 | 1.0 | 1.0 | 0.830342 | 80818.7 | 6910080 | 341 |

Artifacts:

```text
runs/agentcoder_request_value_bytefallback_boundary_seq384_20260703T160000Z/raam_fields1/summary.json
runs/agentcoder_request_value_bytefallback_boundary_seq384_20260703T160000Z/raam_fields1/keyvalue_eval.json
runs/agentcoder_request_value_bytefallback_boundary_seq384_20260703T160000Z/raam_fields2/summary.json
runs/agentcoder_request_value_bytefallback_boundary_seq384_20260703T160000Z/raam_fields2/keyvalue_eval.json
```

Full `target_fields=2` coverage ladder rerun:

```text
RUN_ID=agentcoder_request_value_target_fields2_coverage_seq384_20260703T170000Z
```

Command shape:

```bash
/venv/main/bin/python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml \
  --output-dir "$RUN_ROOT/raam" \
  --device cuda --seed 31 \
  --train-records 96 --train-variants-per-row 4 --eval-cases 48 \
  --steps 1600 --seq-len 384 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only \
  --target-fields 2 --max-new-tokens 96 \
  --clean --no-fail
```

Matched Transformer used the same command with
`configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml`.

| Model | exact pass | seen | covered | held-out | behavior accuracy | value accuracy | val loss | tokens/sec | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM | 133 / 144 | 48 / 48 | 48 / 48 | 37 / 48 | 1.0 | 0.923611 | 2.154128 | 80519.0 | 7095168 |
| Transformer | 144 / 144 | 48 / 48 | 48 / 48 | 48 / 48 | 1.0 | 1.0 | 1.978342 | 62743.6 | 8109824 |

RAAM remaining held-out failures:

- `keyvalue_heldout_002`, requested `parser, fixture`; expected
  `copy_parser_098.py`, `case_fixture_098`; generated
  `copy_parser_098.py` followed by repeated `<|assistant|>ccopy_parser_098.py`
  fragments.
- `keyvalue_heldout_007`, requested `adapter, file`; expected
  `disabled_adapter_103`, `copy_file_103`; generated repeated first-value /
  assistant-marker fragments and only a partial `copy_file_103`.
- `keyvalue_heldout_022`, requested `route, service`; expected
  `allowed_route_118`, `allowed_service_118`; generated repeated
  `allowed_route_118` with assistant markers plus unrelated fragments.

Interpretation:

- The focused blocker is fixed: one-value and two-value held-out
  byte-fallback boundary gates both pass `24 / 24`.
- The broader target_fields=2 ladder is much stronger than the previous
  `79 / 144` capped RAAM diagnostic, but RAAM still trails the matched
  Transformer on held-out value copies.
- The remaining RAAM failures are no longer simple EOS-after-first-value
  failures. They are held-out separator/assistant-marker repetition loops after
  the first value, while seen and tokenizer-covered tiers are now perfect.
- Do not escalate to `target_fields=3` yet. The next modeling step should focus
  on RAAM's held-out second-value stability and special-token repetition after
  generated separators, using the Transformer `144 / 144` run as the matched
  target.

Artifact pull:

- Pulled the two run directories locally under `runs/`, excluding checkpoints
  and packed `.bin` files.

Remote cleanup:

- Stopped Vast instance `43634442`.
- Verified `cur_state=stopped`, `intended_status=stopped`,
  `next_state=stopped`, `actual_status=exited`.

Post-ladder decoder suppression re-eval:

- Added generation-time suppression of special/control tokens inside assistant
  completions while keeping `<eos>` available. This is intended to target the
  remaining RAAM held-out failures that repeat `<|assistant|>` after a
  generated separator.
- Added `AgentCoderTokenizer.generation_suppressed_token_ids()` and wired it
  into `scripts/eval_overfit_sanity.py`, `scripts/generate.py`, and
  `scripts/qualitative_checkpoint_inspect.py`.
- Local verification:

```text
python3 -m py_compile src/raam_lm/tokenization.py scripts/eval_overfit_sanity.py scripts/generate.py scripts/qualitative_checkpoint_inspect.py tests/test_tokenization.py

PYTHONPATH=src:. python3 -m pytest tests/test_tokenization.py -q
# 1 passed
```

- Remote verification after syncing the suppression change back to Vast:

```text
/venv/main/bin/python -m py_compile src/raam_lm/tokenization.py scripts/eval_overfit_sanity.py scripts/generate.py scripts/qualitative_checkpoint_inspect.py tests/test_tokenization.py tests/test_copy_head.py tests/test_agentcoder_keyvalue_copy_generator.py

PYTHONPATH=src:. /venv/main/bin/python -m pytest tests/test_tokenization.py tests/test_config.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py \
  -k "generation_suppressed or resolve_copy_head or byte_fallback or value_boundary or focus_window or request_value_configs or request_key_follow_copies_byte_fallback_offsets" -q
# 6 passed, 36 deselected
```

- Re-evaluated saved `target_fields=2` coverage checkpoints with suppressed
  special/control ids `[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` and `<eos>`
  still allowed:

```text
/venv/main/bin/python scripts/eval_overfit_sanity.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml \
  --tokenizer "$OUT/tokenizer.json" \
  --checkpoint "$OUT/train/checkpoints/last.pt" \
  --device cuda \
  --cases-json "$OUT/generated/keyvalue_eval_cases.json" \
  --output "$OUT/keyvalue_eval_suppressed.json" \
  --max-new-tokens 96 \
  --min-pass-rate 1.0 \
  --no-fail
```

Suppressed-eval result:

| Model | exact pass | seen | covered | held-out | behavior accuracy | value accuracy | failures |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM | 144 / 144 | 48 / 48 | 48 / 48 | 48 / 48 | 1.0 | 1.0 | 0 |
| Transformer | 144 / 144 | 48 / 48 | 48 / 48 | 48 / 48 | 1.0 | 1.0 | 0 |

Suppressed-eval artifacts:

```text
runs/agentcoder_request_value_target_fields2_coverage_seq384_20260703T170000Z/raam/keyvalue_eval_suppressed.json
runs/agentcoder_request_value_target_fields2_coverage_seq384_20260703T170000Z/transformer/keyvalue_eval_suppressed.json
```

Final interpretation:

- The focused held-out byte-fallback/value-boundary blocker is fixed.
- The broader `target_fields=2` coverage ladder now passes for RAAM and the
  matched Transformer when generation masks non-EOS special/control tokens.
- The remaining pre-suppression RAAM failures were decoding/control-token
  failures, not a need to escalate the task to `target_fields=3`.

Final Vast cleanup:

- Original checkpoint instance `43634442` stopped and verified:
  `cur_state=stopped`, `intended_status=stopped`, `next_state=stopped`,
  `actual_status=exited`.
- Scratch instances created while the original instance was unavailable:
  `43715662` and `43715959`; both were stopped and then destroyed with
  `vastai destroy instance ... -y`.

## 2026-07-03 - real Stage 5 pilot training run

Objective:

- Start a real pilot training run after the `target_fields=2` request-value copy
  blocker was fixed, using the real expanded Stage 5 AgentCoder corpus rather
  than another synthetic copy gate.

Preparation:

- Verified local Python syntax for the touched model/tokenizer/training/eval
  paths and shell syntax for the Vast launch/pull scripts.
- Verified targeted local tests:

```text
PYTHONPATH=src:. python3 -m pytest tests/test_tokenization.py tests/test_agentcoder_keyvalue_copy_generator.py -k 'generation_suppressed or focus_window or request_value_configs or byte_fallback' -q
# 4 passed, 12 deselected
```

- Started Vast instance `43634442` and verified `/root/raam-lm`,
  `/root/data/agentcoder_stage5/tokenizer.json`,
  `/root/data/agentcoder_stage5/packed_2048/train.bin`, and
  `/root/data/agentcoder_stage5/packed_2048/val.bin`.
- Synced the current dirty local worktree changes to `/root/raam-lm`.
- Verified remote syntax and targeted tests:

```text
PYTHONPATH=src:. /venv/main/bin/python -m pytest tests/test_tokenization.py tests/test_config.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py \
  -k "generation_suppressed or resolve_copy_head or byte_fallback or value_boundary or focus_window or request_value_configs or request_key_follow_copies_byte_fallback_offsets" -q
# 6 passed, 36 deselected
```

- Added launcher pass-through for `RAAM_MLOPS_PROJECT_PATH` in:
  - `scripts/vast_train_100m_candidate.sh`
  - `scripts/vast_launch_stage5_gate.sh`

Pilot command shape:

```bash
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
RUN_ID=stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RAW_DIR=/root/data/agentcoder_stage5/raw \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048 \
TOKENIZER=/root/data/agentcoder_stage5/tokenizer.json \
RAAM_MLOPS_PROJECT_PATH=/root/raam-lm \
STEPS=2000 \
RESUME_STEPS=2200 \
SAVE_EVERY=200 \
EVAL_EVERY=100 \
EXPORT_CHECKPOINT=1 \
KEEP_TRAINING_CHECKPOINTS=1 \
bash scripts/vast_launch_stage5_gate.sh
```

Run evidence:

- Remote run root:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z`
- Local artifact pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current`
- Local MLOps run id: `live-train-cea845692806`
- `.mlops` metric rows after summary logging: `2201`
- Model: compression-only RAAM 100M Stage 5, `lr: 5e-5`, reconstruction/MTP
  disabled in the objective, fallback gated-conv mixer backend.
- Tokens per step: `65536`.
- Training completed through the planned resume endpoint at step `2199`.
- Vast cleanup: instance `43634442` verified `cur_state=stopped`,
  `intended_status=stopped`, `next_state=stopped`, `actual_status=exited`.

Metrics:

| Metric | Value |
| --- | ---: |
| Logged train rows | 2200 |
| Tokens seen | 144179200 |
| First validation loss | 10.389572143554688 |
| Best validation loss | 3.0210491180419923 at step 800 |
| Final validation loss | 3.2564679265022276 at step 2199 |
| Final train loss | 2.822740316390991 |
| Final tokens/sec | 238189.41361988886 |
| Peak allocated VRAM MB | 12310.53662109375 |
| Agentic JSON tool-call validity | 0.0 |
| Agentic mean patch apply rate | 0.0 |
| Best-step qualitative useful completions | 0 / 8 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.389572143554688 |
| 100 | 7.264174485206604 |
| 200 | 5.9533222913742065 |
| 300 | 4.944909036159515 |
| 400 | 4.30405638217926 |
| 500 | 3.5682665586471556 |
| 600 | 3.1166778326034548 |
| 700 | 3.04752801656723 |
| 800 | 3.0210491180419923 |
| 900 | 3.0360927820205688 |
| 1000 | 3.1705517292022707 |
| 1100 | 3.2336891055107118 |
| 1200 | 3.1605583310127257 |
| 1300 | 3.2766310930252076 |
| 1400 | 3.275023567676544 |
| 1500 | 3.2647506475448607 |
| 1600 | 3.1681557416915895 |
| 1700 | 3.201487684249878 |
| 1800 | 3.2474425196647645 |
| 1900 | 3.1019840836524963 |
| 1999 | 3.2698657631874086 |
| 2000 | 3.269370412826538 |
| 2100 | 3.296204650402069 |
| 2199 | 3.2564679265022276 |

Pulled artifacts:

```text
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/runner.log
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/train/train_log.jsonl
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/train/agentic_eval.json
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/train/generation_smoke.txt
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/best_step_000800_qualitative_samples.json
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/best_step_000800_qualitative_samples.md
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/train/checkpoints/model_only_step_000800_fp16.pt
runs/vast_backups/stage5_raam_agentcoder_100m_real_pilot_lr5e5_20260703T164256Z/current/train/checkpoints/model_only_fp16.pt
```

Model-only export sizes:

- Best step 800 export: `201316691` bytes.
- Final step 2199 export: `201315043` bytes.

Interpretation:

- We did begin real pilot training on the expanded Stage 5 corpus and completed
  the planned `2000 -> 2200` resume run.
- The training path is operational: real data, full 2048-token packing,
  checkpoint/resume, live `.mlops` metrics, generation smoke, agentic eval,
  compact final export, and compact best-step export all completed.
- The best validation point still occurs early, around step `800`, and the
  final checkpoint is worse than that best point. This repeats the earlier
  capped-LR pattern rather than clearing the model for a larger spend.
- The best-step qualitative inspection and final agentic eval still show no
  useful chat/coding behavior. Treat `model_only_step_000800_fp16.pt` as the
  current best base-LM pilot artifact, not as a usable assistant model.
- Next highest-value work: do not simply run longer at `5e-5`; test either a
  lower/decayed LR schedule after step `800`, better chat/code objective
  weighting, or a data/tokenizer/template audit that explains why validation
  improves while qualitative assistant behavior remains incoherent.

## 2026-07-03 - assistant-loss masked decay pilot

Objective:

- Test the immediate Phase 1 blocker from the real pilot: the model was
  learning the packed corpus loss but not assistant behavior, and validation
  drifted after the best region.

Changes under test:

- Added assistant-only loss-mask packing to the Vast Stage 5 wrappers.
- Added delayed cosine decay support: warm up to `5e-5`, hold through step
  `800`, then decay toward `1e-5`.
- Config under test:
  `configs/scratch/raam_agentcoder_100m_stage5_lr5e5_masked_decay.yaml`.
- Packed corpus:
  `/root/data/agentcoder_stage5/packed_2048_assistant_loss`.

Run:

```bash
RUN_ID=stage5_raam_agentcoder_100m_masked_decay_20260703T172643Z \
CONFIG=configs/scratch/raam_agentcoder_100m_stage5_lr5e5_masked_decay.yaml \
PACKED_DIR=/root/data/agentcoder_stage5/packed_2048_assistant_loss \
ASSISTANT_LOSS_ONLY=1 \
STEPS=2000 RESUME_STEPS=2200 SAVE_EVERY=200 EVAL_EVERY=100 \
EXPORT_CHECKPOINT=1 KEEP_TRAINING_CHECKPOINTS=1 \
bash scripts/vast_launch_stage5_gate.sh
```

Local artifact pull:
`runs/vast_backups/stage5_raam_agentcoder_100m_masked_decay_20260703T172643Z/current`.
The pull includes logs, manifest, agentic eval, generation smoke, and the compact
`model_only_fp16.pt` export. Full optimizer checkpoints were not pulled. Vast
instance `43634442` was stopped and verified `exited` after the pull.

Metrics:

| Metric | Value |
| --- | ---: |
| Logged train rows | 2200 |
| Tokens seen | 144179200 |
| Total parameters | 83857922 |
| Non-embedding parameters | 67080706 |
| First validation loss | 10.436189889907837 |
| Best validation loss | 2.8247740983963014 at step 1900 |
| Final validation loss | 3.06810177564621 at step 2199 |
| Final train loss | 2.0745818614959717 |
| Final tokens/sec | 236448.31849039317 |
| Peak allocated VRAM MB | 12310.66162109375 |
| Agentic JSON tool-call validity | 0.0 |
| Agentic mean patch apply rate | 0.0 |

Validation curve:

| Step | Val next-token loss |
| ---: | ---: |
| 0 | 10.436189889907837 |
| 100 | 7.6992237091064455 |
| 200 | 6.742557954788208 |
| 300 | 5.711968731880188 |
| 400 | 4.869263696670532 |
| 500 | 4.086435151100159 |
| 600 | 3.6132681012153625 |
| 700 | 3.594025444984436 |
| 800 | 3.257691729068756 |
| 900 | 3.1018675565719604 |
| 1000 | 3.1586724519729614 |
| 1100 | 3.0839218378067015 |
| 1200 | 3.073712611198425 |
| 1300 | 3.144188177585602 |
| 1400 | 3.039172911643982 |
| 1500 | 3.0964579701423647 |
| 1600 | 2.897498893737793 |
| 1700 | 2.924788475036621 |
| 1800 | 2.920748841762543 |
| 1900 | 2.8247740983963014 |
| 1999 | 3.047560155391693 |
| 2000 | 3.090535855293274 |
| 2100 | 2.881490647792816 |
| 2199 | 3.06810177564621 |

Interpretation:

- Phase 1 training infrastructure is working on real data: assistant loss masks,
  2048-token packing, 84M-param training, optimizer resume, validation, agentic
  eval, model-only export, artifact pull, and instance shutdown all completed.
- The masked objective plus decay materially improved validation versus the
  previous real pilot (`2.8248` best versus `3.0210`, `3.0681` final versus
  `3.2565`), and it moved the best point from step `800` to step `1900`.
- The drift problem is reduced but not solved: final loss is still `+0.2433`
  above the best measured loss.
- The useful-behavior gate is still not cleared. Final agentic eval remains
  `0.0` for JSON validity and patch apply, and generation is still incoherent.
- One procedural issue: the best step `1900` was evaluated but not checkpointed
  because saves were every `200` steps. Future gates should align save/eval
  cadence around candidate best regions, or save best-on-validation directly.

## 2026-07-03 - best-checkpoint restore and guarded 84M useful-behavior run

Objective:

- Turn the repeated post-best validation drift into a non-promoted artifact state,
  while preserving the first nonzero useful qualitative behavior at the 84M scale.

Implementation changes:

- Added `train.save_best`, `train.early_stop_patience_evals`,
  `train.early_stop_min_delta`, `train.early_stop_min_step`, and
  `train.restore_best_on_finish`.
- `scripts/train.py` now saves `checkpoints/best.pt` on validation improvement,
  can stop after repeated non-improving validation checks, and can restore the
  best checkpoint before writing `checkpoints/last.pt`.
- Vast wrappers forward the new controls.
- Stage 5 guarded defaults:
  - broad masked/records-only configs: patience `4`, minimum stop step `1800`,
    restore best on finish.
  - curated SFT config: patience `3`, minimum stop step `2500`, restore best on
    finish.

Validation:

```bash
python3 -m py_compile src/raam_lm/config.py scripts/train.py tests/test_config.py tests/test_agentcoder_pipeline.py
bash -n scripts/vast_train_50m.sh scripts/vast_train_100m_candidate.sh scripts/vast_launch_stage5_gate.sh
python3 -m pytest -q tests/test_config.py
```

Local pytest that imports `raam_lm.agent_data` cannot run on the workstation
Python because `torch` is not installed. Remote RTX 5090 validation passed:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_config.py tests/test_agentcoder_pipeline.py \
  -k "train_resume_generate_and_agentic_eval or packing_can_focus or packing_can_disable or filter_long or pack_dataset_cli"
```

Remote result: `5 passed, 11 deselected`.

Forced tiny drift-control smoke:

- With `early_stop_patience_evals=1`, `early_stop_min_delta=999`, and
  `restore_best_on_finish=true`, training stopped at step `1`, restored
  `last.pt` to best step `0`, and marked `last_restored_from_best=true`.

Corrected 84M guarded curated SFT run:

- Remote run:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_curated_guarded_late_20260703T194229Z`.
- Local pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_curated_guarded_late_20260703T194229Z/current`.
- MLOps train backfill: `backfill-train-b676d676a172`.
- Vast instance `43634442` was stopped and verified `exited` after artifact pull.

Metrics:

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Non-embedding parameters | 67080706 |
| Best validation loss | 0.4507140815258026 at step 2500 |
| Early stop step | 2800 |
| Final/promoted checkpoint step | 2500 |
| Restored best on finish | true |
| Curated eval pass rate | 9 / 10 |
| Behavior accuracy | 10 / 10 |
| Qualitative useful samples | 4 / 8 |
| Compact model export | `train/checkpoints/model_only_guarded_fp16.pt` |

Validation curve excerpt:

| Step | Val next-token loss | Best step | No-improve evals | Stopped |
| ---: | ---: | ---: | ---: | --- |
| 1500 | 0.9468620866537094 | 1500 | 0 | false |
| 1600 | 0.5646604672074318 | 1600 | 0 | false |
| 1700 | 0.8782196938991547 | 1600 | 1 | false |
| 1800 | 0.77409528195858 | 1600 | 2 | false |
| 1900 | 0.9934779927134514 | 1600 | 3 | false |
| 2000 | 0.9929693937301636 | 1600 | 4 | false |
| 2100 | 0.7492213994264603 | 1600 | 5 | false |
| 2200 | 0.819332629442215 | 1600 | 6 | false |
| 2300 | 0.6929627060890198 | 1600 | 7 | false |
| 2400 | 0.6331205368041992 | 1600 | 8 | false |
| 2500 | 0.4507140815258026 | 2500 | 0 | false |
| 2600 | 0.8772342354059219 | 2500 | 1 | false |
| 2700 | 0.8555787801742554 | 2500 | 2 | false |
| 2800 | 0.9369257986545563 | 2500 | 3 | true |

Interpretation:

- Raw validation loss still drifts after the best checkpoint, but the trainer now
  stops the late degradation and writes the promoted `last.pt` from the best
  validation state. This fixes the artifact-promotion failure mode behind the
  post-step-800 drift.
- The 84M model has verified nonzero useful qualitative behavior on the guarded
  checkpoint: `4 / 8` useful samples, with useful examples across chat, coding,
  software-engineering, and agentic-coding categories.
- This is still a curated curriculum result, not proof of a broadly strong
  chat/code model. The broad records-only run remains the next generalization
  gate.

## 2026-07-03 - validation LR backoff drift experiment

Objective:

- Reduce raw post-best validation drift, not just restore the promoted artifact
  to the best checkpoint.

Implementation changes:

- Added validation-triggered LR backoff controls:
  `train.validation_lr_decay_patience_evals`,
  `train.validation_lr_decay_factor`,
  `train.validation_lr_decay_min_scale`, and
  `train.validation_lr_decay_min_step`.
- `scripts/train.py` now logs `scheduled_learning_rate`,
  `validation_lr_decay_scale`, `validation_lr_decay_count`, and per-eval decay
  events. LR backoff is separate from early stopping.
- Vast wrappers forward the new controls.
- Stage 5 defaults:
  - broad masked/records-only configs back off from step `800`.
  - curated SFT backs off from step `2500`.

Validation:

```bash
python3 -m py_compile src/raam_lm/config.py scripts/train.py tests/test_config.py tests/test_agentcoder_pipeline.py
bash -n scripts/vast_train_50m.sh scripts/vast_train_100m_candidate.sh scripts/vast_launch_stage5_gate.sh
python3 -m pytest -q tests/test_config.py
```

Local result: `2 passed`.

Remote RTX 5090 result:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_config.py tests/test_agentcoder_pipeline.py \
  -k "train_resume_generate_and_agentic_eval or packing_can_focus or packing_can_disable or filter_long or pack_dataset_cli"
```

Remote result: `5 passed, 11 deselected`.

Forced tiny LR-backoff smoke:

- With `validation_lr_decay_patience_evals=1`,
  `validation_lr_decay_factor=0.5`, and `validation_lr_decay_min_scale=0.25`,
  the trainer applied two LR backoffs at steps `1` and `2`, ending with
  `validation_lr_decay_scale=0.25`.

Curated plateau-LR run:

- Remote run:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_curated_plateau_lr_20260703T200010Z`.
- Local pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_curated_plateau_lr_20260703T200010Z/current`.
- MLOps train backfill: `backfill-train-6c445e2fb55a`.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.4507140815258026 at step 2500 |
| Early stop step | 2800 |
| Final/promoted checkpoint step | 2500 |
| Validation LR backoffs | 2 |
| Final LR scale | 0.25 |
| Raw drift after best | +0.488767609000206 |
| Curated eval pass rate | 9 / 10 |
| Qualitative useful samples | 4 / 8 |

Interpretation:

- The backoff machinery works and preserves the useful 84M curated checkpoint,
  but it does not materially reduce curated post-best drift. The curated drift
  appears to be dominated by tiny validation-set instability/overfit rather than
  a simple LR overshoot.

Broad records-only plateau-LR run:

- Remote run:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_records_plateau_lr_20260703T201012Z`.
- Local pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_records_plateau_lr_20260703T201012Z/current`.
- MLOps train backfill: `backfill-train-696be62cff28`.
- Vast instance `43634442` was stopped and verified `exited` after artifact
  pulls.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 3.286919629573822 at step 1500 |
| Early stop step | 1900 |
| Final/promoted checkpoint step | 1500 |
| Validation LR backoffs | 2 |
| Final LR scale | 0.25 |
| Raw drift after best | +0.05094232559204093 |
| Agentic JSON tool-call validity | 0.0 |
| Agentic mean patch apply rate | 0.0 |
| Qualitative useful samples | 0 / 8 |

Validation curve:

| Step | Val next-token loss | LR scale | LR backoff | Stopped |
| ---: | ---: | ---: | --- | --- |
| 1500 | 3.286919629573822 | 1.0 | false | false |
| 1600 | 3.368794345855713 | 1.0 | true -> 0.5 | false |
| 1700 | 3.2997918367385863 | 0.5 | true -> 0.25 | false |
| 1800 | 3.3051330924034117 | 0.25 | false | false |
| 1900 | 3.337861955165863 | 0.25 | false | true |

Interpretation:

- LR backoff reduced broad records-only raw drift amplitude versus the earlier
  records-cap run (`+0.0509` versus `+0.1706`), but the absolute best validation
  loss was worse and useful behavior stayed at zero.
- Current best practical state remains: use guarded/best-restored checkpoints for
  promotion, keep the curated 84M checkpoint as proof of nonzero useful behavior,
  and treat broad useful behavior as the next generalization blocker.

## 2026-07-03 - mixed broad/curated curriculum bridge

Objective:

- Find a Stage 5 curriculum that keeps some broad real records in training while
  preserving nonzero useful qualitative behavior and reducing post-best drift.

Implementation:

- Added `scripts/make_agentcoder_mixed_curriculum.py`, which samples structured
  real AgentCoder records and repeats the curated behavior anchors into one
  mixed JSONL.
- Added `configs/scratch/raam_agentcoder_100m_mixed_stage5_sft.yaml`, an 84M
  bridge config with 512-token training, best checkpoint restore, early stop, and
  validation LR backoff.
- Added `tests/test_mixed_curriculum.py`.

Validation:

```bash
python3 -m py_compile scripts/make_agentcoder_mixed_curriculum.py tests/test_mixed_curriculum.py
python3 -m pytest -q tests/test_mixed_curriculum.py tests/test_config.py
PYTHONPATH=src python3 - <<'PY'
from raam_lm.config import load_config
cfg = load_config("configs/scratch/raam_agentcoder_100m_mixed_stage5_sft.yaml")
print(cfg.d_model, cfg.n_layers, cfg.train.seq_len, cfg.train.validation_lr_decay_patience_evals)
PY
```

Local result: `3 passed`.

Remote RTX 5090 result:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_mixed_curriculum.py tests/test_config.py
```

Remote result: `3 passed`.

Broad-heavy mixed run:

- Remote run:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_mixed_curriculum_20260703T202712Z`.
- Local pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_mixed_curriculum_20260703T202712Z/current`.
- MLOps train backfill: `backfill-train-a9e8c6e18384`.
- Mix: `2000` real records + `1920` curated records.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 2.6351894438266754 at step 2000 |
| Early stop step | 2400 |
| Final/promoted checkpoint step | 2000 |
| Raw drift after best | +0.2767331600189209 |
| Curated eval pass rate | 2 / 10 |
| Qualitative useful samples | 0 / 8 |

Interpretation:

- The broad-heavy mix improved validation versus pure records-only, but it
  washed out useful behavior. This ratio should not be promoted.

Curated-dominant mixed run:

- Remote run:
  `/root/raam-lm/runs/stage5_raam_agentcoder_100m_mixed_curated_dominant_20260703T203910Z`.
- Local pull:
  `runs/vast_backups/stage5_raam_agentcoder_100m_mixed_curated_dominant_20260703T203910Z/current`.
- MLOps train backfill: `backfill-train-adda7bfcc70a`.
- Mix: `500` real records + `2880` curated records.
- Vast instance `43634442` was stopped and verified `exited` after artifact
  pulls.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 2.0162661224603653 at step 1900 |
| Early stop step | 2300 |
| Final/promoted checkpoint step | 1900 |
| Restored best on finish | true |
| Validation LR backoffs | 2 |
| Raw drift after best | +0.009236752986907959 |
| Curated eval pass rate | 3 / 10 |
| Behavior accuracy | 7 / 10 |
| Agentic JSON validity | 0.0 |
| Agentic mean patch apply rate | 0.0 |
| Qualitative useful samples | 2 / 8 |
| Compact model export | `train/checkpoints/model_only_mixed_curated_dominant_fp16.pt` |

Validation curve:

| Step | Val next-token loss | Best step | LR scale | Stopped |
| ---: | ---: | ---: | ---: | --- |
| 1500 | 2.339496925473213 | 1500 | 1.0 | false |
| 1600 | 2.1321469247341156 | 1600 | 1.0 | false |
| 1700 | 2.223752737045288 | 1600 | 1.0 | false |
| 1800 | 2.140855960547924 | 1600 | 1.0 | false |
| 1900 | 2.0162661224603653 | 1900 | 1.0 | false |
| 2000 | 2.1169208586215973 | 1900 | 1.0 -> 0.5 | false |
| 2100 | 2.027666814625263 | 1900 | 0.5 -> 0.25 | false |
| 2200 | 2.070050060749054 | 1900 | 0.25 | false |
| 2300 | 2.0255028754472733 | 1900 | 0.25 | true |

Interpretation:

- This is the first single 84M run that satisfies both target signals together:
  post-best drift is reduced to near-flat (`+0.0092`), and qualitative behavior is
  nonzero (`2 / 8` useful samples).
- It is not a broadly useful coding agent yet: agentic JSON/tool and patch evals
  are still zero, and curated exact pass rate is only `3 / 10`. Treat this as a
  Phase 1 gate pass, not a strong-model claim.

## 2026-07-04 - coding ladder repair SFT pilot

Objective:

- Move from tiny/memorized code behavior toward verified held-out coding tasks.
- Add a repo-owned coding ladder generator and strict eval, then run a real
  repair SFT pilot on an RTX 5090 from the current 84M curated-dominant
  checkpoint.

Implementation:

- Added `scripts/make_agentcoder_coding_ladder_sft.py`.
- Added `scripts/eval_coding_ladder.py`.
- Added `configs/scratch/raam_agentcoder_100m_coding_ladder_repair_sft.yaml`.
- Added `tests/test_coding_ladder.py`.

Local validation:

```bash
python3 -m py_compile scripts/make_agentcoder_coding_ladder_sft.py scripts/eval_coding_ladder.py tests/test_coding_ladder.py
python3 -m pytest -q tests/test_coding_ladder.py
```

Result: `3 passed`.

Local packing smoke remained blocked on this CPU environment because local
Python did not have `torch` installed. The actual pack/train/eval path was
validated on Vast.ai with the CUDA image.

Vast.ai execution:

- The cheap `Type #27594213` offer disappeared before it could be rented.
- A known reachable RTX 5090 instance was used instead:
  `43806336`, `NVIDIA GeForce RTX 5090`, torch `2.12.0+cu130`, CUDA available.
- A cheaper follow-up instance `43807428` was tried but SSH refused connections,
  so it was stopped.
- All instances were verified `exited` after artifact pulls:
  `43627905`, `43634442`, `43806336`, and `43807428`.

Baseline ladder eval, before repair:

- Checkpoint:
  `runs/vast_backups/stage5_raam_agentcoder_100m_mixed_curated_dominant_20260703T203910Z/current/train/checkpoints/model_only_mixed_curated_dominant_fp16.pt`.
- Ladder pass rate: `0 / 10`.
- Function pass count: `1`.
- JSON pass count: `0`.
- Patch pass count: `0`.
- Nonsense failures: `1`.

First repair run:

- Remote run:
  `/root/raam-lm/runs/coding_ladder_repair_20260704T_remote`.
- Local pull:
  `runs/vast_backups/coding_ladder_repair_20260704T_remote/current`.
- Dataset: `602` train records, `10` eval cases, no exact train/eval prompt
  overlaps.
- Packed data: `542` train docs, `60` validation docs, `121051` train tokens,
  `14277` validation tokens.
- Resume mode: `model_only`, starting after the curated-dominant checkpoint.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.1912618987262249 at step 2150 |
| Final/promoted checkpoint step | 2150 |
| Restored best on finish | true |
| Validation LR backoffs | 2 |
| Coding ladder pass rate | 0 / 10 |
| Ladder function pass count | 3 |
| Ladder JSON pass count | 0 |
| Ladder patch pass count | 0 |
| Ladder nonsense failures | 0 |
| Curated eval pass rate | 3 / 10 |
| Curated behavior accuracy | 7 / 10 |
| Qualitative useful samples | 1 / 8 |
| Compact model export | `train/checkpoints/model_only_coding_ladder_repair_fp16.pt` |

Interpretation:

- The first repair run reduced nonsense and improved some raw function behavior,
  but strict ladder pass stayed `0 / 10`.
- Raw generations showed a stop-control problem: after correct code, outputs
  continued into learned final/test-command fragments.
- Root cause: ladder records trained a `trace` followed by a `final`, while
  generation suppresses special control tokens such as `<|final|>`. The model
  learned to continue beyond the desired assistant answer instead of ending.

Stop-control repair run:

- Generator adjusted so ladder records end after the assistant trace, training
  EOS immediately after the desired code/patch/JSON answer.
- Remote run:
  `/root/raam-lm/runs/coding_ladder_repair_stop_control_20260704T_remote`.
- Local pull:
  `runs/vast_backups/coding_ladder_repair_stop_control_20260704T_remote/current`.
- Dataset: `820` train records, `10` eval cases, no exact train/eval prompt
  overlaps.
- Packed data: `738` train docs, `82` validation docs, `153400` train tokens,
  `16780` validation tokens.
- Resume mode: `optimizer`, continuing from the first repair run.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.203645471483469 at step 2225 |
| Final/promoted checkpoint step | 2225 |
| Restored best on finish | true |
| Validation LR backoffs | 1 |
| Coding ladder pass rate | 3 / 10 |
| Passed ladder cases | `is_even`, `is_odd`, `filter_even` |
| Failed ladder cases | `count_even`, `safe_int`, `parse_port`, patch 0, patch 1, pytest, JSON |
| Ladder function pass count | 3 |
| Ladder JSON pass count | 0 |
| Ladder patch pass count | 0 |
| Ladder nonsense failures | 0 |
| Curated eval pass rate | 3 / 10 |
| Curated behavior accuracy | 8 / 10 |
| Qualitative useful samples | 1 / 8 |
| Compact model export | `train/checkpoints/model_only_coding_ladder_stop_control_fp16.pt` |

Interpretation:

- This is a real improvement over the baseline ladder eval (`0 / 10` to
  `3 / 10`) and it fixed the most obvious trailing-nonsense failure for tiny
  functions.
- It does not satisfy the stronger promotion gate yet: `count_even`, `safe_int`,
  `parse_port`, patches, pytest generation, and JSON command output still fail.
- The run should be kept as evidence and as a repair checkpoint, but not claimed
  as a strong coding model.

Next scaling step:

- Build a narrower no-final repair dataset focused on the exact failed frontier:
  `count_even`, `safe_int`, `parse_port`, one-file diffs with exact file headers,
  pytest files, and strict JSON command responses.
- Keep tiny functions as anchors, but reduce their share so the model cannot win
  only by repeating `is_even`/`filter_even`.
- Add more syntactic variations for medium functions and reject malformed train
  examples with the same evaluator used for held-out tests.
- Run another short repair SFT from
  `model_only_coding_ladder_stop_control_fp16.pt`.
- Promote only if held-out ladder pass rises above `3 / 10` and at least one of
  `count_even`, `safe_int`, or `parse_port` passes without hurting curated eval
  below `3 / 10`.

Medium frontier repair pilot, July 4, 2026:

- Built and tested a no-final medium-repair generator for the failed frontier:
  `count_even`, `safe_int`, `parse_port`, exact unified diffs, pytest generation,
  strict JSON command output, and a small number of tiny-function anchors.
- Remote run:
  `/root/raam-lm/runs/medium_repair_20260704T_remote`.
- Local pull:
  `runs/vast_backups/medium_repair_20260704T_remote/current`.
- Dataset: `1428` train records, `20` eval cases, `0` non-empty final fields,
  no exact train/eval prompt overlaps.
- Packed data: `1285` train docs, `143` validation docs, `152717` train loss
  tokens, `16539` validation loss tokens, assistant-loss-only.
- Hardware: Vast.ai `43811933`, `NVIDIA GeForce RTX 5090`, torch `2.12.0+cu130`,
  CUDA available. The instance was destroyed after evidence was pulled.
- Resume mode: `model_only`, from
  `model_only_coding_ladder_stop_control_fp16.pt` at checkpoint step `2225`.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.22265831753611565 at step 2350 |
| Early stop step | 2450 |
| Final checkpoint step after restore | 2350 |
| Restored best on finish | true |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 1 / 20 |
| Curated eval after pilot | 2 / 10 |
| Curated behavior accuracy after pilot | 2 / 10 |
| Gate decision | fail, do not promote |

Interpretation:

- The run was a valid GPU pilot, but it made the model worse on the gate:
  expanded medium eval fell from `3 / 20` to `1 / 20`, and curated eval fell
  from the previous `3 / 10` floor to `2 / 10`.
- None of the target frontier functions (`count_even`, `safe_int`, `parse_port`)
  passed after the pilot.
- Qualitative samples still showed mixed fragments from adjacent patterns and
  occasional premature `<eos>`.
- The exported compact checkpoint existed remotely, but it was not retained
  locally because the gate failed and transfer speed was too slow to justify
  saving a non-promoted checkpoint. Evidence, logs, manifests, evals, and remote
  MLOps files were pulled locally.

Next scaling step:

- Do not promote the medium repair pilot. Keep
  `model_only_coding_ladder_stop_control_fp16.pt` as the current best repair
  checkpoint.
- Rebalance the repair data before the next GPU run: increase anchor weight for
  `is_odd` and `filter_even`, separate function-completion, patch, pytest, and
  JSON batches more aggressively, and add local eval checks that reject any run
  that forgets the tiny-function floor before training deeper on medium tasks.

Medium repair curriculum diagnosis:

- The failed medium pilot used only `8` tiny-anchor records out of `1428` total
  records and did not include `is_odd` as a medium-repair anchor. After training,
  only `is_even` still passed; `is_odd` and `filter_even` regressed.
- Function examples, patch examples, pytest examples, and JSON command examples
  were all interleaved with similar wording around tests and commands. The
  failed generations show cross-family contamination: function answers contain
  pytest command fragments and diff fragments, JSON answers contain Python/code
  text, and patch answers collapse into unrelated add/subtract diffs.
- Several post-train outputs are syntactically malformed, such as incomplete
  comprehensions, invalid `try = int` statements, and partial pytest bodies.
- Some samples terminate early with `<eos>`, while others continue with repeated
  command fragments. This means the no-final stop-control fix helped the earlier
  tiny ladder, but the medium curriculum still teaches incompatible answer
  shapes too close together.
- Next design requirement: anchor the tiny floor explicitly, separate answer
  families with stricter system/task wording, reject malformed train examples at
  generation time, and add local tests that fail if the generated eval/gate no
  longer protects `is_even`, `is_odd`, and `filter_even`.

Medium repair v2 local preflight:

- Generator format updated to `agentcoder-medium-frontier-repair-v2`.
- Added `is_odd` to the tiny-function anchor floor and raised the default
  anchor repeats. The local v2 preflight produced `1468` train records, `20`
  eval cases, and `16` records each for `is_even`, `is_odd`, and `filter_even`.
- Added family-contamination validation so function answers reject diff/test/JSON
  fragments, JSON answers reject code fences and function bodies, and pytest
  answers reject diff/JSON/test-command fragments.
- Added `scripts/preflight_medium_repair.py`. The GPU preflight command must be
  run before training without `--skip-checkpoint-eval`; it generates the v2 data,
  evaluates the stop-control checkpoint on the expanded medium eval, and fails if
  the baseline pass count is below `3` or if any tiny-floor case fails.
- Added lower-risk config
  `configs/scratch/raam_agentcoder_100m_medium_repair_v2_sft.yaml`: resume from
  the stop-control checkpoint, train only to step `2400`, use LR `1e-5`, and
  restore the best checkpoint.
- Local verification passed:
  `python3 -m py_compile scripts/make_agentcoder_medium_repair_sft.py scripts/preflight_medium_repair.py scripts/eval_coding_ladder.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  and `python3 -m pytest -q tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `6 passed`.

Medium repair v2 GPU pilot, July 4, 2026:

- Remote run:
  `/root/raam-lm/runs/medium_repair_v2_20260704T_remote`.
- Local pull:
  `runs/vast_backups/medium_repair_v2_20260704T_remote/current`.
- Hardware: Vast.ai `43813976`, `NVIDIA GeForce RTX 5090`, torch
  `2.12.0+cu130`, CUDA available. The instance was destroyed after evidence was
  pulled.
- Preflight gate passed before training: expanded medium baseline was `3 / 20`,
  and `is_even`, `is_odd`, and `filter_even` all passed.
- Dataset: `1468` train records, `20` eval cases, `48` tiny-anchor records, no
  exact train/eval prompt overlaps.
- Packed data: `1321` train docs, `147` validation docs, `145124` train loss
  tokens, `17292` validation loss tokens.
- Resume mode: `model_only`, from
  `model_only_coding_ladder_stop_control_fp16.pt` at checkpoint step `2225`.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.2425982914865017 at step 2399 |
| Final checkpoint step after restore | 2399 |
| Restored best on finish | true |
| Validation LR backoffs | 2 |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 3 / 20 |
| Tiny floor after pilot | 3 / 3 |
| Target frontier functions after pilot | 0 / 6 |
| Curated eval after pilot | 2 / 10 |
| Curated behavior accuracy after pilot | 2 / 10 |
| Qualitative useful samples | 2 / 8 |
| Gate decision | fail, do not promote |

Interpretation:

- The v2 curriculum fixed the specific forgetting failure from the previous
  medium pilot: `is_even`, `is_odd`, and `filter_even` all remained passing.
- It still did not improve the actual medium frontier. Expanded medium eval
  stayed at the `3 / 20` baseline, and all `count_even`, `safe_int`, and
  `parse_port` target cases failed.
- Curated eval remained below the promotion floor at `2 / 10`.
- The compact v2 checkpoint was exported remotely but not retained locally
  because the gate failed. Evidence, logs, manifests, evals, qualitative samples,
  and MLOps files were pulled.
- Current best remains
  `model_only_coding_ladder_stop_control_fp16.pt`.

Next scaling step:

- Do not keep pushing mixed-family SFT in one pass. The v2 result shows anchor
  preservation is achievable, but medium skills are not being acquired.
- Next attempt should train in staged family-specific phases or use much smaller
  updates: first medium functions only with tiny anchors, then evaluate; only
  after a function gain should patch/pytest/JSON families be introduced.

Function-only medium repair Stage A, July 4, 2026:

- Added a function-only Stage A generator:
  `scripts/make_agentcoder_function_repair_sft.py`.
- Added GPU preflight:
  `scripts/preflight_function_repair.py`.
- Added focused tests:
  `tests/test_function_repair.py`.
- Added config:
  `configs/scratch/raam_agentcoder_100m_function_repair_sft.yaml`.
- Local verification passed:
  `python3 -m py_compile scripts/make_agentcoder_function_repair_sft.py scripts/preflight_function_repair.py tests/test_function_repair.py`
  and
  `python3 -m pytest -q tests/test_function_repair.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `9 passed`.
- Local packing with the stop-control tokenizer succeeded with assistant-only
  masks: `1361` train docs, `151` validation docs, `109981` train loss tokens,
  and `11483` validation loss tokens.

Function-only medium repair GPU pilot, July 4, 2026:

- Remote run:
  `/root/raam-lm/runs/function_repair_20260704T_remote`.
- Local pull:
  `runs/vast_backups/function_repair_20260704T_remote/current`.
- Hardware: Vast.ai `43815937`, `NVIDIA GeForce RTX 5090`, torch
  `2.12.0+cu130`, CUDA available. The instance was destroyed after evidence was
  pulled, and `vastai show instances-v1 --raw` reported no remaining instances.
- Preflight gate passed before training: expanded medium baseline was `3 / 20`,
  tiny floor was `3 / 3`, and target frontier functions were `0 / 6`.
- Dataset: `1512` train records, all `function_completion`, with
  `360` examples each for `count_even`, `safe_int`, and `parse_port`, plus
  `144` examples each for `is_even`, `is_odd`, and `filter_even`.
- Packed data: `1361` train docs, `151` validation docs, `109981` train loss
  tokens, `11483` validation loss tokens, assistant-loss-only.
- Resume mode: `model_only`, from
  `model_only_coding_ladder_stop_control_fp16.pt` at checkpoint step `2225`.

| Metric | Value |
| --- | ---: |
| Total parameters | 83857922 |
| Best validation loss | 0.16017264872789383 at step 2675 |
| Final checkpoint step after restore | 2675 |
| Restored best on finish | true |
| Validation LR backoffs | 2 |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 3 / 20 |
| Tiny floor after pilot | 3 / 3 |
| Target frontier functions after pilot | 0 / 6 |
| Curated eval after pilot | 1 / 10 |
| Curated behavior accuracy after pilot | 1 / 10 |
| Qualitative useful samples | 1 / 8 |
| Gate decision | fail, do not promote |

Interpretation:

- The function-only Stage A pilot preserved the tiny-function floor:
  `is_even`, `is_odd`, and `filter_even` all passed after training.
- It did not teach the medium frontier. `count_even`, `safe_int`, and
  `parse_port` all still failed in both ladder and medium held-out cases.
- Curated eval regressed to `1 / 10`, below the non-promotion floor.
- The run is a valid negative result. Current best remains
  `model_only_coding_ladder_stop_control_fp16.pt`.
- The compact checkpoint was not pulled because the run failed the gate.
  Manifests, train logs, eval JSON, qualitative samples, summary, packed
  manifests, and MLOps metadata were pulled locally.
- Post-train target completions show syntax/form failure, not merely wrong edge
  cases. Examples include `return sum  for the`, `return sum  = int`,
  `def safe_int(value, default if n % 2 == 0)`, and `try = int ... 2 == 0)`.
  The model still emits tiny `is_even` correctly, but medium functions collapse
  into malformed snippets and copied tiny-function tails.

Next scaling step:

- Do not move to larger parameter counts yet.
- The next attempt should not simply repeat SFT on more paraphrases of the same
  function answers. The model is preserving memorized tiny functions but not
  acquiring even concentrated medium-function behavior.
- Before another GPU run, inspect the post-train completions for the six target
  function cases and decide whether the blocker is prompt mismatch, code-fence
  formatting, insufficient update strength, tokenizer/code-token weakness, or
  architecture/optimizer limits.
- Add a tiny target-function memorization probe before the next paid pilot:
  train/eval should prove the model can exactly generate valid `count_even`,
  `safe_int`, and `parse_port` on train-like prompts before expecting held-out
  medium generalization.

Function-only probe update after Stage A failure:

- Updated `scripts/make_agentcoder_function_repair_sft.py` and
  `scripts/preflight_function_repair.py` to emit
  `function_probe_cases.json`.
- The probe contains exact train-like function-generation cases for
  `count_even`, `safe_int`, `parse_port`, `is_even`, `is_odd`, and
  `filter_even`.
- This probe intentionally overlaps Stage A train prompts and is diagnostic
  only; it is separate from the held-out expanded medium gate.
- Held-out medium train/eval prompts remain disjoint.
- Local verification passed:
  `python3 -m pytest -q tests/test_function_repair.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `10 passed`.

Function-only exact probe diagnostic GPU pilot, July 4, 2026:

- Remote run:
  `/root/raam-lm/runs/function_probe_memorize_20260704T_remote`.
- Local pull:
  `runs/vast_backups/function_probe_memorize_20260704T_remote/current`.
- Hardware: Vast.ai `43817371`, `NVIDIA GeForce RTX 5090`, torch
  `2.12.0+cu130`, CUDA available. The instance was destroyed after evidence was
  pulled, and `vastai show instances-v1 --raw` reported no remaining instances.
- Purpose: diagnostic only. This run tested whether the current model can
  exactly reproduce train-like `count_even`, `safe_int`, and `parse_port`
  answers when trained on a tiny repeated function-only curriculum.
- Local and remote tests passed before training:
  `python3 -m pytest -q tests/test_function_repair.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `11 passed`.
- Preflight passed on CUDA before training: baseline expanded medium was
  `3 / 20`, baseline target functions were `0 / 6`, and baseline tiny floor was
  `3 / 3`.
- Dataset: `900` train records, all `function_completion`, with `180` examples
  each for `count_even`, `safe_int`, and `parse_port`, plus `120` examples each
  for `is_even`, `is_odd`, and `filter_even`.
- Packed data: `810` train docs, `90` validation docs, `60380` train loss
  tokens, `6640` validation loss tokens, assistant-loss-only.
- Resume mode: from
  `model_only_coding_ladder_stop_control_fp16.pt`.

| Metric | Value |
| --- | ---: |
| Checkpoint step evaluated | 2474 |
| Best validation loss seen | 0.12365658953785896 |
| Exact probe after pilot | 3 / 6 |
| Exact target probe after pilot | 0 / 3 |
| Exact anchor probe after pilot | 3 / 3 |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 3 / 20 |
| Medium tiny floor after pilot | 3 / 3 |
| Medium target frontier functions after pilot | 0 / 6 |
| Curated eval after pilot | 1 / 10 |
| Curated behavior accuracy after pilot | 0.1 |
| Qualitative useful samples | 1 / 8 |
| Gate decision | diagnostic fail, do not promote |

Interpretation:

- The exact memorization diagnostic failed the important part of the probe:
  `count_even`, `safe_int`, and `parse_port` all failed even on train-like
  prompts after focused repeated SFT.
- The tiny-function floor remained intact: `is_even`, `is_odd`, and
  `filter_even` passed in both exact probe and expanded medium eval.
- Expanded medium stayed flat at the baseline `3 / 20`, and curated eval stayed
  at `1 / 10`. This is not a hidden improvement masked by the frontier gate.
- The failure points away from mixed-family contamination as the only blocker.
  Even isolated function-only data is not enough for the current setup to learn
  these slightly larger function bodies reliably.
- Current best remains
  `model_only_coding_ladder_stop_control_fp16.pt`. The diagnostic checkpoints
  were not pulled because the run failed the gate.

Next scaling step:

- Do not increase parameter count yet.
- First fix the inability to memorize exact medium function bodies. The next
  experiment should be a smaller and more controlled local or GPU probe that
  isolates architecture/update mechanics: one target function at a time, longer
  training or higher effective update strength, and direct inspection of logits
  or generations at fixed intervals.
- If a one-function probe cannot learn `count_even` exactly, scaling the
  curriculum or model size is premature. If it can learn one function exactly,
  reintroduce `safe_int` and `parse_port` one at a time before returning to the
  held-out expanded medium gate.

One-function `count_even` exact probe GPU pilot, July 4, 2026:

- Remote run:
  `/root/raam-lm/runs/function_count_even_probe_20260704T_remote`.
- Local pull:
  `runs/vast_backups/function_count_even_probe_20260704T_remote/current`.
- Hardware: Vast.ai `43819726`, `NVIDIA GeForce RTX 5090`, torch
  `2.12.0+cu130`, CUDA available. The instance was destroyed after artifacts
  and remote MLOps data were pulled; `vastai show instances-v1 --raw` reported
  `total_instances: 0`.
- Purpose: diagnostic only. This run tested whether the current checkpoint can
  learn one selected medium function, `count_even`, while preserving the tiny
  floor.
- Local validation before training passed:
  `python3 -m py_compile scripts/make_agentcoder_function_repair_sft.py scripts/preflight_function_repair.py scripts/eval_function_memorization_probe.py scripts/eval_function_probe_timeline.py tests/test_function_repair.py`
  and
  `python3 -m pytest -q tests/test_function_repair.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `14 passed`.
- CUDA preflight before training passed: baseline expanded medium was `3 / 20`,
  baseline target functions were `0`, and baseline tiny floor was `3 / 3`.
- Dataset: `720` train records, all `function_completion`, with `360`
  `count_even` examples and `120` examples each for `is_even`, `is_odd`, and
  `filter_even`.
- Packed data: `648` train docs, `72` validation docs, `32313` train loss
  tokens, `3567` validation loss tokens, assistant-loss-only.
- Resume mode: from
  `model_only_coding_ladder_stop_control_fp16.pt`.

| Metric | Value |
| --- | ---: |
| Checkpoint step evaluated | 3024 |
| Best validation loss seen | 0.0045226526708574966 |
| Exact probe after pilot | 4 / 4 |
| Exact target probe after pilot | 1 / 1 |
| Exact anchor probe after pilot | 3 / 3 |
| First checkpoint with target pass | step 2250 |
| First checkpoint with target and anchors pass | step 2300 |
| Timeline checkpoints evaluated | 19 |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 5 / 20 |
| Medium tiny floor after pilot | 3 / 3 |
| Medium `count_even` cases after pilot | 2 / 2 |
| Medium target frontier functions after pilot | 2 |
| Curated eval after pilot | 1 / 10 |
| Curated behavior accuracy after pilot | 0.1 |
| Qualitative useful samples | 1 / 8 |
| Diagnostic decision | one target exact pass |
| Promotion gate | diagnostic fail, do not promote |

Medium passed cases after this pilot:

- `ladder_is_even`
- `ladder_is_odd`
- `ladder_count_even`
- `ladder_filter_even`
- `medium_count_even_negatives`

Interpretation:

- The prior exact-memorization failure was not a hard architecture or optimizer
  impossibility. With a single selected target and stronger update pressure,
  the model learned `count_even` exactly and preserved `is_even`, `is_odd`, and
  `filter_even`.
- The improvement transferred to held-out medium `count_even` cases, raising
  expanded medium from `3 / 20` to `5 / 20`.
- The run still failed promotion because the stronger gates require broader
  medium performance, curated behavior did not recover above `1 / 10`, and
  `safe_int`, `parse_port`, patch, pytest, and JSON families remain failed.
- Current best remains
  `model_only_coding_ladder_stop_control_fp16.pt`. No checkpoint from this run
  was promoted or pulled as current best.

Next scaling step:

- Stay at the current parameter count.
- Use the now-validated one-target curriculum path to add `safe_int` as the
  next isolated target, then `parse_port`, while keeping the tiny anchors and
  timeline eval.
- Only after all three target functions can be learned without hurting curated
  behavior should the medium curriculum reintroduce pytest, JSON, and patch
  families.

### 2026-07-04 safe_int one-target diagnostic pilot

- Remote run:
  `/root/raam-lm/runs/function_safe_int_probe_20260704T_remote`.
- Local pull:
  `runs/vast_backups/function_safe_int_probe_20260704T_remote/current`.
- Hardware: Vast.ai `43822979`, `NVIDIA GeForce RTX 5090`, torch
  `2.12.0+cu130`, CUDA available. The instance was destroyed after artifacts
  were pulled; `vastai show instances-v1 --raw` reported `total_instances: 0`.
- Remote MLOps note: `/root/raam-lm/.mlops/experiments` did not exist on this
  instance, so no remote MLOps experiment artifacts were available to pull.
- Purpose: diagnostic only. This run tested whether the current checkpoint can
  learn one selected medium function, `safe_int`, while preserving the tiny
  floor.
- Local validation before training passed:
  `python3 -m py_compile scripts/make_agentcoder_function_repair_sft.py scripts/preflight_function_repair.py scripts/eval_function_memorization_probe.py scripts/eval_function_probe_timeline.py tests/test_function_repair.py`
  and
  `python3 -m pytest -q tests/test_function_repair.py tests/test_medium_repair.py tests/test_coding_ladder.py`
  produced `15 passed`.
- CUDA preflight before training passed: baseline expanded medium was `3 / 20`,
  baseline target functions were `0`, and baseline tiny floor was `3 / 3`.
- Dataset: `720` train records, all `function_completion`, with `360`
  `safe_int` examples and `120` examples each for `is_even`, `is_odd`, and
  `filter_even`.
- Packed data: `648` train docs, `72` validation docs, `38833` train loss
  tokens, `4247` validation loss tokens, assistant-loss-only.
- Resume mode: from
  `model_only_coding_ladder_stop_control_fp16.pt`.

| Metric | Value |
| --- | ---: |
| Checkpoint step evaluated | 3024 |
| Best validation loss seen | 0.0546931610442698 |
| Exact probe after pilot | 3 / 4 |
| Exact target probe after pilot | 1 / 1 |
| Exact anchor probe after pilot | 2 / 3 |
| Failed exact anchor | `filter_even` |
| First checkpoint with target pass | step 2500 |
| First checkpoint with target and anchors pass | none |
| Timeline checkpoints evaluated | 19 |
| Expanded medium baseline | 3 / 20 |
| Expanded medium after pilot | 3 / 20 |
| Medium tiny floor after pilot | 1 / 3 |
| Medium `safe_int` cases after pilot | 2 / 2 |
| Medium target frontier functions after pilot | 2 |
| Curated eval after pilot | 1 / 10 |
| Curated behavior accuracy after pilot | 0.1 |
| Qualitative useful samples | 0 / 8 |
| Diagnostic decision | one target exact fail |
| Promotion gate | diagnostic fail, do not promote |

Medium passed cases after this pilot:

- `ladder_is_odd`
- `ladder_safe_int`
- `medium_safe_int_defaults`

Interpretation:

- The model learned `safe_int` under isolated target pressure, including both
  held-out medium `safe_int` cases.
- The same run damaged the tiny floor. Exact probe kept `is_even` and `is_odd`
  but corrupted `filter_even`; expanded medium kept only `is_odd` from the
  tiny floor.
- The model also over-specialized on `safe_int`, answering unrelated function,
  patch, pytest, and JSON prompts with `safe_int` or malformed `filter_even`
  snippets.
- No timeline checkpoint had both the target and all anchors passing. The first
  target-pass checkpoint was step `2500`, but `is_odd` had already failed
  there. Later checkpoints restored `is_odd` but lost `filter_even`.
- Current best remains
  `model_only_coding_ladder_stop_control_fp16.pt`. No checkpoint from this run
  was promoted or pulled as current best.

Next scaling step:

- Do not scale parameters yet.
- Fix one-target retention before running another target: shorten or interrupt
  training around the first target pass, reduce update pressure, and/or
  increase anchor sampling with special attention to `filter_even`.
- Add a hard timeline selector/gate that refuses to promote unless target
  probes and all tiny anchors pass in the same checkpoint.
- After `safe_int` can pass without anchor damage, repeat the same diagnostic
  for `parse_port`; only then attempt a combined function-only frontier run.

## 2026-07-04 - Executable Coding Data/Eval Pipeline

Verified current local state before editing:

- Best Stage 5 base-LM evidence remains the `lr5e5` region around step `800`.
  Local `.mlops` run `live-train-cea845692806` reports best validation
  `3.0210491180419923` at step `800` and worse final validation
  `3.2564679265022276` at step `2199`.
- Current local key/value coverage ladder evidence still shows RAAM behind the
  Transformer baseline on exact held-out binding:
  `runs/agentcoder_request_value_target_fields2_coverage_seq384_20260703T170000Z/raam/summary.json`
  reports overall pass rate `0.9236111111111112` and held-out-slot pass rate
  `0.7708333333333334`; the matched Transformer summary reports `1.0` and
  `1.0`.
- No model checkpoint was promoted. Agentic/tool-call and patch quality should
  still be treated as unresolved.

Implemented:

- Added `scripts/make_agentcoder_executable_sft.py`, a reproducible SFT data
  builder for the next executable coding repair run.
- The builder emits canonical RAAM-AgentCoder JSONL from the local coding ladder
  plus optional local JSONL or Hugging Face streaming sources for
  `nvidia/OpenCodeInstruct`, `Samip/Scotch`, `KAKA22/CodeRM-UnitTest`, and
  `bigcode/commitpackft`.
- It filters toward Python examples with passing/high-score test signal,
  function bodies/docstrings, pytest records, and small single-file diffs.
- It writes `agentcoder_executable_train.jsonl`,
  `agentcoder_executable_eval_cases.json`, and
  `agentcoder_executable_manifest.json`.
- It records behavior/topic/source counts, filter settings, and exact train/eval
  user-prompt overlaps.
- It avoids executing arbitrary public-source code during preparation; held-out
  executable eval cases are only created from structured JSON `args`/`expected`
  tests that can later be consumed by `scripts/eval_coding_ladder.py`.
- Aligned the matched key-follow configs with the current key/value window tests
  by raising `max_seq_len`, `train.seq_len`, and `eval.long_context_lengths` to
  `384` in both
  `configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml` and
  `configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml`.

Smoke command:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/make_agentcoder_executable_sft.py \
  --output-dir runs/agentcoder_executable_sft_smoke \
  --ladder-repeats 1 \
  --curated-anchor-repeats 0
```

Smoke artifact:

```text
runs/agentcoder_executable_sft_smoke/agentcoder_executable_manifest.json
```

Smoke result:

| Metric | Value |
| --- | ---: |
| Train records | 41 |
| Eval cases | 10 |
| Exact train/eval user-prompt overlaps | 0 |
| Function-completion records | 24 |
| Patch records | 9 |
| Pytest-generation records | 3 |
| JSON-tool-command records | 5 |

Attempted MLOps logging for this ad hoc smoke via MCP failed because the run id
did not already exist (`run not found: local-executable-sft-smoke-20260704`), so
the manifest remains the authoritative metric artifact.

Validation:

```bash
PATH="$PWD/.venv/bin:$PATH" python -m py_compile scripts/make_agentcoder_executable_sft.py
PATH="$PWD/.venv/bin:$PATH" python -m pytest -q tests/test_executable_sft.py
PATH="$PWD/.venv/bin:$PATH" python -m pytest -q tests/test_executable_sft.py tests/test_coding_ladder.py tests/test_prepare_agentcoder_research_data.py tests/test_function_repair.py tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_config.py
PATH="$PWD/.venv/bin:$PATH" python -m pytest -q
git diff --check
```

Results:

- `tests/test_executable_sft.py`: `3 passed`.
- Focused adjacent suite: `59 passed, 1 skipped, 1 warning`.
- Full suite: `148 passed, 1 skipped, 1 warning`.
- `git diff --check`: passed.

Interpretation:

- The executable coding data/eval pipeline is now concretely staged and locally
  verified.
- The measured RAAM-vs-Transformer answer on the held-out key/value binding gate
  remains negative for RAAM, so scaling is still not cleared.
- The next decision gate should run the new executable SFT builder with a small
  real-source sample, pack with assistant-only loss, and compare RAAM against the
  matched Transformer on function, patch, pytest, JSON, and request-value held-out
  evals before any larger continuation run.

## 2026-07-04 - HF Viewer Executable Tiny Gate

Implemented follow-up fixes:

- `scripts/make_agentcoder_executable_sft.py` now falls back to the read-only
  Hugging Face Dataset Viewer API when the optional `datasets` package is not
  installed.
- The direct HF defaults now match the observed Dataset Viewer configs:
  `nvidia/OpenCodeInstruct` config `train`, `Samip/Scotch` config `python`,
  `KAKA22/CodeRM-UnitTest` config `default`, and `bigcode/commitpackft` config
  `python`.
- The Python-language filter now accepts rows whose metadata is generic but
  whose answer/code fields contain fenced Python or a Python function definition.
  This fixed the initial OpenCodeInstruct skip.
- The CodeRM converter now reads the actual Hub field `code_ground_truth`.
- Added `scripts/run_agentcoder_executable_gate.py`, a bounded train/eval runner
  for executable coding data. It builds or reuses executable SFT data, trains a
  tokenizer, packs assistant-only loss masks, runs `scripts/train.py`, logs local
  `.mlops` metrics, evaluates with `scripts/eval_coding_ladder.py`, and writes a
  `summary.json`.
- Added matched CPU-sized configs:
  `configs/scratch/raam_agentcoder_executable_tiny_gate.yaml` and
  `configs/scratch/transformer_agentcoder_executable_tiny_gate.yaml`.

HF Viewer sample command:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/make_agentcoder_executable_sft.py \
  --output-dir runs/agentcoder_executable_hf_viewer_sample_v2_20260704T000000Z \
  --use-hf \
  --opencode-limit 20 \
  --scotch-limit 30 \
  --coderm-unittest-limit 5 \
  --commitpackft-limit 20 \
  --ladder-repeats 2 \
  --curated-anchor-repeats 0 \
  --eval-source-fraction 0.05 \
  --max-answer-chars 2400 \
  --max-tests-chars 5000 \
  --max-function-lines 80 \
  --max-diff-lines 80 \
  --max-file-chars 4000
```

HF Viewer sample artifact:

```text
runs/agentcoder_executable_hf_viewer_sample_v2_20260704T000000Z/agentcoder_executable_manifest.json
```

HF Viewer sample result:

| Metric | Value |
| --- | ---: |
| Train records | 148 |
| Eval cases | 10 |
| Exact train/eval user-prompt overlaps | 0 |
| Local ladder records | 82 |
| OpenCodeInstruct records | 17 |
| Scotch records | 29 |
| commitpackft records | 20 |
| CodeRM-UnitTest records | 0 |

CodeRM contributed no records in this bounded sample because the first rows were
large and exceeded the current answer/test length filters. That should be fixed
with a small-test extractor or selective higher limits before relying on CodeRM
for the next larger repair run.

Matched tiny gate commands:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/raam_agentcoder_executable_tiny_gate.yaml \
  --output-dir runs/agentcoder_executable_hf_viewer_tiny_compare_20260704T000000Z/raam \
  --data-dir runs/agentcoder_executable_hf_viewer_sample_v2_20260704T000000Z \
  --steps 80 \
  --eval-batches 1 \
  --eval-every 20 \
  --device cpu \
  --seq-len 384 \
  --vocab-size 2048 \
  --no-fail \
  --mlops-run-id executable-hf-viewer-tiny-raam-20260704

PATH="$PWD/.venv/bin:$PATH" python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/transformer_agentcoder_executable_tiny_gate.yaml \
  --output-dir runs/agentcoder_executable_hf_viewer_tiny_compare_20260704T000000Z/transformer \
  --data-dir runs/agentcoder_executable_hf_viewer_sample_v2_20260704T000000Z \
  --steps 80 \
  --eval-batches 1 \
  --eval-every 20 \
  --device cpu \
  --seq-len 384 \
  --vocab-size 2048 \
  --no-fail \
  --mlops-run-id executable-hf-viewer-tiny-transformer-20260704
```

Matched tiny gate artifacts:

```text
runs/agentcoder_executable_hf_viewer_tiny_compare_20260704T000000Z/raam/summary.json
runs/agentcoder_executable_hf_viewer_tiny_compare_20260704T000000Z/transformer/summary.json
.mlops/experiments/executable-hf-viewer-tiny-raam-20260704/metrics.json
.mlops/experiments/executable-hf-viewer-tiny-transformer-20260704/metrics.json
```

Matched tiny gate result:

| Model | Best val loss | Best step | Eval pass | Function | Patch | JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM tiny | 3.752042055130005 | 60 | 0/10 | 0 | 0 | 0 |
| Transformer tiny | 3.8542072772979736 | 60 | 0/10 | 0 | 0 | 0 |

Interpretation:

- The executable real-source data/eval loop now runs end to end on this CPU box
  and records local MLOps metrics.
- The tiny gate is still a negative capability result: both models fail all 10
  executable held-out ladder cases, so it does not show useful coding ability.
- RAAM has slightly lower validation loss than the tiny Transformer in this
  smoke, but exact executable correctness is tied at zero. Validation loss alone
  is not a promotion criterion.
- The separate request/value binding evidence remains the stronger blocker:
  RAAM still trails the matched Transformer on held-out exact binding, so scaling
  remains blocked.

Next decision gate:

- Fix the binding/copy gap first: run the cheapest matched request-value
  ablations that can close RAAM held-out exact binding against Transformer.
- In parallel, improve the executable data builder by extracting small CodeRM
  tests and adding source-derived held-out function evals that are not exact
  prompt overlaps.
- Then rerun the executable gate with a larger but still bounded sample and
  require non-zero function/patch/JSON pass rates before any 100M continuation.

## 2026-07-04 - Vast request/value binding ablation

Objective:

- Move the binding/copy blocker back onto GPU instead of CPU.
- Compare the matched Transformer control against RAAM variants on the same
  request/value coverage ladder.
- Identify the cheapest RAAM variant that matches Transformer on held-out exact
  binding before any scaling claim.

Implementation:

- Added `--mlops-project-path` and `--mlops-run-id` pass-through to
  `scripts/run_agentcoder_keyvalue_copy_gate.py`.
- Added `configs/scratch/raam_agentcoder_keyvalue_request_value_no_compression_gate.yaml`.
- Added `configs/scratch/raam_agentcoder_keyvalue_request_value_all_anchor_gate.yaml`.

Remote setup:

- Vast instance: `43829419`.
- Hardware: `NVIDIA GeForce RTX 5090`, driver `580.95.05`, torch
  `2.12.0+cu130`, CUDA available.
- Template: Vast recommended PyTorch image `vastai/pytorch:cuda-13.0.3-auto`.
- Synced the current dirty local worktree to `/root/raam-lm`.
- Remote focused validation passed:

```text
PYTHONPATH=src:. python -m pytest -q tests/test_agentcoder_keyvalue_copy_generator.py tests/test_copy_head.py tests/test_config.py
# 42 passed, 1 warning
```

Remote command shape:

```bash
ROOT=runs/agentcoder_request_value_vast_ablation_20260704T191900Z
COMMON=(--steps 1600 --seq-len 384 --vocab-size 2048 \
  --eval-mode coverage_ladder --completion-mode value_only --target-fields 2 \
  --train-records 384 --train-variants-per-row 1 --eval-cases 48 \
  --max-new-tokens 48 --eval-batches 1 --device cuda --clean --no-fail \
  --mlops-project-path /root/raam-lm)

python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml \
  --output-dir "$ROOT/transformer_steps1600" \
  "${COMMON[@]}" \
  --mlops-run-id request-value-vast-transformer-steps1600-20260704

python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_no_compression_gate.yaml \
  --output-dir "$ROOT/raam_no_compression_steps1600" \
  "${COMMON[@]}" \
  --mlops-run-id request-value-vast-raam-no-compression-steps1600-20260704

python scripts/run_agentcoder_keyvalue_copy_gate.py \
  --config configs/scratch/raam_agentcoder_keyvalue_request_value_all_anchor_gate.yaml \
  --output-dir "$ROOT/raam_all_anchor_steps1600" \
  "${COMMON[@]}" \
  --mlops-run-id request-value-vast-raam-all-anchor-steps1600-20260704
```

Local artifact pull:

```text
runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current
```

Matched result:

| Model | Config | Pass rate | Value sequence | Best val loss | Final val loss | FLOPs/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | `transformer_agentcoder_keyvalue_request_value_gate.yaml` | 144/144 | 144/144 | 1.8367745876312256 | 1.898492693901062 | 8153856 |
| RAAM no compression | `raam_agentcoder_keyvalue_request_value_no_compression_gate.yaml` | 144/144 | 144/144 | 1.5094492435455322 | 1.5094492435455322 | 7372544 |
| RAAM all-anchor | `raam_agentcoder_keyvalue_request_value_all_anchor_gate.yaml` | 144/144 | 144/144 | 1.503405213356018 | 1.503405213356018 | 8008064 |

Authoritative summaries:

```text
runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current/transformer_steps1600/summary.json
runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current/raam_no_compression_steps1600/summary.json
runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current/raam_all_anchor_steps1600/summary.json
runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current/ablation_summary.json
.mlops/experiments/request-value-vast-transformer-steps1600-20260704/metrics.json
.mlops/experiments/request-value-vast-raam-no-compression-steps1600-20260704/metrics.json
.mlops/experiments/request-value-vast-raam-all-anchor-steps1600-20260704/metrics.json
```

Interpretation:

- This is the first measured answer in this cycle where a RAAM variant matches
  the Transformer control on the 144-case request/value coverage ladder.
- The cheapest passing variant is RAAM no-compression: it matches exact binding
  with lower estimated FLOPs/token than the Transformer and all-anchor RAAM.
- The result points at dynamic hourglass compression as the current binding
  failure source. It does not prove useful agentic coding ability: executable
  function, JSON/tool-call, and patch gates remain the next blocker.
- The original aggregate `ablation_summary.json` written on remote used stale
  key names; the pulled local aggregate was corrected from the per-run
  `summary.json` and `keyvalue_eval.json` files.

Cleanup note:

- Remote workload finished and GPU was idle after artifact pull.
- Vast CLI cleanup was blocked by `Session expired. Please log in again.`
  Instance `43829419` still needs to be destroyed from the Vast console or after
  refreshing CLI auth.

Next decision gate:

- Promote `raam_agentcoder_keyvalue_request_value_no_compression_gate.yaml` as
  the cheapest exact-binding diagnostic variant.
- Rerun a function-repair/executable coding gate from the no-compression
  binding variant before any 100M continuation or frontier-style claim.

## 2026-07-04 - Vast executable no-compression GPU smoke

Objective:

- Check whether the no-compression RAAM variant that fixed the request/value
  binding gate transfers to a real executable coding gate.
- Keep the comparison bounded and matched against the tiny Transformer control
  on the same tokenizer, data, sequence length, eval cases, and step budget.

Implementation:

- Added `configs/scratch/raam_agentcoder_executable_no_compression_tiny_gate.yaml`
  as a tiny RAAM executable-gate config with dynamic compression disabled.
- Updated `scripts/run_agentcoder_executable_gate.py` and
  `scripts/run_agentcoder_keyvalue_copy_gate.py` summaries to include train-log
  fields such as `best_val_loss`, `final_val_loss`, `final_step`, and final
  throughput.

Remote command shape:

```bash
ROOT=runs/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z
DATA=runs/agentcoder_executable_hf_viewer_sample_v2_20260704T000000Z
COMMON=(--data-dir "$DATA" --steps 200 --eval-batches 1 --eval-every 50 \
  --device cuda --seq-len 384 --vocab-size 2048 --max-new-tokens 180 \
  --no-fail --clean --mlops-project-path /root/raam-lm)

python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/raam_agentcoder_executable_no_compression_tiny_gate.yaml \
  --output-dir "$ROOT/raam_no_compression_steps200" \
  "${COMMON[@]}" \
  --mlops-run-id executable-gpu-raam-no-compression-steps200-20260704

python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/transformer_agentcoder_executable_tiny_gate.yaml \
  --output-dir "$ROOT/transformer_steps200" \
  "${COMMON[@]}" \
  --mlops-run-id executable-gpu-transformer-steps200-20260704
```

Local artifact pull:

```text
runs/vast_backups/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z/current
```

Matched result:

| Model | Config | Eval pass | Function | Patch | JSON | Best val loss | Final val loss | FLOPs/token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RAAM no compression | `raam_agentcoder_executable_no_compression_tiny_gate.yaml` | 0/10 | 0 | 0 | 0 | 2.5497891902923584 | 2.9630250930786133 | 476416 |
| Transformer | `transformer_agentcoder_executable_tiny_gate.yaml` | 0/10 | 0 | 0 | 0 | 2.7539620399475098 | 3.2945966720581055 | 671744 |

Authoritative summaries:

```text
runs/vast_backups/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z/current/executable_compare_summary.json
runs/vast_backups/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z/current/raam_no_compression_steps200/summary.json
runs/vast_backups/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z/current/transformer_steps200/summary.json
.mlops/experiments/executable-gpu-raam-no-compression-steps200-20260704/metrics.json
.mlops/experiments/executable-gpu-transformer-steps200-20260704/metrics.json
```

Interpretation:

- No-compression RAAM keeps the lower validation loss and lower estimated
  FLOPs/token pattern from the binding diagnostic, but executable correctness is
  still tied at zero.
- All 10 ladder eval cases failed for both models: function completion, patch,
  pytest, and JSON/tool-call correctness remain unsolved.
- This is not evidence of useful coding ability. It is evidence that the
  executable data/eval pipeline runs on GPU and that the next blocker is
  executable supervision/eval quality rather than request/value binding alone.

Next decision gate:

- Keep no-compression RAAM as the cheapest diagnostic architecture for the next
  small coding runs.
- Improve the executable corpus and add a smaller function-only held-out gate
  that can produce nonzero pass rates before spending on broader 100M training.

## 2026-07-04 - Function-only executable gate

Objective:

- Narrow the executable blocker from mixed function/patch/pytest/JSON behavior
  down to held-out Python function synthesis.
- Check whether low validation loss on tiny executable runs corresponds to any
  exact executable correctness.
- Compare no-compression RAAM against the matched tiny Transformer under the
  same generated data, tokenizer size, sequence length, eval cases, and step
  budget.

Implementation:

- Added eval-case filters to `scripts/eval_coding_ladder.py`:
  `--expected-behavior` and `--topic-contains`.
- Added train/eval behavior filters to
  `scripts/make_agentcoder_executable_sft.py`:
  `--train-behavior`, `--train-topic-contains`,
  `--eval-expected-behavior`, and `--eval-topic-contains`.
- Wired the same filters through `scripts/run_agentcoder_executable_gate.py`.
- Added tests proving function-only train/eval artifacts are written with
  manifest filters and nonempty filtered cases.

Local staged data artifact:

```bash
python scripts/make_agentcoder_executable_sft.py \
  --output-dir runs/agentcoder_executable_function_train_eval_gate_20260704T213000Z \
  --ladder-repeats 4 \
  --curated-anchor-repeats 0 \
  --train-behavior function_completion \
  --eval-expected-behavior function_completion
```

Artifact:

```text
runs/agentcoder_executable_function_train_eval_gate_20260704T213000Z/agentcoder_executable_manifest.json
```

Manifest summary:

- `train_records=96`
- `eval_cases=6`
- `behavior_counts={"function_completion": 96}`
- `exact_train_eval_user_prompt_overlaps=[]`

Remote mixed-train/function-eval diagnostic:

```text
runs/vast_backups/agentcoder_executable_function_only_gpu_compare_20260704T211500Z/current/function_only_compare_summary.json
```

Result: RAAM no-compression `0/6`, Transformer `0/6`. This showed that simply
filtering eval to function cases was not enough; both models failed when trained
on the mixed executable ladder.

Remote function-train/function-eval command shape:

```bash
ROOT=runs/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z
COMMON=(--steps 600 --eval-batches 1 --eval-every 150 --device cuda \
  --seq-len 384 --vocab-size 2048 --max-new-tokens 180 \
  --ladder-repeats 4 --curated-anchor-repeats 0 \
  --train-behavior function_completion \
  --eval-expected-behavior function_completion \
  --no-fail --clean --mlops-project-path /root/raam-lm)

python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/raam_agentcoder_executable_no_compression_tiny_gate.yaml \
  --output-dir "$ROOT/raam_no_compression_steps600" \
  "${COMMON[@]}" \
  --mlops-run-id executable-function-train-eval-raam-no-compression-steps600-20260704

python scripts/run_agentcoder_executable_gate.py \
  --config configs/scratch/transformer_agentcoder_executable_tiny_gate.yaml \
  --output-dir "$ROOT/transformer_steps600" \
  "${COMMON[@]}" \
  --mlops-run-id executable-function-train-eval-transformer-steps600-20260704
```

Local artifact pull:

```text
runs/vast_backups/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z/current
```

Matched result:

| Model | Config | Function pass | Best val loss | Final val loss | FLOPs/token |
| --- | --- | ---: | ---: | ---: | ---: |
| RAAM no compression | `raam_agentcoder_executable_no_compression_tiny_gate.yaml` | 0/6 | 0.04705824702978134 | 0.04705824702978134 | 269952 |
| Transformer | `transformer_agentcoder_executable_tiny_gate.yaml` | 1/6 | 0.018568092957139015 | 0.018568092957139015 | 465280 |

Authoritative summaries:

```text
runs/vast_backups/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z/current/function_train_eval_compare_summary.json
runs/vast_backups/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z/current/raam_no_compression_steps600/summary.json
runs/vast_backups/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z/current/transformer_steps600/summary.json
.mlops/experiments/executable-function-train-eval-raam-no-compression-steps600-20260704/metrics.json
.mlops/experiments/executable-function-train-eval-transformer-steps600-20260704/metrics.json
```

Qualitative failure:

- RAAM no-compression often generated a syntactically valid but wrong function,
  for example `def is_odd(n): return n % 2 == 0` for `is_even`,
  `filter_even`, and other prompts.
- The Transformer passed only `ladder_filter_even`; all other function cases
  still failed.

Interpretation:

- This is the first nonzero executable held-out result in this cycle, but it is
  a Transformer-only result. RAAM no-compression still trails on executable
  function correctness even after it matched Transformer on request/value
  binding.
- Low validation loss is not a reliable promotion signal here: RAAM reached
  `0.047` final validation loss and still passed `0/6`.
- Scaling RAAM remains blocked until it matches or beats the Transformer on this
  held-out function gate.

Next decision gate:

- Add a prompt-binding/function-name ablation for function completion, because
  RAAM's main failure is selecting the wrong function identity while emitting
  syntactically valid code.
- Consider function-name copy hints or a function-signature copy route before
  broader patch/JSON training.
