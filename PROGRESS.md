# RAAM-LM Progress

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
