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
