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
