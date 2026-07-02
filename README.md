# RAAM-LM / RAAM-AgentCoder

RAAM-LM means **Reconstructive Anchor-Attention Mamba Language Model**. This repository is a research-grade PyTorch prototype of a decoder-only efficient language model that combines a gated attention-island Mamba-like backbone, reconstructive dynamic hourglass compression, anchor-preserved local-global processing, and optional curriculum multi-token prediction.

RAAM-AgentCoder is the from-scratch training track built on this prototype. It targets chat-first, agentic software-engineering behavior: repository understanding, code editing, patch generation, debugging from stack traces, test-driven repair, code review, tool-use transcripts, and concise final summaries.

It is not a benchmark claim, a Transformer replacement claim, or a real Mamba implementation when `mamba-ssm` is unavailable. If `mamba-ssm` is not installed, the code uses a clearly logged `fallback_gated_conv` mixer.

## Install

```bash
python -m pip install -e .
```

Optional packages such as `mamba-ssm`, `datasets`, `wandb`, `deepspeed`, and `codecarbon` are feature-gated and are not required for debug runs.

## Core Commands

Run tests:

```bash
python -m pytest -q
```

Run tiny smoke training:

```bash
python scripts/smoke_train.py --config configs/debug/raam_tiny.yaml --steps 30 --device auto
```

Run the debug ablation matrix:

```bash
python scripts/run_debug_ablation_matrix.py --steps 30 --device auto
```

Run profiling:

```bash
python scripts/profile_step.py --config configs/debug/raam_tiny.yaml --device auto
```

Run synthetic probes:

```bash
python scripts/eval_probes.py --config configs/debug/raam_tiny.yaml --device auto
```

Estimate activated FLOPs/token:

```bash
python scripts/estimate_flops.py --config configs/debug/raam_tiny.yaml
```

Smoke results are not architecture evidence. They only verify code, causality, and measurement plumbing.

## Scratch AgentCoder Pipeline

Train a tiny local tokenizer:

```bash
python scripts/train_tokenizer.py examples/tiny_agentic.jsonl --output runs/agentcoder_e2e/tokenizer.json --vocab-size 512
```

Pack local agent/code data:

```bash
python scripts/pack_dataset.py examples/tiny_agentic.jsonl --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/packed --seq-len 64
```

Train from packed data with validation and checkpoints:

```bash
python scripts/train.py \
  --config configs/scratch/raam_agentcoder_debug.yaml \
  --train-bin runs/agentcoder_e2e/packed/train.bin \
  --val-bin runs/agentcoder_e2e/packed/val.bin \
  --tokenizer runs/agentcoder_e2e/tokenizer.json \
  --output-dir runs/agentcoder_e2e/train \
  --steps 20 \
  --device auto
```

Resume:

```bash
python scripts/train.py \
  --config configs/scratch/raam_agentcoder_debug.yaml \
  --train-bin runs/agentcoder_e2e/packed/train.bin \
  --val-bin runs/agentcoder_e2e/packed/val.bin \
  --tokenizer runs/agentcoder_e2e/tokenizer.json \
  --output-dir runs/agentcoder_e2e/train \
  --resume runs/agentcoder_e2e/train/checkpoints/last.pt \
  --steps 25 \
  --device auto
```

Generate from a checkpoint:

```bash
python scripts/generate.py \
  --config configs/scratch/raam_agentcoder_debug.yaml \
  --tokenizer runs/agentcoder_e2e/tokenizer.json \
  --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt \
  --prompt "<|user|>\nFix add().\n<|assistant|>\n" \
  --device auto
```

Run agentic coding eval smoke tests:

```bash
python scripts/eval_agentic_coding.py \
  --config configs/scratch/raam_agentcoder_debug.yaml \
  --tokenizer runs/agentcoder_e2e/tokenizer.json \
  --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt \
  --device auto
```

Run Vast preflight:

```bash
bash scripts/vast_preflight.sh configs/scratch/raam_agentcoder_debug.yaml runs/vast_preflight
```

For Vast, the simplest path is publishing this source repo on GitHub and cloning it on the
instance. Keep datasets, checkpoints, packed token files, logs, and tokens out of Git; see
`docs/VAST_TRAINING.md`.

Prepare the researched Vast dataset recipe:

```bash
python scripts/prepare_agentcoder_research_data.py \
  --output-dir /data/agentcoder/raw \
  --max-open-swe 20000 \
  --max-swe-zero 20000 \
  --max-wildchat 20000 \
  --max-oasst 10000 \
  --starcoder2-extras documentation=20000 issues=20000 stackoverflow=20000 kaggle=10000
```

See `docs/TRAINING_DATA_AND_VAST_RESEARCH.md` for the exact dataset mix, Vast.ai RTX 5090 setup, and first paid-run gates.

## Models

- `transformer`: dense Transformer baseline with causal SDPA attention in every layer.
- `pure_mamba_like`: fallback recurrent/mixer baseline with no compression, anchors, or attention islands.
- `raam`: full RAAM prototype with mechanism toggles.

The primary toggles are:

- `use_mamba_or_fallback_backbone`
- `use_dynamic_hourglass_compression`
- `use_anchor_preserved_local_global`
- `use_attention_islands`
- `use_curriculum_mtp`

## Outputs

Training writes JSONL logs under each config's `train.output_dir`. Profiling emits a JSON manifest with parameter counts, approximate activated FLOPs/token, throughput, step timing, peak memory, device, dtype, config hash, and git SHA when available.

Scratch training writes `manifest.json`, `config.yaml`, `tokenizer.json`, `train_log.jsonl`, and checkpoints under `checkpoints/`. Validation loss is logged as `val_next_token_loss`.
