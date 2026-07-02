# Scratch Training Plan

RAAM-AgentCoder is intentionally trained from scratch. Do not fine-tune an existing pretrained model for this track.

## Curriculum

Stage A: code/text pretraining  
Stage B: chat formatting and instruction following  
Stage C: code explanation and code generation  
Stage D: bug fixing from stack traces  
Stage E: repo-edit patch generation  
Stage F: agentic tool-use transcripts  
Stage G: test-driven repair and self-correction

The pipeline supports these stages by accepting local folders or JSONL files per stage. Each stage should produce packed train/validation token streams and a manifest.

The current researched dataset plan is in `docs/TRAINING_DATA_AND_VAST_RESEARCH.md`.
The first Vast corpus is a staged mix of selected StarCoder2 extra subsets, OASST1,
WildChat, NVIDIA Open-SWE-Traces, NVIDIA SWE-Zero OpenHands trajectories, and local
permissive code/patch/test data. SWE-bench is held out for evaluation, not training.

## First Serious Target

The first serious target is `configs/scratch/raam_agentcoder_100m.yaml`. The current debug evidence favors `raam_no_anchors`, so the 50M and 100M RAAM configs disable anchors and attention islands by default. Full RAAM remains available for later long-context and probe experiments.

## Debug Gate

Before any paid training, run:

```bash
python -m pytest -q
python scripts/train_tokenizer.py examples/tiny_agentic.jsonl --output runs/agentcoder_e2e/tokenizer.json --vocab-size 512
python scripts/pack_dataset.py examples/tiny_agentic.jsonl --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/packed --seq-len 64
python scripts/train.py --config configs/scratch/raam_agentcoder_debug.yaml --train-bin runs/agentcoder_e2e/packed/train.bin --val-bin runs/agentcoder_e2e/packed/val.bin --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/train --steps 20 --device auto
python scripts/train.py --config configs/scratch/raam_agentcoder_debug.yaml --train-bin runs/agentcoder_e2e/packed/train.bin --val-bin runs/agentcoder_e2e/packed/val.bin --tokenizer runs/agentcoder_e2e/tokenizer.json --output-dir runs/agentcoder_e2e/train --steps 25 --resume runs/agentcoder_e2e/train/checkpoints/last.pt --device auto
python scripts/generate.py --config configs/scratch/raam_agentcoder_debug.yaml --tokenizer runs/agentcoder_e2e/tokenizer.json --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt --prompt "<|user|>\nFix add().\n<|assistant|>\n" --device auto
python scripts/eval_agentic_coding.py --config configs/scratch/raam_agentcoder_debug.yaml --tokenizer runs/agentcoder_e2e/tokenizer.json --checkpoint runs/agentcoder_e2e/train/checkpoints/last.pt --device auto
```

Smoke output is not evidence that the model is good. It proves the pipeline can train, resume, generate, and evaluate.
