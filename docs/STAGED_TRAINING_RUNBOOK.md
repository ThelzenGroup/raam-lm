# RAAM-AgentCoder Staged Training Runbook

Last updated: 2026-07-02

This is the execution plan for training RAAM-AgentCoder from scratch for chat and
agentic software engineering while keeping training and inference cheap.

## Stage 0: Hardware and Persistence Gate

Required before any paid run longer than a short rehearsal:

- `bash scripts/vast_preflight.sh configs/scratch/raam_agentcoder_debug.yaml runs/vast_preflight`
- `scripts/vast_pull_artifacts.sh` proven against the target instance
- checkpoint resume proven from `checkpoints/last.pt`
- `/workspace` persistence checked with `vast-capabilities | jq '.instance.workspace_is_volume'`

If `/workspace` is not a volume, keep `scripts/vast_pull_artifacts.sh` running from the
local machine during long training.

## Stage 1: First Real Dataset

Use a staged corpus, not a single monolithic dump:

- code/docs base: selected StarCoder2 extras and local permissive repos
- chat/instruction: OpenAssistant and filtered WildChat
- agentic SWE traces: Open-SWE-Traces and SWE-Zero OpenHands trajectories
- patch/test traces: local diffs, stack traces, and test repair examples

The small rehearsal command is:

```bash
cd /workspace/raam-lm
STEPS=20 RESUME_STEPS=25 bash scripts/vast_train_50m.sh
```

The larger first corpus command is:

```bash
MAX_OPEN_SWE=20000 \
MAX_SWE_ZERO=20000 \
MAX_WILDCHAT=20000 \
MAX_OASST=10000 \
STARCODER2_EXTRAS='documentation=20000 issues=20000 stackoverflow=20000 kaggle=10000' \
STEPS=1000 RESUME_STEPS=1100 \
bash scripts/vast_train_50m.sh
```

## Stage 2: 50M RAAM Rehearsal

Purpose:

- estimate RTX 5090 memory headroom
- confirm tokens/sec and step time
- confirm loss decreases on a real mixed corpus
- confirm checkpoint/resume
- confirm generation and agentic eval scripts complete

Pass criteria:

- no NaNs or infs
- validation loss is available and generally lower than the initial value
- `generation_smoke.txt` is non-empty
- `agentic_eval.json` is produced
- artifacts are pulled to local storage before instance stop or destroy

## Stage 3: Matched Baselines

Run the same packed dataset, tokenizer, optimizer, sequence length, and step budget:

- `configs/scratch/transformer_agentcoder_50m.yaml`
- `configs/scratch/pure_mamba_like_agentcoder_50m.yaml`
- `configs/scratch/raam_agentcoder_50m.yaml`
- `configs/scratch/transformer_agentcoder_100m.yaml`
- `configs/scratch/pure_mamba_like_agentcoder_100m.yaml`
- `configs/scratch/raam_agentcoder_100m.yaml`

Compare:

- validation loss vs tokens
- validation loss vs estimated FLOPs
- tokens/sec and wall-clock to target loss
- generation smoke behavior
- agentic eval output validity

First run the 50M gate:

```bash
cd /workspace/raam-lm
STEPS=20 RESUME_STEPS=25 bash scripts/vast_stage3_baselines.sh
```

For a cheaper OOM/resume smoke, shrink runtime-only training settings without
editing the matched configs:

```bash
cd /workspace/raam-lm
BATCH_SIZE=1 TRAIN_SEQ_LEN=512 GRAD_ACCUMULATION_STEPS=1 EVAL_BATCHES=1 \
STEPS=1 RESUME_STEPS=2 SAVE_EVERY=1 EVAL_EVERY=1 \
bash scripts/vast_stage3_baselines.sh
```

This writes:

```text
runs/stage3_baselines/summary.json
runs/stage3_baselines/summary.md
```

Use the 100M configs only after the 50M matched gate runs without OOMs or broken
resume/eval behavior.

## Stage 4: 100M Scratch Candidate

Only start after Stage 2 and Stage 3 pass. Use the 100M configs with the same corpus,
tokenizer, and schedule. Keep MTP and RAAM mechanisms isolated with ablations before
claiming any architecture benefit.

First prove fit/resume with the 100M matched configs:

```bash
cd /root/raam-lm
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_fit_gate \
CONFIGS='configs/scratch/transformer_agentcoder_100m.yaml configs/scratch/pure_mamba_like_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m.yaml' \
STEPS=5 RESUME_STEPS=6 SAVE_EVERY=5 EVAL_EVERY=5 EXPORT_CHECKPOINT=0 \
bash scripts/vast_stage3_baselines.sh
```

Then isolate the actual RAAM mechanisms before longer training:

```bash
cd /root/raam-lm
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_raam_mechanisms \
CONFIGS='configs/scratch/raam_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m_no_attention_islands.yaml configs/scratch/raam_agentcoder_100m_no_anchors.yaml configs/scratch/raam_agentcoder_100m_full.yaml' \
STEPS=5 RESUME_STEPS=6 SAVE_EVERY=5 EVAL_EVERY=5 EXPORT_CHECKPOINT=0 \
bash scripts/vast_stage3_baselines.sh
```

The existing `raam_agentcoder_100m.yaml` is the cheapest compression-only RAAM
variant. `raam_agentcoder_100m_full.yaml` enables learned anchors and two exact
attention islands on the compressed/global stream.

Before any longer Stage 4 run, keep `scripts/vast_pull_artifacts.sh` running from the
local machine. If direct SSH is unreliable, set `SSH_HOST` and `SSH_PORT` to the Vast
relay endpoint shown by `vastai show instances`.

For the next longer quality/efficiency gate, compare only the current quality leader
against the cheapest useful RAAM candidate and delete optimizer checkpoints after eval:

```bash
cd /root/raam-lm
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder \
PACKED_DIR=/root/data/agentcoder/packed_2048 \
TOKENIZER=/root/data/agentcoder/tokenizer.json \
RUN_ROOT=/root/raam-lm/runs/stage4_100m_two_way_1000step \
CONFIGS='configs/scratch/pure_mamba_like_agentcoder_100m.yaml configs/scratch/raam_agentcoder_100m.yaml' \
STEPS=1000 RESUME_STEPS=1100 SAVE_EVERY=0 EVAL_EVERY=50 \
EXPORT_CHECKPOINT=0 KEEP_TRAINING_CHECKPOINTS=0 \
bash scripts/vast_stage3_baselines.sh
```

This keeps train logs, manifests, generation smoke output, agentic eval JSON, and
summary files, but removes `last.pt` and `step_*.pt` files after each config has
completed generation/eval.

## Stage 5: Chat and Agentic Tuning

After a base LM checkpoint is stable:

- increase chat/instruction share
- increase resolved agentic SWE traces
- add local patch/test repair examples
- evaluate with held-out tiny agentic coding tasks

Do not train on SWE-bench gold patches intended for held-out evaluation.

The current Stage 5 entrypoint uses the stable 100M compression-only RAAM
candidate, a separate expanded data root, and one resumable `last.pt` plus a
compact model-only export. This config disables early reconstruction loss and
curriculum MTP because the first expanded Stage 5 candidate run learned early,
then destabilized after auxiliary losses were active:

```bash
cd /root/raam-lm
BASE_DIR=/root/raam-lm \
DATA_ROOT=/root/data/agentcoder_stage5 \
RUN_DIR=/root/raam-lm/runs/stage5_raam_agentcoder_100m_stage5_stable \
STEPS=5000 RESUME_STEPS=5500 \
bash scripts/vast_train_100m_candidate.sh
```

Use `CONFIG=configs/scratch/raam_agentcoder_100m.yaml` only when intentionally
rerunning the older auxiliary-loss schedule for comparison.

For a pure smoke of the Stage 5 wrapper, lower the sample and step counts with env
overrides rather than editing the script.

## Stage 6: Decision Gate

Continue RAAM only if it has evidence under matched comparisons:

- same tokenizer
- same packed data order
- same optimizer and schedule
- same token budget
- parameter/FLOP/token/wall-clock views

If gains vanish under matched comparisons, keep the codebase but redirect training
budget to the simpler baseline.
