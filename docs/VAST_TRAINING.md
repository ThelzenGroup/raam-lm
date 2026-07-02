# Vast.ai RTX 5090 Training

Use Vast only after the local debug gate passes. Do not start long paid training automatically.

For the current dataset recipe and full Vast runbook, see
`docs/TRAINING_DATA_AND_VAST_RESEARCH.md`.

## Environment Target

- RTX 5090 class GPU
- CUDA 12.8+ compatible runtime
- PyTorch version compatible with RTX 50-series GPUs
- checkpoint directory on persistent storage

Use a recommended Vast PyTorch template with the `[Automatic]` version tag for RTX 5-series GPUs.
Do not change the Docker image for the first run; the RTX 5-series path depends on CUDA 12.8
and PyTorch 2.7+ compatibility.

Set disk at instance creation time. Use at least 250GB for a small rehearsal, 500GB for a
useful first dataset, and 1TB+ if you keep raw data plus multiple checkpoints on the instance.

## Public GitHub Code Transfer

Public GitHub is the simplest Vast workflow. Keep datasets, packed token files, checkpoints,
logs, local caches, and tokens out of Git; `.gitignore` is configured for that.

Publish from this machine:

```bash
git init
git branch -M main
git add .
git status --short
git commit -m "Initial RAAM-AgentCoder prototype"
gh repo create raam-lm --public --source=. --remote=origin --push
```

Clone on Vast:

```bash
cd /workspace
git clone https://github.com/<YOUR_GITHUB_USER>/raam-lm.git
cd /workspace/raam-lm
. /venv/main/bin/activate || true
python -m pip install -U pip
python -m pip install -e .
python -m pip install datasets tqdm huggingface_hub
```

If you create the GitHub repo manually, set `origin` yourself and push `main`:

```bash
git remote add origin git@github.com:<YOUR_GITHUB_USER>/raam-lm.git
git push -u origin main
```

## Preflight

Run:

```bash
bash scripts/vast_preflight.sh configs/scratch/raam_agentcoder_debug.yaml runs/vast_preflight
```

The script checks:

- `nvidia-smi`
- PyTorch CUDA availability
- RTX 5090 GPU name and capability
- CUDA 12.8+ runtime
- PyTorch 2.7+ compatibility
- VRAM
- repo tests
- tiny profile step
- checkpoint/resume with a tiny local dataset
- checkpoint/output write access

For a non-5090 rehearsal host, set `ALLOW_NON_5090=1`. Do not use that override for the actual RTX 5090 readiness gate.

## First Vast Run

Because `/workspace` may not be persistent, always set up artifact pull before a paid
run. From the local machine, this one-shot command copies the current run off the instance:

```bash
INSTANCE_ID=43627905 \
REMOTE_RUN_DIR=/workspace/raam-lm/runs/raam_agentcoder_50m_rehearsal \
LOCAL_DIR=runs/vast_backups/raam_agentcoder_50m_rehearsal \
bash scripts/vast_pull_artifacts.sh
```

This skips large checkpoint `.pt` files by default so logs, manifests, evals, and
tokenizers are copied reliably. To copy compact model-only exports too, set
`INCLUDE_MODEL_EXPORT=1`. To copy every optimizer-resumable checkpoint, set
`INCLUDE_CHECKPOINTS=1`.

The pull helper uses tar transport by default to avoid accidental large checkpoint
partials during watch pulls. Use `PULL_TRANSPORT=rsync` only when you intentionally
want rsync behavior and have verified the exclude rules against the target run.

For runs where checkpoints are needed only for resume, generation, and eval, set
`KEEP_TRAINING_CHECKPOINTS=0` on `scripts/vast_train_50m.sh` or
`scripts/vast_stage3_baselines.sh`. The runner will delete `last.pt` and `step_*.pt`
after eval/export for each completed config while keeping logs, manifests, summaries,
and optional `model_only_*.pt` exports.

To watch and pull periodically during a longer run:

```bash
WATCH_INTERVAL=300 \
INSTANCE_ID=43627905 \
REMOTE_RUN_DIR=/workspace/raam-lm/runs/raam_agentcoder_50m_rehearsal \
LOCAL_DIR=runs/vast_backups/raam_agentcoder_50m_rehearsal \
bash scripts/vast_pull_artifacts.sh
```

If direct SSH closes but the Vast relay works, override the SSH endpoint:

```bash
WATCH_INTERVAL=300 \
INSTANCE_ID=43634442 \
SSH_HOST=ssh1.vast.ai \
SSH_PORT=34442 \
REMOTE_RUN_DIR=/root/raam-lm/runs/stage4_100m_fit_gate \
LOCAL_DIR=runs/vast_backups/stage4_100m_fit_gate \
bash scripts/vast_pull_artifacts.sh
```

If the instance has a persistent mount, also pass `--sync-dir` to `scripts/train.py`
or set `SYNC_DIR=/path/to/persistent/mount` when using `scripts/vast_train_50m.sh`.

Prepare the first research corpus manually:

```bash
python scripts/prepare_agentcoder_research_data.py \
  --output-dir /data/agentcoder/raw \
  --open-swe-config openhands \
  --open-swe-split minimax_m25 \
  --max-open-swe 20000 \
  --max-swe-zero 20000 \
  --max-wildchat 20000 \
  --max-oasst 10000 \
  --starcoder2-extras documentation=20000 issues=20000 stackoverflow=20000 kaggle=10000
```

Then train the tokenizer and pack data:

```bash
python scripts/train_tokenizer.py /data/agentcoder/raw \
  --output /data/agentcoder/tokenizer.json \
  --vocab-size 32768

python scripts/pack_dataset.py /data/agentcoder/raw \
  --tokenizer /data/agentcoder/tokenizer.json \
  --output-dir /data/agentcoder/packed \
  --seq-len 2048 \
  --val-fraction 0.02
```

Start with a short rehearsal:

```bash
python scripts/train.py \
  --config configs/scratch/raam_agentcoder_50m.yaml \
  --train-bin /data/agentcoder/packed/train.bin \
  --val-bin /data/agentcoder/packed/val.bin \
  --tokenizer /data/agentcoder/tokenizer.json \
  --output-dir /workspace/runs/raam_agentcoder_50m_rehearsal \
  --steps 1000 \
  --device cuda
```

Then test resume:

```bash
python scripts/train.py \
  --config configs/scratch/raam_agentcoder_50m.yaml \
  --train-bin /data/agentcoder/packed/train.bin \
  --val-bin /data/agentcoder/packed/val.bin \
  --tokenizer /data/agentcoder/tokenizer.json \
  --output-dir /workspace/runs/raam_agentcoder_50m_rehearsal \
  --resume /workspace/runs/raam_agentcoder_50m_rehearsal/checkpoints/last.pt \
  --steps 1100 \
  --device cuda
```

Only move to `raam_agentcoder_100m` after validation loss, checkpoint resume, generation, and eval scripts are stable.

## One-Command 50M Rehearsal

On the Vast instance:

```bash
cd /workspace/raam-lm
STEPS=20 RESUME_STEPS=25 bash scripts/vast_train_50m.sh
```

The script:

- installs repo and optional dataset dependencies
- streams a small real dataset sample by default
- trains the tokenizer
- packs the dataset at sequence length 2048
- runs a bounded 50M rehearsal
- resumes from `checkpoints/last.pt`
- runs generation and agentic coding smoke evals

Scale the dataset sample with:

```bash
MAX_OPEN_SWE=20000 \
MAX_SWE_ZERO=20000 \
MAX_WILDCHAT=20000 \
MAX_OASST=10000 \
STARCODER2_EXTRAS='documentation=20000 issues=20000 stackoverflow=20000 kaggle=10000' \
STEPS=1000 RESUME_STEPS=1100 \
bash scripts/vast_train_50m.sh
```

Do not treat the default tiny rehearsal as quality evidence. It only proves the
dataset path, GPU memory, checkpoint/resume, generation, eval plumbing, and artifact
sync workflow.

## Matched Baseline Gate

After `scripts/vast_train_50m.sh` has created a tokenizer and packed dataset, run:

```bash
cd /workspace/raam-lm
STEPS=20 RESUME_STEPS=25 bash scripts/vast_stage3_baselines.sh
```

For the first paid smoke, use the same script with runtime-only memory overrides:

```bash
cd /workspace/raam-lm
BATCH_SIZE=1 TRAIN_SEQ_LEN=512 GRAD_ACCUMULATION_STEPS=1 EVAL_BATCHES=1 \
STEPS=1 RESUME_STEPS=2 SAVE_EVERY=1 EVAL_EVERY=1 \
bash scripts/vast_stage3_baselines.sh
```

This compares `transformer_agentcoder_50m`, `pure_mamba_like_agentcoder_50m`, and
`raam_agentcoder_50m` on the same tokenizer, packed data, sequence length, and step
budget. It writes `runs/stage3_baselines/summary.json` and
`runs/stage3_baselines/summary.md`.

To run the same fit/resume gate with the 100M configs:

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

To isolate RAAM's 100M mechanisms:

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

For the longer two-way 100M quality/efficiency gate after the mechanism smoke:

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
