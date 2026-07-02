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

Prepare the first research corpus:

```bash
python scripts/prepare_agentcoder_research_data.py \
  --output-dir /data/agentcoder/raw \
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
