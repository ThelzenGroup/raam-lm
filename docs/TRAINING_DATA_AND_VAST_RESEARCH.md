# Training Data and Vast Setup Research

Last updated: 2026-07-02

## Short Answer

We are training RAAM-AgentCoder from scratch on a staged chat and software-engineering corpus, not on one monolithic dataset.

The first paid Vast.ai run should use `configs/scratch/raam_agentcoder_50m.yaml` as a rehearsal. The first serious target is `configs/scratch/raam_agentcoder_100m.yaml`. The current research candidate is the RAAM compressed model with anchors and attention islands disabled, because the tiny multi-seed comparison favored that variant. Full RAAM stays available for long-context and ablation runs.

## Dataset Choice

Use this order:

1. Base code and documentation pretraining
2. General chat and instruction formatting
3. Agentic software-engineering trajectories
4. Patch, test-output, and repair traces
5. SWE-bench style evaluation only

The recommended first mix is:

| Slice | Target Share | Sources | Purpose |
| --- | ---: | --- | --- |
| Code/docs base | 55% | local permissive repos, selected `bigcode/starcoder2data-extras` subsets | Teach syntax, APIs, docs, issue language, and code-adjacent text |
| Chat/instruction | 15% | `OpenAssistant/oasst1`, `allenai/WildChat` | Teach conversational behavior and turn-taking |
| Agentic SWE traces | 25% | `nvidia/Open-SWE-Traces`, `nvidia/SWE-Zero-openhands-trajectories` | Teach repo navigation, tool observations, patches, and issue-solving style |
| Patch/test traces | 5% | local patches, diffs, stack traces, pytest logs | Teach test-driven repair and final engineering summaries |

For the first rehearsal, target 50M to 100M packed tokens. For a more serious 100M run, target 200M to 1B packed tokens. This is deliberately small compared with frontier code models, but it is the right scale for proving whether our scratch architecture learns cheaply on one RTX 5090.

## Source Notes

`bigcode/the-stack-v2` is not the first dataset to download. It is a huge source-code corpus with access terms, Software Heritage content retrieval, original-license obligations, and removal-update obligations. The full corpus is about 67.5TB, deduplicated is about 32.1TB, and the training split is around 900B tokens. Treat it as a later base-code source after credentials, filtering, provenance, and storage are solved.

Use `bigcode/starcoder2data-extras` first for code-adjacent text. The useful subsets are `documentation`, `issues`, `stackoverflow`, `kaggle`, `ir_python`, `ir_rust`, and `ir_cpp`. Start small and check subset licensing/source terms before any public or commercial release.

Use `OpenAssistant/oasst1` for clean general assistant turns. Hugging Face lists it as Apache-2.0 with train and validation splits.

Use `allenai/WildChat` for broader chat style. Its dataset card lists ODC-BY, about 650K conversations in this version, non-toxic filtering, PII de-identification, and fields for moderation and redaction. Filter to English, non-toxic records, and non-empty user turns.

Use `nvidia/Open-SWE-Traces` as the primary agentic coding source. It is a 200k+ software-engineering trajectory dataset with system/user/assistant/tool roles, repo/license/language fields, tool definitions, and resolved status. Its card says the issue statements come from permissive-license sources and the dataset is ready for commercial and non-commercial use.

Use `nvidia/SWE-Zero-openhands-trajectories` as the second agentic source. It has 318,115 trajectories over 118,092 issues, uses OpenHands-style trajectories, includes model patches, and its card lists CC BY 4.0 plus MIT, Apache-2.0, BSD-2-Clause, and BSD-3-Clause source licenses.

Use SWE-bench for evaluation, not training. The current docs list SWE-bench Full at 2,294 instances, Lite at 534, Verified at 500, Multimodal at 100 dev / 500 test, and Multilingual at 300.

## Vast.ai Instance Setup

In Vast.ai:

1. Select a recommended PyTorch template with the `[Automatic]` version tag.
2. Filter for RTX 5090 or RTX 5-series GPU offers.
3. Do not change the Docker image for the first run. Vast's RTX 5-series guide says Blackwell/RTX 5-series requires CUDA 12.8 and PyTorch 2.7 or greater.
4. Use SSH or Jupyter launch mode. SSH is best for long training.
5. Set disk before renting. Vast docs note disk is fixed at creation time. Use at least 250GB for a small rehearsal, 500GB for a useful first run, and 1TB+ if keeping raw data and multiple checkpoints on the same instance.
6. Prefer a machine with enough host RAM and CPU to stream/pack datasets without choking. Cheap GPU offers with tiny disk or weak CPU can waste more time than they save.

## Public GitHub Code Transfer

Public GitHub is the simplest Vast workflow. Keep the source code public, but keep datasets,
checkpoints, logs, tokens, local caches, and packed `.bin` files out of Git. The repository
includes a `.gitignore` for those files.

Before publishing, check what Git would include:

```bash
git init
git branch -M main
git status --short
git add .
git status --short
```

Create the public repository and push with GitHub CLI:

```bash
git commit -m "Initial RAAM-AgentCoder prototype"
gh repo create raam-lm --public --source=. --remote=origin --push
```

If you create the GitHub repo manually instead:

```bash
git remote add origin git@github.com:<YOUR_GITHUB_USER>/raam-lm.git
git push -u origin main
```

Then on the Vast instance:

```bash
cd /workspace
git clone https://github.com/<YOUR_GITHUB_USER>/raam-lm.git
cd raam-lm
```

Private transfer still works if you change your mind later: use `rsync`, SCP, or a private repo
with a short-lived deploy key.

After the code is present on the Vast instance:

```bash
cd /workspace/raam-lm
. /venv/main/bin/activate || true

python -m pip install -U pip
python -m pip install -e .
python -m pip install datasets tqdm huggingface_hub

export HF_HOME=/data/hf
export HF_DATASETS_CACHE=/data/hf/datasets
export TOKENIZERS_PARALLELISM=false
mkdir -p /data/agentcoder/raw /data/agentcoder/packed /workspace/runs
```

If a dataset requires a Hugging Face token:

```bash
huggingface-cli login
```

Run the hardware and repo gate:

```bash
bash scripts/vast_preflight.sh configs/scratch/raam_agentcoder_debug.yaml runs/vast_preflight
```

Do not start paid long training until that preflight passes on the actual RTX 5090 instance.

## Build the First Dataset

Create a small first corpus:

```bash
python scripts/prepare_agentcoder_research_data.py \
  --output-dir /data/agentcoder/raw \
  --max-open-swe 20000 \
  --max-swe-zero 20000 \
  --max-wildchat 20000 \
  --max-oasst 10000 \
  --starcoder2-extras documentation=20000 issues=20000 stackoverflow=20000 kaggle=10000
```

This writes canonical JSONL under:

```text
/data/agentcoder/raw/base_code_docs/
/data/agentcoder/raw/chat/
/data/agentcoder/raw/agent_traces/
```

Add your own local permissive repos, docs, diffs, and logs under the same tree:

```text
/data/agentcoder/raw/local_code/
/data/agentcoder/raw/patch_test/
```

Train the tokenizer:

```bash
python scripts/train_tokenizer.py /data/agentcoder/raw \
  --output /data/agentcoder/tokenizer.json \
  --vocab-size 32768
```

Pack the data:

```bash
python scripts/pack_dataset.py /data/agentcoder/raw \
  --tokenizer /data/agentcoder/tokenizer.json \
  --output-dir /data/agentcoder/packed \
  --seq-len 2048 \
  --val-fraction 0.02
```

## First Vast Training Run

Start with the 50M rehearsal:

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

Run generation and eval smoke checks:

```bash
python scripts/generate.py \
  --config configs/scratch/raam_agentcoder_50m.yaml \
  --tokenizer /data/agentcoder/tokenizer.json \
  --checkpoint /workspace/runs/raam_agentcoder_50m_rehearsal/checkpoints/last.pt \
  --prompt $'<|user|>\nFix this failing unit test.\n<|assistant|>\n' \
  --device cuda \
  --max-new-tokens 128

python scripts/eval_agentic_coding.py \
  --config configs/scratch/raam_agentcoder_50m.yaml \
  --tokenizer /data/agentcoder/tokenizer.json \
  --checkpoint /workspace/runs/raam_agentcoder_50m_rehearsal/checkpoints/last.pt \
  --device cuda \
  --output /workspace/runs/raam_agentcoder_50m_rehearsal/agentic_eval.json
```

Only move to `configs/scratch/raam_agentcoder_100m.yaml` after:

- validation loss decreases during the 50M rehearsal
- checkpoint resume works
- generation does not collapse into repeats immediately
- agentic eval runs to completion
- GPU memory leaves enough headroom for the selected batch and sequence length

## What We Are Not Doing Yet

We are not training from the full Stack v2 dump on one RTX 5090.

We are not using SWE-bench gold patches as training data for benchmark evaluation. Keep SWE-bench Lite and Verified for held-out evaluation.

We are not claiming RAAM is good until it beats the Transformer and pure Mamba-like baselines under matched token, parameter, and FLOP budgets.

## Sources

- https://docs.vast.ai/rtx-5-series
- https://docs.vast.ai/guides/instances/choosing/templates
- https://docs.vast.ai/guides/instances/docker-environment
- https://docs.vast.ai/guides/templates/advanced-setup
- https://huggingface.co/datasets/bigcode/the-stack-v2
- https://huggingface.co/datasets/bigcode/starcoder2data-extras
- https://huggingface.co/datasets/OpenAssistant/oasst1
- https://huggingface.co/datasets/allenai/WildChat
- https://huggingface.co/datasets/nvidia/Open-SWE-Traces
- https://huggingface.co/datasets/nvidia/SWE-Zero-openhands-trajectories
- https://www.swebench.com/SWE-bench/guides/datasets/
