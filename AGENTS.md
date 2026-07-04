# RAAM-LM Agent Guidance

## Project Scope

- This repo is the from-scratch RAAM-LM effort targeting a cheap-to-train/use chat and agentic coding/software-engineering model.
- Work from this repository when touching code, configs, docs, training scripts, evaluation scripts, or experiment tracking for RAAM.
- The GitHub repository is `ThelzenGroup/raam-lm`.

## Experiment Tracking

- Use the MLOps MCP first for questions about run history, best runs, comparisons, metrics, artifacts, and model-card style summaries.
- The local `.mlops` store is generated experiment state and should stay out of git.
- Validate user-facing claims about model quality with artifacts, eval logs, generated samples, and MLOps metrics. Do not treat low loss on tiny synthetic/overfit runs as proof of useful chat or coding ability.

## Training And Evaluation

- Favor real evaluation evidence over optimistic summaries: next-token validation loss, generated chat/code samples, pass rates, patch-apply/tool-call validity, stability, throughput, and VRAM usage.
- Before full training, check data quality, tokenizer compatibility, config consistency, checkpoint/resume behavior, eval coverage, logging, and sample generation.
- For Vast.ai work, verify the active instance, SSH endpoint, repo path, Python environment, checkpoints, and data paths before giving commands.

## Tool Routing

- Use Hugging Face MCP/skills for datasets, Hub uploads/downloads, model/dataset discovery, paper lookup, Spaces, and training ecosystem work.
- Use Trackio for live training dashboards, alerts, and metric visualization when setting up or monitoring runs.
- Use Context7 for current documentation about training libraries, CLI tools, Python packages, and cloud services.
- Use GitHub tooling for repo publishing, PRs, issues, and CI-related work.
