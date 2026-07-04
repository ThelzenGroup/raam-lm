# RAAM-LM MLOps Tracker

This project can publish experiment evidence to the native `.mlops/experiments`
store used by `mlops-mcp-server`.

The tracker is local-only by default and is ignored by Git. It should contain
run summaries, metrics, and artifact references, not large model weights.

## Global MCP Server

Codex is configured to start the global MCP server from:

```bash
/home/lumalgo/.codex/bin/mlops-mcp-server
```

Codex config:

```toml
[mcp_servers.mlops]
command = "/home/lumalgo/.codex/bin/mlops-mcp-server"
args = []
```

Restart Codex after changing MCP config.

## Backfill Historical Runs

Use the backfill script to import pulled run evidence into `.mlops/experiments`:

```bash
python scripts/backfill_mlops_runs.py \
  --project-path /home/lumalgo/Documents/exp2 \
  --source-root /home/lumalgo/Documents/Codex/2026-07-02/g/outputs
```

By default, artifact files are referenced by path. Add `--copy-artifacts` only
if you want small JSON/YAML/Markdown evidence copied into `.mlops`.
Checkpoint files are always stored as references only.

## Live Training Logging

Future packed-data training can log live metrics by passing:

```bash
python scripts/train.py \
  --config configs/scratch/raam_agentcoder_100m_stage5_lr5e5.yaml \
  --train-bin /path/to/train.bin \
  --val-bin /path/to/val.bin \
  --tokenizer /path/to/tokenizer.json \
  --output-dir /path/to/run/train \
  --mlops-project-path /home/lumalgo/Documents/exp2
```

Equivalent environment variable:

```bash
export RAAM_MLOPS_PROJECT_PATH=/home/lumalgo/Documents/exp2
```

Live logging records training loss, validation loss, learning rate, grad norm,
tokens/sec, step time, memory, FLOP estimates, config/tokenizer/manifest
references, and checkpoint path references.

## Query Through MCP

After Codex restarts with the `mlops` MCP loaded, use the experiment tools with:

```text
project_path=/home/lumalgo/Documents/exp2
```

Useful queries:

- list runs
- get best run by `val_next_token_loss` with direction `min`
- compare selected run IDs
- inspect artifact references for a run
