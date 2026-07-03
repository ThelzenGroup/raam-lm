#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-43634442}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"

BASE_DIR="${BASE_DIR:-/root/raam-lm}"
DATA_ROOT="${DATA_ROOT:-/root/data/agentcoder_stage5}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw}"
PACKED_DIR="${PACKED_DIR:-$DATA_ROOT/packed_2048}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/tokenizer.json}"
CONFIG="${CONFIG:-configs/scratch/raam_agentcoder_100m_stage5_lr75e6.yaml}"
RUN_ID="${RUN_ID:-stage5_raam_agentcoder_100m_gate_$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_RUN_ROOT="${REMOTE_RUN_ROOT:-$BASE_DIR/runs/$RUN_ID}"
RUN_DIR="${RUN_DIR:-$REMOTE_RUN_ROOT/train}"

STEPS="${STEPS:-1000}"
RESUME_STEPS="${RESUME_STEPS:-1100}"
SAVE_EVERY="${SAVE_EVERY:-0}"
EVAL_EVERY="${EVAL_EVERY:-100}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-0}"
KEEP_TRAINING_CHECKPOINTS="${KEEP_TRAINING_CHECKPOINTS:-0}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-}"
EVAL_BATCHES="${EVAL_BATCHES:-}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
SEQ_LEN="${SEQ_LEN:-2048}"

parse_ssh_url() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
if url.scheme != "ssh" or not url.hostname or not url.port:
    raise SystemExit(f"unexpected ssh url: {sys.argv[1]}")
print(url.hostname)
print(url.port)
PY
}

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
  ssh_url="$(vastai ssh-url "$INSTANCE_ID")"
  mapfile -t parsed < <(parse_ssh_url "$ssh_url")
  SSH_HOST="${parsed[0]}"
  SSH_PORT="${parsed[1]}"
fi

ssh -i "$SSH_KEY" \
  -o IdentitiesOnly=yes \
  -o BatchMode=yes \
  -o StrictHostKeyChecking=accept-new \
  -p "$SSH_PORT" \
  "root@$SSH_HOST" \
  BASE_DIR="$BASE_DIR" \
  DATA_ROOT="$DATA_ROOT" \
  RAW_DIR="$RAW_DIR" \
  PACKED_DIR="$PACKED_DIR" \
  TOKENIZER="$TOKENIZER" \
  CONFIG="$CONFIG" \
  RUN_ID="$RUN_ID" \
  REMOTE_RUN_ROOT="$REMOTE_RUN_ROOT" \
  RUN_DIR="$RUN_DIR" \
  STEPS="$STEPS" \
  RESUME_STEPS="$RESUME_STEPS" \
  SAVE_EVERY="$SAVE_EVERY" \
  EVAL_EVERY="$EVAL_EVERY" \
  EXPORT_CHECKPOINT="$EXPORT_CHECKPOINT" \
  KEEP_TRAINING_CHECKPOINTS="$KEEP_TRAINING_CHECKPOINTS" \
  BATCH_SIZE="$BATCH_SIZE" \
  TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
  GRAD_ACCUMULATION_STEPS="$GRAD_ACCUMULATION_STEPS" \
  EVAL_BATCHES="$EVAL_BATCHES" \
  VOCAB_SIZE="$VOCAB_SIZE" \
  SEQ_LEN="$SEQ_LEN" \
  'bash -s' <<'REMOTE'
set -euo pipefail

cd "$BASE_DIR"
mkdir -p "$REMOTE_RUN_ROOT"
(
  nohup env \
    BASE_DIR="$BASE_DIR" \
    DATA_ROOT="$DATA_ROOT" \
    RAW_DIR="$RAW_DIR" \
    PACKED_DIR="$PACKED_DIR" \
    TOKENIZER="$TOKENIZER" \
    RUN_DIR="$RUN_DIR" \
    CONFIG="$CONFIG" \
    STEPS="$STEPS" \
    RESUME_STEPS="$RESUME_STEPS" \
    SAVE_EVERY="$SAVE_EVERY" \
    EVAL_EVERY="$EVAL_EVERY" \
    EXPORT_CHECKPOINT="$EXPORT_CHECKPOINT" \
    KEEP_TRAINING_CHECKPOINTS="$KEEP_TRAINING_CHECKPOINTS" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_SEQ_LEN="$TRAIN_SEQ_LEN" \
    GRAD_ACCUMULATION_STEPS="$GRAD_ACCUMULATION_STEPS" \
    EVAL_BATCHES="$EVAL_BATCHES" \
    VOCAB_SIZE="$VOCAB_SIZE" \
    SEQ_LEN="$SEQ_LEN" \
    bash scripts/vast_train_100m_candidate.sh \
    > "$REMOTE_RUN_ROOT/runner.log" 2>&1 < /dev/null &
  echo "$!" > "$REMOTE_RUN_ROOT/runner.pid"
)

printf 'launched run_id=%s pid=%s log=%s run_dir=%s\n' \
  "$RUN_ID" \
  "$(cat "$REMOTE_RUN_ROOT/runner.pid")" \
  "$REMOTE_RUN_ROOT/runner.log" \
  "$RUN_DIR"
REMOTE
