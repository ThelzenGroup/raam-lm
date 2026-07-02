#!/usr/bin/env bash
set -euo pipefail

INSTANCE_ID="${INSTANCE_ID:-43627905}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/workspace/raam-lm/runs/raam_agentcoder_50m_rehearsal}"
LOCAL_DIR="${LOCAL_DIR:-runs/vast_backups/$(basename "$REMOTE_RUN_DIR")}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
WATCH_INTERVAL="${WATCH_INTERVAL:-0}"
INCLUDE_CHECKPOINTS="${INCLUDE_CHECKPOINTS:-0}"
INCLUDE_MODEL_EXPORT="${INCLUDE_MODEL_EXPORT:-0}"

parse_ssh_url() {
  python - "$1" <<'PY'
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
if url.scheme != "ssh" or not url.hostname or not url.port:
    raise SystemExit(f"unexpected ssh url: {sys.argv[1]}")
print(url.hostname)
print(url.port)
PY
}

pull_once() {
  local ssh_url host port
  ssh_url="$(vastai ssh-url "$INSTANCE_ID")"
  mapfile -t parsed < <(parse_ssh_url "$ssh_url")
  host="${parsed[0]}"
  port="${parsed[1]}"
  mkdir -p "$LOCAL_DIR"

  if command -v rsync >/dev/null 2>&1 && ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -p "$port" "root@$host" 'command -v rsync >/dev/null 2>&1'; then
    rsync_args=(-az --partial --delete)
    if [[ "$INCLUDE_CHECKPOINTS" != "1" ]]; then
      if [[ "$INCLUDE_MODEL_EXPORT" == "1" ]]; then
        rsync_args+=(--include '/checkpoints/' --include '/checkpoints/model_only_*.pt')
      fi
      rsync_args+=(--exclude '/checkpoints/*.pt')
    fi
    rsync "${rsync_args[@]}" \
      -e "ssh -i $SSH_KEY -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p $port" \
      "root@$host:$REMOTE_RUN_DIR/" "$LOCAL_DIR/"
  else
    rm -rf "$LOCAL_DIR/.incoming"
    mkdir -p "$LOCAL_DIR/.incoming"
    tar_exclude=""
    if [[ "$INCLUDE_CHECKPOINTS" != "1" ]]; then
      if [[ "$INCLUDE_MODEL_EXPORT" == "1" ]]; then
        tar_exclude="--exclude='$(basename "$REMOTE_RUN_DIR")/checkpoints/last.pt' --exclude='$(basename "$REMOTE_RUN_DIR")/checkpoints/step_*.pt'"
      else
        tar_exclude="--exclude='$(basename "$REMOTE_RUN_DIR")/checkpoints/*.pt'"
      fi
    fi
    ssh -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p "$port" "root@$host" \
      "tar -C '$(dirname "$REMOTE_RUN_DIR")' $tar_exclude -cf - '$(basename "$REMOTE_RUN_DIR")'" \
      | tar -C "$LOCAL_DIR/.incoming" -xf -
    rm -rf "$LOCAL_DIR/current"
    mv "$LOCAL_DIR/.incoming/$(basename "$REMOTE_RUN_DIR")" "$LOCAL_DIR/current"
    rmdir "$LOCAL_DIR/.incoming"
  fi

  printf 'pulled instance=%s remote=%s local=%s\n' "$INSTANCE_ID" "$REMOTE_RUN_DIR" "$LOCAL_DIR"
}

if [[ "$WATCH_INTERVAL" == "0" ]]; then
  pull_once
else
  while true; do
    pull_once
    sleep "$WATCH_INTERVAL"
  done
fi
