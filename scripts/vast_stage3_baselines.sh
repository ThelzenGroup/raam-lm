#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/workspace/raam-lm}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/agentcoder}"
PACKED_DIR="${PACKED_DIR:-$DATA_ROOT/packed_2048}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/tokenizer.json}"
RUN_ROOT="${RUN_ROOT:-/workspace/raam-lm/runs/stage3_baselines}"
STEPS="${STEPS:-20}"
RESUME_STEPS="${RESUME_STEPS:-25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-}"
EVAL_BATCHES="${EVAL_BATCHES:-}"
SYNC_DIR="${SYNC_DIR:-}"
SYNC_EVERY="${SYNC_EVERY:-5}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-1}"
DEVICE="${DEVICE:-cuda}"
CONFIGS="${CONFIGS:-configs/scratch/transformer_agentcoder_50m.yaml configs/scratch/pure_mamba_like_agentcoder_50m.yaml configs/scratch/raam_agentcoder_50m.yaml}"

cd "$BASE_DIR"
if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

uv pip install -e .

if [[ ! -f "$TOKENIZER" || ! -f "$PACKED_DIR/train.bin" || ! -f "$PACKED_DIR/val.bin" ]]; then
  echo "Missing packed data or tokenizer. Run scripts/vast_train_50m.sh first." >&2
  exit 1
fi

sync_args=()
if [[ -n "$SYNC_DIR" ]]; then
  sync_args=(--sync-dir "$SYNC_DIR" --sync-every "$SYNC_EVERY")
fi

train_overrides=()
if [[ -n "$BATCH_SIZE" ]]; then
  train_overrides+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$TRAIN_SEQ_LEN" ]]; then
  train_overrides+=(--seq-len "$TRAIN_SEQ_LEN")
fi
if [[ -n "$GRAD_ACCUMULATION_STEPS" ]]; then
  train_overrides+=(--grad-accumulation-steps "$GRAD_ACCUMULATION_STEPS")
fi
if [[ -n "$EVAL_BATCHES" ]]; then
  train_overrides+=(--eval-batches "$EVAL_BATCHES")
fi

run_dirs=()
for config in $CONFIGS; do
  name="$(basename "$config" .yaml)"
  run_dir="$RUN_ROOT/$name"
  mkdir -p "$run_dir"
  run_dirs+=("$run_dir")

  python scripts/train.py \
    --config "$config" \
    --train-bin "$PACKED_DIR/train.bin" \
    --val-bin "$PACKED_DIR/val.bin" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$run_dir" \
    --steps "$STEPS" \
    --device "$DEVICE" \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    "${train_overrides[@]}" \
    "${sync_args[@]}"

  if (( RESUME_STEPS > STEPS )); then
    python scripts/train.py \
      --config "$config" \
      --train-bin "$PACKED_DIR/train.bin" \
      --val-bin "$PACKED_DIR/val.bin" \
      --tokenizer "$TOKENIZER" \
      --output-dir "$run_dir" \
      --resume "$run_dir/checkpoints/last.pt" \
      --steps "$RESUME_STEPS" \
      --device "$DEVICE" \
      --save-every "$SAVE_EVERY" \
      --eval-every "$EVAL_EVERY" \
      "${train_overrides[@]}" \
      "${sync_args[@]}"
  fi

  python scripts/generate.py \
    --config "$config" \
    --tokenizer "$TOKENIZER" \
    --checkpoint "$run_dir/checkpoints/last.pt" \
    --prompt $'<|user|>\nFix this failing unit test.\n<|assistant|>\n' \
    --device "$DEVICE" \
    --max-new-tokens 64 \
    > "$run_dir/generation_smoke.txt"

  python scripts/eval_agentic_coding.py \
    --config "$config" \
    --tokenizer "$TOKENIZER" \
    --checkpoint "$run_dir/checkpoints/last.pt" \
    --device "$DEVICE" \
    --output "$run_dir/agentic_eval.json"

  if [[ "$EXPORT_CHECKPOINT" == "1" ]]; then
    python scripts/export_checkpoint.py \
      --checkpoint "$run_dir/checkpoints/last.pt" \
      --output "$run_dir/checkpoints/model_only_fp16.pt" \
      --dtype fp16
  fi
done

python scripts/compare_training_runs.py "${run_dirs[@]}" \
  --output-json "$RUN_ROOT/summary.json" \
  --output-md "$RUN_ROOT/summary.md"

if [[ -n "$SYNC_DIR" ]]; then
  mkdir -p "$SYNC_DIR"
  cp -a "$RUN_ROOT" "$SYNC_DIR/"
fi

printf 'vast_stage3_baselines_complete run_root=%s\n' "$RUN_ROOT"
