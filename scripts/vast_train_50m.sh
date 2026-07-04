#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scratch/raam_agentcoder_50m.yaml}"
BASE_DIR="${BASE_DIR:-/workspace/raam-lm}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/agentcoder}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw}"
PACKED_DIR="${PACKED_DIR:-$DATA_ROOT/packed_2048}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/tokenizer.json}"
RUN_DIR="${RUN_DIR:-/workspace/raam-lm/runs/raam_agentcoder_50m_rehearsal}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
SYNC_DIR="${SYNC_DIR:-}"
SYNC_EVERY="${SYNC_EVERY:-5}"
STEPS="${STEPS:-20}"
RESUME_STEPS="${RESUME_STEPS:-25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-}"
EVAL_BATCHES="${EVAL_BATCHES:-}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
SEQ_LEN="${SEQ_LEN:-2048}"
ASSISTANT_LOSS_ONLY="${ASSISTANT_LOSS_ONLY:-0}"
AGENT_RECORDS_ONLY="${AGENT_RECORDS_ONLY:-0}"
SCORE_PLAIN_TEXT_LOSS="${SCORE_PLAIN_TEXT_LOSS:-1}"
MAX_PACK_DOCUMENTS="${MAX_PACK_DOCUMENTS:-0}"
MAX_PACK_DOCUMENT_CHARS="${MAX_PACK_DOCUMENT_CHARS:-0}"
SAVE_BEST="${SAVE_BEST:-}"
EARLY_STOP_PATIENCE_EVALS="${EARLY_STOP_PATIENCE_EVALS:-}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-}"
EARLY_STOP_MIN_STEP="${EARLY_STOP_MIN_STEP:-}"
RESTORE_BEST_ON_FINISH="${RESTORE_BEST_ON_FINISH:-}"
VALIDATION_LR_DECAY_PATIENCE_EVALS="${VALIDATION_LR_DECAY_PATIENCE_EVALS:-}"
VALIDATION_LR_DECAY_FACTOR="${VALIDATION_LR_DECAY_FACTOR:-}"
VALIDATION_LR_DECAY_MIN_SCALE="${VALIDATION_LR_DECAY_MIN_SCALE:-}"
VALIDATION_LR_DECAY_MIN_STEP="${VALIDATION_LR_DECAY_MIN_STEP:-}"
USE_BEST_CHECKPOINT_FOR_EVAL="${USE_BEST_CHECKPOINT_FOR_EVAL:-1}"
RUN_QUALITATIVE_INSPECT="${RUN_QUALITATIVE_INSPECT:-0}"
QUALITATIVE_SEEDS="${QUALITATIVE_SEEDS:-17}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-1}"
KEEP_TRAINING_CHECKPOINTS="${KEEP_TRAINING_CHECKPOINTS:-1}"

# Small defaults keep the first paid rehearsal cheap. Raise these for the real corpus.
MAX_OPEN_SWE="${MAX_OPEN_SWE:-200}"
MAX_SWE_ZERO="${MAX_SWE_ZERO:-200}"
MAX_WILDCHAT="${MAX_WILDCHAT:-200}"
MAX_OASST="${MAX_OASST:-100}"
STARCODER2_EXTRAS="${STARCODER2_EXTRAS:-documentation=200 issues=200 stackoverflow=200 kaggle=100}"

cd "$BASE_DIR"
if [[ -f /venv/main/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi

uv pip install -e .
uv pip install datasets tqdm huggingface_hub

mkdir -p "$RAW_DIR" "$PACKED_DIR" "$RUN_DIR"

if [[ ! -f "$RAW_DIR/manifest.json" ]] && ! find "$RAW_DIR" -type f -name '*.jsonl' -size +0c | grep -q .; then
  read -r -a extras <<< "$STARCODER2_EXTRAS"
  if ! python scripts/prepare_agentcoder_research_data.py \
      --output-dir "$RAW_DIR" \
      --max-open-swe "$MAX_OPEN_SWE" \
      --max-swe-zero "$MAX_SWE_ZERO" \
      --max-wildchat "$MAX_WILDCHAT" \
      --max-oasst "$MAX_OASST" \
      --continue-on-source-error \
      --starcoder2-extras "${extras[@]}"; then
    if [[ -f "$RAW_DIR/manifest.json" ]] && python -m json.tool "$RAW_DIR/manifest.json" >/dev/null; then
      echo "dataset preparation exited non-zero after writing a valid manifest; continuing with $RAW_DIR" >&2
    else
      exit 1
    fi
  fi
elif [[ ! -f "$RAW_DIR/manifest.json" ]]; then
  echo "raw data exists without manifest; reusing existing JSONL files in $RAW_DIR"
fi

if [[ ! -f "$TOKENIZER" ]]; then
  python scripts/train_tokenizer.py "$RAW_DIR" --output "$TOKENIZER" --vocab-size "$VOCAB_SIZE"
fi

pack_args=()
if [[ "$ASSISTANT_LOSS_ONLY" == "1" ]]; then
  pack_args+=(--assistant-loss-only)
fi
if [[ "$AGENT_RECORDS_ONLY" == "1" ]]; then
  pack_args+=(--agent-records-only)
fi
if [[ "$SCORE_PLAIN_TEXT_LOSS" == "0" ]]; then
  pack_args+=(--no-score-plain-text-loss)
fi
if (( MAX_PACK_DOCUMENTS > 0 )); then
  pack_args+=(--max-documents "$MAX_PACK_DOCUMENTS")
fi
if (( MAX_PACK_DOCUMENT_CHARS > 0 )); then
  pack_args+=(--max-document-chars "$MAX_PACK_DOCUMENT_CHARS")
fi

needs_pack=0
if [[ ! -f "$PACKED_DIR/train.bin" || ! -f "$PACKED_DIR/val.bin" ]]; then
  needs_pack=1
elif [[ "$ASSISTANT_LOSS_ONLY" == "1" && ( ! -f "$PACKED_DIR/train_loss_mask.bin" || ! -f "$PACKED_DIR/val_loss_mask.bin" ) ]]; then
  needs_pack=1
fi
if [[ "$needs_pack" == "0" && ! -f "$PACKED_DIR/manifest.json" ]]; then
  needs_pack=1
fi
if [[ "$needs_pack" == "0" && -f "$PACKED_DIR/manifest.json" ]]; then
  if ! python - "$PACKED_DIR/manifest.json" "$ASSISTANT_LOSS_ONLY" "$AGENT_RECORDS_ONLY" "$SCORE_PLAIN_TEXT_LOSS" "$MAX_PACK_DOCUMENTS" "$MAX_PACK_DOCUMENT_CHARS" <<'PY'
import json
import sys

manifest = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected = {
    "assistant_loss_only": sys.argv[2] == "1",
    "agent_records_only": sys.argv[3] == "1",
    "score_plain_text_loss": sys.argv[4] != "0",
    "max_documents": int(sys.argv[5]),
    "max_document_chars": int(sys.argv[6]),
}
raise SystemExit(0 if all(manifest.get(key) == value for key, value in expected.items()) else 1)
PY
  then
    needs_pack=1
  fi
fi

if [[ "$needs_pack" == "1" ]]; then
  python scripts/pack_dataset.py "$RAW_DIR" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$PACKED_DIR" \
    --seq-len "$SEQ_LEN" \
    --val-fraction 0.02 \
    "${pack_args[@]}"
fi

sync_args=()
if [[ -n "$SYNC_DIR" ]]; then
  sync_args=(--sync-dir "$SYNC_DIR" --sync-every "$SYNC_EVERY")
fi

train_overrides=()
resume_args=()
if [[ -n "$START_CHECKPOINT" ]]; then
  resume_args=(--resume "$START_CHECKPOINT")
fi
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
if [[ "$SAVE_BEST" == "1" ]]; then
  train_overrides+=(--save-best)
elif [[ "$SAVE_BEST" == "0" ]]; then
  train_overrides+=(--no-save-best)
fi
if [[ -n "$EARLY_STOP_PATIENCE_EVALS" ]]; then
  train_overrides+=(--early-stop-patience-evals "$EARLY_STOP_PATIENCE_EVALS")
fi
if [[ -n "$EARLY_STOP_MIN_DELTA" ]]; then
  train_overrides+=(--early-stop-min-delta "$EARLY_STOP_MIN_DELTA")
fi
if [[ -n "$EARLY_STOP_MIN_STEP" ]]; then
  train_overrides+=(--early-stop-min-step "$EARLY_STOP_MIN_STEP")
fi
if [[ "$RESTORE_BEST_ON_FINISH" == "1" ]]; then
  train_overrides+=(--restore-best-on-finish)
elif [[ "$RESTORE_BEST_ON_FINISH" == "0" ]]; then
  train_overrides+=(--no-restore-best-on-finish)
fi
if [[ -n "$VALIDATION_LR_DECAY_PATIENCE_EVALS" ]]; then
  train_overrides+=(--validation-lr-decay-patience-evals "$VALIDATION_LR_DECAY_PATIENCE_EVALS")
fi
if [[ -n "$VALIDATION_LR_DECAY_FACTOR" ]]; then
  train_overrides+=(--validation-lr-decay-factor "$VALIDATION_LR_DECAY_FACTOR")
fi
if [[ -n "$VALIDATION_LR_DECAY_MIN_SCALE" ]]; then
  train_overrides+=(--validation-lr-decay-min-scale "$VALIDATION_LR_DECAY_MIN_SCALE")
fi
if [[ -n "$VALIDATION_LR_DECAY_MIN_STEP" ]]; then
  train_overrides+=(--validation-lr-decay-min-step "$VALIDATION_LR_DECAY_MIN_STEP")
fi

python scripts/train.py \
  --config "$CONFIG" \
  --train-bin "$PACKED_DIR/train.bin" \
  --val-bin "$PACKED_DIR/val.bin" \
  --tokenizer "$TOKENIZER" \
  --output-dir "$RUN_DIR" \
  --steps "$STEPS" \
  --device cuda \
  --save-every "$SAVE_EVERY" \
  --eval-every "$EVAL_EVERY" \
  "${resume_args[@]}" \
  "${train_overrides[@]}" \
  "${sync_args[@]}"

if (( RESUME_STEPS > STEPS )); then
  python scripts/train.py \
    --config "$CONFIG" \
    --train-bin "$PACKED_DIR/train.bin" \
    --val-bin "$PACKED_DIR/val.bin" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$RUN_DIR" \
    --resume "$RUN_DIR/checkpoints/last.pt" \
    --steps "$RESUME_STEPS" \
    --device cuda \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    "${train_overrides[@]}" \
    "${sync_args[@]}"
fi

POST_TRAIN_CHECKPOINT="$RUN_DIR/checkpoints/last.pt"
if [[ "$USE_BEST_CHECKPOINT_FOR_EVAL" == "1" && -f "$RUN_DIR/checkpoints/best.pt" ]]; then
  POST_TRAIN_CHECKPOINT="$RUN_DIR/checkpoints/best.pt"
fi
printf 'post_train_checkpoint path=%s\n' "$POST_TRAIN_CHECKPOINT"

python scripts/generate.py \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$POST_TRAIN_CHECKPOINT" \
  --prompt $'<|user|>\nFix this failing unit test.\n<|assistant|>\n' \
  --device cuda \
  --max-new-tokens 64 \
  > "$RUN_DIR/generation_smoke.txt"

python scripts/eval_agentic_coding.py \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$POST_TRAIN_CHECKPOINT" \
  --device cuda \
  --output "$RUN_DIR/agentic_eval.json"

if [[ "$RUN_QUALITATIVE_INSPECT" == "1" ]]; then
  python scripts/qualitative_checkpoint_inspect.py \
    --config "$CONFIG" \
    --tokenizer "$TOKENIZER" \
    --checkpoint "$POST_TRAIN_CHECKPOINT" \
    --device cuda \
    --seeds "$QUALITATIVE_SEEDS" \
    --output-json "$RUN_DIR/qualitative_samples.json" \
    --output-md "$RUN_DIR/qualitative_samples.md"
fi

if [[ "$EXPORT_CHECKPOINT" == "1" ]]; then
  python scripts/export_checkpoint.py \
    --checkpoint "$POST_TRAIN_CHECKPOINT" \
    --output "$RUN_DIR/checkpoints/model_only_fp16.pt" \
    --dtype fp16
  if [[ "$POST_TRAIN_CHECKPOINT" == "$RUN_DIR/checkpoints/best.pt" ]]; then
    cp "$RUN_DIR/checkpoints/model_only_fp16.pt" "$RUN_DIR/checkpoints/model_only_best_fp16.pt"
  fi
fi

if [[ "$KEEP_TRAINING_CHECKPOINTS" != "1" ]]; then
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type f \( -name 'last.pt' -o -name 'step_*.pt' \) -delete
fi

printf 'vast_train_50m_complete run_dir=%s raw_dir=%s packed_dir=%s tokenizer=%s\n' "$RUN_DIR" "$RAW_DIR" "$PACKED_DIR" "$TOKENIZER"
