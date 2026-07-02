#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scratch/raam_agentcoder_50m.yaml}"
BASE_DIR="${BASE_DIR:-/workspace/raam-lm}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/agentcoder}"
RAW_DIR="${RAW_DIR:-$DATA_ROOT/raw}"
PACKED_DIR="${PACKED_DIR:-$DATA_ROOT/packed_2048}"
TOKENIZER="${TOKENIZER:-$DATA_ROOT/tokenizer.json}"
RUN_DIR="${RUN_DIR:-/workspace/raam-lm/runs/raam_agentcoder_50m_rehearsal}"
SYNC_DIR="${SYNC_DIR:-}"
SYNC_EVERY="${SYNC_EVERY:-5}"
STEPS="${STEPS:-20}"
RESUME_STEPS="${RESUME_STEPS:-25}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_EVERY="${EVAL_EVERY:-5}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
SEQ_LEN="${SEQ_LEN:-2048}"
EXPORT_CHECKPOINT="${EXPORT_CHECKPOINT:-1}"

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
  python scripts/prepare_agentcoder_research_data.py \
    --output-dir "$RAW_DIR" \
    --max-open-swe "$MAX_OPEN_SWE" \
    --max-swe-zero "$MAX_SWE_ZERO" \
    --max-wildchat "$MAX_WILDCHAT" \
    --max-oasst "$MAX_OASST" \
    --continue-on-source-error \
    --starcoder2-extras "${extras[@]}"
elif [[ ! -f "$RAW_DIR/manifest.json" ]]; then
  echo "raw data exists without manifest; reusing existing JSONL files in $RAW_DIR"
fi

if [[ ! -f "$TOKENIZER" ]]; then
  python scripts/train_tokenizer.py "$RAW_DIR" --output "$TOKENIZER" --vocab-size "$VOCAB_SIZE"
fi

if [[ ! -f "$PACKED_DIR/train.bin" || ! -f "$PACKED_DIR/val.bin" ]]; then
  python scripts/pack_dataset.py "$RAW_DIR" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$PACKED_DIR" \
    --seq-len "$SEQ_LEN" \
    --val-fraction 0.02
fi

sync_args=()
if [[ -n "$SYNC_DIR" ]]; then
  sync_args=(--sync-dir "$SYNC_DIR" --sync-every "$SYNC_EVERY")
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
    "${sync_args[@]}"
fi

python scripts/generate.py \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$RUN_DIR/checkpoints/last.pt" \
  --prompt $'<|user|>\nFix this failing unit test.\n<|assistant|>\n' \
  --device cuda \
  --max-new-tokens 64 \
  > "$RUN_DIR/generation_smoke.txt"

python scripts/eval_agentic_coding.py \
  --config "$CONFIG" \
  --tokenizer "$TOKENIZER" \
  --checkpoint "$RUN_DIR/checkpoints/last.pt" \
  --device cuda \
  --output "$RUN_DIR/agentic_eval.json"

if [[ "$EXPORT_CHECKPOINT" == "1" ]]; then
  python scripts/export_checkpoint.py \
    --checkpoint "$RUN_DIR/checkpoints/last.pt" \
    --output "$RUN_DIR/checkpoints/model_only_fp16.pt" \
    --dtype fp16
fi

printf 'vast_train_50m_complete run_dir=%s raw_dir=%s packed_dir=%s tokenizer=%s\n' "$RUN_DIR" "$RAW_DIR" "$PACKED_DIR" "$TOKENIZER"
