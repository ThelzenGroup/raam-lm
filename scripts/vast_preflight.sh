#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/scratch/raam_agentcoder_debug.yaml}"
OUTDIR="${2:-runs/vast_preflight}"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "$OUTDIR"

echo "== nvidia-smi =="
nvidia-smi || true

echo "== torch/cuda =="
"$PYTHON_BIN" - <<'PY'
import os
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda or "0.0"
    print("gpu_name", name)
    print("capability", capability)
    print("torch_cuda_version", cuda_version)
    props = torch.cuda.get_device_properties(0)
    print("total_vram_gb", round(props.total_memory / (1024**3), 2))
    if "5090" not in name and os.environ.get("ALLOW_NON_5090") != "1":
        raise SystemExit(f"expected RTX 5090; got {name}. Set ALLOW_NON_5090=1 only for rehearsal.")
    major_minor = tuple(int(part) for part in cuda_version.split(".")[:2])
    if major_minor < (12, 8):
        raise SystemExit(f"expected CUDA 12.8+ runtime; got {cuda_version}")
    torch_version = tuple(int(part.split("+")[0]) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 7):
        raise SystemExit(f"expected PyTorch 2.7+ for RTX 50-series; got {torch.__version__}")
else:
    raise SystemExit("CUDA is not available; Vast RTX 5090 preflight failed")
PY

echo "== tests =="
"$PYTHON_BIN" -m pytest -q

echo "== tiny profile =="
"$PYTHON_BIN" scripts/profile_step.py --config "$CONFIG" --device auto --steps 1 --output "$OUTDIR/profile_manifest.json"

echo "== tiny checkpoint/resume test =="
"$PYTHON_BIN" scripts/train_tokenizer.py examples/tiny_agentic.jsonl --output "$OUTDIR/tokenizer.json" --vocab-size 512
"$PYTHON_BIN" scripts/pack_dataset.py examples/tiny_agentic.jsonl --tokenizer "$OUTDIR/tokenizer.json" --output-dir "$OUTDIR/packed" --seq-len 64 --val-fraction 0.34
"$PYTHON_BIN" scripts/train.py --config "$CONFIG" --train-bin "$OUTDIR/packed/train.bin" --val-bin "$OUTDIR/packed/val.bin" --tokenizer "$OUTDIR/tokenizer.json" --output-dir "$OUTDIR/train" --steps 1 --device auto
"$PYTHON_BIN" scripts/train.py --config "$CONFIG" --train-bin "$OUTDIR/packed/train.bin" --val-bin "$OUTDIR/packed/val.bin" --tokenizer "$OUTDIR/tokenizer.json" --output-dir "$OUTDIR/train" --steps 2 --resume "$OUTDIR/train/checkpoints/last.pt" --device auto

echo "== checkpoint write test =="
"$PYTHON_BIN" - <<PY
from pathlib import Path
path = Path("$OUTDIR") / "write_test.txt"
path.write_text("ok\\n")
print(path)
PY

echo "vast_preflight_complete output_dir=$OUTDIR"
