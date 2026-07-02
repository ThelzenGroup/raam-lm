# RAAM-LM Implementation Plan

RAAM-LM is implemented as an executable PyTorch research prototype, not as a benchmark claim.

## Milestone Checklist

- [x] Inspect repository and project instructions.
- [x] Create installable Python package layout.
- [x] Add typed config loader and debug/scale configs.
- [x] Implement RMSNorm, SwiGLU, RoPE, SDPA attention, fallback mixer, and optional Mamba wrapper.
- [x] Implement Dense Transformer and pure mixer baselines.
- [x] Implement RAAM compression, compressed/global stream, attention islands, causal expansion, and LM output.
- [x] Implement next-token, reconstruction, and curriculum MTP losses.
- [x] Add deterministic tiny data and synthetic probes.
- [x] Add smoke training, profiling, FLOP estimate, probe, and ablation scripts.
- [x] Add tests for shapes, causality, compression, anchors, MTP, training, FLOPs, and baselines.
- [x] Add README, DESIGN, EXPERIMENTS, and RESULTS_TEMPLATE.
- [x] Run `python -m pytest -q`.
- [x] Run `python scripts/smoke_train.py --config configs/debug/raam_tiny.yaml --steps 30 --device auto`.
- [x] Run `python scripts/run_debug_ablation_matrix.py --steps 30 --device auto`.
- [x] Run `python scripts/profile_step.py --config configs/debug/raam_tiny.yaml --device auto`.
- [x] Update PROGRESS.md with exact command outcomes.
