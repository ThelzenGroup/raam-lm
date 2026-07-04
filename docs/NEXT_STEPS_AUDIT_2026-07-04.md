# RAAM-AgentCoder Next Steps Audit - 2026-07-04

## Executive Decision

Stop the synthetic eval loop. The next useful step is not another tiny gate and not a blind bigger training run. The next useful step is:

1. Fix the real-code data builder so it produces source-derived held-out executable evals and extracts the intended CodeRM/unit-test rows.
2. Build a bounded real-code SFT dataset.
3. Run one Vast.ai GPU continuation pilot from the best current 84M-ish RAAM path.
4. Promote only if it beats the current executable baselines on held-out, runnable tasks.

The reason this decision is stable: every useful signal so far points to data/objective quality as the bottleneck, not raw parameter count alone.

## Why The Recommendation Looked Inconsistent

Three different questions were being mixed together:

- Binding diagnostic: can RAAM route requested values when compression is disabled or anchored?
- Executable coding: can the model emit runnable functions, patches, JSON, or tests?
- Scaling readiness: is the data/eval pipeline ready for a larger run?

The binding diagnostic looked good for RAAM no-compression. The executable coding gates did not. The older 84M runs showed some coding signal, but mostly on synthetic/local families. The real-data audit showed the public-data pipeline is not promotion-ready yet. Those are not the same result, and treating the newest tiny gate as the main answer caused the plan to drift.

## Evidence Inventory

### Stage 5 Base LM

`EXPERIMENTS.md` records the best Stage 5 base-LM candidate around the `lr5e5` step-800 export, with validation loss near `3.0210`. Later continuation did not beat it, and agentic scores stayed at `0.0`.

Conclusion: base-LM loss progress exists, but it has not translated into useful chat, coding, or tool behavior.

### Request/Value Binding Ablation

Artifact: `runs/vast_backups/agentcoder_request_value_vast_ablation_20260704T191900Z/current/ablation_summary.json`

Results:

- Transformer: `144/144`, best validation loss `1.8368`, estimated FLOPs/token `8,153,856`.
- RAAM no-compression: `144/144`, best validation loss `1.5094`, estimated FLOPs/token `7,372,544`.
- RAAM all-anchor: `144/144`, best validation loss `1.5034`, estimated FLOPs/token `8,008,064`.

Conclusion: dynamic compression was a likely binding failure source. No-compression RAAM plus request routing can solve the synthetic request/value diagnostic cheaply. This does not prove coding ability.

### Tiny Executable Gates

Artifacts:

- `runs/vast_backups/agentcoder_executable_no_compression_gpu_compare_20260704T194500Z/current/executable_compare_summary.json`
- `runs/vast_backups/agentcoder_executable_function_only_gpu_compare_20260704T211500Z/current/function_only_compare_summary.json`
- `runs/vast_backups/agentcoder_executable_function_train_eval_gpu_compare_20260704T213000Z/current/function_train_eval_compare_summary.json`

Results:

- Mixed executable data, tiny no-compression RAAM: `0/10`.
- Mixed executable data, tiny Transformer: `0/10`.
- Function-only train/eval, tiny no-compression RAAM: `0/6`, best validation loss `0.0471`.
- Function-only train/eval, tiny Transformer: `1/6`, best validation loss `0.0186`.

Conclusion: validation loss was misleading. A low loss tiny RAAM still failed exact executable behavior. These gates were useful diagnostics, but continuing to repeat them is low value.

### 84M-ish Coding Repair And Probe Runs

Key artifacts:

- `runs/vast_backups/coding_ladder_repair_stop_control_20260704T_remote/current/coding_ladder_eval.json`
- `runs/vast_backups/function_repair_20260704T_remote/current/summary.json`
- `runs/vast_backups/function_count_even_probe_20260704T_remote/current/summary.json`
- `runs/vast_backups/function_safe_int_probe_20260704T_remote/current/summary.json`

Results:

- Coding ladder repair with stop control: `3/10`, passing simple function tasks such as even/odd/filter-even families.
- Function repair: `3/20`, with target function pass count `0`; promotion failed.
- Count-even targeted probe: taught the count-even family, reaching `5/20`, but did not generalize to safe-int, parse-port, patch, or JSON tasks.
- Safe-int targeted probe: taught safe-int family, but regressed elsewhere and still failed promotion.

Conclusion: the larger 84M-ish path has more executable signal than tiny no-compression, but targeted synthetic training overfits narrow function families. It is not a general coding model yet.

### Real-Data Builder Audit

Artifact: `runs/audit_executable_real_data_yield_20260704/agentcoder_executable_manifest.json`

Command used:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/make_agentcoder_executable_sft.py \
  --output-dir runs/audit_executable_real_data_yield_20260704 \
  --use-hf \
  --opencode-limit 120 \
  --scotch-limit 120 \
  --coderm-unittest-limit 80 \
  --commitpackft-limit 120 \
  --ladder-repeats 0 \
  --curated-anchor-repeats 0 \
  --eval-source-fraction 0.1 \
  --max-answer-chars 2400 \
  --max-tests-chars 5000 \
  --max-function-lines 80 \
  --max-diff-lines 80 \
  --max-file-chars 4000
```

Results:

- Train records: `364`.
- Eval cases: `10`.
- Source counts: `Samip/Scotch` `118`, `bigcode/commitpackft` `119`, `local_ladder` `41`, `nvidia/OpenCodeInstruct` `86`.
- Behavior counts: function completion `213`, patch generation `119`, code generation `15`, JSON/tool command `5`, and small local ladder behaviors.
- `coderm_unittest:hf`: `80` input rows, `0` train records, `0` eval cases, `80` skipped rows.
- All 10 eval cases came from local ladder; the public-source rows added no held-out executable eval cases.

Conclusion: the builder can fetch public rows, but it is not ready as a promotion-grade real-code pipeline. CodeRM/unit-test extraction is currently ineffective on sampled rows, and held-out executable eval remains local-only.

### External Dataset Fit

Current dataset choices are directionally reasonable:

- [`nvidia/OpenCodeInstruct`](https://huggingface.co/datasets/nvidia/OpenCodeInstruct): coding instruction/question-answer data.
- [`Samip/Scotch`](https://huggingface.co/datasets/Samip/Scotch): permissive GitHub functions, including Python and docstring/code pairs.
- [`KAKA22/CodeRM-UnitTest`](https://huggingface.co/datasets/KAKA22/CodeRM-UnitTest): Python unit-test style data, useful if the converter extracts it correctly.
- [`bigcode/commitpackft`](https://huggingface.co/datasets/bigcode/commitpackft): commit-message and code-diff style data.

The issue is not that these datasets are bad. The issue is that the current builder is not yet converting enough of them into runnable, held-out tasks.

## What We Actually Learned

1. Synthetic request/value binding is mostly solved by disabling compression or using stronger anchoring.
2. Tiny no-compression RAAM is not enough for executable coding behavior.
3. Validation loss is not a reliable promotion metric for this project.
4. Larger 84M-ish synthetic runs can learn simple function families, but they overfit and fail transfer.
5. Public real-code data is necessary, but the current conversion/eval path is incomplete.
6. Local CPU training should stop being used for anything except smoke tests, data validation, and short script checks. Actual training belongs on Vast.ai or equivalent GPU infrastructure.

## Best Next Steps

### Phase 0: Stop The Current Loop

No more repeated tiny synthetic gates unless they answer a new, named hypothesis.

Allowed local CPU work:

- dataset-builder unit tests
- manifest inspection
- tiny smoke training for script correctness only
- eval parser validation

Not allowed local CPU work:

- meaningful model training
- long synthetic overfit runs
- repeated promotion eval loops without a new data or code change

### Phase 1: Fix The Real-Code Data Pipeline

Required changes:

- Inspect sampled rows from `KAKA22/CodeRM-UnitTest` and update the converter to match the actual schema.
- Add source-derived executable eval case generation for at least function completion and unit-test generation.
- Add manifest fields for source ingest, skipped-row reasons, source-derived eval counts, and sampled examples per source.
- Enforce exact train/eval prompt-overlap checks.
- Keep local ladder anchors small, ideally `<= 5-10%` of the final pilot set.

Acceptance criteria:

- CodeRM/unit-test sampled rows produce nonzero train records.
- Public-source rows produce nonzero held-out eval cases.
- The manifest reports source-specific skip reasons.
- Exact train/eval prompt overlap is zero.

### Phase 2: Build A Bounded Real-Code Pilot Dataset

Target first pilot:

- `5,000-20,000` training records.
- Mix: functions, small patches, tests, and JSON/tool format.
- Mostly public-source records, with local ladder anchors capped.
- Python first. Do not expand to broad agent traces until function/test/patch behavior is measurable.

Example build command after the builder is fixed:

```bash
PATH="$PWD/.venv/bin:$PATH" python scripts/make_agentcoder_executable_sft.py \
  --output-dir runs/agentcoder_realcode_pilot_20260704 \
  --use-hf \
  --opencode-limit 6000 \
  --scotch-limit 6000 \
  --coderm-unittest-limit 3000 \
  --commitpackft-limit 5000 \
  --ladder-repeats 1 \
  --curated-anchor-repeats 1 \
  --eval-source-fraction 0.05 \
  --max-answer-chars 2400 \
  --max-tests-chars 5000 \
  --max-function-lines 80 \
  --max-diff-lines 80 \
  --max-file-chars 4000
```

### Phase 3: Run One Vast.ai GPU Pilot

Use Vast.ai for the real training run. Do not use local CPU for the actual pilot.

Starting point:

- Prefer the best current 84M-ish RAAM path over the tiny no-compression path, because the 84M runs have the only current executable coding signal.
- Do not assume the tiny no-compression result scales. Treat 84M no-compression as a separate architecture experiment only after estimating VRAM, FLOPs, and checkpoint compatibility.

Training shape:

- short continuation SFT
- low learning rate
- Track validation loss, generated samples, executable pass rate, and throughput
- stop on degradation or no executable improvement

### Phase 4: Run One Promotion Gate

The promotion gate should be fixed before training starts.

Minimum gate:

- source-derived held-out function completion
- source-derived unit-test generation
- small patch apply
- JSON/tool-command validity
- existing local ladder smoke cases

Compare against:

- current coding-ladder repair baseline: `3/10`
- current function repair baseline: `3/20`
- optionally a matched Transformer if GPU budget allows

Promotion criteria:

- beats current RAAM baseline on executable pass rate
- does not regress the simple function anchors
- produces syntactically valid Python/JSON at materially higher rate
- shows sample-level improvements, not just lower validation loss

## What Not To Do Next

- Do not keep running the same synthetic evals.
- Do not train bigger on CPU.
- Do not scale to a large frontier-style run before real held-out eval exists.
- Do not claim frontier progress from validation loss alone.
- Do not promote a checkpoint that only learns one synthetic family.

## Frontier-Model Reality Check

Making RAAM a frontier model is a much larger program than these experiments. The credible path is:

1. prove the architecture can learn and generalize on real executable tasks;
2. scale data quality and coverage;
3. scale model size only after the promotion gate is meaningful;
4. add instruction/chat/tool/SWE traces after the code model can pass small runnable tasks;
5. evaluate against external benchmarks and real patch tasks.

The current frontier-relevant bottleneck is not parameter count. It is the absence of a reliable real-data training/eval loop that turns GPU spend into measured capability.
