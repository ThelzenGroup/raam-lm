# Agentic Coding Evals

The initial evals are smoke tests, not a benchmark. They check whether generation and scoring plumbing works for chat-first software-engineering behavior.

`scripts/eval_agentic_coding.py` covers:

- bug-fix patch prompt
- stack trace diagnosis prompt
- tool-call formatting prompt

Logged fields include:

- response length
- latency
- syntax validity where applicable
- JSON/tool-call validity
- exact patch apply rate
- unit test pass rate field where available
- qualitative sample output

Future evals should add real repository tasks with expected patches and test commands.

## Curated Overfit Sanity Gate

Before paying for larger chat/coding runs, run the tiny overfit gate:

```bash
python scripts/run_agentcoder_overfit_sanity.py \
  --config configs/scratch/raam_agentcoder_overfit.yaml \
  --data examples/agentcoder_overfit_sanity.jsonl \
  --output-dir runs/agentcoder_overfit_sanity \
  --device auto \
  --clean
```

This is not a benchmark. It deliberately mirrors the tiny dataset into both
train and validation splits, then checks whether the model can memorize exact
chat/software-engineering behaviors:

- minimal bug-fix patch plus `pytest` command
- valid JSON command response
- risky-edit clarifying question
- plain-English debugging process
- simple Python function completion
- stack-trace diagnosis
- repo-context lookup
- default package test command

If this gate fails, do not treat more tokens or bigger paid runs as the next
fix. Inspect formatting, tokenizer coverage, prompt boundaries, EOS behavior,
loss/objective settings, and generation settings first.

## Small Non-Mirrored Slice Gate

After the mirrored overfit gate passes, run the small slice gate:

```bash
python scripts/run_agentcoder_slice_gate.py \
  --config configs/scratch/raam_agentcoder_slice_gate.yaml \
  --data examples/agentcoder_slice_train.jsonl \
  --cases-json examples/agentcoder_slice_eval_cases.json \
  --output-dir runs/agentcoder_slice_gate \
  --device auto \
  --clean
```

This gate uses normal train/validation splitting and held-out eval prompts. It
is still deliberately tiny, so failure is not a model-quality verdict. Its job
is to catch whether the format learned in the mirrored gate survives a small
non-mirrored slice before moving back to larger paid runs.

## Curated Paraphrase SFT Gate

If the small slice gate shows nearest-neighbor memorization, run the broader
curated gate:

```bash
python scripts/run_agentcoder_curated_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir runs/agentcoder_curated_gate \
  --device auto \
  --clean
```

This gate generates deterministic synthetic SFT records with balanced behavior
coverage, then evaluates separate held-out prompts. It covers bug-fix patches,
strict JSON command output, risky-edit clarification, plain debugging, function
completion, stack-trace diagnosis, repo-context lookup, test-command
recommendation, command-intent disambiguation, code review, boolean-flag patching,
and commit summaries.

The eval summary also reports a behavior confusion matrix. Exact pass/fail still
comes from required substrings and expected JSON, but the confusion matrix shows
whether a failed answer looked like the wrong behavior family, for example
answering a risky-edit question with a JSON command or answering a boolean-flag
patch prompt with a generic addition patch.

Passing this gate still does not prove a useful model. It is a cheap control
that checks whether the pipeline can learn reusable behavior patterns before
spending on a larger supervised or continuation run.
