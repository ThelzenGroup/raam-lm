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

The eval summary also reports a behavior confusion matrix. Exact pass/fail comes
from required substrings, forbidden substrings, and expected JSON. The confusion
matrix shows whether a failed answer looked like the wrong behavior family, for
example answering a risky-edit question with a JSON command or answering a
boolean-flag patch prompt with a generic addition patch.

Some cases also include forbidden substrings to catch slot-copying failures:
right behavior, wrong symbol/file/literal. For example, a repo lookup can have
the correct "implemented in" style while naming a stale function or file from a
nearby training example. Those failures are reported as `slot_error` when the
behavior family is correct but required text is missing or forbidden text is
present.

Repo-lookup training records include distractor files with familiar definitions
such as `add`, `slugify`, and `parse_port`. The model must bind the requested
symbol to the matching `def` line instead of replaying a frequent symbol/file
pair from another training example.

Boolean-flag patch cases are intentionally shaped differently from arithmetic
bug-fix patches. They name themselves as boolean flag repair, avoid focused
pytest-command language, and require the exact file/helper/enabled-literal
slots. This keeps a model from passing by emitting the familiar addition-patch
template when the task is really a feature-flag literal fix.

Passing this gate still does not prove a useful model. It is a cheap control
that checks whether the pipeline can learn reusable behavior patterns before
spending on a larger supervised or continuation run.

## Comparing Gate Runs

Use the gate comparison script after pulling one or more curated gate artifact
directories:

```bash
python scripts/compare_agentcoder_gates.py \
  runs/agentcoder_curated_gate_a \
  runs/agentcoder_curated_gate_b \
  --output-json runs/agentcoder_gate_comparison.json \
  --output-md runs/agentcoder_gate_comparison.md
```

The report compares exact pass rate, validation loss, behavior accuracy, failed
cases, and behavior confusions. This is useful for deciding whether a new
curriculum actually improves held-out behavior or only lowers validation loss
while shifting mistakes between behavior families.

By default, the comparison script recomputes behavior labels from completions
with the current heuristic. Pass `--prefer-stored-behavior` when you need to
replay the labels saved inside the original artifact exactly.

When exact pass rate stalls but behavior accuracy improves, inspect
`slot_error`, `missing_required_substrings`, and `present_forbidden_substrings`.
Typical examples are choosing the correct repo-lookup style but naming the wrong
file, or producing the right kind of patch for the wrong helper/value pair.
If behavior accuracy drops, inspect the confusion matrix first; a flag patch
predicted as `patch_addition` means the curriculum still has patch-family
collision, not merely poor slot copying.
