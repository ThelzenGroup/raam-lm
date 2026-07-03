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
pair from another training example. The tiny curriculum also avoids near-duplicate
`title_tools.py` examples around the held-out `normalize_title` case, and instead
uses nearby but distinct `title_case` and `normalize_*` drills to test slot
binding without handing the exact answer to the model. Repo-lookup prompts now
explicitly tell the model to start with the exact requested symbol so failures
show up as slot-copy errors instead of vague behavior misses.

Boolean-flag patch cases are intentionally shaped differently from arithmetic
bug-fix patches. They name themselves as boolean flag repair, avoid focused
pytest-command language, and require the exact file/helper/enabled-literal
slots. This keeps a model from passing by emitting the familiar addition-patch
template when the task is really a feature-flag literal fix.

Arithmetic bug-fix patch cases are also shaped away from generic test-command
recommendations: prompts explicitly ask for the diff first and the focused test
command second, while naming the file path, helper, and return expression that
must be copied from repo context. A response that only names
`python -m pytest -q` is a behavior confusion, not a patch pass; a response that
patches the wrong arithmetic helper is a slot-copy failure.

Passing this gate still does not prove a useful model. It is a cheap control
that checks whether the pipeline can learn reusable behavior patterns before
spending on a larger supervised or continuation run.

## Atomic Copy Gate

If the copy-only ladder still fails, run the atomic copy gate before changing
architecture. This is the smallest control: one slot family, two fields
(`symbol` and `file`), no distractors, short `key=value` output, and mirrored
packed validation by default.

```bash
python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/raam_agentcoder_atomic_copy_gate.yaml \
  --output-dir runs/agentcoder_atomic_copy_gate_raam \
  --device auto \
  --clean
```

Run the tiny Transformer baseline with:

```bash
python scripts/run_agentcoder_atomic_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_atomic_copy_gate.yaml \
  --output-dir runs/agentcoder_atomic_copy_gate_transformer \
  --device auto \
  --clean
```

The default `--eval-mode mirror` evaluates slots that appeared in training, and
the runner defaults to `--mirror-val`. Passing this does not prove useful coding
ability; failing it means the pipeline cannot yet learn the most basic exact
copy control. After mirror passes, use `--eval-mode ladder --no-mirror-val` to
add held-out slots back in.

When one-record mirror overfit passes but larger mirror runs fail, locate the
binding break point with the cardinality sweep:

```bash
python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --output-dir runs/agentcoder_atomic_cardinality_sweep \
  --device auto \
  --clean
```

By default this runs RAAM and the tiny Transformer baseline at
`1,2,4,8,16,32,64` train records, with mirrored eval cases matched to the train
record count. The aggregate `summary.json` records the first cardinality below
the selected pass-rate threshold for each model. The sweep intentionally uses
`--no-fail` for each sub-run unless `--fail-on-gate` is passed, because a low
pass rate is the measurement.

If full RAAM breaks before the Transformer baseline, isolate dynamic hourglass
compression with:

```bash
python scripts/run_agentcoder_atomic_cardinality_sweep.py \
  --models raam \
  --raam-config configs/scratch/raam_agentcoder_atomic_no_compression_gate.yaml \
  --train-records 4,8,16,32,64 \
  --output-dir runs/agentcoder_atomic_cardinality_sweep_raam_no_compression \
  --device auto \
  --clean
```

## Copy-Only Slot Binding Gate

If exact slot-copy failures appear, run the copy-only gate before another
patch-formatting gate:

```bash
python scripts/run_agentcoder_copy_gate.py \
  --config configs/scratch/raam_agentcoder_copy_gate.yaml \
  --output-dir runs/agentcoder_copy_gate_raam \
  --device auto \
  --clean
```

Run the tiny Transformer baseline on the same generated data shape with:

```bash
python scripts/run_agentcoder_copy_gate.py \
  --config configs/scratch/transformer_agentcoder_copy_gate.yaml \
  --output-dir runs/agentcoder_copy_gate_transformer \
  --device auto \
  --clean
```

This gate removes diff formatting, prose, and test-command wording. The model
only has to emit short `key=value` lines copied from the current context:

- `repo_lookup_copy`: `symbol=<symbol>` and `file=<file>`.
- `patch_return_copy`: `file=<file>`, `helper=<helper>`, `return=<expr>`, and
  `test=<path>`.
- `patch_literal_copy`: `file=<file>`, `helper=<helper>`, `literal=<value>`,
  and `test=<path>`.

The default ladder emits `144` train records and `96` eval cases, split into
`48` seen-slot cases and `48` held-out-slot cases. If a model cannot pass
seen-slot cases here, the blocker is basic context copying/memorization rather
than patch generation. If the Transformer baseline passes and RAAM fails, the
RAAM compression path is suspect. If both fail, fix the objective, tokenizer,
prompt/data shape, or training recipe before scaling.

## Slot-Copy Diagnostic Gate

If the curated gate reaches the right behavior family but fails exact file,
function, symbol, literal, or test-command slots, run the larger slot-copy gate:

```bash
python scripts/run_agentcoder_slotcopy_gate.py \
  --config configs/scratch/raam_agentcoder_curated_gate.yaml \
  --output-dir runs/agentcoder_slotcopy_gate \
  --device auto \
  --clean
```

This gate generates a programmatic curriculum plus a seen-vs-heldout diagnostic
ladder:

- `repo_lookup`: copy the requested symbol and defining file from a repo context
  that includes several unrelated definitions.
- `patch_return`: patch the exact arithmetic helper/file/return expression from
  repo context while ignoring other similar buggy helpers.
- `patch_literal`: patch the exact boolean-flag helper/file/enabled literal from
  repo context while ignoring other flag helpers.

The runner default uses `--eval-mode ladder`, which writes `144` training records
and `96` eval cases: `48` train records, `16` seen-slot eval cases, and `16`
held-out-slot eval cases per slot family. `seen_slot` cases reuse slot tuples
from training with regenerated context, while `heldout_slot` cases keep expected
slot tuples disjoint from training. The eval cases include `slot_family`,
`eval_tier`, and `expected_slots`; the runner summarizes pass rate, behavior
accuracy, and slot-error count by family and by ladder tier.

Use `--eval-mode heldout` if you want the older `48` case held-out-only gate.
This is still not a benchmark; it is a preflight for context binding before
spending on broader chat/coding training.

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
