# Agentic Data Format

RAAM-AgentCoder uses local JSONL, text, code, Markdown, patch, and log files. No dataset downloader is required by default.

## Canonical JSONL

Each line may contain:

```json
{
  "system": "Optional behavior instruction.",
  "repo_context": "Relevant files, snippets, tree, or issue context.",
  "messages": [
    {"role": "user", "content": "Request"},
    {"role": "assistant", "content": "Optional prior reply"}
  ],
  "trace": [
    {"type": "assistant", "content": "Plan or reasoning summary."},
    {"type": "tool_call", "content": "pytest -q"},
    {"type": "tool_result", "content": "1 failed"},
    {"type": "patch", "content": "--- a/file.py\n+++ b/file.py\n..."},
    {"type": "test_output", "content": "1 passed"}
  ],
  "final": "Concise final answer."
}
```

The renderer maps records into explicit tokens such as `<|user|>`, `<|assistant|>`, `<|repo_context|>`, `<|patch|>`, and `<|test_output|>`.

## Supported Local Inputs

- `.txt`
- `.md`
- `.jsonl`
- `.json`
- code folders
- `.diff` and `.patch`
- stack trace or terminal `.log`

## Tiny Example

The repository includes `examples/tiny_agentic.jsonl` for CPU debug:

```bash
python scripts/train_tokenizer.py examples/tiny_agentic.jsonl --output runs/tokenizer.json --vocab-size 512
python scripts/pack_dataset.py examples/tiny_agentic.jsonl --tokenizer runs/tokenizer.json --output-dir runs/packed_tiny --seq-len 64
```

