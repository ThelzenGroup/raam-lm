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

