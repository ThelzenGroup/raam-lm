# RAAM-LM Experiment Plan

## First Tiny Experiment

Run:

- `transformer_tiny`
- `pure_mamba_like_tiny`
- `raam_tiny`
- `raam_no_compression`
- `raam_no_anchors`
- `raam_no_attention_islands`
- `raam_no_mtp`

Use the same generated token stream, tokenizer identity, optimizer, schedule, batch size, and sequence length. Report loss versus tokens, approximate FLOPs, wall-clock time, throughput, memory, and causal-test status.

## Small-Scale Experiment

Use the 70M-style configs as practical templates. Run 10, 20, and 40 tokens-per-parameter sweeps when resources allow. Use seeds 17 and 29 for tiny/small runs if budget permits. Keep tokenizer, data order, optimizer, schedule, and sequence length matched across models.

## RAAM-AgentCoder Scratch Track

First prove the end-to-end pipeline locally:

- tokenizer training on `examples/tiny_agentic.jsonl`
- dataset packing
- 20-step train
- 5-step resume
- generation from checkpoint
- agentic coding eval smoke run

Then run a Vast rehearsal:

- `raam_agentcoder_50m`
- 1000 steps
- validation every 500 steps
- checkpoint save/resume
- generation samples
- agentic eval outputs

Only promote `raam_agentcoder_100m` after the 50M rehearsal shows stable validation loss, usable throughput, and reliable checkpoint resume.

Current Vast evidence: in a matched 100M `1000 -> 1100` step gate on the first
packed AgentCoder corpus, compression-only `raam_agentcoder_100m` beat
`pure_mamba_like_agentcoder_100m` on validation loss while running faster and using
less peak VRAM. This promotes compression-only RAAM as the next scaling default, but
it is still base-LM evidence only: chat/tool-call validity and patch-apply eval
scores are not useful yet.

Expanded Stage 5 evidence: the full Stage 5 data path successfully built a larger
raw corpus, tokenizer, and packed 2048-token corpus, but the older
`raam_agentcoder_100m` auxiliary-loss schedule was not stable. Validation reached
its best point early, then worsened by the final evaluation with collapsed
generation and `0.0` agentic scores. The next candidate is
`raam_agentcoder_100m_stage5_stable`, which keeps compression-only RAAM but disables
early reconstruction loss and curriculum MTP until a clean base-LM curve exists.

Stable Stage 5 gate evidence: `raam_agentcoder_100m_stage5_stable` avoids the
catastrophic auxiliary-loss blow-up, but it still peaks early on the expanded
corpus. In the `1000 -> 1100` step gate, validation improved from `10.3872` to a
best `3.1310` at step 500, then worsened to `4.5944` by step 1099. The next
experiment should keep the stable loss setup but lower/cap the learning rate or
shorten warmup before any full training spend.

The first capped-LR candidate is
`configs/scratch/raam_agentcoder_100m_stage5_lr1e4.yaml`: same data, architecture,
and disabled auxiliary-loss setup as the stable gate, but with `lr: 0.0001` and
`warmup_steps: 500`.

Capped-LR gate evidence: `raam_agentcoder_100m_stage5_lr1e4` improved the final
validation loss materially versus the stable config (`3.3375` versus `4.5944` at
step 1099), and had a better best point (`3.0503` at step 500 versus `3.1310`).
It still peaked at step 500 and drifted upward afterward, so the next gate should
test an even safer continuation policy: shorter `500 -> 600` checkpoint export for
the current best region, or a lower `5e-5`/`7.5e-5` LR gate if training beyond
500 steps is required.

The next lower-LR candidate is
`configs/scratch/raam_agentcoder_100m_stage5_lr75e6.yaml`, which keeps the same
loss and architecture settings as `lr1e4` but caps LR at `0.000075`.

## Ablations

- No compression.
- Fixed compression instead of learned anchors.
- No anchors.
- No attention islands.
- 1 versus 2 versus 3 attention islands.
- MTP off, static, and curriculum.
- Fallback mixer versus `mamba_ssm` if available.

## Fair Comparisons

Parameter-matched: match non-embedding parameter count where possible.

FLOP-matched: compare by approximate activated training FLOPs/token and total token budget.

Token-budget-matched: use the same tokenizer, data order, optimizer, schedule, and number of tokens.

Long-context-matched: use the same length adaptation recipe and evaluation lengths.

## Falsification Criteria

- Kill the attention-island hypothesis if RAAM does not improve associative recall, passkey, or copy probes over `pure_mamba_like` and does not improve loss-per-FLOP or wall-clock-to-loss against Transformer.
- Kill the dynamic compression hypothesis if it cannot achieve meaningful compression without a validation-loss or probe regression, or if fixed compression matches it at the same compression ratio.
- Kill the anchor hypothesis if learned anchors retain too many tokens or rare-token/code/copy probes regress versus no-compression.
- Kill the MTP add-on if next-token validation loss or time-to-target-loss is not improved versus MTP-off, or if calibration visibly worsens.
- Kill the overall idea if gains vanish when tokenizer, data order, optimizer, sequence length, and schedule are matched.
