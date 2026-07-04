from __future__ import annotations

from raam_lm.tokenization import AgentCoderTokenizer, SPECIAL_TOKENS


def test_generation_suppressed_token_ids_excludes_eos_but_blocks_role_tokens():
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    tokenizer = AgentCoderTokenizer(vocab)

    suppressed = tokenizer.generation_suppressed_token_ids()

    assert tokenizer.vocab["<eos>"] not in suppressed
    assert tokenizer.vocab["<|assistant|>"] in suppressed
    assert tokenizer.vocab["<|user|>"] in suppressed
    assert tokenizer.vocab["<pad>"] in suppressed
