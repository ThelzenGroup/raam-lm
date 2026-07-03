from __future__ import annotations

import torch
import pytest

from raam_lm.config import CopyHeadConfig, ModelConfig
from raam_lm.copy_head import CausalCopyHead
from raam_lm.registry import build_model


def zero_copy_projection_weights(head: CausalCopyHead) -> None:
    torch.nn.init.zeros_(head.query.weight)
    torch.nn.init.zeros_(head.key.weight)


def test_causal_copy_head_boosts_visible_context_tokens():
    config = CopyHeadConfig(enabled=True, d_copy=4, logit_scale=4.0)
    head = CausalCopyHead(d_model=4, vocab_size=16, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 4, 4)
    input_ids = torch.tensor([[3, 5, 5, 9]])
    lm_logits = torch.zeros(1, 4, 16)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 2, 5] > out[0, 2, 9]
    assert out[0, 2, 3] > out[0, 2, 9]
    assert out[0, 2, 7] < out[0, 2, 5]


def test_causal_copy_head_consistency_bias_prefers_same_source_row():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        consistency_strength=8.0,
        consistency_recent_tokens=2,
        consistency_source_window=2,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 9, 4)
    input_ids = torch.tensor([[10, 101, 20, 201, 10, 102, 20, 202, 101]])
    lm_logits = torch.zeros(1, 9, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 8, 201] > out[0, 8, 202]


def test_causal_copy_head_binding_carry_prefers_target_after_rare_recent_anchor():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        binding_carry_strength=10.0,
        binding_carry_recent_tokens=4,
        binding_carry_source_window=3,
        binding_carry_min_source_gap=2,
        binding_carry_max_anchor_occurrences=3,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 9, 4)
    input_ids = torch.tensor([[101, 77, 201, 102, 77, 202, 101, 77, 55]])
    lm_logits = torch.zeros(1, 9, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 8, 201] > out[0, 8, 202]


def test_causal_copy_head_key_follow_prefers_source_value_before_boundary():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_strength=10.0,
        key_follow_recent_tokens=1,
        key_follow_value_offset=3,
        key_follow_value_span=1,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 11, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 5, 31, 9, 9, 202, 31, 8]])
    lm_logits = torch.zeros(1, 11, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 10, 201] > out[0, 10, 202]


def test_causal_copy_head_key_follow_aligns_multitoken_value_offset():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_strength=10.0,
        key_follow_recent_tokens=4,
        key_follow_value_offset=3,
        key_follow_value_span=1,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        key_follow_align_value_offset=True,
        key_follow_separator_token_id=8,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 11, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 202, 203, 5, 31, 8, 201, 202]])
    lm_logits = torch.zeros(1, 11, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 8, 201] > out[0, 8, 202]
    assert out[0, 9, 202] > out[0, 9, 201]
    assert out[0, 10, 203] > out[0, 10, 201]


def test_causal_copy_head_key_follow_matches_generated_value_prefix():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_strength=10.0,
        key_follow_recent_tokens=4,
        key_follow_value_offset=3,
        key_follow_value_span=1,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        key_follow_align_value_offset=True,
        key_follow_match_value_prefix=True,
        key_follow_separator_token_id=8,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 15, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 202, 31, 9, 9, 203, 204, 5, 31, 8, 203, 204]])
    lm_logits = torch.zeros(1, 15, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 13, 203] > out[0, 13, 201]
    assert out[0, 14, 204] > out[0, 14, 202]


def test_causal_copy_head_key_follow_requires_recent_key_after_assistant_boundary():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_strength=10.0,
        key_follow_recent_tokens=4,
        key_follow_value_offset=3,
        key_follow_value_span=1,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        key_follow_align_value_offset=True,
        key_follow_separator_token_id=8,
        key_follow_recent_after_token_id=6,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 14, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 32, 9, 9, 202, 5, 31, 8, 6, 31, 8]])
    lm_logits = torch.zeros(1, 14, 256)

    out = head(hidden, input_ids, lm_logits)

    torch.testing.assert_close(out[0, 11, 201], out[0, 11, 202])
    assert out[0, 13, 201] > out[0, 13, 202]


def test_causal_copy_head_request_key_follow_prefers_requested_value_before_assistant():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        request_key_follow_strength=10.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=4,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 12, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 32, 9, 9, 202, 5, 31, 6, 8]])
    lm_logits = torch.zeros(1, 12, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 11, 201] > out[0, 11, 202]


def test_causal_copy_head_request_key_follow_continues_value_prefix():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        request_key_follow_strength=10.0,
        request_key_follow_continuation_strength=12.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=4,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 17, 4)
    input_ids = torch.tensor(
        [[31, 9, 9, 201, 202, 203, 32, 9, 9, 204, 205, 206, 5, 31, 6, 8, 201]]
    )
    lm_logits = torch.zeros(1, 17, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 16, 202] > out[0, 16, 204]
    assert out[0, 16, 202] > out[0, 16, 201]


def test_causal_copy_head_request_key_follow_uses_query_window_and_prompt_suffix():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        request_key_follow_strength=10.0,
        request_key_follow_continuation_strength=12.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=4,
        request_key_follow_source_after_token_id=4,
        request_key_follow_query_after_token_id=8,
        request_key_follow_query_before_token_ids=[9],
        request_key_follow_query_ignore_token_ids=[7],
        request_key_follow_prompt_suffix_tokens=1,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 23, 4)
    input_ids = torch.tensor(
        [[32, 7, 7, 202, 4, 31, 7, 7, 201, 32, 7, 7, 202, 5, 99, 32, 8, 7, 31, 9, 100, 6, 10]]
    )
    lm_logits = torch.zeros(1, 23, 256)

    out = head(hidden, input_ids, lm_logits)

    assert out[0, 22, 201] > out[0, 22, 202]
    assert out[0, 22, 201] > out[0, 22, 31]


def test_causal_copy_head_request_key_follow_does_not_cross_stop_tokens():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        key_follow_stop_token_ids=[10],
        request_key_follow_strength=10.0,
        request_key_follow_continuation_strength=20.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=8,
        request_key_follow_query_after_token_id=8,
        request_key_follow_query_before_token_ids=[9],
        request_key_follow_prompt_suffix_tokens=1,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    input_ids = torch.tensor([[31, 7, 7, 201, 10, 32, 7, 7, 202, 5, 99, 8, 31, 9, 6, 10, 201, 10]])
    seq_len = input_ids.shape[1]
    source_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril().unsqueeze(0)

    _, continuation = head._request_key_follow_probs_by_pos(input_ids, source_mask)

    assert continuation[0, 17, 5] == 0


def test_causal_copy_head_request_key_follow_eval_only_skips_train_route():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=2,
        key_follow_source_until_token_id=5,
        request_key_follow_strength=10.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=4,
        request_key_follow_eval_only=True,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 12, 4)
    input_ids = torch.tensor([[31, 9, 9, 201, 32, 9, 9, 202, 5, 31, 6, 8]])
    lm_logits = torch.zeros(1, 12, 256)

    head.eval()
    eval_out = head(hidden, input_ids, lm_logits)
    head.train()
    train_out = head(hidden, input_ids, lm_logits)

    assert eval_out[0, 11, 201] > eval_out[0, 11, 202]
    torch.testing.assert_close(train_out[0, 11, 201], train_out[0, 11, 202])


def test_causal_copy_head_ignores_future_token_ids_for_earlier_logits():
    config = CopyHeadConfig(enabled=True, d_copy=4, logit_scale=4.0)
    head = CausalCopyHead(d_model=4, vocab_size=32, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 5, 4)
    lm_logits = torch.zeros(1, 5, 32)
    base_ids = torch.tensor([[2, 4, 6, 8, 10]])
    changed_future_ids = torch.tensor([[2, 4, 6, 21, 22]])

    base = head(hidden, base_ids, lm_logits)
    changed = head(hidden, changed_future_ids, lm_logits)

    torch.testing.assert_close(base[:, :3], changed[:, :3])


def test_causal_copy_head_key_follow_ignores_future_token_ids_for_earlier_logits():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_strength=10.0,
        key_follow_recent_tokens=1,
        key_follow_value_offset=3,
        key_follow_value_span=1,
        key_follow_min_source_gap=1,
        key_follow_source_until_token_id=5,
    )
    head = CausalCopyHead(d_model=4, vocab_size=64, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 8, 4)
    lm_logits = torch.zeros(1, 8, 64)
    base_ids = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]])
    changed_future_ids = torch.tensor([[2, 4, 6, 21, 22, 23, 24, 25]])

    base = head(hidden, base_ids, lm_logits)
    changed = head(hidden, changed_future_ids, lm_logits)

    torch.testing.assert_close(base[:, :3], changed[:, :3])


def test_causal_copy_head_request_key_follow_ignores_future_token_ids_for_earlier_logits():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        key_follow_value_offset=3,
        key_follow_min_source_gap=1,
        key_follow_source_until_token_id=5,
        request_key_follow_strength=10.0,
        request_key_follow_continuation_strength=10.0,
        request_key_follow_recent_tokens=8,
        request_key_follow_after_token_id=5,
        request_key_follow_before_token_id=6,
        request_key_follow_value_span=4,
    )
    head = CausalCopyHead(d_model=4, vocab_size=256, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 12, 4)
    lm_logits = torch.zeros(1, 12, 256)
    base_ids = torch.tensor([[31, 9, 9, 201, 32, 9, 9, 202, 5, 31, 6, 8]])
    changed_future_ids = torch.tensor([[31, 9, 9, 201, 32, 9, 9, 99, 5, 32, 6, 8]])

    base = head(hidden, base_ids, lm_logits)
    changed = head(hidden, changed_future_ids, lm_logits)

    torch.testing.assert_close(base[:, :7], changed[:, :7])


def test_causal_copy_head_binding_carry_ignores_future_token_ids_for_earlier_logits():
    config = CopyHeadConfig(
        enabled=True,
        d_copy=4,
        logit_scale=4.0,
        binding_carry_strength=10.0,
        binding_carry_recent_tokens=4,
        binding_carry_source_window=3,
        binding_carry_min_source_gap=1,
        binding_carry_max_anchor_occurrences=3,
    )
    head = CausalCopyHead(d_model=4, vocab_size=64, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 8, 4)
    lm_logits = torch.zeros(1, 8, 64)
    base_ids = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]])
    changed_future_ids = torch.tensor([[2, 4, 6, 21, 22, 23, 24, 25]])

    base = head(hidden, base_ids, lm_logits)
    changed = head(hidden, changed_future_ids, lm_logits)

    torch.testing.assert_close(base[:, :3], changed[:, :3])


def test_causal_copy_head_promotes_lower_precision_logits_for_stability():
    config = CopyHeadConfig(enabled=True, d_copy=4, logit_scale=4.0)
    head = CausalCopyHead(d_model=4, vocab_size=16, config=config)
    zero_copy_projection_weights(head)
    hidden = torch.zeros(1, 4, 4)
    input_ids = torch.tensor([[3, 5, 5, 9]])
    lm_logits = torch.zeros(1, 4, 16, dtype=torch.bfloat16)

    out = head(hidden, input_ids, lm_logits)

    assert out.dtype == torch.float32


def test_causal_copy_head_backward_is_finite_with_lower_precision_logits():
    config = CopyHeadConfig(enabled=True, d_copy=4, logit_scale=4.0)
    head = CausalCopyHead(d_model=4, vocab_size=16, config=config)
    hidden = torch.randn(2, 6, 4, requires_grad=True)
    input_ids = torch.tensor([[3, 5, 5, 9, 4, 8], [2, 7, 7, 1, 6, 3]])
    lm_logits = torch.randn(2, 6, 16, dtype=torch.bfloat16).requires_grad_()

    out = head(hidden, input_ids, lm_logits)
    loss = out[:, :-1].float().mean()
    loss.backward()

    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(head.query.weight.grad).all()
    assert torch.isfinite(head.key.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression requires CUDA")
def test_causal_copy_head_cuda_autocast_backward_is_finite():
    config = CopyHeadConfig(enabled=True, d_copy=8, logit_scale=4.0)
    head = CausalCopyHead(d_model=8, vocab_size=32, config=config).cuda()
    hidden = torch.randn(2, 8, 8, device="cuda", requires_grad=True)
    input_ids = torch.tensor(
        [[3, 5, 5, 9, 4, 8, 7, 1], [2, 7, 7, 1, 6, 3, 9, 4]],
        device="cuda",
    )
    lm_logits = torch.randn(2, 8, 32, device="cuda", requires_grad=True)

    with torch.autocast(device_type="cuda"):
        out = head(hidden, input_ids, lm_logits)
        loss = out[:, :-1].mean()
    loss.backward()

    assert out.dtype == torch.float32
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(head.query.weight.grad).all()
    assert torch.isfinite(head.key.weight.grad).all()


def test_registry_models_enable_copy_head_from_config():
    for model_name in ["raam", "transformer", "pure_mamba_like"]:
        config = ModelConfig(
            model_name=model_name,
            vocab_size=32,
            max_seq_len=16,
            d_model=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            d_ff=32,
        )
        config.copy_head.enabled = True
        config.compression.block_size = 4
        config.compression.anchors_per_block = 1
        config.compression.token_id_anchor_count = 1
        config.compression.anchor_selection = "hybrid_token_id_learned"
        if model_name != "raam":
            config.compression.enabled = False
            config.use_dynamic_hourglass_compression = False
            config.use_anchor_preserved_local_global = False
            config.use_attention_islands = False

        model = build_model(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 12))
        out = model(input_ids, labels=input_ids)

        assert out["logits"].shape == (2, 12, config.vocab_size)
        assert out["aux"]["copy_head_enabled"] is True
