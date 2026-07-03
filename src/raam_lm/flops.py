"""Approximate activated FLOP accounting."""

from __future__ import annotations

from .config import ModelConfig


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_non_embedding_parameters(model) -> int:
    embedding_ids = set()
    for module in [getattr(model, "tok_embeddings", None), getattr(model, "lm_head", None)]:
        if module is None:
            continue
        for param in module.parameters(recurse=False):
            embedding_ids.add(id(param))
    return sum(p.numel() for p in model.parameters() if id(p) not in embedding_ids)


def estimate_flops_per_token(config: ModelConfig) -> int:
    d = config.d_model
    ff = config.d_ff
    layers = config.n_layers
    seq = config.train.seq_len
    vocab = config.vocab_size
    attn_proj = 8 * d * d
    attn_scores = 4 * seq * d
    ffn = 6 * d * ff
    mixer = 8 * d * d + 10 * d
    head = 2 * d * vocab

    if config.model_name == "transformer":
        body = layers * (attn_proj + attn_scores + ffn)
    elif config.model_name == "pure_mamba_like":
        body = layers * (mixer + ffn)
    else:
        comp = config.compression
        entries = comp.pooled_chunks_per_block + max(comp.anchors_per_block, 0)
        ratio = entries / max(comp.block_size, 1) if comp.enabled else 1.0
        attention_islands = len(config.attention_island_layers) if config.use_attention_islands else 0
        cheap_layers = max(layers - attention_islands, 0)
        body = cheap_layers * (mixer + ffn)
        body += int(attention_islands * ratio * (attn_proj + attn_scores * ratio + ffn))
        if comp.enabled:
            body += int(4 * d * d / max(comp.block_size, 1) + entries * d)

    mtp = 0
    if config.mtp.enabled:
        mtp = int(sum(config.mtp.horizon_weights.values()) * head)
    copy = 0
    if config.copy_head.enabled:
        d_copy = int(config.copy_head.d_copy or d)
        copy = 2 * d * d_copy + 4 * seq * d_copy + seq
        if config.copy_head.consistency_strength:
            recent = max(0, int(config.copy_head.consistency_recent_tokens))
            window = max(0, int(config.copy_head.consistency_source_window))
            copy += seq * recent * (window + 1)
        if config.copy_head.binding_carry_strength:
            recent = max(0, int(config.copy_head.binding_carry_recent_tokens))
            window = max(0, int(config.copy_head.binding_carry_source_window))
            occurrence_scan = seq if config.copy_head.binding_carry_max_anchor_occurrences else 0
            copy += seq * recent * (window + occurrence_scan + 1)
    return int(body + head + mtp + copy)
