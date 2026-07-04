from __future__ import annotations

from raam_lm.config import CopyHeadConfig, ModelConfig, load_config, resolve_copy_head_token_ids


class FakeTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<0x0A>": 23,
            " ": 269,
            ":": 271,
            ",": 334,
            "\n": 333,
            ".": 335,
        }

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        del add_bos, add_eos
        return [self.vocab[text]]


def test_resolve_copy_head_token_ids_uses_trained_tokenizer_ids():
    config = ModelConfig(
        copy_head=CopyHeadConfig(
            key_follow_stop_token_ids=[23, 328],
            request_key_follow_query_after_token_id=271,
            request_key_follow_source_separator_token_id=271,
            request_key_follow_query_before_token_ids=[273],
            request_key_follow_query_ignore_token_ids=[23, 269, 272, 328],
        )
    )

    resolve_copy_head_token_ids(config, FakeTokenizer())

    assert config.copy_head.key_follow_stop_token_ids == [23, 333]
    assert config.copy_head.request_key_follow_query_after_token_id == 271
    assert config.copy_head.request_key_follow_source_separator_token_id == 271
    assert config.copy_head.request_key_follow_query_before_token_ids == [335]
    assert config.copy_head.request_key_follow_query_ignore_token_ids == [23, 333, 269, 334]


def test_train_config_loads_early_stop_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train:",
                "  save_best: true",
                "  early_stop_patience_evals: 2",
                "  early_stop_min_delta: 0.01",
                "  early_stop_min_step: 800",
                "  restore_best_on_finish: true",
                "  validation_lr_decay_patience_evals: 1",
                "  validation_lr_decay_factor: 0.25",
                "  validation_lr_decay_min_scale: 0.125",
                "  validation_lr_decay_min_step: 900",
            ]
        )
        + "\n"
    )

    config = load_config(config_path)

    assert config.train.save_best is True
    assert config.train.early_stop_patience_evals == 2
    assert config.train.early_stop_min_delta == 0.01
    assert config.train.early_stop_min_step == 800
    assert config.train.restore_best_on_finish is True
    assert config.train.validation_lr_decay_patience_evals == 1
    assert config.train.validation_lr_decay_factor == 0.25
    assert config.train.validation_lr_decay_min_scale == 0.125
    assert config.train.validation_lr_decay_min_step == 900
