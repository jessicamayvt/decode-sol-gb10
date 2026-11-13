import json
from pathlib import Path

import pytest

from decode_sol.roofline import (
    ModelDims,
    UnsupportedModelError,
    compute_kv_bytes,
    compute_num_weight_params_read,
    load_model_dims,
    parse_model_dims,
    predict_decode_step,
)

QWEN3_06B_CONFIG_PATH = Path(__file__).parent / "fixtures" / "qwen3-0.6b-config.json"

# From the Qwen/Qwen3-0.6B safetensors header (revision c1899de): 751,632,384 parameters in total,
# of which 155,582,464 are model.embed_tokens.weight. Decode reads every tensor except the
# embedding table (only batch_size rows of it), so 751,632,384 - 155,582,464.
QWEN3_06B_NUM_WEIGHT_PARAMS_READ = 596_049_920

# A config small enough to count by hand. Per layer: q_proj 8*8=64, k_proj 8*4=32, v_proj 32,
# o_proj 64, MLP 3*8*16=384, norms 2*8 + 2*2=20, total 596. Two layers 1192, plus final norm 8
# and lm_head 10*8=80, gives 1280.
TINY_DIMS = ModelDims(
    num_layers=2,
    hidden_size=8,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=2,
    intermediate_size=16,
    vocab_size=10,
)
TINY_NUM_WEIGHT_PARAMS_READ = 1280


def _load_qwen3_06b_config() -> dict[str, object]:
    config: dict[str, object] = json.loads(QWEN3_06B_CONFIG_PATH.read_text())
    return config


def test_num_weight_params_read_matches_hand_count_for_tiny_config() -> None:
    num_params = compute_num_weight_params_read(TINY_DIMS)

    assert num_params == TINY_NUM_WEIGHT_PARAMS_READ


def test_num_weight_params_read_matches_qwen3_06b_safetensors() -> None:
    dims = load_model_dims(QWEN3_06B_CONFIG_PATH)

    num_params = compute_num_weight_params_read(dims)

    assert num_params == QWEN3_06B_NUM_WEIGHT_PARAMS_READ


def test_kv_bytes_matches_hand_count() -> None:
    # 3 sequences * 5 tokens * 2 layers * (K and V) * 2 kv heads * head_dim 2 * 2 bytes
    expected_kv_bytes = 3 * 5 * 2 * 2 * 2 * 2 * 2

    kv_bytes = compute_kv_bytes(TINY_DIMS, batch_size=3, context_len=5, kv_element_bytes=2)

    assert kv_bytes == expected_kv_bytes


def test_fp8_kv_cache_halves_kv_bytes() -> None:
    bf16_kv_bytes = compute_kv_bytes(TINY_DIMS, batch_size=3, context_len=5, kv_element_bytes=2)

    fp8_kv_bytes = compute_kv_bytes(TINY_DIMS, batch_size=3, context_len=5, kv_element_bytes=1)

    assert fp8_kv_bytes * 2 == bf16_kv_bytes


def test_predict_decode_step_matches_hand_computation() -> None:
    # weights 1280 params * 2 bytes = 2560, KV 480 bytes, at 1000 bytes/s: 3.04 s per step.
    prediction = predict_decode_step(
        TINY_DIMS,
        batch_size=3,
        context_len=5,
        kv_element_bytes=2,
        bandwidth_bytes_per_s=1000.0,
    )

    assert prediction.weight_bytes == 2560
    assert prediction.kv_bytes == 480
    assert prediction.step_time_s == pytest.approx(3.04)
    assert prediction.tokens_per_s == pytest.approx(3 / 3.04)


def test_predict_decode_step_with_empty_context_reads_only_weights() -> None:
    prediction = predict_decode_step(
        TINY_DIMS,
        batch_size=1,
        context_len=0,
        kv_element_bytes=2,
        bandwidth_bytes_per_s=1000.0,
    )

    assert prediction.kv_bytes == 0
    assert prediction.step_time_s == pytest.approx(2.56)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_predict_decode_step_rejects_non_positive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        predict_decode_step(
            TINY_DIMS,
            batch_size=batch_size,
            context_len=5,
            kv_element_bytes=2,
            bandwidth_bytes_per_s=1000.0,
        )


def test_predict_decode_step_rejects_negative_context_len() -> None:
    with pytest.raises(ValueError, match="context_len"):
        predict_decode_step(
            TINY_DIMS,
            batch_size=1,
            context_len=-1,
            kv_element_bytes=2,
            bandwidth_bytes_per_s=1000.0,
        )


def test_parse_model_dims_reads_qwen3_06b_config() -> None:
    dims = parse_model_dims(_load_qwen3_06b_config())

    assert dims == ModelDims(
        num_layers=28,
        hidden_size=1024,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        intermediate_size=3072,
        vocab_size=151936,
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("model_type", "llama", "model_type"),
        ("attention_bias", True, "attention_bias"),
        ("quantization_config", {"quant_method": "fp8"}, "quantization_config"),
    ],
)
def test_parse_model_dims_rejects_configs_the_formula_does_not_cover(
    key: str, value: object, message: str
) -> None:
    config = _load_qwen3_06b_config()
    config[key] = value

    with pytest.raises(UnsupportedModelError, match=message):
        parse_model_dims(config)


def test_parse_model_dims_rejects_config_without_head_dim() -> None:
    config = _load_qwen3_06b_config()
    del config["head_dim"]

    with pytest.raises(UnsupportedModelError, match="head_dim"):
        parse_model_dims(config)
