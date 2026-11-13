"""Memory-bandwidth ceiling for one decode step. Pure functions, no GPU.

The derivation and its assumptions are in docs/MODEL.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

BF16_BYTES = 2
SUPPORTED_MODEL_TYPE = "qwen3"


class UnsupportedModelError(ValueError):
    """The config describes a model the weight-count formula does not cover."""


@dataclass(frozen=True, slots=True)
class ModelDims:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int


@dataclass(frozen=True, slots=True)
class DecodeStepPrediction:
    weight_bytes: int
    kv_bytes: int
    step_time_s: float
    tokens_per_s: float


def _require_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnsupportedModelError(f"config needs an integer {key!r}, got {value!r}")
    return value


def parse_model_dims(config: dict[str, object]) -> ModelDims:
    """Parse a Hugging Face config.json for a dense bf16 Qwen3 model.

    Raises UnsupportedModelError for anything else, because the weight count in
    compute_num_weight_params_read assumes Qwen3's exact layer structure.
    """
    model_type = config.get("model_type")
    if model_type != SUPPORTED_MODEL_TYPE:
        raise UnsupportedModelError(f"model_type {model_type!r} is not {SUPPORTED_MODEL_TYPE!r}")
    if config.get("attention_bias") is not False:
        raise UnsupportedModelError("attention_bias must be false; biases are not counted")
    if "quantization_config" in config:
        raise UnsupportedModelError("quantization_config present; only bf16 weights are modeled")

    return ModelDims(
        num_layers=_require_int(config, "num_hidden_layers"),
        hidden_size=_require_int(config, "hidden_size"),
        num_attention_heads=_require_int(config, "num_attention_heads"),
        num_kv_heads=_require_int(config, "num_key_value_heads"),
        head_dim=_require_int(config, "head_dim"),
        intermediate_size=_require_int(config, "intermediate_size"),
        vocab_size=_require_int(config, "vocab_size"),
    )


def load_model_dims(config_path: Path) -> ModelDims:
    config: dict[str, object] = json.loads(config_path.read_text())
    return parse_model_dims(config)


def compute_num_weight_params_read(dims: ModelDims) -> int:
    """Count the parameters one decode step reads: every layer, the final norm and lm_head.

    The input embedding is excluded because a step reads only batch_size rows of it. When
    embeddings are tied, the same table is read in full as lm_head, which is counted once here,
    so the count does not depend on tie_word_embeddings.
    """
    q_size = dims.num_attention_heads * dims.head_dim
    kv_size = dims.num_kv_heads * dims.head_dim

    attention_params = dims.hidden_size * q_size  # q_proj
    attention_params += 2 * dims.hidden_size * kv_size  # k_proj, v_proj
    attention_params += q_size * dims.hidden_size  # o_proj
    mlp_params = 3 * dims.hidden_size * dims.intermediate_size  # gate_proj, up_proj, down_proj
    norm_params = 2 * dims.hidden_size + 2 * dims.head_dim  # input/post-attention norms, q/k norms
    params_per_layer = attention_params + mlp_params + norm_params

    final_norm_params = dims.hidden_size
    lm_head_params = dims.vocab_size * dims.hidden_size
    return dims.num_layers * params_per_layer + final_norm_params + lm_head_params


def compute_kv_bytes(
    dims: ModelDims, *, batch_size: int, context_len: int, kv_element_bytes: int
) -> int:
    """Bytes of K and V read by one decode step across all sequences and layers."""
    kv_elements_per_token = dims.num_layers * 2 * dims.num_kv_heads * dims.head_dim
    return batch_size * context_len * kv_elements_per_token * kv_element_bytes


def predict_decode_step(
    dims: ModelDims,
    *,
    batch_size: int,
    context_len: int,
    kv_element_bytes: int,
    bandwidth_bytes_per_s: float,
) -> DecodeStepPrediction:
    """Fastest possible decode step if it only had to read weights and KV cache once.

    Weights are bf16. context_len is the number of tokens already in each sequence's KV cache.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")

    weight_bytes = compute_num_weight_params_read(dims) * BF16_BYTES
    kv_bytes = compute_kv_bytes(
        dims, batch_size=batch_size, context_len=context_len, kv_element_bytes=kv_element_bytes
    )
    step_time_s = (weight_bytes + kv_bytes) / bandwidth_bytes_per_s
    return DecodeStepPrediction(
        weight_bytes=weight_bytes,
        kv_bytes=kv_bytes,
        step_time_s=step_time_s,
        tokens_per_s=batch_size / step_time_s,
    )
