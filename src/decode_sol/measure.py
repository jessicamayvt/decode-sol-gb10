"""Time decode in vLLM at one grid point, with GPU energy, and append a raw record.

The same batch is generated at a short and a long output length, alternately. Both pay
the same prefill, so analyze.py takes their difference as pure decode.
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Final

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from decode_sol.energy import (
    IDLE_DURATION_S,
    GpuEnergyWindow,
    measure_gpu_energy,
    measure_idle_gpu_energy,
    open_nvml_device,
)
from decode_sol.guard import collect_run_metadata, ensure_gpu_is_free, parse_gpu_memory_utilization

logger = logging.getLogger(__name__)

SHORT_OUTPUT_LEN = 16
LONG_OUTPUT_LEN = 144
NUM_WARMUP_PAIRS = 2
NUM_TIMED_PAIRS = 10
PROMPT_SEED = 0
WEIGHT_DTYPE: Final = "bfloat16"


def make_random_prompts(
    *, batch_size: int, context_len: int, num_token_ids: int, seed: int
) -> list[TokensPrompt]:
    """Exact-length prompts of random token ids in [0, num_token_ids), one per sequence."""
    rng = np.random.default_rng(seed)
    token_ids = rng.integers(0, num_token_ids, size=(batch_size, context_len))
    prompts: list[TokensPrompt] = []
    for sequence_token_ids in token_ids.tolist():
        prompts.append(TokensPrompt(prompt_token_ids=sequence_token_ids))
    return prompts


def time_generation(
    llm: LLM, prompts: list[TokensPrompt], *, output_len: int, device: object
) -> GpuEnergyWindow:
    """Generate exactly output_len tokens per sequence, timed and with GPU energy.

    llm.generate returns only after the GPU has finished, so the window covers all its work.
    """
    sampling_params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    with measure_gpu_energy(device) as window:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    for output in outputs:
        num_output_tokens = len(output.outputs[0].token_ids)
        if num_output_tokens != output_len:
            raise RuntimeError(f"expected {output_len} output tokens, got {num_output_tokens}")
    return window


def time_paired_generations(
    llm: LLM, prompts: list[TokensPrompt], *, num_pairs: int, device: object
) -> list[dict[str, dict[str, int]]]:
    """Alternate short and long generations, so slow drift affects both alike."""
    raw_pairs: list[dict[str, dict[str, int]]] = []
    for _ in range(num_pairs):
        short_window = time_generation(llm, prompts, output_len=SHORT_OUTPUT_LEN, device=device)
        long_window = time_generation(llm, prompts, output_len=LONG_OUTPUT_LEN, device=device)
        raw_pairs.append({"short": asdict(short_window), "long": asdict(long_window)})
    return raw_pairs


def load_model_config(model: str, revision: str) -> dict[str, object]:
    config_path = hf_hub_download(model, "config.json", revision=revision)
    config: dict[str, object] = json.loads(Path(config_path).read_text())
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure vLLM decode at one grid point.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("results/raw/decode.jsonl"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    gpu_memory_utilization = parse_gpu_memory_utilization(args.gpu_memory_utilization)

    ensure_gpu_is_free()
    record = collect_run_metadata()
    revision = HfApi().model_info(args.model).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face returned no revision for {args.model!r}")
    max_model_len = args.context_len + LONG_OUTPUT_LEN
    llm = LLM(
        model=args.model,
        revision=revision,
        dtype=WEIGHT_DTYPE,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        seed=PROMPT_SEED,
    )
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_token_ids = llm.get_tokenizer().vocab_size  # ordinary tokens only, no special tokens
    prompts = make_random_prompts(
        batch_size=args.batch_size,
        context_len=args.context_len,
        num_token_ids=num_token_ids,
        seed=PROMPT_SEED,
    )

    with open_nvml_device() as device:
        # Idle with the model loaded: the baseline the decode windows sit on top of.
        idle_window = measure_idle_gpu_energy(device, duration_s=IDLE_DURATION_S)
        time_paired_generations(llm, prompts, num_pairs=NUM_WARMUP_PAIRS, device=device)
        raw_pairs = time_paired_generations(llm, prompts, num_pairs=NUM_TIMED_PAIRS, device=device)

    record.update(
        {
            "model": args.model,
            "model_revision": revision,
            "model_config": load_model_config(args.model, revision),
            "weight_dtype": WEIGHT_DTYPE,
            "kv_cache_dtype": args.kv_cache_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "enable_prefix_caching": False,
            "enforce_eager": False,
            "kv_cache_size_tokens": cache_config.kv_cache_size_tokens,
            "num_gpu_blocks": cache_config.num_gpu_blocks,
            "block_size": cache_config.block_size,
            "batch_size": args.batch_size,
            "context_len": args.context_len,
            "short_output_len": SHORT_OUTPUT_LEN,
            "long_output_len": LONG_OUTPUT_LEN,
            "prompt_seed": PROMPT_SEED,
            "num_warmup_pairs": NUM_WARMUP_PAIRS,
            "idle_gpu_energy_window": asdict(idle_window),
            "idle_gpu_power_w": idle_window.mean_power_w,
            "paired_generations": raw_pairs,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output_file:
        output_file.write(json.dumps(record) + "\n")
    logger.info("Appended %d paired generations to %s", len(raw_pairs), args.output)


if __name__ == "__main__":
    main()
