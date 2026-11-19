"""Time decode in vLLM over a grid of points, with GPU energy, and append one raw record each.

The model is loaded once per process. At every point the same batch is generated at a
short and a long output length, alternately. Both pay the same prefill, so analyze.py
takes their difference as pure decode.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from decode_sol.energy import (
    IDLE_DURATION_S,
    GpuEnergyWindow,
    measure_gpu_energy,
    measure_idle_gpu_energy,
    open_nvml_device,
)
from decode_sol.grid import (
    GRID_BY_NAME,
    LONG_OUTPUT_LEN,
    NUM_TIMED_PAIRS,
    NUM_WARMUP_PAIRS,
    SHORT_OUTPUT_LEN,
    Grid,
    GridPoint,
    find_skip_reason,
    load_model_config,
)
from decode_sol.guard import collect_run_metadata, ensure_gpu_is_free, parse_gpu_memory_utilization

logger = logging.getLogger(__name__)

PROMPT_SEED = 0
WEIGHT_DTYPE: Final = "bfloat16"


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """One process: a model at one KV cache dtype over a named grid.

    The engine is always sized for the whole grid; points can be a subset of it, to top up
    points that did not fit in earlier sweeps.
    """

    model: str
    grid_name: str
    kv_cache_dtype: str
    gpu_memory_utilization: float
    output: Path
    points: tuple[GridPoint, ...]

    @property
    def grid(self) -> Grid:
        return GRID_BY_NAME[self.grid_name]


def parse_grid_points(point_specs: list[str], grid: Grid) -> tuple[GridPoint, ...]:
    """Parse BATCHxCONTEXT specs such as 8x8192 into points, in the grid's order."""
    requested_points: set[GridPoint] = set()
    for point_spec in point_specs:
        batch_size_text, separator, context_len_text = point_spec.partition("x")
        if not separator:
            raise ValueError(f"expected BATCHxCONTEXT such as 8x8192, got {point_spec!r}")
        point = GridPoint(batch_size=int(batch_size_text), context_len=int(context_len_text))
        if point not in grid.points:
            raise ValueError(f"{point_spec!r} is not a point of the grid")
        requested_points.add(point)

    points: list[GridPoint] = []
    for point in grid.points:
        if point in requested_points:
            points.append(point)
    return tuple(points)


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


def measure_point(
    llm: LLM, point: GridPoint, *, num_token_ids: int, device: object
) -> list[dict[str, dict[str, int]]]:
    """Warm up, then time NUM_TIMED_PAIRS pairs of the batch at one point."""
    prompts = make_random_prompts(
        batch_size=point.batch_size,
        context_len=point.context_len,
        num_token_ids=num_token_ids,
        seed=PROMPT_SEED,
    )
    time_paired_generations(llm, prompts, num_pairs=NUM_WARMUP_PAIRS, device=device)
    return time_paired_generations(llm, prompts, num_pairs=NUM_TIMED_PAIRS, device=device)


def load_llm(config: SweepConfig, *, revision: str) -> LLM:
    """Load the model sized for the grid's largest batch and longest sequence."""
    return LLM(
        model=config.model,
        revision=revision,
        dtype=WEIGHT_DTYPE,
        kv_cache_dtype=config.kv_cache_dtype,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.grid.max_model_len,
        max_num_seqs=config.grid.max_batch_size,
        max_num_batched_tokens=config.grid.max_num_batched_tokens,
        enable_prefix_caching=False,
        seed=PROMPT_SEED,
    )


def collect_sweep_metadata(
    config: SweepConfig, llm: LLM, *, revision: str, num_gpu_blocks: int
) -> dict[str, object]:
    """Everything a record shares with the other records of the same process."""
    cache_config = llm.llm_engine.vllm_config.cache_config
    return {
        **collect_run_metadata(),
        "model": config.model,
        "model_revision": revision,
        "model_config": load_model_config(config.model, revision),
        "weight_dtype": WEIGHT_DTYPE,
        "kv_cache_dtype": config.kv_cache_dtype,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "grid": config.grid_name,
        "requested_points": [[point.batch_size, point.context_len] for point in config.points],
        "max_model_len": config.grid.max_model_len,
        "max_num_seqs": config.grid.max_batch_size,
        "max_num_batched_tokens": config.grid.max_num_batched_tokens,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "kv_cache_size_tokens": cache_config.kv_cache_size_tokens,
        "num_gpu_blocks": num_gpu_blocks,
        "block_size": cache_config.block_size,
        "short_output_len": SHORT_OUTPUT_LEN,
        "long_output_len": LONG_OUTPUT_LEN,
        "prompt_seed": PROMPT_SEED,
        "num_warmup_pairs": NUM_WARMUP_PAIRS,
    }


def append_record(record: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as output_file:
        output_file.write(json.dumps(record) + "\n")


def parse_sweep_config() -> SweepConfig:
    parser = argparse.ArgumentParser(description="Measure vLLM decode over a grid of points.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--grid", choices=sorted(GRID_BY_NAME), required=True)
    parser.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("results/raw/decode.jsonl"))
    parser.add_argument(
        "--points",
        nargs="+",
        metavar="BATCHxCONTEXT",
        help="measure only these grid points (default: the whole grid)",
    )
    args = parser.parse_args()
    grid = GRID_BY_NAME[args.grid]
    points = tuple(grid.points) if args.points is None else parse_grid_points(args.points, grid)
    return SweepConfig(
        model=args.model,
        grid_name=args.grid,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=parse_gpu_memory_utilization(args.gpu_memory_utilization),
        output=args.output,
        points=points,
    )


def main() -> None:
    config = parse_sweep_config()
    logging.basicConfig(level=logging.INFO)

    ensure_gpu_is_free()
    revision = HfApi().model_info(config.model).sha
    if revision is None:
        raise RuntimeError(f"Hugging Face returned no revision for {config.model!r}")
    llm = load_llm(config, revision=revision)
    cache_config = llm.llm_engine.vllm_config.cache_config
    num_gpu_blocks = cache_config.num_gpu_blocks
    if num_gpu_blocks is None:
        raise RuntimeError("vLLM did not report its KV cache size")
    num_token_ids = llm.get_tokenizer().vocab_size  # ordinary tokens only, no special tokens
    sweep_metadata = collect_sweep_metadata(
        config, llm, revision=revision, num_gpu_blocks=num_gpu_blocks
    )

    with open_nvml_device() as device:
        # Idle with the model loaded: the baseline every decode window sits on top of.
        idle_window = measure_idle_gpu_energy(device, duration_s=IDLE_DURATION_S)
        for point in config.points:
            skip_reason = find_skip_reason(
                point,
                num_gpu_blocks=num_gpu_blocks,
                block_size=cache_config.block_size,
                max_num_batched_tokens=config.grid.max_num_batched_tokens,
            )
            if skip_reason is not None:
                logger.warning("Skipping %s: %s", point, skip_reason)
                continue
            raw_pairs = measure_point(llm, point, num_token_ids=num_token_ids, device=device)
            append_record(
                {
                    **sweep_metadata,
                    "batch_size": point.batch_size,
                    "context_len": point.context_len,
                    "idle_gpu_energy_window": asdict(idle_window),
                    "idle_gpu_power_w": idle_window.mean_power_w,
                    "paired_generations": raw_pairs,
                },
                config.output,
            )
            logger.info("Appended %d paired generations for %s", len(raw_pairs), point)


if __name__ == "__main__":
    main()
