from pathlib import Path

import pytest

from decode_sol.profile_decode import ProfileConfig, ProfilerKind, to_vllm_profiler_config


@pytest.mark.parametrize("profiler", ["torch", "cuda"])
def test_vllm_accepts_the_profiler_config_for_each_profiler(profiler: ProfilerKind) -> None:
    config = ProfileConfig(
        model="Qwen/Qwen3-1.7B",
        kv_cache_dtype="auto",
        batch_size=1,
        context_len=512,
        profiler=profiler,
        output_dir=Path("/tmp/profiles"),
        gpu_memory_utilization=0.25,
        soak_minutes=0.0,
    )

    vllm_profiler_config = to_vllm_profiler_config(config)

    assert vllm_profiler_config.profiler == profiler
