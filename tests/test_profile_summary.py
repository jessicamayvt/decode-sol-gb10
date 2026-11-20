import pytest

from decode_sol.profile_summary import (
    KernelLaunch,
    ProfiledPoint,
    StepRange,
    compute_bytes_per_decode_step,
    parse_profiled_point,
    split_into_decode_steps,
    summarize_decode_steps,
    to_kernel_family,
)
from decode_sol.roofline import ModelDims

MATMUL = "cutlass::Kernel2"
ATTENTION = "kernel_mha"


@pytest.mark.parametrize(
    ("demangled_name", "expected_family"),
    [
        (
            "std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv_bfloat16, "
            "float, (bool)0>(T13)",
            "internal::gemvx::kernel",
        ),
        (
            "void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128, (int)64>, "
            "(bool)0>(Flash_fwd_params)",
            "flash::flash_fwd_splitkv_kernel",
        ),
        (
            "void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out("
            "const c10::Scalar &)>(T1, T2)",
            "elementwise_kernel_with_index",
        ),
        (
            "nvjet_sm121_tst_mma_128x144x64_2_32x72x64_tmaAB_bz_TNNN",
            "nvjet_sm121_tst_mma_128x144x64_2_32x72x64_tmaAB_bz_TNNN",
        ),
        ("triton_poi_fused_mul_silu_slice_1", "triton_poi_fused_mul_silu_slice_1"),
    ],
)
def test_kernel_family_is_the_qualified_name_without_templates_or_params(
    demangled_name: str, expected_family: str
) -> None:
    assert to_kernel_family(demangled_name) == expected_family


def make_kernel(name: str, *, launch_ns: int, start_ns: int, end_ns: int) -> KernelLaunch:
    return KernelLaunch(name=name, launch_ns=launch_ns, start_ns=start_ns, end_ns=end_ns)


# Three step ranges on the CPU. Each step's matmul runs on the GPU after its range has closed,
# and its attention kernel is launched after the range, the way vLLM launches the LM head.
STEP_RANGES = [
    StepRange(start_ns=0, end_ns=10),
    StepRange(start_ns=100, end_ns=110),
    StepRange(start_ns=200, end_ns=210),
]
KERNELS = [
    make_kernel(MATMUL, launch_ns=5, start_ns=20, end_ns=80),
    make_kernel(ATTENTION, launch_ns=50, start_ns=80, end_ns=95),
    make_kernel(MATMUL, launch_ns=105, start_ns=120, end_ns=180),
    make_kernel(ATTENTION, launch_ns=150, start_ns=180, end_ns=190),
    make_kernel(MATMUL, launch_ns=205, start_ns=230, end_ns=290),
]
TOY_WEIGHT_BYTES = 6_000
TOY_KV_BYTES = 1_250


def test_decode_step_includes_kernels_launched_after_its_range() -> None:
    steps = split_into_decode_steps(KERNELS, STEP_RANGES)

    kernel_names_by_step = [[kernel.name for kernel in step.kernels] for step in steps]
    assert kernel_names_by_step == [[MATMUL, ATTENTION], [MATMUL, ATTENTION]]


def test_last_step_range_only_ends_the_step_before_it() -> None:
    steps = split_into_decode_steps(KERNELS, STEP_RANGES)

    assert [(step.start_ns, step.end_ns) for step in steps] == [(20, 120), (120, 230)]


def test_split_raises_when_a_step_range_launches_no_kernels() -> None:
    empty_range = StepRange(start_ns=300, end_ns=310)

    with pytest.raises(ValueError, match="no kernels launched"):
        split_into_decode_steps(KERNELS, [*STEP_RANGES, empty_range])


def test_summary_splits_step_time_into_gpu_busy_and_idle() -> None:
    summary = summarize_decode_steps(
        "toy", KERNELS, STEP_RANGES, weight_bytes=TOY_WEIGHT_BYTES, kv_bytes=TOY_KV_BYTES
    )

    # Step 1: window 100 ns, kernels 75 ns. Step 2: window 110 ns, kernels 70 ns.
    assert summary.num_decode_steps == 2
    assert summary.gpu_step_time_ms == pytest.approx(105e-6)
    assert summary.gpu_busy_ms == pytest.approx(72.5e-6)
    assert summary.gpu_idle_ms == pytest.approx(32.5e-6)
    assert summary.cpu_step_range_ms == pytest.approx(10e-6)


def test_top_kernels_are_ordered_by_gpu_time_with_per_step_counts() -> None:
    summary = summarize_decode_steps(
        "toy", KERNELS, STEP_RANGES, weight_bytes=TOY_WEIGHT_BYTES, kv_bytes=TOY_KV_BYTES
    )

    matmul, attention = summary.top_kernels
    assert (matmul.name, attention.name) == (MATMUL, ATTENTION)
    assert matmul.num_launches_per_step == 1.0
    assert matmul.gpu_time_ms_per_step == pytest.approx(60e-6)
    assert matmul.fraction_of_gpu_busy_time + attention.fraction_of_gpu_busy_time == pytest.approx(
        1.0
    )
    assert matmul.fraction_of_gpu_busy_time == pytest.approx(120 / 145)


def test_summary_needs_at_least_two_step_ranges() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize_decode_steps(
            "toy", KERNELS, STEP_RANGES[:1], weight_bytes=TOY_WEIGHT_BYTES, kv_bytes=TOY_KV_BYTES
        )


def test_read_bandwidths_are_bytes_over_the_gpu_time_of_their_kernels() -> None:
    summary = summarize_decode_steps(
        "toy", KERNELS, STEP_RANGES, weight_bytes=TOY_WEIGHT_BYTES, kv_bytes=TOY_KV_BYTES
    )

    # Matmul: 60 ns per step, 6,000 bytes. Attention: 15 and 10 ns, so 12.5 ns, 1,250 bytes.
    assert summary.weight_read.gpu_time_ms_per_step == pytest.approx(60e-6)
    assert summary.weight_read.bandwidth_gb_per_s == pytest.approx(100.0)
    assert summary.kv_cache_read.gpu_time_ms_per_step == pytest.approx(12.5e-6)
    assert summary.kv_cache_read.bandwidth_gb_per_s == pytest.approx(100.0)
    assert summary.kv_cache_read.num_bytes_per_step == TOY_KV_BYTES


def test_summary_raises_when_no_attention_kernel_ran() -> None:
    matmul_only = [kernel for kernel in KERNELS if kernel.name == MATMUL]

    with pytest.raises(ValueError, match="no kernel of"):
        summarize_decode_steps(
            "toy", matmul_only, STEP_RANGES, weight_bytes=TOY_WEIGHT_BYTES, kv_bytes=TOY_KV_BYTES
        )


@pytest.mark.parametrize("profile", ["Qwen3-4B_fp8_b4_c32768", "Qwen3-4B_fp8_b4_c32768_kepler"])
def test_profiled_point_parses_with_or_without_a_label(profile: str) -> None:
    point = parse_profiled_point(profile)

    assert point == ProfiledPoint(
        model_name="Qwen3-4B", kv_cache_dtype="fp8", batch_size=4, context_len=32768
    )


@pytest.mark.parametrize("profile", ["Qwen3-4B_bf16_b4_c32768", "Qwen3-4B_fp8_c32768", "notes"])
def test_parse_profiled_point_rejects_other_names(profile: str) -> None:
    with pytest.raises(ValueError, match="is not <model>"):
        parse_profiled_point(profile)


# Weights: 1280 parameters at 2 bytes (tests/test_roofline.py counts them by hand). KV cache:
# 3 sequences * 5 tokens * 2 layers * (K and V) * 2 kv heads * head_dim 2 = 240 elements.
TINY_DIMS = ModelDims(
    num_layers=2,
    hidden_size=8,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=2,
    intermediate_size=16,
    vocab_size=10,
)


@pytest.mark.parametrize(("kv_cache_dtype", "expected_kv_bytes"), [("auto", 480), ("fp8", 240)])
def test_bytes_per_decode_step_use_bf16_weights_and_the_kv_cache_dtype(
    kv_cache_dtype: str, expected_kv_bytes: int
) -> None:
    point = ProfiledPoint(
        model_name="tiny", kv_cache_dtype=kv_cache_dtype, batch_size=3, context_len=5
    )

    weight_bytes, kv_bytes = compute_bytes_per_decode_step(point, TINY_DIMS)

    assert (weight_bytes, kv_bytes) == (2560, expected_kv_bytes)
