import pytest

from decode_sol.profile_summary import (
    KernelLaunch,
    StepRange,
    split_into_decode_steps,
    summarize_decode_steps,
    to_kernel_family,
)


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
# and its LM head is launched after the range, the way vLLM launches it.
STEP_RANGES = [
    StepRange(start_ns=0, end_ns=10),
    StepRange(start_ns=100, end_ns=110),
    StepRange(start_ns=200, end_ns=210),
]
KERNELS = [
    make_kernel("matmul", launch_ns=5, start_ns=20, end_ns=80),
    make_kernel("lm_head", launch_ns=50, start_ns=80, end_ns=95),
    make_kernel("matmul", launch_ns=105, start_ns=120, end_ns=180),
    make_kernel("lm_head", launch_ns=150, start_ns=180, end_ns=190),
    make_kernel("matmul", launch_ns=205, start_ns=230, end_ns=290),
]


def test_decode_step_includes_kernels_launched_after_its_range() -> None:
    steps = split_into_decode_steps(KERNELS, STEP_RANGES)

    kernel_names_by_step = [[kernel.name for kernel in step.kernels] for step in steps]
    assert kernel_names_by_step == [["matmul", "lm_head"], ["matmul", "lm_head"]]


def test_last_step_range_only_ends_the_step_before_it() -> None:
    steps = split_into_decode_steps(KERNELS, STEP_RANGES)

    assert [(step.start_ns, step.end_ns) for step in steps] == [(20, 120), (120, 230)]


def test_split_raises_when_a_step_range_launches_no_kernels() -> None:
    empty_range = StepRange(start_ns=300, end_ns=310)

    with pytest.raises(ValueError, match="no kernels launched"):
        split_into_decode_steps(KERNELS, [*STEP_RANGES, empty_range])


def test_summary_splits_step_time_into_gpu_busy_and_idle() -> None:
    summary = summarize_decode_steps("toy", KERNELS, STEP_RANGES)

    # Step 1: window 100 ns, kernels 75 ns. Step 2: window 110 ns, kernels 70 ns.
    assert summary.num_decode_steps == 2
    assert summary.gpu_step_time_ms == pytest.approx(105e-6)
    assert summary.gpu_busy_ms == pytest.approx(72.5e-6)
    assert summary.gpu_idle_ms == pytest.approx(32.5e-6)
    assert summary.cpu_step_range_ms == pytest.approx(10e-6)


def test_top_kernels_are_ordered_by_gpu_time_with_per_step_counts() -> None:
    summary = summarize_decode_steps("toy", KERNELS, STEP_RANGES)

    matmul, lm_head = summary.top_kernels
    assert (matmul.name, lm_head.name) == ("matmul", "lm_head")
    assert matmul.num_launches_per_step == 1.0
    assert matmul.gpu_time_ms_per_step == pytest.approx(60e-6)
    assert matmul.fraction_of_gpu_busy_time + lm_head.fraction_of_gpu_busy_time == pytest.approx(
        1.0
    )
    assert matmul.fraction_of_gpu_busy_time == pytest.approx(120 / 145)


def test_summary_needs_at_least_two_step_ranges() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        summarize_decode_steps("toy", KERNELS, STEP_RANGES[:1])
