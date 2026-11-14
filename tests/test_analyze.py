import pytest

from decode_sol.analyze import (
    PairedGeneration,
    compute_decode_energy_per_token_j,
    compute_decode_step_time_s,
    compute_effective_context_len,
    compute_median_and_bootstrap_ci,
)

# A batch of 2, generating 16 then 144 tokens per sequence: 128 extra decode steps, 256 extra
# tokens. The long run takes 1.28 s more and uses 12.8 J more, so a step is 10 ms and the
# GPU averaged 10 W over the extra time.
HAND_PAIR = PairedGeneration(
    short_duration_s=0.5,
    long_duration_s=1.78,
    short_energy_j=4.0,
    long_energy_j=16.8,
)


def test_decode_step_time_is_extra_time_over_extra_steps() -> None:
    step_time_s = compute_decode_step_time_s(HAND_PAIR, short_output_len=16, long_output_len=144)

    assert step_time_s == pytest.approx(0.01)


def test_decode_energy_per_token_divides_by_every_sequence_in_the_batch() -> None:
    energy_per_token_j = compute_decode_energy_per_token_j(
        HAND_PAIR, batch_size=2, short_output_len=16, long_output_len=144, idle_power_w=0.0
    )

    assert energy_per_token_j == pytest.approx(12.8 / 256)


def test_decode_energy_per_token_subtracts_idle_power_over_the_extra_time() -> None:
    # Idle 5 W over the extra 1.28 s is 6.4 J, leaving 6.4 J for 256 tokens.
    energy_per_token_j = compute_decode_energy_per_token_j(
        HAND_PAIR, batch_size=2, short_output_len=16, long_output_len=144, idle_power_w=5.0
    )

    assert energy_per_token_j == pytest.approx(6.4 / 256)


def test_effective_context_is_prompt_plus_mean_of_output_lengths() -> None:
    effective_context_len = compute_effective_context_len(
        context_len=512, short_output_len=16, long_output_len=144
    )

    assert effective_context_len == 592


def test_median_and_ci_of_identical_samples_collapse_to_that_value() -> None:
    summary = compute_median_and_bootstrap_ci([3.0, 3.0, 3.0, 3.0, 3.0], seed=0)

    assert summary.median == 3.0
    assert summary.ci_low == 3.0
    assert summary.ci_high == 3.0


def test_bootstrap_ci_brackets_the_median_and_stays_inside_the_samples() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 10.0]

    summary = compute_median_and_bootstrap_ci(samples, seed=0)

    assert summary.median == 3.0
    assert 1.0 <= summary.ci_low <= 3.0 <= summary.ci_high <= 10.0
    assert summary.ci_low < summary.ci_high


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    samples = [1.0, 2.0, 3.0, 4.0, 10.0]

    first = compute_median_and_bootstrap_ci(samples, seed=7)
    second = compute_median_and_bootstrap_ci(samples, seed=7)

    assert first == second
