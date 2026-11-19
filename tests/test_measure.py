import pytest

from decode_sol.grid import GRID_BY_NAME, GridPoint
from decode_sol.measure import make_random_prompts, parse_grid_points


def test_random_prompts_have_exact_length_and_stay_below_the_token_id_limit() -> None:
    prompts = make_random_prompts(batch_size=3, context_len=512, num_token_ids=100, seed=0)

    assert len(prompts) == 3
    for prompt in prompts:
        token_ids = prompt["prompt_token_ids"]
        assert len(token_ids) == 512
        assert min(token_ids) >= 0
        assert max(token_ids) < 100


def test_random_prompts_differ_across_sequences_in_a_batch() -> None:
    prompts = make_random_prompts(batch_size=2, context_len=64, num_token_ids=1000, seed=0)

    assert prompts[0]["prompt_token_ids"] != prompts[1]["prompt_token_ids"]


def test_random_prompts_are_deterministic_for_a_seed() -> None:
    first = make_random_prompts(batch_size=2, context_len=64, num_token_ids=1000, seed=5)
    second = make_random_prompts(batch_size=2, context_len=64, num_token_ids=1000, seed=5)

    assert first == second


def test_grid_points_are_parsed_in_grid_order_without_duplicates() -> None:
    grid = GRID_BY_NAME["full"]

    points = parse_grid_points(["8x8192", "2x32768", "8x8192"], grid)

    assert points == (
        GridPoint(batch_size=8, context_len=8192),
        GridPoint(batch_size=2, context_len=32768),
    )


@pytest.mark.parametrize(
    ("point_spec", "message"),
    [("3x512", "not a point of the grid"), ("8-8192", "expected BATCHxCONTEXT")],
)
def test_grid_point_parsing_rejects_points_outside_the_grid(point_spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_grid_points([point_spec], GRID_BY_NAME["full"])
