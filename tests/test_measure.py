from decode_sol.measure import make_random_prompts


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
