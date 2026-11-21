import pytest
from vllm.v1.core.block_pool import BlockPool

from decode_sol.block_order import (
    compute_num_contiguous_runs,
    load_head_free_block_ids,
    shuffle_free_block_queue,
    sort_free_block_queue,
)

NUM_GPU_BLOCKS = 64
BLOCK_SIZE = 16


def make_block_pool() -> BlockPool:
    return BlockPool(
        num_gpu_blocks=NUM_GPU_BLOCKS, enable_caching=False, hash_block_size=BLOCK_SIZE
    )


def free_last_first(block_pool: BlockPool, block_ids: list[int]) -> None:
    """Free blocks the way vLLM frees a finished sequence: its last block first."""
    blocks = []
    for block_id in reversed(block_ids):
        blocks.append(block_pool.blocks[block_id])
    block_pool.free_blocks(blocks)


@pytest.mark.parametrize(
    ("block_ids", "expected_num_runs"),
    [
        ([], 0),
        ([5], 1),
        ([1, 2, 3, 4], 1),
        ([4, 3, 2, 1], 1),
        ([1, 2, 3, 10, 11, 12], 2),
        ([1, 3, 5, 7], 4),
        ([3, 2, 1, 7, 8], 2),
    ],
)
def test_contiguous_runs_count_jumps_between_non_adjacent_blocks(
    block_ids: list[int], expected_num_runs: int
) -> None:
    assert compute_num_contiguous_runs(block_ids) == expected_num_runs


def test_fresh_pool_hands_out_blocks_in_id_order() -> None:
    block_pool = make_block_pool()

    head_block_ids = load_head_free_block_ids(block_pool, num_blocks=8)

    # Block 0 is vLLM's null block, never handed out.
    assert head_block_ids == [1, 2, 3, 4, 5, 6, 7, 8]


def test_freeing_two_sequences_scatters_the_head_and_sorting_restores_it() -> None:
    block_pool = make_block_pool()
    first_sequence_block_ids = [block.block_id for block in block_pool.get_new_blocks(4)]
    second_sequence_block_ids = [block.block_id for block in block_pool.get_new_blocks(4)]
    free_last_first(block_pool, first_sequence_block_ids)
    free_last_first(block_pool, second_sequence_block_ids)
    scattered_block_ids = load_head_free_block_ids(block_pool, num_blocks=10)

    sort_free_block_queue(block_pool)

    assert scattered_block_ids == [8, 7, 6, 5, 4, 3, 2, 1, 9, 10]
    assert load_head_free_block_ids(block_pool, num_blocks=10) == list(range(1, 11))
    assert block_pool.get_num_free_blocks() == NUM_GPU_BLOCKS - 1


def test_shuffle_keeps_every_free_block_and_is_seeded() -> None:
    first_pool = make_block_pool()
    second_pool = make_block_pool()

    shuffle_free_block_queue(first_pool, seed=0)
    shuffle_free_block_queue(second_pool, seed=0)

    first_block_ids = load_head_free_block_ids(first_pool, num_blocks=NUM_GPU_BLOCKS)
    second_block_ids = load_head_free_block_ids(second_pool, num_blocks=NUM_GPU_BLOCKS)
    assert first_block_ids == second_block_ids
    assert sorted(first_block_ids) == list(range(1, NUM_GPU_BLOCKS))
    assert compute_num_contiguous_runs(first_block_ids) > NUM_GPU_BLOCKS // 2
    assert first_pool.get_new_blocks(1)[0].block_id == first_block_ids[0]
