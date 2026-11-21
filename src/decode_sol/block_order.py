"""Read and reset the order of vLLM's free KV cache blocks.

With prefix caching off, vLLM puts freed blocks back at the front of its free queue, each
sequence's blocks last-first, and hands out new blocks from the front. So a long-running
process gives a decode batch the blocks of earlier batches, in an order that every batch of
a different size or length scatters further. A fresh process hands them out in block id
order.

get_block_pool reaches into vLLM's scheduler, so it needs the engine in this process:
VLLM_ENABLE_V1_MULTIPROCESSING=0 before the LLM is created.
"""

import random
from itertools import pairwise

from vllm import LLM
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from vllm.v1.engine.core_client import InprocClient


def get_block_pool(llm: LLM) -> BlockPool:
    engine_core_client = llm.llm_engine.engine_core
    if not isinstance(engine_core_client, InprocClient):
        raise RuntimeError(
            "the vLLM engine runs in another process; set VLLM_ENABLE_V1_MULTIPROCESSING=0"
            f" before creating the LLM (client is {type(engine_core_client).__name__})"
        )
    scheduler = engine_core_client.engine_core.scheduler
    return scheduler.kv_cache_manager.block_pool  # type: ignore[attr-defined, no-any-return]


def load_head_free_block_ids(block_pool: BlockPool, *, num_blocks: int) -> list[int]:
    """Ids of the first num_blocks free blocks, the ones the next batch will get."""
    free_blocks = block_pool.free_block_queue.get_all_free_blocks()
    head_block_ids: list[int] = []
    for block in free_blocks[:num_blocks]:
        head_block_ids.append(block.block_id)
    return head_block_ids


def compute_num_contiguous_runs(block_ids: list[int]) -> int:
    """Number of maximal runs in which each block id is next to the previous one, either way.

    A fresh queue is one run. Blocks that are adjacent in memory, ascending or descending,
    stay in the same run; any jump starts a new one.
    """
    if not block_ids:
        return 0
    num_runs = 1
    for previous_id, block_id in pairwise(block_ids):
        if abs(block_id - previous_id) != 1:
            num_runs += 1
    return num_runs


def replace_free_block_queue(block_pool: BlockPool, ordered_blocks: list[KVCacheBlock]) -> None:
    num_free_blocks = block_pool.free_block_queue.num_free_blocks
    if len(ordered_blocks) != num_free_blocks:
        raise ValueError(f"got {len(ordered_blocks)} blocks for a queue of {num_free_blocks}")
    block_pool.free_block_queue = FreeKVCacheBlockQueue(ordered_blocks)


def sort_free_block_queue(block_pool: BlockPool) -> None:
    """Put the free blocks back in block id order, as in a fresh process."""
    free_blocks = block_pool.free_block_queue.get_all_free_blocks()
    sorted_blocks = sorted(free_blocks, key=lambda block: block.block_id)
    replace_free_block_queue(block_pool, sorted_blocks)


def shuffle_free_block_queue(block_pool: BlockPool, *, seed: int) -> None:
    free_blocks = block_pool.free_block_queue.get_all_free_blocks()
    random.Random(seed).shuffle(free_blocks)
    replace_free_block_queue(block_pool, free_blocks)
