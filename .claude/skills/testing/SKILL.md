---
name: testing
description: Writes pytest tests that catch regressions in Python, PyTorch and GPU code, including numerical comparisons against reference implementations. Use when adding a feature, fixing any bug, changing behavior, writing a kernel, or when asked to add tests.
---

# Testing

A test is worth having only if it fails when the behavior breaks. Coverage percentage is not the goal; catching the next regression is.

## Bug fixes start with a failing test

1. Write a test that reproduces the bug. Run it and watch it fail for the right reason.
2. Fix the code.
3. Watch the test pass.

A fix without a reproducing test is a guess.

## What to test

- **Behavior through the public interface**, not private helpers. A refactor that keeps behavior should not break a test.
- **The edges**: empty batch, batch of one, the maximum sequence length, just over it, non-power-of-two sizes, all-masked rows, `-inf` logits.
- **Every error path** a caller can hit, with `pytest.raises(SpecificError, match=...)`.

## Numerical tests

Optimized code is tested against a simple reference, never against hardcoded numbers copied from a previous run.

```python
@pytest.mark.gpu
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_rope_matches_reference(num_tokens: int, head_dim: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    query = torch.randn(num_tokens, NUM_HEADS, head_dim, dtype=dtype, device="cuda")
    positions = torch.arange(num_tokens, device="cuda")

    expected = rope_reference(query, positions)
    actual = fused_rope(query, positions)

    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])
```

- Keep tolerances in one table, set per dtype and justified (for example fp32 `atol=1e-5`, fp16 `atol=1e-3`, bf16 `atol=1e-2`). Never loosen a tolerance to make a test pass without explaining why in the commit.
- Parametrize over shapes and dtypes instead of writing one test per case.
- For end-to-end correctness, compare greedy outputs (or top-k logprobs) against a Hugging Face Transformers reference on a small model and a fixed set of prompts.

## Pick the right level

| Level | For | Where it runs |
| --- | --- | --- |
| Unit | Pure logic: scheduling decisions, block accounting, metric math, config parsing | Any machine |
| Kernel / op | A kernel against its reference, `torch.library.opcheck` | GPU, marked `gpu` |
| Integration | The engine on a tiny model; the server via an HTTP test client | GPU, marked `gpu` |

Most tests are unit tests of the pure core and run in seconds without a GPU. Mark every GPU test with `@pytest.mark.gpu` and skip it cleanly when CUDA is unavailable.

## Names are sentences

The test name states the behavior, so a failure reads like a bug report:

```python
def test_allocator_raises_out_of_blocks_when_cache_is_full() -> None: ...
def test_scheduler_preempts_newest_sequence_first() -> None: ...
```

Not `test_allocator`, `test_case_2` or `test_edge`.

## Structure

- Arrange, act, assert, separated by blank lines.
- One behavior per test.
- No branching logic in a test body; use `@pytest.mark.parametrize`.
- Shared setup goes in fixtures with clear names (`tiny_model`, `empty_allocator`). Expensive fixtures like a loaded model use `scope="module"` or `"session"`.

## Fakes over mocks

- Mock only at process boundaries: the network, the hub, the clock, randomness.
- Prefer a small fake with the same interface over a `MagicMock` that asserts which methods were called; interaction assertions break on every refactor.
- Never download models in unit tests. Integration tests use a small model cached in CI.

## Determinism

- Seed `torch`, `random` and `numpy` in every test that uses randomness.
- Freeze time with an injected clock, never `time.sleep`.
- Tests don't depend on each other or on run order.

A flaky test is a failing test. Fix it or delete it; never retry it into passing.

## Prove the test can fail

After writing a test, break the code on purpose: invert the condition you added, return early, or perturb the kernel output beyond the tolerance. Run the test. If it stays green, it doesn't test the change. Restore the code and strengthen the test.

## Checklist

- [ ] Bug fixes have a test that failed before the fix.
- [ ] Optimized code is compared against a reference implementation, parametrized over shapes and dtypes.
- [ ] Tolerances come from the shared per-dtype table.
- [ ] GPU tests are marked and skip cleanly without CUDA.
- [ ] Names describe behavior as sentences.
- [ ] Seeds set; no sleeps; no network in unit tests.
- [ ] Breaking the code on purpose makes at least one test fail.
