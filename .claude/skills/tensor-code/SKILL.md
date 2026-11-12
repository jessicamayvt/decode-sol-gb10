---
name: tensor-code
description: Writes and reviews PyTorch, Triton and CUDA code for inference - shapes, dtypes, memory, host-device syncs, CUDA graphs, torch.compile and custom ops. Use whenever code creates, reshapes or computes on tensors, runs on a GPU, or adds a kernel.
---

# Tensor Code

On a GPU, the slow parts are rarely where they look. A single `.item()` in a decode loop can cost more than the matmul next to it. Write tensor code so its shapes are obvious, its memory is planned, and it never stalls the GPU by accident.

## Shapes and dtypes are part of the contract

- Every tensor created or reshaped gets a shape comment using the glossary's dimension names:
  `logits = hidden @ lm_head.T  # [num_seqs, vocab_size], fp32`
- Public functions document input and output shapes and dtypes in their docstring.
- Check shapes at the boundary of a public function, not inside the loop that calls it.
- Be explicit about dtype at every precision change. Accumulate reductions (softmax, layer norm, sums over long dimensions) in fp32 even when inputs are bf16 or fp16.
- Prefer named reshapes (`view(num_tokens, num_heads, head_dim)`) over `-1` when more than one dimension is inferred.

## Never sync the host in the hot path

These force the CPU to wait for the GPU and serialize the pipeline:

- `.item()`, `.tolist()`, `.cpu()`, `.numpy()`, `print(tensor)`.
- Python `if` on a tensor value (`if mask.any():`), `len()` of a data-dependent result, `torch.nonzero` without a size bound.
- `torch.cuda.synchronize()` outside benchmarking code.
- Creating a tensor from a Python list on the GPU every step (`torch.tensor(lst, device="cuda")`); build it pinned on the CPU and copy with `non_blocking=True`.

If a sync is truly needed, do it once per step, outside the per-layer code, and leave a comment saying why.

## Plan memory

- Preallocate buffers that are reused every step (the KV cache, sampling workspaces, metadata tensors) and write into them. Don't allocate per token.
- Know when an op copies: `.contiguous()`, `.reshape()` on non-contiguous input, advanced indexing, `torch.cat`. Avoid them in loops.
- Use in-place ops (`add_`, `copy_`, `out=`) only on buffers you own, never on a tensor autograd or another caller might still read.
- Run inference under `torch.inference_mode()`.
- Measure peak memory (`torch.cuda.max_memory_allocated()`) for any change that touches allocation, and record it with the benchmark.

## Vectorize, and name the steps

Python loops over tokens, heads or batch elements are a bug in the hot path. Replace them with batched tensor ops, and name each intermediate so the math stays readable:

```python
# Before: one kernel launch per sequence
for i in range(num_seqs):
    probs[i] = torch.softmax(logits[i] / temperatures[i], dim=-1)

# After: one launch for the batch
scaled_logits = logits / temperatures.unsqueeze(1)  # [num_seqs, vocab_size]
probs = torch.softmax(scaled_logits, dim=-1)         # [num_seqs, vocab_size], fp32
```

## Stay compatible with CUDA graphs and torch.compile

- Static shapes on the captured path: pad to a fixed set of batch sizes instead of producing a new shape every step.
- No data-dependent Python control flow, no host syncs, no allocation inside captured regions.
- Avoid graph breaks under `torch.compile`: no `print`, no `.item()`, no unsupported Python inside compiled functions. Check with `TORCH_LOGS=graph_breaks`.
- Keep the eager path working. Compiled and graph-captured paths must produce the same output as eager, and a test proves it.

## Kernels

Write a custom kernel only after a profile shows the op is a bottleneck (see `benchmarking`). Then:

1. **Reference first.** Write the plain PyTorch version and keep it in the repo next to the kernel. It is the specification.
2. **Test against the reference** across shapes (including non-power-of-two, size 1 and the largest supported) and dtypes, with tolerances set per dtype (see `testing`).
3. **Register it properly** as a PyTorch custom op with a schema, a fake/meta implementation for shape inference, and check the registration with `torch.library.opcheck`.
4. **Benchmark it** against the reference and against the best existing alternative (a fused PyTorch op, an existing vLLM or FlashInfer kernel), at the shapes that matter for real models.
5. Document the launch configuration choices (block size, number of warps, tiling) with the reason, since the code can't say why 128 beat 64.

Triton is the default for a new kernel. Drop to CUDA only when Triton can't express it or a profile shows a gap Triton can't close.

## Determinism and correctness

- Seed everything in tests and benchmarks: `torch.manual_seed`, `random.seed`, `numpy.random.seed`.
- Know which ops are non-deterministic on GPU (atomics in scatter, some reductions) and don't compare their outputs bit-for-bit.
- Check numerics on edge values: very long sequences, all-masked rows, `-inf` logits, empty batches.

## Checklist

- [ ] Every created or reshaped tensor has a shape comment.
- [ ] Reductions accumulate in fp32.
- [ ] No `.item()`, `.cpu()`, `.tolist()`, `print` or tensor-valued `if` in the hot path.
- [ ] No per-step allocation of buffers that could be preallocated.
- [ ] No Python loops over tokens, heads or batch elements.
- [ ] Captured and compiled paths match eager in a test.
- [ ] Any new kernel has a reference implementation, a comparison test, `opcheck` coverage and a benchmark.
