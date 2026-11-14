# Milestones

A summary of each stage: what was done, what was measured, and anything a reader needs to trace a result. One section is added at every checkpoint. Every number here comes from a file in `results/`.

## M0: Environment

**Commit:** `e375e1a`

- vLLM 0.33.0 (PyPI aarch64 wheel) with PyTorch 2.14.0+cu130 and Triton 3.8.0, on driver 580.178.04 and CUDA 13.0. The GB10 is sm_121 and runs PyTorch's sm_120 binaries.
- vLLM picks the `FLASH_ATTN` attention backend.
- One workaround: Triton compiles a C launcher at runtime and needs `Python.h`, which the machine lacks. The matching Ubuntu dev packages are unpacked into the venv and found through `CPATH` (`docs/SETUP.md`). It does not change performance.
- `torch.compile` and CUDA graphs (piecewise and full) both work. `enforce_eager` is **not** used.
- Smoke test: Qwen/Qwen3-0.6B generated text under `gpu_memory_utilization=0.25`, with 26.67 GiB of KV cache (249,712 tokens).

## M1: Instruments

**Commit:** `c64d342`

- `roofline.py` predicts the fastest decode step as (weight bytes + KV bytes) / bandwidth, from the model's `config.json`. Its weight count matches the Qwen3-0.6B checkpoint exactly (596,049,920 parameters read per step).
- `bandwidth.py`: achievable read bandwidth is **265.1 GB/s, 97.1% of the 273 GB/s spec**. Six repeats over two runs fall between 265.10 and 265.24 GB/s (`results/raw/bandwidth.jsonl`).
- `energy.py`: idle GPU power with no process on the GPU is **4.81 and 4.85 W** over two 30 s windows (`results/raw/idle_power.jsonl`). The NVML counter covers the GPU only and advances every 100 ms.
- `guard.py` enforces the shared-machine rules: the 0.25 memory cap, no other GPU users, and run metadata in every record.

## M2: First point

**Commits:** `11182e4` (code), `b8f6f20` (results)

Qwen/Qwen3-1.7B (revision `70d244c`), batch 1, context 512, bf16 weights and KV cache. `measure.py` generates the same prompt at 16 and 144 output tokens, alternately: 2 warmup pairs, then 10 timed pairs. The difference between the two is pure decode. The effective context over those steps is 592 tokens. The roofline formula's weight count matches the checkpoint exactly: 1,720,574,976 parameters read per step.

| | Step time | Tokens/s | Efficiency |
| --- | --- | --- | --- |
| Measured | 20.433 ms (95% CI 20.423–20.441) | 48.9 | |
| Ceiling at achievable 265.1 GB/s | 13.235 ms | 75.6 | 0.648 |
| Ceiling at spec 273 GB/s | 12.854 ms | 77.8 | 0.629 |

- **GPU energy per token:** 0.573 J including idle power, 0.357 J above idle. GPU power is about 28 W while decoding.
- **Idle with the model loaded:** 10.57 W, about twice the 4.8 W from M1. Energy "above idle" in this project subtracts this loaded baseline.
- **KV cache capacity:** 214,976 tokens under the 0.25 cap.
- **Sanity checks:** the measurement is below the ceiling and above 20% of it, so no investigation was needed. About 7.2 ms of each step goes to something other than reading memory; M4's profiling will look at it.

**Cross-check:** `vllm bench latency` at input 512, batch 1, 2 warmup + 10 iterations, output 16 and 144, differenced the same way, gives **20.48 ms per step**, 0.2% from `measure.py` (`results/raw/bench_latency_qwen3-1.7b_b1_c512_o*.json`). It samples at temperature 1.0 from token ids below 10,000, where `measure.py` is greedy over the full ordinary vocabulary.

**Tracing the records** in `results/raw/decode.jsonl`:

1. The first record is this point run before the code was committed (`has_uncommitted_changes: true`). It agrees with the second within 0.2%. It is kept because results are append-only.
2. The second record, the one in the table above, names commit `04ee59e`. That commit was later rewritten as `11182e4` to change only its message. The code is identical.
