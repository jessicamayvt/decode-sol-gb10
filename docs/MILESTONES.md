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

## M3: Sweep (in progress)

**Commits:** `1c52d03` (sweep tooling), `59f84c5` (warmup fix), `32f512b`, `40d8afd`, `e634e6e` (repeat 1 results)

The grid is Qwen3-1.7B, 4B and 8B (bf16 weights) × KV cache dtype auto (bf16) and fp8 × batch 1–64 × context 512, 2048, 8192 and 32768: 6 sweeps of 28 points each, run 3 times. `python -m decode_sol.grid` printed the feasible points and estimated 689 GPU minutes per repeat before anything ran. Each sweep runs through `scripts/run_sweep.sh` under the 0.25 memory cap. `measure.py` skips any point whose KV cache won't fit in the blocks vLLM actually allocates and logs why.

### Repeat 1: complete

| Model | KV dtype | Points | KV cache blocks (planned / allocated) | Efficiency range | b1, c512 step time | Energy/token above idle (b1, c512) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | bf16 | 22 / 22 | 12,402 / 13,296 | 0.678–0.934 | 19.353 ms | 0.323 J |
| Qwen3-1.7B | fp8 | 24 / 24 | 24,804 / 24,881 | 0.672–0.869 | 19.510 ms | 0.317 J |
| Qwen3-4B | bf16 | 19 / 19 | 7,694 / 7,934 | 0.723–0.923 | 42.040 ms | 0.997 J |
| Qwen3-4B | fp8 | 22 / 22 | 15,389 / 15,825 | 0.722–0.912 | 42.096 ms | 1.085 J |
| Qwen3-8B | bf16 | **16 / 19** | 4,688 / **3,840** | 0.828–0.914 | 69.322 ms | 1.777 J |
| Qwen3-8B | fp8 | 22 / 22 | 9,377 / 9,202 | 0.808–0.898 | 68.960 ms | 1.756 J |

"Points" is measured / feasible according to the planner. Efficiency uses the achievable 265.1 GB/s ceiling and the median step time, as in M2. The bootstrap CIs are in the raw records.

- **Sanity checks pass.** No point beats the ceiling, and none falls below 20% of it. Every record has `has_uncommitted_changes: false`. Each record lists one other GPU process at start, which is the sweep's own vLLM engine subprocess.
- **The planner's KV-capacity estimate is off in both directions.** It was 18% too high for Qwen3-8B bf16, so 3 points were skipped at runtime: (32, 2048), (8, 8192) and (2, 32768). It was also 2% too high for Qwen3-8B fp8, but no point needed those blocks. For the other four sweeps it was 0.3–7% too low. Allocated capacity also changes from run to run. Qwen3-4B bf16 got 7,934 blocks in repeat 1 and 8,219 in repeat 2.
- **Differs from M2:** Qwen3-1.7B bf16 at batch 1, context 512 measured 19.353 ms here and 20.433 ms in M2, 5.3% faster. M4 should check whether this is run-to-run variance or the code change between them.

### Repeats 2 and 3: running

They run one sweep after another from 2026-11-18. The first sweep of repeat 2, Qwen3-1.7B bf16, ran at 06:19 UTC at `40d8afd`. The rest of the queue started at 18:01 UTC. These rows are not committed yet. They are told apart from repeat 1 by `timestamp_utc`.
