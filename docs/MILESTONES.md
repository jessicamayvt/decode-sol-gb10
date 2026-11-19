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

## M3: Sweep

**Commits:** `1c52d03` (sweep tooling), `59f84c5` (warmup fix), `32f512b`, `40d8afd`, `e634e6e` (repeat 1 results), `320f285` (repeat 2 results), `5553bab` (repeat 3 results)

The grid is Qwen3-1.7B, 4B and 8B (bf16 weights) × KV cache dtype auto (bf16) and fp8 × batch 1–64 × context 512, 2048, 8192 and 32768: 6 sweeps of 28 points each, run 3 times. `python -m decode_sol.grid` printed the feasible points and estimated 689 GPU minutes per repeat before anything ran. Each sweep ran through `scripts/run_sweep.sh` under the 0.25 memory cap. `measure.py` skips any point whose KV cache won't fit in the blocks vLLM actually allocates and logs why.

Repeat 1 ran on 2026-11-17 and 18. Repeats 2 and 3 ran one sweep after another from 2026-11-18 06:19 UTC to 2026-11-19 07:20 UTC, about 0.7× the planner's time estimate. In total there are **374 records over 128 distinct grid points**. Repeats are told apart by `timestamp_utc`.

### Repeat 1

| Model | KV dtype | Points | KV cache blocks (planned / allocated) | Efficiency range | b1, c512 step time | GPU energy/token above idle (b1, c512) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | bf16 | 22 / 22 | 12,402 / 13,296 | 0.678–0.934 | 19.353 ms | 0.323 J |
| Qwen3-1.7B | fp8 | 24 / 24 | 24,804 / 24,881 | 0.672–0.869 | 19.510 ms | 0.317 J |
| Qwen3-4B | bf16 | 19 / 19 | 7,694 / 7,934 | 0.723–0.923 | 42.040 ms | 0.997 J |
| Qwen3-4B | fp8 | 22 / 22 | 15,389 / 15,825 | 0.722–0.912 | 42.096 ms | 1.085 J |
| Qwen3-8B | bf16 | **16 / 19** | 4,688 / **3,840** | 0.828–0.914 | 69.322 ms | 1.777 J |
| Qwen3-8B | fp8 | 22 / 22 | 9,377 / 9,202 | 0.808–0.898 | 68.960 ms | 1.756 J |

"Points" is measured / feasible according to the planner. Efficiency uses the achievable 265.1 GB/s ceiling and the median step time, as in M2. The bootstrap CIs are in the raw records.

### All three repeats

Points measured, with KV cache blocks allocated in brackets:

| Model | KV dtype | Repeat 1 | Repeat 2 | Repeat 3 | b1, c512 step time (repeats 1 / 2 / 3) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-1.7B | bf16 | 22 (13,296) | 22 (12,883) | 22 (13,179) | 19.353 / 20.653 / 19.851 ms |
| Qwen3-1.7B | fp8 | 24 (24,881) | 24 (25,073) | 24 (26,650) | 19.510 / 19.103 / 19.712 ms |
| Qwen3-4B | bf16 | 19 (7,934) | 19 (8,219) | **20** (8,273) | 42.040 / 42.450 / 42.715 ms |
| Qwen3-4B | fp8 | 22 (15,825) | 22 (16,189) | 22 (16,127) | 42.096 / 42.281 / 42.549 ms |
| Qwen3-8B | bf16 | 16 (3,840) | 16 (3,912) | **18** (4,197) | 69.322 / 68.446 / 68.461 ms |
| Qwen3-8B | fp8 | 22 (9,202) | **19** (8,147) | **21** (8,357) | 68.960 / 68.127 / 68.205 ms |

**Coverage.** 122 of the 128 planned feasible points have all 3 repeats. The other 6, plus one point the planner did not expect to fit, sit at the edge of what fits: they need just more KV cache blocks than vLLM allocated in some runs, but fewer than in others.

| Point (model, KV dtype, batch, context) | Repeats | Blocks needed |
| --- | --- | --- |
| Qwen3-4B bf16, 4, 32768 | 1 (not planned as feasible) | 8,228 |
| Qwen3-8B bf16, 2, 32768 | 1 | 4,114 |
| Qwen3-8B bf16, 8, 8192 | 1 | 4,168 |
| Qwen3-8B bf16, 32, 2048 | **0** (planned, never fit) | 4,384 |
| Qwen3-8B fp8, 64, 2048 | 1 | 8,768 |
| Qwen3-8B fp8, 16, 8192 | 2 | 8,336 |
| Qwen3-8B fp8, 4, 32768 | 2 | 8,228 |

### Notes

- **Sanity checks pass on all 374 records.** Efficiency ranges from 0.641 to 0.934, so no point beats the ceiling and none falls below 20% of it. Every record has `has_uncommitted_changes: false`. Each record lists one other GPU process at start, the sweep's own vLLM engine subprocess. Idle GPU power with the model loaded was 10.3–13.7 W across sweeps.
- **KV cache capacity is not fixed.** The planner's estimate was 18% too high for Qwen3-8B bf16 and 0.3–7% too low for most other sweeps. vLLM's allocation also varies between runs with identical settings, by up to 13% (Qwen3-8B fp8: 8,147 to 9,202 blocks). That variation is what causes the uneven coverage above.
- **M2's point is within run-to-run variance.** Qwen3-1.7B bf16 at batch 1, context 512 measured 19.353–20.653 ms over the three repeats, a 6.7% spread, and M2's 20.433 ms falls inside it. The other sweeps vary by 1.1–3.2% at this point. M4 should report the spread across repeats, not only the CI within one.
- **Commit labels.** The code is identical from `40d8afd` onward. Later records name `dddccd6` (docs only) or `320f285` (results only), whichever was HEAD when that sweep started.
