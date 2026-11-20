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

## M4: Analysis

**Commits:** `00393c8` (summary table), `e6bffb8` (plots and gap model), `4b86f04` (`--points` option for top-ups), `e306709` (profiling), `83b4b43` (top-up results), `b03c96a` (lint fix), `714fd95` (batch-1 terms in the gap model), `6f26e93` (physical gap model)

- `analyze.py` pools the repeats of each grid point into `results/summary.csv`: median step time, efficiency and GPU energy per token, each with a 95% CI from a bootstrap that resamples repeats, then the samples within them.
- `plots.py` draws `plots/measured_vs_predicted_tokens_per_s.png`, `plots/efficiency_heatmaps.png` and `plots/gpu_energy_per_token_vs_batch.png`.
- `gap_model.py` fits step time as the bytes of each kind of read over that read's own bandwidth, plus fixed time (`results/gap_model.json`). It replaced a linear model of efficiency.
- `profile_summary.py` reads Nsight Systems profiles and writes the top kernels and GPU idle time per decode step (`results/profile_summary.json`).

### Coverage after the top-up

A top-up run on 2026-11-19 and 20 measured only the 7 short points, through `--points`. It added 9 records, so there are now **383 records over 128 grid points**. 127 points have 3 repeats. Qwen3-8B fp8 at batch 64, context 2048 has 2: on two of its three attempts vLLM allocated 8,506 and 8,672 KV cache blocks, short of the 8,768 it needs. Qwen3-8B bf16 at batch 32, context 2048 never fit.

At every topped-up point the old repeat was the slowest, by 2.6–4.5%. The median spread across repeats elsewhere is 1.6%. The CIs of these points widened to match.

### Findings

1. **The gap is at batch 1, and it is the matrix-vector kernel.** Efficiency is 0.667–0.849 at batch 1 (24 points) and 0.811–0.929 at batch 2 and above (104 points). At batch 1, cuBLAS `internal::gemvx::kernel` takes 92–97% of GPU busy time and reads the weights at about 183 GB/s (Qwen3-1.7B) and 194 GB/s (Qwen3-4B), 69–73% of the achievable 265.1 GB/s. At batch 2, `cutlass::Kernel2` reads the same Qwen3-4B weights at about 238 GB/s (90%). So a batch 2 step is faster than a batch 1 step: 19.85 → 16.24 ms for Qwen3-1.7B, 42.45 → 34.23 ms for Qwen3-4B, 68.47 → 64.25 ms for Qwen3-8B (bf16 KV cache, context 512). Qwen3-8B reaches 0.84 at batch 1 because the same kernel reads its larger matrices faster: 225 GB/s (85%), 98% of GPU busy time. The gap model, fitted before that profile was taken, predicted 225.3 GB/s.
2. **CPU overhead is not the cause.** The GPU idles 0.24–0.33 ms per decode step in all eight profiles, 0.3–1.4% of the step. CUDA graphs keep it fed. The profiled step times agree with the measured ones within 1.5%, except Qwen3-4B bf16 at batch 2, context 512 (35.58 vs 34.23 ms, 3.9%) and Qwen3-4B fp8 at batch 4, context 32768 (73.62 vs 76.38 ms, 3.6%; see finding 4).
3. **The smallest gaps are at long context, where attention reads most of the bytes.** Qwen3-4B bf16 at batch 2, context 32768 (0.923): `flash::flash_fwd_splitkv_kernel` takes 53% of GPU busy time and reads the KV cache at about 253 GB/s (95%). The matmuls take 46% at about 243 GB/s. The highest efficiency after the top-up is 0.929, Qwen3-4B bf16 at batch 4, context 32768.
4. **An fp8 KV cache is faster but less efficient.** It reads half the KV bytes, so steps are up to 37% shorter (Qwen3-1.7B, batch 64, context 2048: 79.8 → 50.3 ms; 31% at Qwen3-4B, batch 4, context 32768: 111.3 → 76.4 ms). But efficiency falls (0.929 → 0.876 at the Qwen3-4B point), and the gap model fits fp8 attention at 227.7 GB/s against 245.8 GB/s for bf16. With an fp8 KV cache, attention runs FlashInfer's paged kernel (`kernel_mha` in Nsight Systems) instead of FlashAttention's split-KV kernel. In the profile of that point it reads the KV cache at about 245 GB/s, close to bf16's 253 GB/s, so the profiled step implies an efficiency of 0.91. The measured step is 3.6% slower than the profiled one in all three repeats (fastest 76.31 ms), and that is where most of fp8's lower efficiency comes from. The torch profile agrees with Nsight Systems. *Corrected in M5:* later decode steps are not what is slower. In a fresh process the measurement agrees with the profile; the point was measured about 69 minutes into its sweep, and long runs get slower. See "The gap" in `docs/REPORT.md` and `results/raw/fp8_gap_check.jsonl`.
5. **Batching cuts GPU energy per token 20–37×, but energy is far noisier than time.** At context 512, from batch 1 to 64, energy above idle falls from 0.443 to 0.022 J for Qwen3-1.7B bf16 (20×) and from 1.853 to 0.062 J for Qwen3-8B bf16 (30×). Including idle power, the drop is 24–37× across the six sweeps. Across repeats of the same point, the median step time varies by 1.6% at the median point (6.5% at most). GPU energy above idle varies by 7% (21% at the 90th percentile, 54% at most). Qwen3-1.7B bf16 at batch 1, context 512: 0.323, 0.444 and 0.469 J above idle, while the step time stayed between 19 and 21 ms.

**Gap model.** Step time = weight bytes / matmul bandwidth + KV bytes / attention bandwidth + fixed time + batch size × weight GB × time per sequence per weight GB. Fitted to 128 points by least squares on relative error, it predicts efficiency with R² = 0.958 (leave-one-out 0.952), residual std 0.013. CIs are 95% bootstraps over points.

| Term | Fit (95% CI) | Profiles |
| --- | --- | --- |
| Batch-1 matmul bandwidth, Qwen3-1.7B | 186.1 GB/s (182.8–189.0) | about 183 GB/s |
| Batch-1 matmul bandwidth, Qwen3-4B | 195.4 GB/s (194.2–196.6) | about 194 GB/s |
| Batch-1 matmul bandwidth, Qwen3-8B | 225.3 GB/s (224.2–226.3) | 225 GB/s |
| Batch ≥ 2 matmul bandwidth | 244.0 GB/s (242.5–245.7) | about 238–243 GB/s |
| bf16 attention bandwidth | 245.8 GB/s (242.9–248.5) | about 253 GB/s |
| fp8 attention bandwidth | 227.7 GB/s (224.8–230.3) | about 245 GB/s (finding 4) |
| Fixed time | 1.09 ms (0.86–1.31) | GPU idle 0.24–0.33 ms; the rest is small kernels |
| Time per sequence per weight GB | 0.0043 ms (0.0020–0.0060) | not profiled |

The last term grows with batch size × weights, the shape of the matmul arithmetic, and costs Qwen3-8B 4.2 ms of its 93.7 ms step at batch 64. The profiles don't show yet that it is arithmetic. The linear model of efficiency this replaced had R² = 0.868 with batch-1 terms (`results/gap_model.json` at `714fd95`) and 0.327 without, fitted before the top-up (at `e6bffb8`).

### Notes

- **Profiles.** Nsight Systems and torch profiles of 8 points: Qwen3-1.7B bf16 at batch 1, context 512 and 2048; Qwen3-1.7B fp8 at batch 1, context 512; Qwen3-4B bf16 at batch 1 and 2, context 512, and at batch 2, context 32768; Qwen3-4B fp8 at batch 4, context 32768; Qwen3-8B bf16 at batch 1, context 512. The reports are in `profiles/`, which git ignores. Only the summary is committed.
- **Profiles start at the prefill.** vLLM 0.33 starts profiling at the prefill despite `delay_iterations`, so each profile holds the prefill and fewer decode steps than asked. `profile_summary.py` keeps only decode steps: 18 per profile, 16 and 14 at context 32768, where the prefill takes more of the profiled engine steps.
- **Kernel time per step comes from the GPU clock.** The LM head and sampling launch after vLLM's NVTX range for the step closes. So a step runs from its first kernel to the next step's first kernel, not to the end of the range.
- **Commit labels.** The first top-up record names `52e8a33`. That commit was later rewritten as `e306709` to change only its message, and the code is identical. All top-up records have `has_uncommitted_changes: false`.
- **Sanity checks still pass.** Efficiency is 0.667–0.929, so no point beats the ceiling.
