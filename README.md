# decode-sol-gb10

How close does vLLM get to the speed of light for LLM decode on an NVIDIA GB10?

During decode, every step reads all model weights plus the KV cache from memory, so memory bandwidth sets a hard ceiling on tokens per second. This project derives that ceiling from first principles for each model, batch size, context length and precision. It then measures what vLLM actually achieves and uses profiler traces to explain the gap. GPU energy per token is reported alongside speed.

**Status: work in progress.** No results yet.

## Questions

1. What fraction of the memory-bandwidth ceiling does vLLM reach during decode, across model size, batch size, context length and precision?
2. Where does the remaining time go, according to profiler traces?
3. How does GPU energy per token change across the same settings?
4. Can a small statistical model predict vLLM's efficiency from those settings, with honest uncertainty?

## Setup

- NVIDIA GB10 (sm_121) with 128 GB unified LPDDR5x memory shared by CPU and GPU, 273 GB/s peak bandwidth (spec)
- vLLM offline API, Qwen3 1.7B / 4B / 8B in bf16 and FP8
- Energy comes from the NVML GPU energy counter only. This machine exposes no CPU or memory power sensors, so every energy figure is **GPU energy**.

## Plan

### Method

1. **The ceiling.** For batch size B and context length C, one decode step must read every weight plus the KV cache:
   `step_time_min = (weight_bytes + kv_bytes) / bandwidth`, and `tokens/s = B / step_time`.
   Model dimensions come from each model's `config.json`. Each prediction is reported against two bandwidths: the 273 GB/s spec peak and the achievable bandwidth from a read-only microbenchmark.
2. **Measured decode.** vLLM's offline API with exact-length random prompts, greedy sampling, `ignore_eos` and prefix caching off. Each batch is timed at two output lengths, and the difference gives the decode step time with prefill removed. Each point gets at least 5 timed iterations, reported as the median with a bootstrap 95% CI.
3. **Energy.** The NVML energy counter is read around each timed region and differenced in the same way as time. Every result is reported both with and without the idle baseline subtracted.
4. **The gap.** Efficiency is predicted step time divided by measured step time. The points with the largest and smallest gaps are profiled with the PyTorch profiler and Nsight Systems. A regression model with bootstrap CIs predicts efficiency from batch size, context length, model size and dtype.

### Sweep grid

| Dimension | Values |
| --- | --- |
| Model | Qwen3-1.7B, 4B, 8B (bf16), plus FP8 where an official checkpoint exists |
| Batch size | 1, 2, 4, 8, 16, 32, 64 |
| Context length | 512, 2048, 8192, 32768 (points that don't fit are skipped and logged) |
| KV cache dtype | bf16, fp8 |
| Repeats | 3 independent repeats of the full grid |

### Milestones

Each milestone ends at a checkpoint, where work stops for review.

| | Milestone | Checkpoint |
| --- | --- | --- |
| M0 | Environment: venv, vLLM for CUDA 13 on aarch64 | Versions, workarounds, one short generation |
| M1 | Instruments: roofline model, bandwidth benchmark, energy meter, shared-machine guard | Achievable bandwidth vs. 273 GB/s, idle power |
| M2 | First point: Qwen3-1.7B, batch 1, context 512, end to end | Measurement vs. prediction looks sane |
| M3 | Full sweep, one model at a time | Feasible grid and GPU-minute estimate approved first |
| M4 | Analysis: summary table, plots, profiles, gap model | Plots and the five most interesting findings |
| M5 | Write-up in `docs/REPORT.md`, plus a draft vLLM issue proposing energy reporting in `vllm bench` | Draft ready for human rewrite |

### Ground rules

This runs on a shared teaching GPU, so vLLM is capped at `gpu_memory_utilization=0.25`. Every run is guarded by a timeout and a check for other GPU users, and starts only while a human is watching. Every number in the report traces to a raw, append-only record in `results/`.

## Documents

- [`docs/PROJECT_BRIEF.md`](docs/PROJECT_BRIEF.md): full method, sweep grid and milestones
- [`docs/SETUP.md`](docs/SETUP.md): exact install steps and workarounds
- [`docs/MODEL.md`](docs/MODEL.md): roofline derivation and assumptions
- `docs/REPORT.md`: write-up of the results (coming in milestone M5)
