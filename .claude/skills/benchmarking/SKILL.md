---
name: benchmarking
description: Measures latency, throughput, memory and energy of inference code and kernels, reproducibly. Use before any optimization, for any claim that something is faster, smaller or cheaper, and when comparing configurations, hardware or engines.
---

# Benchmarking

"It feels faster" is not a result. A benchmark is an experiment: a stated question, a controlled setup, enough repetitions to see the noise, and raw numbers someone else can reproduce.

## Profile before you optimize

1. Measure the baseline end to end.
2. Profile it (`torch.profiler` with CUDA activities, or `nsys profile` for the full timeline) and find where the time actually goes: a kernel, a host-side gap, a sync, a copy.
3. Optimize only that.
4. Measure again with the same setup. Keep the change only if the improvement is larger than the run-to-run noise.

Put the profile summary (top ops by CUDA time) in the commit message or results write-up.

## Timing GPU code correctly

GPU work is asynchronous, so wall-clock timing around a launch measures nothing.

- **Warm up** first (at least 10 iterations, or until timings are stable) to absorb compilation, autotuning, allocator growth and clock ramp-up.
- Time with **CUDA events**, or call `torch.cuda.synchronize()` before reading the clock.
- Run **many iterations** and report the median and p99, not the mean of one run, not the minimum.
- Repeat the whole measurement **at least 3 times** and report the spread. A difference smaller than the spread is noise.
- For microbenchmarks, use `triton.testing.do_bench` or `torch.utils.benchmark.Timer`; they handle warmup and cache flushing.
- Keep the machine quiet: no other GPU jobs, fixed clocks if you can set them, and note it if you can't.

## Serving metrics

For end-to-end serving, report the standard metrics, with these names:

| Metric | Meaning |
| --- | --- |
| TTFT | Time to first token (prefill latency as the user sees it) |
| TPOT / ITL | Time per output token / inter-token latency (decode speed) |
| E2E latency | Request start to last token |
| Output throughput | Generated tokens per second across all requests |
| Request throughput | Completed requests per second |
| Goodput | Requests per second that met the latency targets (state the targets) |

- Report p50, p90 and p99 for every latency, not just the mean.
- Sweep the request rate (or concurrency) and plot latency against throughput. A single operating point hides the knee of the curve.
- Fix the workload: dataset, input and output length distributions, number of requests, random seed. Use `vllm bench serve` (or the same methodology) so numbers are comparable with published ones.

## Memory and energy

- Record peak GPU memory (`torch.cuda.max_memory_allocated()`, reset before each run) and the KV cache capacity it leaves, in tokens.
- For energy, sample board power with NVML (`pynvml`) on a background thread at a fixed interval (10 to 100 ms) during the measured window only. Integrate power over time to get joules.
- Report energy per output token (J/token) and average power (W) alongside throughput. Faster is not always cheaper: higher batch sizes often raise power while lowering J/token.
- Subtract or report idle power separately, and say which you did.

## Record everything

Every run writes one JSON file to `benchmarks/results/` with:

- **Question**: one sentence on what this run is meant to answer.
- **Hardware**: GPU model and count, interconnect, CPU, power limit, clock settings.
- **Software**: driver, CUDA, PyTorch, vLLM (commit SHA if built from source), Triton, Python, this repo's commit SHA.
- **Config**: model, dtype, quantization, tensor-parallel size, max batch size, every non-default engine argument.
- **Workload**: dataset, length distributions, request rate, number of requests, seed.
- **Raw samples**, not only summaries, so anyone can recompute percentiles.

A result without its setup can't be trusted or reproduced, so it doesn't count.

## Reporting

Lead with a table someone can read in ten seconds, comparing against the baseline (the format below is an example; its numbers are made up):

```
question: does FP8 KV cache cut J/token on L4 without hurting TPOT?
model: Qwen3-8B  gpu: 1x L4 (72 W cap)  vllm: <sha>  rate: 4 req/s  n=500  repeats=3

                      baseline (bf16 KV)   fp8 KV        delta
output tokens/s       412 ± 6              463 ± 5       +12.4%
TTFT p50 / p99 (ms)   180 / 611            176 / 590     -2% / -3%
TPOT p50 / p99 (ms)   38.1 / 52.0          36.9 / 49.8   -3% / -4%
peak memory (GiB)     21.3                 21.4          +0.5%
KV capacity (tokens)  41k                  82k           +100%
energy (J/token)      0.162 ± 0.003        0.147 ± 0.002 -9.3%
```

Then: a latency-versus-throughput plot from the sweep, the profile summary, and a short paragraph on what the numbers mean and what they don't (other GPUs, other models, other lengths).

## Honesty

- Compare against the strongest reasonable baseline, configured well, not a strawman.
- Report regressions in the same table as wins.
- Never drop runs because they look bad. If a run is invalid (thermal throttling, another job on the GPU), say so and rerun all variants.

## Checklist

- [ ] The question is stated before the run.
- [ ] Profiled before optimizing; the optimized op was the measured bottleneck.
- [ ] Warmup, synchronized timing, many iterations, at least 3 repeats.
- [ ] p50/p90/p99 reported, with spread.
- [ ] Hardware, software versions, config and workload recorded in a results JSON with raw samples.
- [ ] Energy (J/token) and peak memory reported when the change could affect them.
- [ ] Compared against a strong baseline, regressions included.
