# Draft issue for vllm-project/vllm (not filed)

**Title:** [Feature]: Report GPU energy per token in `vllm bench`

## Motivation

`vllm bench latency`, `throughput` and `serve` report how fast vLLM is, but not what it costs in energy. Energy per token is becoming a standard figure next to tokens/s, and today everyone who wants it writes their own NVML wrapper, with different choices about idle power, windows and repeats. Their numbers can't be compared.

NVIDIA GPUs expose a cumulative energy counter, `nvmlDeviceGetTotalEnergyConsumption` (millijoules since driver load, Volta and later). Reading it before and after the timed region is cheap, needs no sampling thread, and is already available through the copy of `pynvml` that vLLM vendors in `vllm/third_party/pynvml.py`.

## Proposal

Add `--measure-energy` to `vllm bench latency`, `throughput` and `serve`. When set:

1. Before the run, record idle power over a short window (for example 10 s) with the model loaded.
2. Read the energy counter on every GPU the engine uses, immediately before and after each timed iteration.
3. Report, alongside the existing metrics:
   - GPU energy per output token, with idle included and with idle subtracted, labeled as such;
   - average GPU power during the timed region;
   - the idle power used.
4. Write the raw counter readings to the `--output-json` file, so others can recompute.

Name the metric **GPU energy**: on most systems the counter does not include the CPU, host memory or the rest of the node.

## Pitfalls we hit

From measuring vLLM decode on an NVIDIA GB10 (Qwen3 1.7B–8B, 128 configurations, 3 repeats each):

- **The counter is coarse.** On the GB10 it advances every 100 ms, so a window shorter than a few seconds carries a large quantization error. The benchmark should warn when a timed iteration is short compared with the counter's update interval.
- **Energy varies between runs far more than time does.** Across repeats of the same configuration, decode step time varied by 1.6% at the median, but GPU energy per token above idle varied by 7% (54% at worst). Reporting a CI from iterations within one run would understate this, so the docs should recommend repeated runs.
- **Idle power depends on what is loaded.** With a model loaded, idle power was about 10–14 W against 4.8 W with no process on the GPU. Which baseline is subtracted changes "above idle" figures a lot, so the report should state it.
- **What the counter covers differs between devices.** On datacenter boards it covers the whole board, HBM included. On the GB10 it covers the GPU only, not the unified LPDDR5x memory. Results from different devices should not be compared without saying so.

## Scope

- NVIDIA through NVML first. AMD has a similar counter in `amdsmi`, which could follow.
- With tensor or pipeline parallelism, sum over the engine's GPUs and report the per-GPU values in the JSON.
- No change to the default behavior.

We have a working implementation of the measurement (counter wrapper, idle baseline, differencing of two output lengths to isolate decode) and would be happy to adapt it into a PR.
