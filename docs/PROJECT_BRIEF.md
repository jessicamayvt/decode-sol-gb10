# Project Brief: Speed of Light for LLM Decode on the GB10

## Goal

Predict the fastest possible LLM decode speed on an NVIDIA GB10 from first principles, measure what vLLM actually achieves, and explain the gap. Report GPU energy per token alongside speed.

The finished project answers four questions:

1. What fraction of the memory-bandwidth ceiling does vLLM reach during decode, across model size, batch size, context length and precision?
2. Where does the remaining time go, according to profiler traces?
3. How does GPU energy per token change across the same settings?
4. Can a small statistical model predict vLLM's efficiency from those settings, with honest uncertainty?

Audience: inference engineers (for example at Inferact, the company behind vLLM). The write-up must hold up to an expert reader.

## The machine

Everything below was measured on the machine on 2026-11-12. Re-check anything that matters before relying on it.

| Fact | Value |
| --- | --- |
| Host | `blackwell.rlogin`, reached through `glogin.cs.vt.edu`. It is the only GPU node, and it is shared with other students. |
| GPU | NVIDIA GB10, compute capability 12.1 (sm_121), one GPU |
| CPU | 20 Arm cores (aarch64) |
| Memory | 128 GB unified LPDDR5x shared by CPU and GPU, about 121.6 GiB visible to PyTorch, 273 GB/s peak bandwidth (spec) |
| Software | Ubuntu 24.04, driver 580.178.04, CUDA 13.0 at `/usr/local/cuda`, Python 3.12.3, PyTorch 2.14.1+cu130 works |
| Energy | NVML total energy counter works (`nvmlDeviceGetTotalEnergyConsumption`, millijoules). It covers the **GPU only**: no CPU, memory or module power sensors are exposed. `power.draw` updates too slowly to integrate. Idle draw is about 4.9 W. |
| Clocks | Readable but not settable (no root) |
| Containers | Not available: rootless podman fails with "cannot clone: Permission denied". Install everything with pip into a venv. |
| Scheduler | None. Processes run directly on the shared GPU. |
| Storage | Home directory is network storage with a **50 GB quota**. `/tmp` is the fast local NVMe (3.4 TB free) but is wiped on reboot and cleaned of files older than 30 days. `/scratch` is not writable. |
| Network | Hugging Face and PyPI are reachable |
| Root | None. Never use `sudo`. |

## Rules for using the shared machine

These come from the CS department's rlogin policies. Breaking them can get the account locked.

- Cap vLLM's memory: `gpu_memory_utilization` must never exceed **0.25** (about 30 GB). The code enforces this with a constant, and refuses higher values.
- Every GPU run is wrapped in a timeout (default 45 minutes per sweep) and is started only while the human is logged in and watching.
- Before every sweep, check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and the process list. If another user is running GPU work, stop and tell the human instead of starting.
- Never leave a vLLM process or server running after a run finishes.
- Model weights and caches go in `/tmp/$USER/hf` (`HF_HOME`). Code, results and plots go in the repo in the home directory. Keep the venv in the repo (`.venv`), and keep total home usage well under 50 GB.

## Approach

### 1. The ceiling (speed of light)

During decode, every step reads all model weights plus the KV cache from memory, and the GB10 is limited by memory bandwidth. For one decode step at batch size B and context length C:

```
weight_bytes   = bytes of every weight tensor read per step
                 (all layers plus lm_head; the input embedding only reads B rows,
                 so exclude it unless it is tied to lm_head)
kv_bytes       = B * C * num_layers * 2 * num_kv_heads * head_dim * bytes_per_kv_element
step_time_min  = (weight_bytes + kv_bytes) / bandwidth
tokens_per_s   = B / step_time
```

Use two bandwidth figures and report results against both:
- **Spec peak:** 273 GB/s.
- **Achievable:** measured by a read-only microbenchmark (for example, summing a multi-GB bf16 tensor on the GPU, timed with CUDA events, median of many iterations). This is the honest ceiling.

Read layer counts, head counts, head dimensions and tied-embedding flags from each model's `config.json`. Never hardcode them from memory. Document every assumption in `docs/MODEL.md`.

### 2. Measuring decode in isolation

Use vLLM's offline Python API (`vllm.LLM`) in one long-lived process per model, so that loading the model once serves many grid points.

- Prompts are exact-length random token ID lists (`TokensPrompt`), so the context length is controlled.
- `SamplingParams(temperature=0, max_tokens=N, ignore_eos=True)`.
- Disable prefix caching, so repeated prompts don't skip prefill.
- **Separate decode from prefill by differencing:** time the same batch at two output lengths, O1 and O2 (for example 16 and 144):
  `decode_step_time = (time(O2) - time(O1)) / (O2 - O1)`.
  The effective context for the KV term is `C + (O1 + O2) / 2`.
- Warm up before timing. Time at least 5 iterations per point, and report the median and a bootstrap 95% confidence interval.
- Record vLLM's reported KV cache capacity at startup. Skip grid points that won't fit, and log each skip with the reason.
- Cross-check one grid point against `vllm bench latency` with the same settings.

### 3. Energy

Read the NVML energy counter immediately before and after each timed region:
`gpu_energy_per_token_j = (energy_end_mj - energy_start_mj) / 1000 / tokens_generated`.
Measure idle power for 30 seconds before each sweep. Report energy both with and without the idle baseline subtracted, and say which is which. Use the same differencing as for time to get decode-only energy per token.

### 4. Sweep grid

Start small and widen only after the first results look sane.

| Dimension | Values |
| --- | --- |
| Model | Qwen3-1.7B, Qwen3-4B, Qwen3-8B (bf16), and FP8 variants where an official checkpoint exists. Verify names on Hugging Face. |
| Batch size | 1, 2, 4, 8, 16, 32, 64 |
| Context length | 512, 2048, 8192, 32768 (skip points that don't fit) |
| KV cache dtype | auto (bf16), fp8 |
| Repeats | 3 independent repeats of the full grid, at different times |

Before running, print the grid, the number of feasible points, and an estimate of total GPU minutes.

### 5. Explaining the gap

- **Efficiency** = predicted step time (achievable bandwidth) / measured step time.
- Profile the 2–3 grid points with the biggest gaps, and one with the smallest. Use the PyTorch profiler through vLLM's profiling support (check the current vLLM docs for how to enable it), and Nsight Systems if `nsys` exists (look under `/opt/nvidia`). Summarize the top kernels by GPU time and any CPU-side gaps between steps.
- Fit a model of efficiency against log2(batch), context length, model size and dtype. Start with linear regression with bootstrap confidence intervals. A Bayesian version is a welcome extension.

## Repository layout

```
src/decode_sol/
  roofline.py      # bytes per step, predicted step time and tokens/s
  bandwidth.py     # achievable-bandwidth microbenchmark
  energy.py        # NVML energy counter wrapper (context manager)
  guard.py         # shared-machine checks: memory cap, other GPU users, timeout
  measure.py       # one model, many grid points, writes raw JSONL
  grid.py          # sweep definition and feasibility filtering
  analyze.py       # summary tables, efficiency, gap model
  plots.py         # all figures
scripts/
  setup_env.sh     # creates .venv and installs pinned deps
  run_sweep.sh     # guarded entry point: checks, timeout, runs measure.py
results/raw/       # one JSONL line per measurement, with full metadata
results/summary.csv
plots/
docs/
  PROJECT_BRIEF.md # this file
  MODEL.md         # the roofline derivation and assumptions
  SETUP.md         # exact working install steps, including any workarounds
  REPORT.md        # draft write-up (the human rewrites the final version)
tests/
```

Every raw measurement line records: timestamp, git commit, hostname, driver, CUDA, torch and vLLM versions, model id and revision, dtype, KV dtype, batch, context, O1, O2, timings, energy readings, idle baseline, `gpu_memory_utilization`, vLLM's KV capacity, and the other GPU processes seen at start.

## Milestones and checkpoints

Stop at each checkpoint, report to the human, and wait for a go-ahead.

**M0 – Environment.**
1. Create `.venv`, then install vLLM for CUDA 13 on aarch64 (try `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130`, then the cu130 nightly index at `https://wheels.vllm.ai/nightly/cu130` if needed).
2. Verify that `import vllm` works, and that a 0.6B model generates text under the memory cap.

GB10 support is young, so expect problems:
- If Triton fails with a `ptxas` error about `sm_121a`, try pointing `TRITON_PTXAS_PATH` at `/usr/local/cuda/bin/ptxas`.
- If `torch.compile` or CUDA graphs fail, `enforce_eager=True` is a last resort. It changes performance a lot, so record it prominently.
- Write every workaround into `docs/SETUP.md`.

**Checkpoint:** report the installed versions, any workarounds, and output from one short generation.

**M1 – Instruments.**
1. Implement `roofline.py` with unit tests on hand-computed examples.
2. Implement `bandwidth.py` and `energy.py`, and run the bandwidth benchmark.
3. Implement `guard.py` with tests.

**Checkpoint:** report achievable bandwidth vs. 273 GB/s, and the idle power baseline.

**M2 – First point.** Measure one grid point (Qwen3-1.7B, batch 1, context 512) end to end, including energy, and compare it to the prediction. **Checkpoint:** if the measurement beats the ceiling, or falls below 20% of it, investigate before going further.

**M3 – Sweep.** Print the feasible grid and GPU-minute estimate. **Checkpoint:** wait for approval, then run the sweep model by model under `run_sweep.sh`.

**M4 – Analysis.** Build the summary table, plots (measured vs. predicted tokens/s, efficiency heatmaps, J/token vs. batch), profiles and the gap model. **Checkpoint:** show the plots and the five most interesting findings.

**M5 – Write-up.** Draft `docs/REPORT.md`:
- the question,
- the setup (with every caveat: GPU-only energy, shared machine, eager mode if used),
- the model,
- the results,
- the gap analysis,
- what would change on datacenter GPUs,
- how to reproduce it.

Also draft a short GitHub issue proposing energy reporting in `vllm bench`. Don't file it; the human will.

## Definition of done

- `pre-commit run -a` and `pytest -m "not gpu"` pass.
- `scripts/run_sweep.sh` reproduces the results from a clean checkout, given the steps in `docs/SETUP.md`.
- Every number in `docs/REPORT.md` traces to a file in `results/`.
- No measurement is faster than the spec-peak ceiling. Any measurement faster than the achievable ceiling is explained.
