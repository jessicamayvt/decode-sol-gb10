# CLAUDE.md

decode-sol-gb10 measures how close vLLM gets to the memory-bandwidth speed of light for LLM decode on an NVIDIA GB10, and reports GPU energy per token. The full spec, the machine's facts and the milestones are in `docs/PROJECT_BRIEF.md`. Read it before starting any task.

## How to work

1. **Think before coding.** State your assumptions. If the task has two readings, ask instead of picking one.
2. **Simplicity first.** Write the least code that answers the question. No speculative options, no abstractions with a single caller.
3. **Surgical changes.** Every changed line traces to the task. If you notice an unrelated problem, mention it; don't fix it in the same change.
4. **Verify, then stop.** Every milestone in the brief ends at a checkpoint. Stop there, report, and wait for the human.

## Shared machine: non-negotiable

This runs on a shared teaching machine. Breaking these rules can get the account locked.

- Never use `sudo`, `podman` or `docker`.
- `gpu_memory_utilization` must never exceed 0.25. `src/decode_sol/guard.py` enforces this.
- Start GPU work only through `scripts/run_sweep.sh`, which checks for other GPU users and applies a timeout. Quick single-point tests may call `measure.py` directly, but only wrapped in `timeout 10m`.
- Never start a GPU run without telling the human first. Never leave a vLLM process running when a run ends.
- Model weights go in `/tmp/$USER/hf` (`export HF_HOME=/tmp/$USER/hf`). The home directory has a 50 GB quota.

## Stack

- Python 3.12, a venv at `.venv`, vLLM for CUDA 13 on aarch64, PyTorch cu130.
- `ruff` for lint and format, `mypy` for types, `pytest` for tests, `pre-commit` to run them.
- `nvidia-ml-py` for energy, `matplotlib` for plots, `numpy`/`pandas` for analysis.

## Skills

Load the matching skill before starting the work, not after.

| Task | Skill |
| --- | --- |
| Writing or changing any code | `production-code`, `naming` |
| Code that creates or transforms tensors, or runs on the GPU | `tensor-code` |
| Any measurement, sweep, or claim about speed or energy | `benchmarking` |
| Tests, or any bug fix | `testing` |
| Before every commit | `code-review` |

## Code map

| Path | What lives there |
| --- | --- |
| `src/decode_sol/roofline.py` | Bytes per decode step and predicted step time. Pure functions, no GPU. |
| `src/decode_sol/bandwidth.py` | Achievable-bandwidth microbenchmark |
| `src/decode_sol/energy.py` | NVML energy counter as a context manager |
| `src/decode_sol/guard.py` | Memory cap, other-GPU-user check, run metadata |
| `src/decode_sol/measure.py` | One model per process, many grid points, raw JSONL out |
| `src/decode_sol/grid.py` | Sweep definition and feasibility filter |
| `src/decode_sol/analyze.py` | Pools repeats into `results/summary.csv`: step time, efficiency, GPU energy |
| `src/decode_sol/gap_model.py` | Physical model of step time, one bandwidth per kind of read, fitted to `results/summary.csv` |
| `src/decode_sol/plots.py` | Every figure |
| `src/decode_sol/profile_summary.py` | Top kernels and GPU idle time per decode step, from nsys profiles |
| `src/decode_sol/profile_decode.py` | Profiles decode steps at one grid point (torch profiler or nsys) |
| `src/decode_sol/step_drift.py` | One grid point fresh, after scrambling the KV cache, over and over, or with vLLM's free KV cache blocks reordered, to separate a slow point from a slow process |
| `src/decode_sol/block_order.py` | Reads, sorts and shuffles vLLM's free KV cache block queue (engine in this process only) |
| `scripts/` | `setup_env.sh`, `run_sweep.sh`, `run_profile.sh` |
| `results/raw/` | Append-only measurement records. Never edit or delete them. |
| `docs/` | Brief, model derivation, setup notes, report draft |

## Commands

- `source .venv/bin/activate`
- `pre-commit run -a` must pass.
- `pytest -m "not gpu"` must pass. GPU tests are marked `@pytest.mark.gpu`.
- `bash scripts/run_sweep.sh --model <id> --grid <name>` for any measurement.

## Invariants

- Every number in a plot or the report comes from a file in `results/`.
- Results are append-only. A bad run is marked invalid with a reason, never deleted.
- Model dimensions come from `config.json`, never from memory.
- Energy is always labeled "GPU energy". This machine has no CPU or memory power sensors.
- Any workaround that changes performance (for example `enforce_eager`) is recorded in every affected result and in `docs/SETUP.md`.

## Writing code for this repo

The human reads every line. Prefer explicit `for` loops with named variables over comprehensions that do more than one simple step. See `production-code`.

## Commits

- Imperative subject under 72 characters, with the area in brackets: `[Measure]`, `[Model]`, `[Analysis]`, `[Setup]`, `[Docs]`.
- The body says why. Include numbers for anything that changes results.
- One logical change per commit. Commit at least at every checkpoint.
