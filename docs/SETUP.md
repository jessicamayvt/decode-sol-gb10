# Setup

Exact steps that produced a working environment on `blackwell.rlogin` (NVIDIA GB10, aarch64) on 2026-11-12.

## Installed versions

| Component | Version |
| --- | --- |
| Python | 3.12.3 (system, `/usr/bin/python3.12`) |
| vLLM | 0.33.0 (PyPI wheel, aarch64) |
| PyTorch | 2.14.0+cu130 (pulled in by vLLM; the brief's 2.14.1 was not required) |
| Triton | 3.8.0 |
| CUDA (torch) | 13.0, arch list `sm_80 … sm_120`; sm_121 runs the sm_120 binaries |
| Driver | 580.178.04 |
| Attention backend | `FLASH_ATTN` (chosen by vLLM) |

## Steps

```bash
cd ~/personal_projects/decode-sol-gb10
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
export PIP_CACHE_DIR=/tmp/$USER/pip-cache    # keep the wheel cache off the 50 GB home quota
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu130
```

The venv is about 8.9 GB.

Then the package itself and the dev tools (versions pinned in `pyproject.toml`):

```bash
pip install --no-deps -e .
pip install ruff==0.17.5 mypy==2.4.1 pytest==9.2.0 pre-commit==4.7.0
pre-commit install
```

The pre-commit hooks are `local` hooks that run these tools from `.venv`, so pre-commit never builds its own environments.

## Workaround: Python headers for Triton

**Symptom.** The first `vllm.LLM(...)` fails during `torch.compile` with
`InductorError: CalledProcessError: Command '['/usr/bin/gcc', '.../triton/backends/nvidia/driver.c', ...]'`.
Running the gcc command by hand shows `fatal error: Python.h: No such file or directory`.

**Cause.** Triton compiles a small C launcher at runtime and needs the Python development headers. `python3.12-dev` is not installed on the machine, and we have no root.

**Fix.** Download the exact matching Ubuntu packages without root, unpack them inside the venv, and point gcc at them with `CPATH`:

```bash
mkdir -p /tmp/$USER/pydev && cd /tmp/$USER/pydev
apt-get download libpython3.12-dev python3.12-dev     # 3.12.3-1ubuntu0.17, matches the system Python
cd ~/personal_projects/decode-sol-gb10
for deb in /tmp/$USER/pydev/*.deb; do dpkg -x "$deb" .venv/python3.12-dev; done
```

Then, in every shell that runs vLLM:

```bash
export CPATH=$PWD/.venv/python3.12-dev/usr/include/python3.12:$PWD/.venv/python3.12-dev/usr/include
```

This does not change performance: `torch.compile` and CUDA graphs (piecewise and full) both run. `enforce_eager` is **not** used.

## Environment for every run

```bash
source .venv/bin/activate
export HF_HOME=/tmp/$USER/hf        # model weights on local NVMe, not home
export CPATH=...                    # see above
```

## Smoke test (M0)

Qwen/Qwen3-0.6B, `gpu_memory_utilization=0.25`, greedy, 2 prompts × 40 tokens: generation succeeded.
Engine startup took 122 s (first run, including 10 s of `torch.compile` and about 67 s of CUDA graph capture); later runs reuse the compile cache in `~/.cache/vllm`.
vLLM reported 26.67 GiB of KV cache memory (249,712 tokens) under the 0.25 cap.

## Known warnings (harmless so far)

- `FlashInfer's precompiled kernels are missing` — vLLM picked `FLASH_ATTN`, so FlashInfer is not used.
- `engine core exited unexpectedly; starting cleanup` at shutdown of the offline `LLM` — the process exits cleanly, and no GPU process is left behind.
