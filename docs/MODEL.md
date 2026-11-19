# The roofline model

How `src/decode_sol/roofline.py` predicts the fastest possible decode step, and what it leaves out.

## The ceiling

In one decode step, every sequence in the batch produces one token. The step must read every weight the forward pass touches, plus each sequence's K and V for every token already in its cache. On the GB10, that traffic is far larger than anything else the step does. If the memory system delivered bandwidth `BW` with no other cost:

```
step_time_min = (weight_bytes + kv_bytes) / BW
tokens_per_s  = batch_size / step_time_min
```

### Weight bytes

For a dense Qwen3 model, with `q_size = num_attention_heads * head_dim` and `kv_size = num_key_value_heads * head_dim`, each layer holds:

| Tensor | Parameters |
| --- | --- |
| `q_proj`, `o_proj` | `hidden_size * q_size` each |
| `k_proj`, `v_proj` | `hidden_size * kv_size` each |
| `gate_proj`, `up_proj`, `down_proj` | `hidden_size * intermediate_size` each |
| `input_layernorm`, `post_attention_layernorm` | `hidden_size` each |
| `q_norm`, `k_norm` | `head_dim` each |

The step reads every layer, plus the final norm (`hidden_size`) and `lm_head` (`vocab_size * hidden_size`). It reads only `batch_size` rows of the input embedding, so the embedding table is excluded. When the embeddings are tied, the same table is read in full as `lm_head` and counted once. Either way, the count does not depend on `tie_word_embeddings`.

`weight_bytes` = that parameter count × 2 (bf16).

**Check against a real checkpoint.** For Qwen/Qwen3-0.6B (revision `c1899de`), the safetensors header lists 751,632,384 parameters. That total includes both `lm_head.weight` and `model.embed_tokens.weight` (155,582,464 each), even though the config sets `tie_word_embeddings: true`. Leaving out the embedding gives 596,049,920, which matches the formula exactly. `tests/test_roofline.py` checks this.

### KV bytes

```
kv_bytes = batch_size * context_len * num_layers * 2 * num_key_value_heads * head_dim * kv_element_bytes
```

The factor 2 counts K and V. `kv_element_bytes` is 2 for a bf16 KV cache and 1 for fp8. `context_len` is the number of tokens already in each sequence's cache. When measuring by differencing two output lengths O1 and O2 after a prompt of length C, the effective context is `C + (O1 + O2) / 2`.

All model dimensions are read from `config.json`. `parse_model_dims` refuses any config the formula does not cover: a model type other than `qwen3`, attention biases, or a `quantization_config`.

## Bandwidth

Every prediction is reported against two bandwidths:

- **Spec peak:** 273 GB/s (NVIDIA's LPDDR5x figure for the GB10).
- **Achievable:** the median of `read_buffer.sum()` over a 4 GiB bf16 buffer of seeded random values, timed with CUDA events: 10 warmup and 50 timed iterations, repeated 3 times (`python -m decode_sol.bandwidth`). Raw times are in `results/raw/bandwidth.jsonl`.

The benchmark is read-only because decode traffic is almost entirely reads. The buffer is 4 GiB, far larger than L2. Its values are random so that memory compression cannot shrink the bytes read. An all-ones buffer and a random buffer measured the same bandwidth, so compression had no visible effect.

GB means 10^9 bytes throughout.

## What the model leaves out

These make the prediction a lower bound on step time, and so an upper bound on tokens/s:

- **Activations and logits.** Hidden states, attention scores, and the logits written for sampling (`batch_size * vocab_size` values per step) are ignored. With a vocabulary of 151,936 and fp32 logits, that is about 0.6 MB per sequence per step, so it grows with batch size.
- **Embedding rows.** `batch_size * hidden_size * 2` bytes, negligible.
- **Compute.** The model assumes decode stays memory-bound. At large batch sizes the matmuls approach the compute roofline, and the model stops applying.
- **Imperfect reuse.** The model assumes each weight is read once per step for the whole batch. Real kernels may re-read tiles.
- **Host overhead.** Scheduling, sampling, Python, and kernel-launch gaps that CUDA graphs don't hide.
- **Shared memory bus.** CPU and GPU share the LPDDR5x. CPU-heavy jobs from other users can lower the bandwidth the GPU gets, so this is a source of noise between runs.
- **Weight precision.** Only bf16 weights are modeled. FP8 checkpoints keep some tensors (norms, possibly `lm_head`) in bf16 and add scale tensors, so they need their own byte count before they are added.

## Energy

`src/decode_sol/energy.py` differences the NVML total-energy counter (`nvmlDeviceGetTotalEnergyConsumption`, millijoules) around a timed region. It covers the **GPU only**. On this machine the counter advances every 100 ms, so each end of a window can be stale by up to 100 ms. Windows therefore need to last several seconds to keep that error small. Idle power is measured over 30 s with `python -m decode_sol.energy` and recorded in `results/raw/idle_power.jsonl`.

## Measuring decode

`src/decode_sol/measure.py` loads one model in vLLM's offline `LLM` and generates the same batch of exact-length random prompts (token ids below the tokenizer's `vocab_size`, so no special tokens) at 16 and then 144 output tokens, alternately: 2 warmup pairs, then 10 timed pairs. Sampling is greedy with `ignore_eos=True`, and every output is checked to have exactly the requested length. Prefix caching is off. Detokenization is left on, as in normal serving, so its CPU cost is part of the measured step time.

Both generations of a pair pay the same prefill, so `src/decode_sol/analyze.py` takes the decode step time as `(long_duration - short_duration) / 128` and the GPU energy per token as `(long_energy - short_energy) / (batch_size * 128)`. Each grid point pools the pairs of every repeat of the sweep. Its median comes with a 95% confidence interval from a hierarchical bootstrap: each resample draws whole repeats with replacement, then pairs within each drawn repeat. Differences between repeats (the KV cache allocation, for one) are larger than the spread within a run, so a bootstrap over the pooled pairs alone would give intervals that are too narrow. A point measured in only one repeat gets a plain percentile bootstrap over its pairs.

The idle baseline for decode energy is measured over 30 s **with the model loaded**, just before timing. On the GB10 this reads about 10.6 W, roughly twice the 4.8 W of a GPU with no process on it, so "above idle" figures depend on which baseline is used. Every record stores its own baseline, and energy is reported both with and without it.

## Sweeping the grid

`src/decode_sol/grid.py` defines the grid and decides which points can be measured. A sweep (`scripts/run_sweep.sh`) is one model at one KV cache dtype, in one process, so the model loads once. The process is sized for the whole grid: `max_num_seqs` is the largest batch size and `max_model_len` the longest context plus 144.

A point is skipped, with the reason logged, when either:

- its batch needs more KV cache blocks than vLLM reported at startup (`batch_size * ceil((context_len + 144) / 16)`), because vLLM would then run the batch as several smaller ones instead of failing; or
- its prefill takes more than 16 steps. vLLM prefills a batch in chunks of `max_num_batched_tokens` and starts decoding each sequence as soon as its own prefill is done. The two generations of a pair share the same prefill phase, including the staggered starts, only if no sequence finishes its 16-token short generation before the last sequence starts decoding. `max_num_batched_tokens` is set to 32768 so that every point that fits in the cache also meets this condition, capped at `max_num_seqs * max_model_len` because vLLM's kernel warmup crashes with an illegal memory access when a prefill step could hold more tokens than its request buffers (seen on the one-point smoke grid).

`python -m decode_sol.grid` prints this plan before a sweep, with an estimate of the GPU minutes. The estimate assumes decode at 0.65 of the achievable ceiling (M2's efficiency), prefill at 50 TFLOP/s, 6 GiB of non-KV memory reserved by vLLM, and 3 minutes of startup per process. It is a planning number; the real capacity and times come from the run.

At batch 1, the short generation lasts about 0.35 s, only a few ticks of the 100 ms energy counter, so the energy of a single pair is uncertain by roughly 10%. The median over 10 pairs narrows this to a few percent.
