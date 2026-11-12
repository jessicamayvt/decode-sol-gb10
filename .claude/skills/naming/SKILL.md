---
name: naming
description: Names functions, variables, classes, modules and tensors so Python code reads without comments. Use whenever you write or rename any identifier, and when you feel the urge to add a comment explaining what code does.
---

# Naming

The name says **what**. A comment, when one exists, says **why**. If a reader needs a comment to learn what a function does, the function is misnamed or doing too much.

## The rules

### 1. Follow PEP 8 casing, without exceptions

`snake_case` for functions, methods, variables and modules. `PascalCase` for classes. `UPPER_SNAKE_CASE` for module-level constants. A leading underscore for anything not part of the module's public API.

### 2. Name for the call site

Read the name where it is used, not where it is defined.

```python
# Definition looks fine, call site is vague
from .cache import update
update(cache, seq)

# Call site says exactly what happens
from .kv_cache import append_tokens
append_tokens(kv_cache, sequence)
```

### 3. Functions are verb + object, and the verb is honest

| Verb | Promise it makes |
| --- | --- |
| property / `get_` | Cheap, no side effects, no I/O, no GPU sync |
| `load_` / `fetch_` | I/O (disk, network, hub download); may fail, may be slow |
| `find_` | May return nothing (`-> X | None`) |
| `parse_` | Turns untrusted input into a typed value or raises |
| `compute_` | Pure computation, possibly expensive |
| `to_` / `from_` | Pure conversion (`to_dtype`, `from_config`) |
| `allocate_` / `free_` | Owns memory |
| `is_` / `has_` / `can_` / `should_` | Returns `bool`, no side effects |

A function named `get_block` that allocates a new block is a bug, even if it works. Pick one verb per concept and keep it everywhere.

A function either answers a question or changes something, not both. `check_and_evict` is two functions.

### 4. Counts, sizes and units go in the name

- Counts start with `num_`: `num_tokens`, `num_blocks`, `num_heads`. Never `tokens` for a count.
- Indices end in `_idx` or `_id`: `block_idx`, `seq_id`.
- Sizes say their unit: `block_size` (in tokens, defined once in the glossary), `size_bytes`, `max_model_len`.
- Durations and rates say their unit: `latency_ms`, `timeout_s`, `tokens_per_s`, `power_w`, `energy_j`.
- Collections are plural: `sequences`. Dicts are `value_by_key`: `block_by_id`, `latency_ms_by_batch_size`.
- Booleans are positive yes/no questions: `is_prefill`, `has_finished`. Never `not_ready` or `disable_cache` (write `use_cache`).

### 5. Tensors: the name is the meaning, the comment is the shape

A tensor's name says what it holds. Its shape and dtype are part of its contract, and the type system can't express them, so a shape comment is one of the few comments that always earns its place:

```python
query = qkv[:, :q_size]           # [num_tokens, num_heads * head_dim]
attn_weights = scores.softmax(-1) # [num_seqs, num_heads, q_len, kv_len], fp32
```

Use the same dimension names everywhere (`num_tokens`, `num_heads`, `head_dim`, `hidden_size`) and list them in `docs/CONTEXT.md`. Don't name tensors `x`, `out`, `tmp` or `t` outside of a three-line math helper.

### 6. Length scales with scope

- A loop variable that lives three lines: `i`, `seq`.
- A local in one function: one or two words.
- A public function or class: long enough to be unambiguous at every import site, no longer.

Drop words the context already provides. Inside `class BlockAllocator`, it's `allocate`, not `allocate_block`. A field on `Sequence` is `status`, not `seq_status`.

### 7. Banned words

These carry no information. Replace them with what the thing actually is.

`data`, `info`, `item`, `obj`, `val`, `res`, `result`, `ret`, `tmp`, `temp`, `stuff`, `manager`, `helper`, `util`, `handle`, `process`, `do`, `perform`, `run` (outside an entry point), `Base`, `New`, `2`, `final`.

No `utils.py` or `helpers.py`. Each helper lives next to the code that owns it, in a module named after its concept: `block_allocator.py`, `sampling.py`.

### 8. No type or implementation in the name

`seq_list`, `block_dict`, `str_name`, `load_model_async`, `cuda_tensor` (when callers don't care about the device). Types belong in type hints.

### 9. Classes and errors

- Classes are nouns: `BlockAllocator`, `SamplingParams`. Enum members are states: `SequenceStatus.RUNNING`.
- Exceptions end in `Error` and name the problem: `OutOfBlocksError`, not `CacheError2`.

## When you want to write a comment

Try these first, in order:

1. **Rename** the function or variable so the comment is redundant.
2. **Extract** the block into a function whose name is the comment.
3. **Introduce a variable** whose name explains a condition.

```python
# Before
# skip sequences that already hit the stop condition or the length cap
if seq.output_ids[-1] == eos_id or len(seq.output_ids) >= max_tokens:
    continue

# After
has_hit_eos = seq.output_ids[-1] == eos_token_id
has_hit_max_tokens = len(seq.output_ids) >= max_tokens
if has_hit_eos or has_hit_max_tokens:
    continue
```

Keep a comment only when it carries something the code cannot:
- **Why** a non-obvious choice was made: `# Pad to a multiple of 8 so the GEMM hits tensor cores.`
- A **constraint** from outside the code: a paper, a hardware limit, a bug in a dependency, with a link.
- **Shape and dtype** of a tensor (rule 5).
- The **contract** of a public function the signature can't express: units, what it raises, whether it syncs.

Delete comments that restate the code, narrate history ("changed this to fix X"), or are commented-out code.

## Checklist

- [ ] PEP 8 casing everywhere.
- [ ] Every function name states its effect, and the verb matches the table above.
- [ ] Counts use `num_`, and every number with a unit has the unit in its name.
- [ ] Tensors are named for their meaning and have a shape comment where they are created.
- [ ] No banned words.
- [ ] The same concept has the same name everywhere, and that name is in `docs/CONTEXT.md`.
- [ ] Every remaining comment explains why, a shape, or a contract.
