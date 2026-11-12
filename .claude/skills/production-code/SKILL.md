---
name: production-code
description: Standards for writing production Python the way a senior engineer would. Use whenever you write or change code in this repository, before writing the first line.
---

# Production Code

Code is read far more than it is written. Optimize for the engineer who opens this file in six months with no context, while a benchmark is regressing.

## Shape of a function

- **One job.** If the honest name needs "and", split it.
- **Short.** Aim for under 40 lines. Past that, look for the named steps hiding inside.
- **Flat.** Nesting deeper than two levels means you need guard clauses or an extracted function.
- **Few parameters.** More than four, or several of the same type in a row, becomes a frozen dataclass or keyword-only arguments (`def f(*, block_size: int, ...)`).
- **No boolean flags** that switch behavior. `schedule(seqs, True, False)` is unreadable. Use two functions or an `Enum`.

```python
# Before
def step(seq):
    if seq is not None:
        if seq.status == "waiting":
            if allocator.num_free_blocks >= needed:
                ...  # 30 lines

# After
def try_schedule(seq: Sequence) -> bool:
    if seq.status is not SequenceStatus.WAITING:
        return False
    num_blocks_needed = compute_num_blocks(seq.num_tokens, block_size=BLOCK_SIZE)
    if allocator.num_free_blocks < num_blocks_needed:
        return False
    allocate_and_start(seq, num_blocks_needed)
    return True
```

## Plain over clever

Write the version a tired reader parses on the first pass.

- A plain `for` loop with named variables is the default for building collections.
- A comprehension is fine only for a single, simple mapping or filter that fits on one line.
- Nested comprehensions, comprehensions with both a filter and a transformation, `functools.reduce`, and chained conditional expressions become explicit loops.
- No `x if a else y if b else z`. Use `if`/`return` or a `match`.

```python
# Before
totals = {m: sum(r.energy_j for r in runs if r.model == m and r.ok) / sum(r.num_tokens for r in runs if r.model == m and r.ok) for m in {r.model for r in runs}}

# After
energy_j_by_model: dict[str, float] = defaultdict(float)
num_tokens_by_model: dict[str, int] = defaultdict(int)
for run in runs:
    if not run.ok:
        continue
    energy_j_by_model[run.model] += run.energy_j
    num_tokens_by_model[run.model] += run.num_tokens

joules_per_token_by_model: dict[str, float] = {}
for model, energy_j in energy_j_by_model.items():
    joules_per_token_by_model[model] = energy_j / num_tokens_by_model[model]
```

**Exception: tensor hot paths.** Never loop in Python over tokens, heads or batch elements when a tensor operation does the same thing. There, "plain" means vectorized operations with each intermediate named and shape-commented. See `tensor-code`.

## Types carry the invariants

- Type hints on every function signature and every non-obvious variable. `mypy` passes with no new `# type: ignore`.
- No `Any` except at a boundary that is parsed right away.
- Structured values are `@dataclass(frozen=True, slots=True)`, not dicts with string keys or tuples indexed by position.
- States are an `Enum`, not strings or ints. Every `match` over an enum or union ends with `assert_never`, so a new state breaks type-checking instead of production:

```python
match status:
    case SequenceStatus.WAITING:
        ...
    case SequenceStatus.RUNNING:
        ...
    case SequenceStatus.FINISHED:
        ...
    case _:
        assert_never(status)
```

- `Protocol` for interfaces with more than one implementation. No abstract base class with a single subclass.
- **Parse, don't validate.** Turn untrusted input (CLI args, config files, HTTP bodies, model configs from the hub) into a typed object once, at the edge, with pydantic or a dataclass constructor that raises. Code past the edge trusts the types.

## Errors

- **Fail fast and loud.** Never write a bare `except:` or `except Exception: pass`.
- Catch only what you can handle at that point. Otherwise let it propagate.
- Raise specific exceptions that name the problem (`OutOfBlocksError`) with the values needed to debug: `raise OutOfBlocksError(f"need {num_blocks_needed} blocks, {num_free} free")`.
- When re-raising, keep the cause: `raise ModelLoadError(path) from err`.
- `assert` is for internal invariants that indicate a bug, never for validating input (asserts disappear under `python -O`).

## Resources and side effects

- Use `with` for anything that must be released: files, locks, CUDA streams, profiler sessions, process pools.
- No work at import time. Importing a module never touches the GPU, the network, the filesystem or global state.
- No mutable default arguments. No module-level mutable globals; pass state explicitly or hold it in an object.
- Keep the core pure: scheduling decisions, metric math and config resolution take values and return values. I/O happens at the edges.

## Configuration

- Parse config once at startup into a frozen dataclass (from CLI args or a config file) and pass it down. Never read `os.environ` deep inside library code.
- Magic numbers become named constants with units: `MAX_BATCH_TOKENS = 8192`, `NVML_SAMPLE_INTERVAL_MS = 10`.

## Logging

- `logger = logging.getLogger(__name__)` per module. No `print` in library code.
- Log events with the values that matter: `logger.info("Allocated KV cache: num_blocks=%d size_gib=%.2f", num_blocks, size_gib)`.
- Use `%`-style arguments, not f-strings, so formatting is skipped when the level is disabled. In hot paths, don't log per token or per step at `INFO`.
- Never log tokens, API keys or full prompts.

## Concurrency

- In `asyncio` code, never call blocking functions (file I/O, `time.sleep`, GPU syncs) on the event loop. Every task is awaited or tracked; no fire-and-forget `create_task` without keeping a reference.
- Bound queues and concurrency. An unbounded queue is a memory leak waiting for traffic.

## Dependencies

Before adding a package: can the standard library, PyTorch, or an existing dependency do it? Is it maintained? Pin it in `pyproject.toml` and justify it in the commit message.

## Leave no residue

Before you finish, delete:
- Commented-out code and debug `print`s.
- Unused imports, variables, parameters and functions.
- `TODO`s without an issue link.

## Done means

- [ ] `pre-commit run -a` and `pytest -m "not gpu"` pass; `pytest -m gpu` passes if tensor code changed.
- [ ] The change is covered by a test that fails without it (see `testing`).
- [ ] Every name passes the `naming` checklist.
- [ ] No function over ~40 lines or nested deeper than two levels without a reason you can state.
- [ ] No Python loops over tensor elements in a hot path.
- [ ] Any performance claim has numbers from `benchmarking`.
- [ ] You reviewed your own diff with `code-review`.
